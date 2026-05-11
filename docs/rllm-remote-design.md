# 远程 NPU 训练设计方案

## 1. 概述

rllm 项目当前仅支持 Mac 本地训练（`rllm_train`，基于 TRL，CPU/MPS）。`rllm_remote` 是第三个训练后端，通过 SSH 连接远程华为昇腾 NPU 服务器，将训练任务提交到预先部署的 AgentSDK 容器中执行，并将结果取回本地——从而实现正式规模的 Agent RL 训练。

### 三个后端对比

| | rllm_train | rllm_verl | rllm_remote |
|---|---|---|---|
| 训练框架 | TRL GRPOTrainer | veRL | AgentSDK（veRL / MindSpeed RL） |
| 硬件 | Mac CPU/MPS | NVIDIA GPU（AutoDL） | 华为昇腾 NPU |
| 执行方式 | 本地进程 | 本地 subprocess | SSH + docker exec |
| 规模 | 单机 | 多 GPU | 多 NPU（单节点 8 卡起） |
| 配置格式 | TrainingConfig JSON | VerlTrainConfig + Hydra yaml | RemoteTrainConfig JSON → AgentSDK YAML |

## 2. 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                        Mac（本地）                            │
│                                                              │
│  rllm_remote/                  skill-bank/                   │
│  ┌──────────────┐    ┌──────────────────────────────┐       │
│  │ config.py    │    │ rllm-train（远程路由）          │       │
│  │ ssh.py       │    │ rllm-config（远程配置模式）     │       │
│  │ config_gen..py│   │ rllm-remote-run               │       │
│  │ train.py     │    │ rllm-remote-monitor           │       │
│  └──────┬───────┘    │ rllm-analyze（远程结果）       │       │
│         │            └──────────────────────────────┘       │
│         │ SSH + SCP                                         │
└─────────┼────────────────────────────────────────────────────┘
          │
  ┌───────┴────────────────────────────────────────────────────┐
  │               192.168.9.142（NPU 服务器）                  │
  │                                                            │
  │  ┌──────────────────────────────────────┐                 │
  │  │  容器: agent5.0.0_qjy                │                 │
  │  │                                      │                 │
  │  │  AgentSDK /home/work/AgentSDK/       │                 │
  │  │  ├── aura/run_start_in_local.sh       │                 │
  │  │  ├── aura/aura/start.py              │                 │
  │  │  ├── aura/configs/                   │                 │
  │  │  │   └── remote_<run_id>.yaml (上传)  │                 │
  │  │  └── aura/outputs/<run_id>/           │                 │
  │  │      └── training_log.txt            │                 │
  │  │                                      │                 │
  │  │  Ray → veRL/vLLM → NPU 训练          │                 │
  │  └──────────────────────────────────────┘                 │
  │                                                            │
  │  模型:  /opt/DPC/.../Qwen2.5-7B-Instruct                  │
  │  数据:  /opt/DPC/.../gsm8k/{train,test}.parquet           │
  └────────────────────────────────────────────────────────────┘
```

### 数据流

```
用户输入 → rllm-train Phase 0（识别 backend=remote）
  → rllm-clarify（需求澄清）
  → rllm-config remote 模式（生成 RemoteTrainConfig JSON）
  → rllm-remote-run（生成 AgentSDK YAML → 上传到 configs/ → tmux 启动 run_start_in_local.sh）
  → rllm-remote-monitor（SSH 日志流 → 异常检测 → 下载结果）
  → rllm-analyze（对已下载的本地结果进行分析）
  → 循环调参 或 训练完成
```

## 3. 模块设计

### 3.1 `rllm_remote/config.py` — RemoteTrainConfig

配置数据类，按逻辑分为六组：

**连接参数**：`ssh_host`、`ssh_port`、`ssh_user`、`ssh_key_path`、`container_name`。

**服务器路径**：`model_name_or_path`、`train_data_path`、`val_data_path`——指向服务器上已有的模型和训练数据。默认值基于服务器现有布局 `/opt/DPC/`。

**训练超参**：`num_epochs`、`train_batch_size`、`ppo_mini_batch_size`、`lr`、`kl_coef`、`temperature`、`n_samples_per_prompt`、`max_prompt_length`、`max_response_length`、`entropy_coeff`。这些参数与现有 `TrainingConfig` 字段一一对应或高度对应。

**NPU/集群**：`n_npus_per_node`、`nnodes`、`tensor_parallel_size`、`pipeline_parallel_size`、`gpu_memory_utilization`。

**框架选择**：`train_engine`（"verl" 或 "megatron"）、`cluster_mode`（"hybrid" 或 "one_step_off"）。

**Agent 配置**：`agent_name`、`max_agent_steps`、`n_parallel_agents`、Agent 采样参数。

**输出路径**：`remote_output_dir`、`local_output_dir`、`run_id`。

序列化方法（`to_json` / `from_json`）与 `TrainingConfig` 保持相同契约，确保技能管线可无缝切换后端。

### 3.2 `rllm_remote/ssh.py` — RemoteExecutor

零外部依赖的 SSH 通信层，使用标准库 `subprocess.run(["ssh", ...])` 实现。所有命令通过 `docker exec <容器名>` 在远程容器内执行。

核心方法：

| 方法 | 用途 |
|---|---|
| `run(cmd)` | 在容器内执行命令，返回 CompletedProcess |
| `is_container_running()` | 检查目标容器是否运行中 |
| `upload_file(local, remote)` | scp 到宿主机 → docker cp 到容器 |
| `download_file(remote, local)` | docker cp 出容器 → scp 到本地 |
| `tail_log(path, lines)` | 读取远程日志最后 N 行 |
| `stream_log(path)` | 通过 Popen 实时 tail -f 日志流 |
| `run_tmux(cmd, log, session)` | 在容器内启动分离的 tmux 会话 |
| `check_tmux_alive(session)` | 检查 tmux 会话是否存活 |
| `kill_tmux(session)` | 终止 tmux 会话 |
| `verify_connectivity()` | 全链路连通性检查（SSH、容器、模型、数据） |

### 3.3 `rllm_remote/config_generator.py` — YAML 配置生成

`generate_agentsdk_config(config: RemoteTrainConfig) -> dict` 生成完整的 AgentSDK Hydra YAML 配置。

**参考模板**：`AgentSDK/aura/configs/base_direct_1node_qwen25_7b_train_hybrid_with_verl_fsdp.yaml`

**选择 FSDP 而非 Megatron 的原因**：FSDP 路径更简洁（无需 Megatron 初始化），8 卡 NPU 足以支撑 7B 级模型，且参考配置已在容器环境中经过验证。

**生成的主要配置段**：
- `agentic_ai` — 运行模式（direct）、日志配置
- `hydra` — verl 配置搜索路径、默认模板（ppo_trainer）
- `verl_conf` — 算法（GRPO）、数据路径、Actor（FSDP 策略）、Rollout（vLLM）、Trainer（NPU 设备）
- `train_instances` — 训练任务定义（cluster_mode、train_engine）
- `agent_instances` — rllm 引擎 Math Agent 配置
- `infer_instances` — vLLM proxy 指向已有推理服务

所有动态参数从 `RemoteTrainConfig` 字段注入，生成的字典通过 `yaml.dump()` 写入 YAML 文件。

### 3.4 `rllm_remote/train.py` — CLI 入口

沿用 `rllm_verl/train.py` 的 argparse 模式。

**三种模式**：
- `--dry-run`：生成并打印 YAML 配置，不执行 SSH（用于本地验证）
- `--check`：检查 SSH 连通性、容器状态、模型/数据路径
- 正常模式：完整流程——验证连通 → 生成 YAML → 上传配置 → tmux 启动训练

**执行步骤**：
1. 从 CLI 参数或 JSON 文件构建 `RemoteTrainConfig`
2. 创建 `RemoteExecutor`，确认容器运行中
3. 生成 AgentSDK YAML 配置
4. 本地保存 YAML（留底）
5. 上传配置文件到容器内 `AgentSDK/aura/configs/remote_<run_id>.yaml`（必须放在 configs/ 目录下，因为 `run_start_in_local.sh` 硬编码了 `--config-path`，且 Hydra searchpath 相对路径依赖此位置）
6. 通过分离的 tmux 会话启动训练：`cd /home/work/AgentSDK/aura && bash run_start_in_local.sh --config-name remote_<run_id>`（复用现有启动脚本，自动处理 PYTHONPATH、Ray 启停等环境设置）
7. 输出报告：run_id、服务器、日志路径、tmux 会话名、手动登录命令

## 4. 技能系统集成

### 4.1 rllm-train 后端路由

对 `rllm-train` Phase 0 的补丁增加了远程/NPU 关键词检测。检测到后将 `backend=remote`，Phase 映射相应变化：

| Phase | 本地模式（默认） | 远程模式（backend=remote） |
|---|---|---|
| Phase 1（需求澄清） | rllm-clarify | rllm-clarify（不变） |
| Phase 2（配置生成） | rllm-config | rllm-config（传入 backend=remote） |
| Phase 3（启动训练） | rllm-run | **rllm-remote-run** |
| Phase 4（过程监控） | rllm-monitor | **rllm-remote-monitor** |
| Phase 5（结果分析） | rllm-analyze | rllm-analyze（传入 backend=remote） |

### 4.2 rllm-config 远程配置模式

当 `backend=remote` 时，rllm-config 生成 `RemoteTrainConfig` JSON 而非 `TrainingConfig`。应用 NPU 特定的安全参数范围（更大的 batch_size、更多的 epochs、更低的 lr）。

### 4.3 rllm-remote-monitor 远程监控

通过周期性 SSH 日志轮询监控训练。同时检测通用异常（reward 崩塌、loss 爆炸）和 NPU 特有错误（Ascend OOM、HCCL 超时、驱动错误、Ray Worker 断开）。

### 4.4 rllm-analyze 远程结果处理

分析前从远程服务器下载训练日志和 trajectory 文件。适配 AgentSDK verl 日志格式（指标名称与 rllm_train 不同，如 `reward` → `reward/train`、`loss` → `ppo_loss`）。

## 5. 部署

### 5.1 前置条件

- Mac 上已配置到 192.168.9.142 的 SSH 密钥认证
- 服务器上容器 `agent5.0.0_qjy` 已运行（Dockerfile：`AgentSDK/aura/dockers/Dockerfile`）
- 模型权重和训练数据已在服务器指定路径就位

### 5.2 一次性初始化

```bash
bash deploy/setup_remote.sh
```

验证 SSH 连通性、容器状态、Python 包（pyyaml）、模型和数据路径。

### 5.3 启动训练

```bash
# 命令行直接启动
python -m rllm_remote.train --lr 1e-6 --epochs 100 --batch-size 32

# 通过环境变量
SSH_HOST=192.168.9.142 EPOCHS=50 bash deploy/run_remote.sh

# 通过技能系统
/rllm-train backend=remote, NPU 训练 math agent, reward >= 0.8
```

### 5.4 监控与结果取回

```bash
# SSH 登录并 attach 到 tmux 会话
ssh root@192.168.9.142
docker exec -it agent5.0.0_qjy tmux attach -t train_<run_id>

# 下载训练结果到本地
python -c "
from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor
config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
executor = RemoteExecutor(config)
executor.download_file(
    f'{config.remote_output_dir}/{config.run_id}/training_log.txt',
    f'{config.local_output_dir}/{config.run_id}/training_log.txt'
)
"
```

## 6. 双 CLI 架构协作

远程训练融入现有双 CLI 架构的方式：

- **CLI-1**（rllm-train）：通过 rllm-remote-run / rllm-remote-monitor 编排远程训练。监控技能在每次远程日志轮询后写入本地 heartbeat。
- **CLI-2**（traj-xx）：读取 trajectory 并生成 skill 补丁。不受影响——CLI-2 只读取 `traj_opt/output/`，该目录由 CLI-1 执行期间的本地 hooks 填充。

`RoundState` 和 `status.json` 机制保持不变。训练阶段的 heartbeat 由监控技能的周期性远程日志轮询维持。

## 7. 错误处理

### SSH/连通性错误

| 错误 | 原因 | 处理方式 |
|---|---|---|
| SSH 超时 | 网络问题、服务器宕机 | 重试退避；提示检查网络 |
| 认证失败 | SSH 密钥未配置 | 引导用户配置 SSH 密钥 |
| 容器未运行 | 容器已停止 | 提示 `docker start agent5.0.0_qjy` |

### NPU 特有错误

| 错误 | 日志特征 | 处理方式 |
|---|---|---|
| NPU OOM | `NPU out of memory`、`Ascend OOM` | 减小 batch_size、tp_size、max_response_length |
| HCCL 超时 | `HCCL timeout`、`hcclComm*` | 检查 NPU 间互联；减少 nnodes |
| 驱动错误 | `drv* error`、`acl* error` | 重启容器或联系管理员 |
| vLLM Ascend 错误 | `VLLM_ASCEND*`、`nz error` | 检查 `VLLM_ASCEND_ENABLE_NZ` 环境变量 |

### 训练错误

rllm 的标准错误恢复策略（OOM → 减小 batch、loss 爆炸 → 降低 lr、plateau → 调节 kl_coef）完全适用于远程训练，叠加上述 NPU 特有处理。

## 8. 文件布局

```
rllm_remote/
  __init__.py              # 包文档
  config.py                # RemoteTrainConfig 数据类
  ssh.py                   # RemoteExecutor（SSH + docker exec）
  config_generator.py      # AgentSDK YAML 配置生成
  train.py                 # CLI 入口

deploy/
  setup_remote.sh          # NPU 服务器一次性初始化
  run_remote.sh            # 训练启动脚本（环境变量覆盖）

skill-bank/rllm/
  rllm-remote-run/
    base.md                # 远程训练启动技能
    manifest.yaml
    patches/
  rllm-remote-monitor/
    base.md                # 远程训练监控技能
    manifest.yaml
    patches/
  rllm-train/patches/
    remote-backend-routing.md   # 后端检测 + Phase 路由
  rllm-config/patches/
    remote-mode.md              # 远程配置生成模式
  rllm-analyze/patches/
    remote-results.md           # 远程结果下载

.claude/skills/
  rllm-remote-run/SKILL.md      # 编译产物
  rllm-remote-monitor/SKILL.md  # 编译产物
```

## 9. 关键设计决策

**SSH 使用 subprocess 而非 paramiko/fabric**：零额外 Python 依赖。项目已在 `rllm_verl/train.py` 中使用 `subprocess.run()` 启动 veRL。SSH 命令均为简单的一行调用，无需交互式 SSH 库。

**tmux 分离执行**：训练进程不依赖 SSH 连接存活。用户可随时 SSH 登录并 `tmux attach` 查看训练。相比 systemd 或 nohup，更适合临时训练任务。

**基于模板的 YAML 生成**：AgentSDK 的 YAML schema 复杂且与 `start.py` 使用的 Hydra/OmegaConf 紧密耦合。从已验证的参考模板生成可避免 schema 偏移，确保配置与容器内 AgentSDK 版本兼容。

**独立的 `rllm_remote/` 包**：与 `rllm_verl` 明确分离。虽然底层都使用 veRL，但 `rllm_remote` 涉及完全不同的关注点（SSH 通信、tmux 会话管理、AgentSDK YAML 生成、服务器路径），而 `rllm_verl` 关注本地子进程、Hydra CLI、本地 GPU 检测。

**Config JSON 作为技能契约**：`TrainingConfig` 和 `RemoteTrainConfig` 共享相同的 `to_json()` / `from_json()` 序列化契约，技能管线以相同的数据传递模式处理两种后端（config JSON 路径 → run 技能 → monitor 技能 → analyze 技能）。切换后端只需改变生成的配置对象，无需修改技能间的数据传递逻辑。
