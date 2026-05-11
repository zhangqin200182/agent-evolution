# rllm_verl + AutoDL 部署设计

> veRL 分布式训练后端 + AutoDL 云端部署方案。将 rllm_train 的 agent RL 训练从单机 TRL 扩展到多 GPU 分布式训练。

## 1. 概述

本次重构将项目拆分为三层：

| 层 | 包 | 职责 |
|----|-----|------|
| 共享层 | `rllm_common/` | Agent/Env 抽象、工具定义、解析器 — 所有后端共用 |
| TRL 后端 | `rllm_train/` | 单 GPU TRL/GRPO 训练（Mac/CPU 兼容） |
| veRL 后端 | `rllm_verl/` | 多 GPU veRL 分布式训练（GPU 服务器） |
| 部署脚本 | `deploy/` | AutoDL 一键部署和启动 |

## 2. 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      rllm_common/                            │
│                                                             │
│  base.py        BaseAgent, BaseEnv, Step, Trajectory, ...   │
│  math_env.py    MathCalcEnv, generate_math_problems()       │
│  tools.py       CalculateTool, FinishTool, TOOL_SYSTEM_PROMPT│
│  parsers.py     QwenToolParser                              │
│  tool_agent.py  ToolAgent (multi-step tool execution)       │
└──────────────────────────┬──────────────────────────────────┘
                           │ import
              ┌────────────┼────────────┐
              ▼                         ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│  rllm_train/         │    │  rllm_verl/                   │
│                      │    │                               │
│  TRL GRPOTrainer     │    │  config.py    VerlTrainConfig  │
│  单 GPU, Mac 兼容     │    │  dataset.py   问题→parquet     │
│  HF model.generate() │    │  reward.py    规则 reward      │
│                      │    │  train.py     veRL 启动入口     │
│  适用：开发/调试       │    │  configs/     hydra yaml      │
└──────────────────────┘    │                               │
                            │  适用：多 GPU 生产训练          │
                            └──────────────────────────────┘
```

## 3. rllm_common 模块设计

从 `rllm_train` 提取的共享代码，零外部依赖（仅标准库 + typing）。

### 模块职责

| 模块 | 职责 |
|------|------|
| `base.py` | 核心数据模型：`BaseAgent`, `BaseEnv`, `Action`, `Step`, `Trajectory`, `ToolCall`, `ToolOutput` |
| `math_env.py` | `MathCalcEnv` 计算器环境 + `generate_math_problems()` 数据集生成 |
| `tools.py` | 工具定义（`CalculateTool`, `FinishTool`）、JSON schema、系统提示词 |
| `parsers.py` | `QwenToolParser`：tool call 格式化与解析 |
| `tool_agent.py` | `ToolAgent`：管理对话历史，解析 tool call，格式化 observation |

### 向后兼容

`rllm_train` 中的原有模块改为 re-export：

```python
# rllm_train/base.py
from rllm_common.base import *  # noqa: F401,F403
```

所有现有代码（`from rllm_train.base import BaseAgent`）无需修改。

## 4. rllm_verl 模块设计

### 4.1 VerlTrainConfig

```python
@dataclass
class VerlTrainConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    num_problems: int = 200
    batch_size: int = 8
    num_epochs: int = 3
    rollouts_per_problem: int = 4
    lr: float = 1e-6
    kl_coeff: float = 0.02
    temperature: float = 0.7
    # ... 完整字段见 config.py
```

### 4.2 Dataset 准备

`dataset.py` 将数学问题转换为 veRL 期望的 parquet 格式：

```python
{
    "data_source": "math_calc",
    "prompt": [{"role": "system", ...}, {"role": "user", ...}],
    "reward_model": {"style": "rule", "ground_truth": "42"},
    "extra_info": {"question": ..., "answer": ..., "index": ...}
}
```

### 4.3 Reward Function

规则型 reward，兼容 veRL 的 reward manager 接口：

```python
def reward_function(data_source, solution_str, ground_truth, extra_info=None) -> float:
```

答案提取逻辑：
1. 优先从 `<tool_call>` 中找 `finish` 调用的 `response` 参数
2. Fallback：提取响应中最后一个数字
3. 数值比较（容差 1e-6）

### 4.4 训练入口

`train.py` 是 CLI 入口，执行流程：

```
parse args → VerlTrainConfig → prepare_verl_dataset() → build_verl_command() → subprocess.run()
```

生成的命令调用 `verl.trainer.main_ppo`，通过 hydra override 传递所有参数。

## 5. AutoDL 部署方案

### 5.1 推荐实例配置

| 模型规模 | 推荐实例 | 预估显存 |
|----------|---------|---------|
| 0.5B-3B | RTX 4090 x1-2 | 24-48 GB |
| 7B | A100 40G x2 | 80 GB |
| 14B+ | A100 80G x4 | 320 GB |

### 5.2 部署脚本

| 脚本 | 用途 |
|------|------|
| `deploy/setup_autodl.sh` | 一次性环境配置：安装依赖、下载模型、验证 |
| `deploy/run_verl.sh` | 启动 veRL 分布式训练（多 GPU） |
| `deploy/run_trl.sh` | 启动 TRL 单 GPU 训练（简单场景） |

### 5.3 部署流程

```bash
# 1. 创建 AutoDL 实例（推荐 RTX 4090 x2）
# 2. 上传代码
git clone <repo> /root/rllm && cd /root/rllm

# 3. 一键配置环境
bash deploy/setup_autodl.sh

# 4. 启动训练
bash deploy/run_verl.sh

# 或自定义参数
MODEL=Qwen/Qwen2.5-7B-Instruct NUM_GPUS=2 bash deploy/run_verl.sh
```

### 5.4 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL` | Qwen/Qwen2.5-3B-Instruct | HuggingFace 模型名 |
| `NUM_GPUS` | 自动检测 | GPU 数量 |
| `NUM_PROBLEMS` | 200 | 训练问题数 |
| `BATCH_SIZE` | 8 | 批大小 |
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `WANDB_API_KEY` | (空) | 设置后启用 W&B 日志 |
| `OUTPUT_DIR` | ./outputs/verl_YYYYMMDD_HHMMSS | 输出目录 |

## 6. 与现有系统的关系

### 6.1 rllm_train 保持不变

- 所有现有 import 路径不变（通过 re-export）
- Mac/CPU 开发调试仍用 `rllm_train`
- Skill 系统（rllm-xx skills）继续调用 `rllm_train`

### 6.2 rllm_verl 是生产训练路径

- 多 GPU 分布式训练
- vLLM 加速推理
- veRL 的 GRPO/PPO 实现
- 适合 AutoDL/云端 GPU 服务器

### 6.3 pyproject.toml

项目通过 optional dependencies 管理两个后端：

```toml
[project.optional-dependencies]
trl = ["trl>=0.12.0", "accelerate", "peft"]
verl = ["verl>=0.2.0", "vllm>=0.6.0"]
all = ["rllm[trl,verl]", "wandb"]
```

安装方式：
```bash
pip install -e .           # 仅 common
pip install -e ".[trl]"    # + TRL 后端
pip install -e ".[verl]"   # + veRL 后端
pip install -e ".[all]"    # 全部
```

## 7. 文件清单

```
rllm_common/
├── __init__.py
├── base.py          # BaseAgent, BaseEnv, Action, Step, Trajectory, ToolCall, ToolOutput
├── math_env.py      # MathCalcEnv, generate_math_problems()
├── tools.py         # CalculateTool, FinishTool, TOOL_SYSTEM_PROMPT
├── parsers.py       # QwenToolParser
└── tool_agent.py    # ToolAgent

rllm_verl/
├── __init__.py
├── config.py        # VerlTrainConfig
├── dataset.py       # prepare_verl_dataset(), build_prompt_for_problem()
├── reward.py        # reward_function(), compute_math_reward()
├── train.py         # CLI 入口, build_verl_command()
└── configs/
    └── math_agent.yaml  # veRL hydra 配置

deploy/
├── setup_autodl.sh  # 环境配置
├── run_verl.sh      # veRL 训练启动
└── run_trl.sh       # TRL 训练启动

pyproject.toml       # 项目安装配置
```
