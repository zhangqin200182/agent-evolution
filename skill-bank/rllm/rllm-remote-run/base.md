---
name: rllm-remote-run
description: Launch remote NPU training via AgentSDK on Huawei Ascend server. Connects via SSH, configures, and starts training in detached tmux session.
metadata:
  version: "1.0.0"
  categories:
    - machine-learning
    - agent-training
    - remote-execution
---

# rllm-remote-run — 远程 NPU 训练启动

<!-- section:intro -->
你负责将 Agent RL 训练任务提交到远程 NPU 服务器（<server-ip>）上的 AgentSDK 容器中执行。

## 前置条件

- SSH 密码可登录 <server-ip>
- 服务器上容器 `agent5.0.0_qjy` 已运行
- 服务器上已有模型权重和训练数据
- 远程训练配置文件已生成: `rllm_remote/output/runs/<run_id>/config.json`
- 工作目录: `/Users/kevin/code/MyProject`

## 重要：密码处理

config.json 中不包含密码。启动时必须通过 CLI 传入 `--ssh-password`：

```bash
python -m rllm_remote.train rllm_remote/output/runs/<run_id>/config.json --ssh-password "<your-password>"
```
<!-- /section:intro -->

<!-- section:verify -->
## 连接验证

启动训练前，先验证连通性（需要密码）：

```bash
cd /Users/kevin/code/MyProject && python -m rllm_remote.train --check --ssh-password "<your-password>"
```

如果验证失败：
- SSH 不通 → 检查密码和网络
- 容器未运行 → `docker start agent5.0.0_qjy`
- 模型/数据不存在 → 确认服务器路径正确
<!-- /section:verify -->

<!-- section:launch -->
## 启动流程

### 1. 验证配置

```bash
python3 -c "
from rllm_remote.config import RemoteTrainConfig
config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
print(config.summary())
"
```

### 2. 启动远程训练

使用 nohup 后台模式启动（断连不影响训练）：

```bash
cd /Users/kevin/code/MyProject && python -m rllm_remote.train rllm_remote/output/runs/<run_id>/config.json --ssh-password "<your-password>"
```

启动命令会：
1. 生成 AgentSDK 兼容的 YAML 配置
2. 通过 SCP + docker cp 上传到容器 configs/ 目录
3. 通过 run_start_in_local.sh 在容器内 nohup 后台启动训练
4. 输出：run_id、服务器地址、PID、日志路径

### 3. 确认启动成功

检查 PID 存活 + 日志写入：

```bash
python3 -c "
from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor
config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
config.ssh_password = '<your-password>'
executor = RemoteExecutor(config)
alive = executor.check_pid('<PID>')
print(f'PID alive: {alive}')
log = executor.tail_log(f'{config.remote_output_dir}/{config.run_id}/training_log.txt', lines=5)
print(log[:300])
"
```
<!-- /section:launch -->

<!-- section:output -->
## 输出

启动成功后报告：

```
远程训练已启动：
  Run ID:      <run_id>
  Server:      <server-ip>
  Container:   agent5.0.0_qjy
  PID:         <pid>
  远程日志:    /home/qjy/code/AgentSDK/master/AgentSDK/outputs/<run_id>/training_log.txt
  本地配置:    rllm_remote/output/runs/<run_id>/

SSH 登录查看:
  ssh root@<server-ip>
  docker exec -it agent5.0.0_qjy tail -f /home/qjy/code/AgentSDK/master/AgentSDK/outputs/<run_id>/training_log.txt
```
<!-- /section:output -->

<!-- section:error-handling -->
## 错误处理

| 错误类型 | 处理方式 |
|---|---|
| SSH 连接超时 | 检查密码和网络，确认 <server-ip> 可达 |
| 容器未运行 | 提示: `docker start agent5.0.0_qjy` |
| NPU OOM | 减小 batch_size、ppo_mini_batch_size、tensor_parallel_size |
| 配置文件不存在 | 提示先运行 rllm-config 生成远程配置 |
| 进程启动后立即退出 | 读取日志文件诊断错误 |
| 模型路径不存在 | 确认服务器上模型路径 |
| Ascend 驱动错误 | 服务器 NPU 驱动异常，需联系管理员 |
<!-- /section:error-handling -->

<!-- section:manual-check -->
## 手动检查命令

```bash
# 进入容器
ssh root@<server-ip>
docker exec -it agent5.0.0_qjy bash

# 查看日志
tail -f /home/qjy/code/AgentSDK/master/AgentSDK/outputs/<run_id>/training_log.txt

# 查看进程
ps aux | grep start.py

# 停止训练
kill <PID>
```
<!-- /section:manual-check -->
