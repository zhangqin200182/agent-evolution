---
description: Monitor remote NPU training progress via SSH. Tracks reward trends, loss,
  and detects anomalies (OOM, driver errors, HCCL timeout) on Huawei Ascend servers.
metadata:
  categories:
  - machine-learning
  - agent-training
  - remote-monitoring
  version: 1.0.0
name: rllm-remote-monitor
---


# rllm-remote-monitor — 远程 NPU 训练监控

你负责监控远程 NPU 服务器上的训练进程。通过 SSH 读取日志文件、检测异常，并在训练完成或出错时报告。

## 前置条件

- 训练已通过 rllm-remote-run 在远程服务器上启动
- 已知 run_id、远程日志路径、PID
- 本地有 `rllm_remote/output/runs/<run_id>/config.json`

## 重要：密码处理

config.json 中不包含密码。加载配置后必须手动设置：

```python
config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
config.ssh_password = '<your-password>'  # 或从环境变量读取
```

## 监控方法

### 首选：结构化监控工具

使用 `rllm_remote/monitor.py`，结合训练日志 + TensorBoard events 双数据源：

```bash
# 单次报告（包含 step 进度、reward、loss、性能、ETA、异常检测）
python -m rllm_remote.monitor <run_id> --pid <PID> --ssh-password "<your-password>"

# 持续轮询（每 20s 刷新）
python -m rllm_remote.monitor <run_id> --pid <PID> --ssh-password "<your-password>" --watch
```

数据来源：
- **TensorBoard events**（83 个标量 tag）：step 编号、reward 聚合值、loss、性能指标——每个训练 step 完成后自动写入
- **训练日志**：Agent 轨迹 reward、LLM 耗时、错误检测

进度推算：total_steps ≈ 数据集/batch_size × epochs（从 config.json），current_step 来自 TB。

### 备选：手动日志检查

当 monitor.py 不可用时，手动执行：

```bash
python3 -c "
from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor
config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
config.ssh_password = '<your-password>'
executor = RemoteExecutor(config)
alive = executor.check_pid('<PID>')
log = executor.tail_log(f'{config.remote_output_dir}/{config.run_id}/training_log.txt', lines=30)
print(f'PID alive: {alive}')
print(log)
"
```

## 异常检测

### 通用异常

| 异常 | 远程日志特征 | 处理建议 |
|---|---|---|
| Reward 崩塌 | reward 连续下降超过 50% | 建议 early stop，调大 kl_coef |
| Loss 爆炸 | loss > 100 或 NaN | 建议减小 lr，增大 grad_accum |
| 训练卡住 | 超过 300s 无新日志 | kill PID 并排查 |
| 进程崩溃 | Traceback, exit code != 0 | 读取错误上下文诊断 |

### NPU 特有异常

| 异常 | 远程日志特征 | 处理建议 |
|---|---|---|
| NPU OOM | `NPU out of memory`, `Ascend OOM` | 减小 batch_size、tp_size、max_response_length |
| HCCL 超时 | `HCCL timeout`, `hcclComm*` | 检查 NPU 间网络连接 |
| Ascend 驱动错误 | `drv* error`, `acl* error` | 重启容器或联系管理员 |
| vLLM Ascend 错误 | `VLLM_ASCEND*`, `nz error` | 检查 VLLM_ASCEND_ENABLE_NZ 环境变量 |
| Ray Worker 断开 | `RayActorError`, `ray::* died` | 减少 worker 数 |

### 诊断命令

```bash
python3 -c "
from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor
config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
config.ssh_password = '<your-password>'
executor = RemoteExecutor(config)

# 拉取最近 100 行日志
log = executor.tail_log(f'{config.remote_output_dir}/{config.run_id}/training_log.txt', lines=100)
print(log)

# 检查 NPU 状态
result = executor.run('npu-smi info 2>/dev/null || echo unavailable')
print(result.stdout[:500])
"
```

## 训练完成处理

1. 下载分析所需文件（日志、trajectory）到本地
2. 报告最终 reward 和训练耗时

```bash
python3 -c "
from rllm_remote.config import RemoteTrainConfig
from rllm_remote.ssh import RemoteExecutor
import os

config = RemoteTrainConfig.from_json('rllm_remote/output/runs/<run_id>/config.json')
config.ssh_password = '<your-password>'
executor = RemoteExecutor(config)
remote_dir = f'{config.remote_output_dir}/{config.run_id}'
local_dir = f'{config.local_output_dir}/{config.run_id}'
os.makedirs(local_dir, exist_ok=True)

executor.download_file(f'{remote_dir}/training_log.txt', f'{local_dir}/training_log.txt')

result = executor.run(f'ls {remote_dir}/trajectories.jsonl 2>/dev/null && echo EXISTS || echo MISSING')
if 'EXISTS' in result.stdout:
    executor.download_file(f'{remote_dir}/trajectories.jsonl', f'{local_dir}/trajectories.jsonl')

print(f'Results downloaded to {local_dir}')
"
```

## 输出格式

```
远程训练监控报告
================
Run ID:      <run_id>
Status:      正常完成 / 异常退出
Final Reward: <value>
Total Steps:  <N>
Anomalies:    <list or "none">

结果已下载到: rllm_remote/output/runs/<run_id>/
```
