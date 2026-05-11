# 远程训练监控系统设计

## 1. 问题分析

### 现状

`rllm_train` 本地训练的 `TrainingLogger` 输出结构化进度行（`Step 3/16 | reward: 0.75`），监控直观。

AgentSDK/verl 远程训练日志完全不同——Ray 多 Worker 并发输出，200 行日志 100% 是 AgentExecutor 轨迹行（`step_idx`、`final reward`、`total_llm_time`），**没有 verl 训练 step 级别的进度输出**。

### 数据源

训练过程中有两个数据源：

**训练日志** (`training_log.txt`)：
- AgentExecutor 轨迹行：逐条 `final reward: 1.0/0.0`、推理耗时、token 用量
- 阶段标记：Ray 启动、AgentManager 创建、TrainExecutor 启动
- 错误输出：Traceback、ConfigAttributeError、OOM

**TensorBoard events**（自动写入，无需额外启动）：
- 路径：`tensorboard_log/<project>/<run>/events.out.tfevents.*`
- 83 个结构化标量 tag，每个训练 step 完成后写入一条
- 核心指标：

| Tag | 含义 |
|---|---|
| `training/global_step` | 当前训练步数 |
| `critic/rewards/mean` | 平均 reward |
| `critic/score/mean` | 平均分数 |
| `actor/pg_loss` | Policy gradient loss |
| `actor/grad_norm` | 梯度范数 |
| `actor/kl_loss` | KL 散度 |
| `actor/entropy` | 策略熵 |
| `perf/time_per_step` | 每步耗时（秒） |
| `perf/throughput` | 吞吐量（tok/s） |
| `perf/mfu/actor` | 模型 FLOPs 利用率 |
| `timing_s/gen` | 推理耗时 |
| `timing_s/update_actor` | 训练耗时 |

### 关联

日志中的逐轨迹 reward 和 TB 中的聚合 reward 对应——每个 step 处理一组轨迹后 TB 写入聚合值。日志用于确认 Agent 活动正常和错误检测，TB 用于获取训练进度百分比和趋势。

## 2. 设计

### RemoteMonitor 类 (`rllm_remote/monitor.py`)

```
RemoteMonitor
├── check_alive()          → PID 存活
├── fetch_log(lines)       → 训练日志尾部
├── fetch_tb_metrics()     → 远程解析 TB events，返回 {tag: (step, value)}
├── scan_anomalies(log)    → 异常扫描
├── compute_progress()     → 进度% 和 ETA
├── summary()              → 单行摘要
└── report()               → 完整报告
```

### fetch_tb_metrics() 实现

通过 ssh 在容器内运行脚本，利用已有的 `tensorboard` 包：

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(logdir)
ea.Reload()  # 增量加载
for tag in ea.Tags()['scalars']:
    events = ea.Scalars(tag)
    print(f'{tag}|{events[-1].step}|{events[-1].value}')
```

每次 poll 新建 EventAccumulator 并 Reload()，确保读到最新数据。

### 进度推算

- total_steps ≈ 数据集大小 / batch_size × epochs（从 config.json 读取）
- current_step = `training/global_step` 最新值
- progress% = current_step / total_steps × 100
- ETA = (total_steps - current_step) × 最近 N 步平均 time_per_step

### CLI

```bash
python -m rllm_remote.monitor <run_id>           # 单次报告
python -m rllm_remote.monitor <run_id> --watch    # 持续轮询
```

### 输出格式

```
=== 远程训练监控  remote_xxx ===  Step 3/~235 (1.3%)

Reward   mean=0.633  max=1.000  min=0.000
Actor    loss=0.000  grad_norm=3.57  kl=0.012  entropy=0.18
Perf     277s/step  gen:220s  train:31s  91.8tok/s  MFU:10.9%
ETA      ~17.9h

Agent    轨迹: 256条  success_rate=60.9%
Anomalies  无
```

## 3. 技能层关系

- **工具层**：`monitor.py` 提供数据采集（tail_log + parse TB），输出结构化指标
- **技能层**：`rllm-remote-monitor` 定义监控策略——轮询间隔、异常阈值、何时报警、何时建议 early stop
- 技能通过 Bash 调用 `monitor.py` 获取数据，然后根据指标做决策

## 4. 文件

| 操作 | 文件 |
|---|---|
| CREATE | `rllm_remote/monitor.py` |
| EDIT | `rllm_remote/ssh.py`（新增 run_script 方法） |
| EDIT | `skill-bank/rllm/rllm-remote-monitor/base.md` |
