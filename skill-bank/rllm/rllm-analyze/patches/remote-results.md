---
id: "remote-results"
target_section: "intro"
action: append
description: "Add remote training analysis support via rllm_remote.monitor --analyze. Uses TensorBoard events (83 scalar tags with full step history) as the primary data source instead of local trajectory files."
source: "2026-05-11 Remote NPU training feature"
created: "2026-05-11"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### Remote 模式分析

当 args 中包含 `backend=remote` 时，数据来源与本地完全不同：

| 数据 | 本地 rllm_train | 远程 AgentSDK (verl) |
|---|---|---|
| Reward/Loss | training_log.txt 逐 step 输出 | TensorBoard events（完整 step 历史） |
| 性能 | perf_stats.json | TB `perf/time_per_step`, `perf/throughput` |
| Agent 行为 | trajectories/*.jsonl | 训练日志中的 AgentExecutor 行 |
| 配置 | TrainingConfig JSON | RemoteTrainConfig JSON |

#### 一键分析

使用 monitor.py 的 `--analyze` 模式，自动从远程 TB + 日志提取数据并生成 analysis.json：

```bash
python -m rllm_remote.monitor <run_id> --ssh-password "<your-password>" --analyze
```

此命令会：
1. 从远程 TB events 读取所有 step 的 reward/loss/kl/性能历史
2. 下载 training_log.txt 到本地
3. 生成 analysis.json 写入 `rllm_remote/output/runs/<run_id>/analysis.json`

#### 分析要点

由于 AgentSDK/verl 不产生逐步轨迹文件，分析时注意：
- **Reward 趋势**：从 TB `critic/rewards/mean` 获取完整 step 序列
- **KL 散度**：TB `actor/kl_loss`，远程训练 KL 通常持续上升
- **性能瓶颈**：TB `timing_s/gen`（推理）vs `timing_s/update_actor`（训练），gen 占比通常 >90%
- **异常检测**：扫描训练日志中的 Traceback、Engine core died、RayTaskError

#### 分析 JSON 格式

与本地格式兼容，额外增加 `backend: "remote"` 标识：

```json
{
  "run_id": "remote_xxx",
  "backend": "remote",
  "total_steps": 11,
  "reward": {"start": 0.72, "end": 0.88, "trend": "rising", "values": [...]},
  "kl": {"start": 0.001, "end": 0.059, "values": [...]},
  "performance": {"total_time_s": 2059, "avg_time_per_step": 187, "avg_mfu": 0.105},
  "suggestions": [...]
}
```

分析完成后，编排者从 `rllm_remote/output/runs/<run_id>/analysis.json` 读取结果。
