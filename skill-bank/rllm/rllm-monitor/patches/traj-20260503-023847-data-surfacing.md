---
id: traj-20260503-023847-data-surfacing
target_section: data-surfacing
action: append
description: 训练结束时完整读取日志和 perf_stats，确保全量数据进入轨迹
status: proposed
source: trajectory-analysis
source_sessions: ["490054f5-e0e8-4813-827f-670c72590443"]
---

### 训练结束时的完整数据读取

训练完成后（检测到 Training Report 或后台任务退出），立即执行:

1. 读取 training_log.txt 最后 100 行（包含所有 step 的 reward）:
   ```bash
   tail -100 rllm_train/output/runs/<run_id>/training_log.txt
   ```

2. 用 Read 工具读取 perf_stats.json 完整内容:
   ```
   Read rllm_train/output/runs/<run_id>/perf_stats.json
   ```

这两个操作是 Monitor grep 的必要补充，确保 hooks 捕获完整训练结果数据。
