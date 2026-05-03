---
id: traj-20260502-213850-problem-count-refine
target_section: param-ranges
action: append
description: "收紧 0.5B+mixed num_problems 最优范围: 36-42 (48 后期 forgetting)"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### num_problems 最优范围精细化 (0.5B + mixed, 基于 2 轮数据)

2 轮训练数据收敛出更精确的推荐范围:

| num_problems | 结果 | 证据 |
|-------------|------|------|
| 32 | 无 forgetting 但 loss=0 (无学习) | run_1777723566: avg=0.773, loss=0 全程 |
| 48 | avg=0.849 但 step 41-47 格式退化 | run_1777726900: 后 20% avg=0.475 |
| 40 (推断) | 平衡点 | 32 太简单, 48 后期崩溃 |

更新推荐:
- 首轮训练: num_problems=40 (安全起点)
- avg_reward >= 0.85 且无后期 forgetting: 可尝试 44
- 出现后期 forgetting: 减少到 36
- 禁止 0.5B+mixed 使用 num_problems > 48
