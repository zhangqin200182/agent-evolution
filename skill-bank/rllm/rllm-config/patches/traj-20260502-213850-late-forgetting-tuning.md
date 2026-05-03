---
id: traj-20260502-213850-late-forgetting-tuning
target_section: tuning
action: append
description: "后期 forgetting 自动调参: 后 20% 步骤 avg < 前 80% 的 50% 时减少 num_problems"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### 后期 Forgetting 自动调参

当分析检测到后期 forgetting 时，自动调整下轮配置:

检测条件:
- 后 20% 步骤的 avg_reward < 前 80% 步骤 avg_reward 的 50%

调参规则:
- 下轮 num_problems 减少 15-20% (向下取整到偶数)
- 保持其他参数不变
- 如果连续 2 轮 forgetting: 同时降低 lr 20%

轨迹证据:
- run_1777726900 (48p): 前 80% (steps 1-38) avg=0.980, 后 20% (steps 39-48) avg=0.475
- 后期 avg 仅为前期的 48%，明确的 forgetting 信号
- 推断: 减少到 40p 可避免后期崩溃
