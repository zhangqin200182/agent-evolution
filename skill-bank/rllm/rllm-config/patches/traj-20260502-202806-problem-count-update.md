---
id: traj-20260502-202806-problem-count-update
target_section: param-ranges
action: append
description: "更新 0.5B 模型 num_problems 安全范围: 32-48 for mixed"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### num_problems 精细化范围 (0.5B 模型)

基于 5 轮训练数据更新推荐范围:

| difficulty | 推荐范围 | 依据 |
|-----------|---------|------|
| mixed (20% hard) | 40-48 | 32 太简单 (loss=0), 64 forgetting |
| mixed-hard (50% hard) | 24-32 | hard 比例增加后需减少总量 |
| hard | 16-24 | 64 完全超出能力 (avg=0.19) |

默认推荐配置 (0.5B + 正式训练):
- num_problems=48, difficulty=mixed, lr=5e-6, epochs=1
- 预期: 比 32 problems 更有挑战性，但不会 forgetting

轨迹证据:
- R1/R2 (64p, mixed): catastrophic forgetting at step 14-16
- R3/R5 (32p, mixed): loss=0, 无学习效果
- 推断: 最优点在 32-64 之间，推荐 40-48
