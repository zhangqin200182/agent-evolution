---
id: traj-20260502-202806-difficulty-scaling
target_section: tuning
action: append
description: "渐进式难度升级规则: mixed→mixed-hard→hard，避免直接跳级"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### 渐进式难度升级

调参时的难度调整增加渐进规则:

当前轮次 avg_reward >= 0.7 且 loss=0 时:
- 当前 difficulty=simple → 升级到 mixed
- 当前 difficulty=mixed (20% hard) → 升级到 mixed-hard (50% hard)
- 当前 difficulty=mixed-hard → 升级到 hard
- 同时增加 max_agent_steps: 3 → 5（给模型更多推理空间）

禁止直接从 mixed 跳到 hard:
- 轨迹证据: R3 mixed avg=0.77 → R4 hard avg=0.19（断崖下降）
- 需要中间级别 mixed-hard 作为过渡

difficulty 参数扩展:
- `"mixed-hard"`: 50% simple + 50% hard（新增，介于 mixed 和 hard 之间）
