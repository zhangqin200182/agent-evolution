---
id: traj-20260502-202806-seed-strategy
target_section: param-ranges
action: append
description: "多轮训练时变更 seed，避免零 reward 步骤周期性重复"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### Seed 随机化策略

多轮训练时（traj-loop 或手动多轮）:
- 每轮使用不同 seed: `seed = base_seed + round_number`
- 或启用 dataset shuffle: `shuffle=True`
- 目的: 避免相同问题固定在相同 step，导致零 reward 步骤的周期性模式

轨迹证据:
- R3 和 R5 使用相同 seed=42，零 reward 步骤完全一致 [5,6,12,25,31]
- 训练未改善模型在这些特定问题上的表现
- 变更 seed 可以让模型接触不同的问题排列，获得更多样的学习信号
