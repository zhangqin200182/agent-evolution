---
id: traj-20260503-091930-param-ranges
target_section: param-ranges
action: append
description: "增加'题目太简单'诊断: 当 reward >= 0.8 且 loss=0 时建议提高 difficulty"
status: proposed
source: trajectory-analysis
source_sessions: ["b4d588ba-052e-4153-9c8b-5681a8850d9f"]
---

### 题目难度自动升级

当调参输入满足以下条件时，优先建议提高难度而非调整超参数:
- avg_reward >= 0.8 且 loss 接近 0
- difficulty 当前为 simple 或 mixed

诊断逻辑:
  if reward >= 0.8 and loss ≈ 0:
      if difficulty == "simple": → 建议切换到 mixed
      if difficulty == "mixed": → 建议增加 num_problems 或切换到 hard
      不要调 lr/epochs/batch 等超参数，问题不在训练动态而在数据难度
