---
id: traj-20260502-152342-param-ranges
target_section: param-ranges
action: append
description: "当 loss=0 + reward>0.8 时，自动建议提高 difficulty"
status: proposed
source: trajectory-analysis
source_sessions: ["65dbb67c-7794-4f87-a4ce-f6a7621eb39c"]
---

**Loss=0 诊断**:
| 症状 | 调整 | 原因 |
|------|------|------|
| loss=0 全程 + reward >= 0.8 | difficulty 提升一级 (simple→mixed, mixed→hard) | 题目太简单，模型预训练能力已覆盖，GRPO 无学习信号 |
| loss=0 全程 + reward < 0.5 | 检查 num_generations 和 temperature | reward variance 不足，GRPO baseline 估计有问题 |
