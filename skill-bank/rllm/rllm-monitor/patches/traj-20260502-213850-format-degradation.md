---
id: traj-20260502-213850-format-degradation
target_section: anomaly-detection
action: append
description: "检测格式退化: 连续 3+ 步所有 trajectory 使用 max_agent_steps 且 reward=0"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### 格式退化检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| 格式退化 | 连续 3+ 步: 所有 trajectory 使用 max_agent_steps 且 reward=0 | 建议 early stop |

检测逻辑:
- 当连续 3 步满足以下全部条件时触发:
  - 所有 trajectory 使用了 max_agent_steps (如 3 步)
  - 所有 trajectory reward=0
  - 模型重复调用同一工具而不调用 finish
- 告警: "检测到格式退化: 模型忘记了 finish 工具的使用方式。建议 early stop，下轮减少 num_problems 15-20%。"
- 与 "reward 归零" 检测的区别: 格式退化关注的是 agent 行为模式 (不调用 finish)，而非单纯的 reward 数值

轨迹证据:
- run_1777726900 step 41-43: 所有 trajectory 使用 3 步全部调用 calculate，从不调用 finish
- 与 step 12 不同 (step 12 正确使用了 calculate→finish 流程，只是答案错误)
