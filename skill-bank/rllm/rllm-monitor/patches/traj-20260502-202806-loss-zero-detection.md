---
id: traj-20260502-202806-loss-zero-detection
target_section: anomaly-detection
action: append
description: "检测 loss=0 状态并报告训练无实际学习效果"
status: proposed
source: trajectory-analysis
source_sessions: ["d77ca2b0-fec6-4ac0-aa91-ef36f58fe6e4"]
---

### Loss=0 检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| 训练无效果 (loss=0) | 连续 N 步 loss=0 且 reward > 0 | 报告 "训练无实际学习效果，模型预训练能力已覆盖当前难度" |

检测逻辑:
- 当连续 5 步 loss=0 且 avg_reward > 0.5 时触发
- 建议: "当前难度过低。建议提高难度 (mixed → mixed-hard) 或增加问题数量。"
- 不建议停止训练（reward 仍在达标），但标记为 "无学习信号"

轨迹证据:
- R3/R5: loss=0, avg_reward=0.77，模型参数未更新但 reward 达标
- 这种情况下继续训练是浪费时间，应提高难度获取学习信号
