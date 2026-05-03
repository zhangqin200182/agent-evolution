---
id: traj-20260501-155627-anomaly-detection
target_section: anomaly-detection
action: append
description: "增加 reward 峰值回落检测，提前触发 early stopping"
status: proposed
source: trajectory-analysis
source_sessions: ["run_1777650398"]
---

### Reward 峰值回落检测

在现有异常检测基础上增加:

| 异常 | 检测方式 | 处理 |
|------|---------|------|
| Reward 峰值回落 | 当前 reward < 历史峰值 * 0.5 且持续 3 步 | 建议 early stopping |

实现: Monitor 维护 max_reward 变量，每步更新。当连续 3 步 reward < max_reward * 0.5 时发出 STOP 建议。
