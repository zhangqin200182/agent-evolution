---
id: traj-20260503-094219-param-ranges
target_section: param-ranges
action: append
description: "增加后半段 reward 下降检测: 连续 2 轮出现时建议减少 num_problems"
status: proposed
source: trajectory-analysis
source_sessions: ["c7b850f9-76ad-477b-9477-09aa0ff3f055"]
---

### 后半段 Reward 下降检测

当 analysis.json 显示后半段 avg_reward < 前半段 * 0.85 时:
- 首先排除数据分布因素（不同 seed 的随机波动）
- 如果连续 2 轮出现此模式: 建议减少 num_problems（缩短训练长度避免退化）
- 如果仅单轮出现: 标记为观察项，不调参
- 不要因单轮的后半段下降就降低 lr 或 epochs，这可能是正常波动
