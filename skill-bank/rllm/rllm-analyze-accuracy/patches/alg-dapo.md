---
id: alg-dapo
target_section: algorithm-models
action: append
description: DAPO (Decoupled Alignment from Policy Optimization) 算法完整定义，带 overlong buffer 惩罚机制，控制生成长度
status: active
---

## DAPO (Decoupled Alignment from Policy Optimization)

### 适用场景
需要精细控制生成长度、防止过长回答的场景。数学题/代码题中过长的回答通常意味着模型在 "水字数"。

### 识别条件
- `reward_manager` 类型为 `dapo`

### 额外参数（相比 GRPO）

| 参数 | 默认值 | 作用 | 调整方向 |
|------|--------|------|---------|
| `overlong_buffer_cfg.enable` | False | 启用超长惩罚 | True 时生效 |
| `overlong_buffer_cfg.len` | - | 缓冲长度（token数） | ↑宽容 ↓严格 |
| `overlong_buffer_cfg.penalty_factor` | - | 惩罚系数 | ↑惩罚更重 ↓惩罚更轻 |
| `overlong_buffer_cfg.log` | False | 是否记录超长统计 | 调试时开启 |
| `filter_groups.enable` | False | 启用组过滤 | True 时生效 |
| `filter_groups.metric` | - | 过滤指标 | acc/score/seq_reward/seq_final_reward |
| `filter_groups.max_num_gen_batches` | 0 | 最大保留 batch 数 | 0=不限制 |

### 额外 TB 指标关注点

- `response_length/mean` — 受 overlong_buffer 直接影响
- `critic/rewards/mean` — 受 penalty 影响会比实际正确率低
- `critic/score/mean` — 如果 score 比 reward 高很多，可能是 penalty 过重

### 常见失效模式与调优

| 失效 | TB 信号 | 根因 | 调整 |
|------|--------|------|------|
| 过度惩罚 | reward 整体偏低 + response_length 偏短 | penalty_factor 太大 | ↓penalty_factor, ↑buffer.len |
| 惩罚无效 | response_length 持续增长 | penalty 太小 | ↑penalty_factor, ↓buffer.len |
| 组过滤过激 | 训练样本数急剧减少 | filter metric 阈值太严格 | 放宽 filter_groups.metric, ↑max_num_gen_batches |
| Reward 和 Score 分叉 | reward < score 且差距大 | penalty 太重 | ↓penalty_factor |

### 特殊说明

- DAPO 的 reward 已经包含了 overlong penalty，不是 "纯净" 的正确率
- 比较 `critic/rewards/mean` 和 `critic/score/mean` 可以判断 penalty 的影响程度
- filter_groups 会丢弃整组样本，如果 batch 突然变小可能是过滤过激
