---
id: alg-gae-ppo
target_section: algorithm-models
action: append
description: GAE-PPO (Generalized Advantage Estimation + PPO) 算法完整定义，需要 Critic 网络，适用于有过程监督的场景
status: active
---

## GAE-PPO (Generalized Advantage Estimation + PPO)

### 适用场景
需要 Critic 网络、有过程监督信号（per-token reward）的场景。不同于 GRPO 的 outcome-only 奖励。

### 识别条件
- `algorithm.adv_estimator == "gae"`

### 额外参数（相比 GRPO）

| 参数 | 默认值 | 作用 | 调整方向 |
|------|--------|------|---------|
| `gamma` | 1.0 | 折扣因子 | <1.0: 考虑未来 reward 不确定性 |
| `lam` | 0.95 | GAE λ，bias/variance trade-off | ↑=高方差低偏差 ↓=低方差高偏差 |
| `cliprange_value` | 0.5 | Critic value 裁剪范围 | ↑允许更大 value 更新 |
| `critic_warmup` | 0 | Critic 预热步数 | ↑让 critic 先学再更新 policy |
| `critic.ppo_mini_batch_size` | 1 | Critic mini batch | ↑更稳定 critic 更新 |
| `critic.ppo_epochs` | 1 | Critic PPO epochs | ↑更多 critic 更新 |

### 额外 TB 指标（GRPO 没有的）

```
critic/vf_loss           — Value function loss，持续高位 = critic 不收敛
critic/vf_clipfrac       — Value 裁剪比例，>0.3 = 过度裁剪
critic/vpred_mean        — 预测 V(s) 均值，应与 returns 接近
```

### TB 指标预期模式

- `critic/advantages/mean` 可能不为 0（不同于 GRPO）
- `critic/returns = advantages + values`（GAE 的数学定义）
- `critic/vf_loss` 应随训练下降
- GAE 的 advantage 含时序信息，`lam` 控制衰减

### 常见失效模式与调优

| 失效 | TB 信号 | 根因 | 调整 |
|------|--------|------|------|
| Critic 不收敛 | vf_loss 持续高位 | Critic 学习不足 | ↑critic_warmup, ↑critic.ppo_mini_batch_size |
| Value 预测偏差 | vpred_mean 与 returns 差距大 | Critic 容量不够或 lam 不当 | ↓lam, ↑critic 网络参数 |
| Value 过度裁剪 | vf_clipfrac > 0.3 | cliprange_value 太小 | ↑cliprange_value (0.5→0.8) |
| Advantage 噪声大 | advantages 方差远超 GRPO 模式 | lam 过高 | ↓lam (0.95→0.9) |
| 训练慢 | 每步需要额外 critic forward | GAE 计算开销 | 考虑切换 GRPO 如果不需要过程监督 |

### 特殊说明

- GAE 模式下 `critic/advantages/mean` 不为 0 是正常的
- `critic/vf_loss` 不收敛是 GAE 最常见的失败模式
- 如果只有 outcome reward（非 per-token），GRPO 通常更高效
