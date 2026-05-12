---
id: alg-grpo
target_section: algorithm-models
action: append
description: GRPO (Group Relative Policy Optimization) 算法完整定义，包括参数集、TB指标预期模式、常见失效模式和调优策略
status: active
---

## GRPO (Group Relative Policy Optimization)

### 适用场景
数学推理、代码生成等有明确二元对错的场景。当前服务器默认使用的算法。

### 核心参数

| 参数 | 默认值 | 作用 | 调整方向 |
|------|--------|------|---------|
| `lr` | 1e-6 | 学习率 | ↑加速收敛 ↓防发散 |
| `kl_coef` | 0.001 | KL惩罚系数 | ↑防发散/防遗忘 ↓允许探索 |
| `clip_ratio` | 0.2 | PPO裁剪范围 | ↑允许更大更新 ↓更保守 |
| `entropy_coeff` | 0.0 | 熵奖励系数 | ↑防坍缩 |
| `n_samples_per_prompt` | 2~8 | 组内样本数 | ↑降 advantage 方差 |
| `temperature` | 1.0 | 采样温度 | ↑增多样性 ↓增确定性 |
| `top_p` | 1.0 | nucleus 采样 | ↓减少低概率 token |
| `total_epochs` | 1~2 | 训练轮数 | ↓防过拟合 |
| `kl_loss_type` | low_var_kl | KL loss 类型 | 可选: kl, low_var_kl, abs, mse, full |
| `kl_penalty` | kl | KL惩罚估计方式 | 可选: kl, abs, mse, low_var_kl, full |
| `ppo_mini_batch_size` | 8 | PPO mini batch 大小 | ↑更稳定梯度 ↓更频繁更新 |
| `clip_grad` | 1.0 | 梯度裁剪阈值 | ↑允许更大梯度 ↓更严格限制 |

### TB 指标预期模式

- `critic/advantages/mean ≈ 0`，`max ≈ -min`（组内归一化的数学性质）
- `critic/*` 指标存在但 GRPO 没有 critic 网络（指标名是历史原因）
- `pg_loss` 通常在较小范围内波动
- `pg_clipfrac` 在正常训练时 0.1~0.2

### 常见失效模式与调优

| 失效 | TB 信号 | 根因 | 调整 |
|------|--------|------|------|
| 组内 reward 同质化 | advantages 全0, reward min==max | reward 方差为 0 | ↑n_samples_per_prompt, ↑temperature |
| 策略坍缩 | entropy → 0, response_length 缩短 | 更新太激进，多样性丧失 | ↑temperature, ↑entropy_coeff, ↓lr |
| KL 发散 | kl_loss > 0.1 且持续上升 | 策略偏离参考太远 | ↑kl_coef, ↓lr, 考虑 adaptive KL |
| 灾难性遗忘 | epoch边界 reward 断崖 + catastrophic_token > 0 | 多 epoch 过拟合 | ↓epochs, ↑kl_coef, 启用 rollout_correction |
| 训练停滞 | reward 不升 + pg_loss→0 | 收敛或 lr 太小 | ↓clip_ratio, ↑lr |
| Reward Hacking | reward↑ + rollout_ppl↑ | 模型钻 reward 函数空子 | ↑kl_coef, 检查 reward 函数 |
| 梯度不稳定 | grad_norm 突刺 >3x | 个别 batch 特殊 | ↓lr, ↑ppo_mini_batch_size |

### 特殊说明

- pg_clipfrac=0 全程时：需结合算法判断。GRPO + vanilla PPO 下这是 lr 太小的信号（非正常）
- 组内样本数=1 时 GRPO 退化为 REINFORCE，advantage 恒为 0
- `norm_adv_by_std_in_grpo=True` 是默认行为（除以标准差），`False` 是 Dr.GRPO 模式
