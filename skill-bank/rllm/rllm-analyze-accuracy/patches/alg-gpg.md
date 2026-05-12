---
id: alg-gpg
target_section: algorithm-models
action: append
description: GPG (Group Policy Gradient) 算法完整定义，群组正则化梯度，无显式 PPO clip
status: active
---

## GPG (Group Policy Gradient)

### 适用场景
需要更平滑的梯度更新、不希望 PPO clip 硬截断的场景。通过组内正则化替代显式裁剪。

### 识别条件
- `actor.policy_loss.loss_mode == "gpg"` 或 `algorithm.adv_estimator == "gpg"`

### 关键配置

| 参数 | 要求值 | 作用 |
|------|--------|------|
| `policy_loss.loss_mode` | `gpg` | 使用 GPG loss 函数 |
| `clip_ratio` | 可选（GPG 不显式 clip） | GPG 通过组内正则化实现类似效果 |

### TB 指标预期模式

- **`pg_clipfrac = 0` 全程是正常行为！**（GPG 不 clip）
- **`pg_clipfrac_lower = 0` 全程也是正常的**
- **`ppo_kl = 0` 全程也是正常的**（GPG 不在 PPO 框架内用 KL）
- `pg_loss` 的数值范围可能与 vanilla PPO 不同

### 常见失效模式与调优

| 失效 | TB 信号 | 根因 | 调整 |
|------|--------|------|------|
| 组内正则化不足 | grad_norm 波动大 | 组大小不够 | ↑n_samples_per_prompt |
| 收敛慢 | pg_loss 下降慢但无异常 | GPG 天然收敛慢 | ↑lr 或切换 vanilla PPO |

### 特殊说明

- **不要因为 pg_clipfrac=0 或 ppo_kl=0 而标记异常** — 这是 GPG 的正常行为
- GPG 的 "clip" 效果来自组内正则化而非显式裁剪
- 如果从 vanilla PPO 切换到 GPG，预期会看到 pg_clipfrac 从 0.1-0.2 变为 0
