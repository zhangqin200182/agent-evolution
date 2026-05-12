---
id: alg-gspo
target_section: algorithm-models
action: append
description: GSPO (Group-level Sequence Policy Optimization) 算法完整定义，序列级 loss 聚合，适用于长序列/多轮对话
status: active
---

## GSPO (Group-level Sequence Policy Optimization)

### 适用场景
需要序列级 loss 聚合的长序列生成、多轮对话/Agent 场景。通过 `seq-mean-token-mean` 聚合方式防止长序列被过度惩罚。

### 识别条件
- `actor.policy_loss.loss_mode == "gspo"`

### 关键配置

| 参数 | 要求值 | 作用 |
|------|--------|------|
| `policy_loss.loss_mode` | `gspo` | 使用 GSPO loss 函数 |
| `loss_agg_mode` | `seq-mean-token-mean` | 先序列内平均再序列间平均 |
| `clip_ratio` | 0.2（标准） | PPO 裁剪，GSPO 中行为可能不同 |

### TB 指标预期模式

- `pg_loss` 的值域和变化模式与 vanilla PPO 不同（序列级聚合导致）
- `pg_clipfrac` 的行为可能与 vanilla 不同
- `response_length/mean` 是 GSPO 重点关注的指标（长序列问题）

### 常见失效模式与调优

| 失效 | TB 信号 | 根因 | 调整 |
|------|--------|------|------|
| 长序列被过度惩罚 | response_length 越长 reward 越低 | seq-mean-token-mean 对长序列不利 | 检查 loss_agg_mode 配置 |
| 短序列优势 | response_length/mean 持续缩短 | 模型学得短回答得分更高 | 调整 loss 聚合策略或 reward 函数 |
| 收敛慢 | pg_loss 下降速率慢 | 序列级聚合更新频率低 | ↑ppo_mini_batch_size |

### 特殊说明

- GSPO 的 `seq-mean-token-mean` 聚合让每条序列权重相等，而 `token-mean` 让每个 token 权重相等
- 如果序列长度差异大，GSPO 和 vanilla PPO 的训练结果可能显著不同
- GSPO 通常与 GRPO advantage estimator 配合使用
