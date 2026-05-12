---
description: Precision/accuracy analysis for remote NPU RL training. Reads all accuracy-relevant
  TensorBoard metrics, diagnoses training issues, and generates structured tuning
  suggestions with per-item approval and auto-config-modification.
metadata:
  categories:
  - machine-learning
  - analysis
  - remote
  version: 2.0.0
name: rllm-analyze-accuracy
---


# 远程训练精度分析

你是 veRL/AgentSDK 远程 RL 训练的精度分析专家。你的任务：

1. 从远程服务器抓取全部精度相关 TB 指标（44 个 tags）
2. 基于三层模型进行分析：TB 指标模型 → 算法配置模型 → 诊断与建议
3. 输出详实可信的分析报告（必须包含分析过程和具体配置问题）
4. 生成逐条可审批的调参建议
5. 用户审批后自动修改配置文件

远程连接方式参见 `rllm-remote-connect` skill。

# 数据获取

## 必须执行的步骤

### Step 1: 运行 accuracy_analyzer.py

```bash
python -m rllm_remote.accuracy_analyzer <run_id> --ssh-password "<password>"
```

这会：
- 通过 SSH 连接远程服务器
- 抓取全部 44 个精度 TB tags 的完整 step 历史
- 计算趋势和异常检测
- 生成 `analysis.json` 和 `accuracy_report.md`

### Step 2: 读取生成的文件

必须使用 Read 工具读取：
- `rllm_remote/output/runs/<run_id>/analysis.json` — 结构化分析数据
- `rllm_remote/output/runs/<run_id>/accuracy_report.md` — 人类可读报告
- `rllm_remote/output/runs/<run_id>/config.json` — 当前配置

**重要**：必须显式读取这些文件（不能依赖上下文），这样才能被 trajectory hooks 捕获。

## 如果 Python 脚本不可用

手动方式：
1. 通过 `rllm-remote-connect` 建立连接
2. 在远程容器内执行 TB 读取脚本（使用 EventAccumulator）
3. 获取全部 44 个 tags 的 step-by-step 数据
4. 手动进行趋势计算和异常检测

# 算法识别

从 `config.json` 中识别当前使用的算法。查看以下字段：

1. 检查 verl 配置: `algorithm.adv_estimator`, `actor.policy_loss.loss_mode`
2. 检查 reward_manager 类型: naive / dapo
3. 检查 rollout_correction 是否启用

算法将决定使用哪组特定的诊断规则（参见 algorithm-models 中的各个算法 patch）。

# TB 指标精度模型

所有算法共用的 6 组 44 个精度指标。每组包含指标含义、健康范围、异常模式。

## 组 A：Reward & Score（6 tags）

```
critic/rewards/mean, max, min
critic/score/mean, max, min
```

| 诊断规则 | 触发条件 | 含义 |
|---------|---------|------|
| min == max 全程为 1.0 | 任务太简单 | GRPO advantage=0 |
| min == max 全程为 0.0 | 任务太难或 agent 配置问题 | GRPO advantage=0 |
| mean 在 epoch 边界骤降 >50% | 灾难性遗忘 | 见 D 组 catastrophic_token |
| score 与 rewards 分叉 | multi-verifier 权重问题 | 检查 verifier_weight |

## 组 B：Advantage & Return（6 tags）

```
critic/advantages/mean, max, min
critic/returns/mean, max, min
```

- advantages/mean ≈ 0 且 max ≈ -min → GRPO 组内归一化正常
- advantages/max 收缩（1.5→0.1）→ 策略收敛
- advantages 全为 0 → 组内 reward 全同，无学习信号

## 组 C：Actor 训练指标（8 tags）

```
actor/pg_loss, actor/entropy, actor/grad_norm
actor/kl_loss, actor/ppo_kl
actor/pg_clipfrac, actor/pg_clipfrac_lower, actor/kl_coef
```

**pg_loss**：
- =0 全程：无学习信号或特定算法正常行为（GPG）
- >0：实现 bug
- 负值趋近于 0：正常收敛

**entropy**：
- 从 ~0.2 缓降至 ~0.14：健康
- <0.01：策略坍缩
- 不降反升（entropy_coeff=0 时）：训练不稳定
- 下降速率 >0.01/step：坍缩过快

**grad_norm**：
- 平稳 2-8：正常
- 突刺 >3x 均值：不稳定，该 step 破坏性更新
- 持续在 clip_grad 附近：裁剪一直生效

**kl_loss**：
- 从 0 逐步上升至 0.05~0.06：正常
- >0.1 且持续上升：发散风险
- epoch 边界跳跃 >2x：epoch 间分布突变

**pg_clipfrac**：
- 0.1~0.2：健康（vanilla PPO）
- 全程为 0：lr 太小或 GPG 模式（正常）
- >0.3：裁剪过激

## 组 D：Rollout-Training 分布差异（15 tags）

```
rollout_corr/kl, k3_kl
rollout_corr/chi2_token, chi2_seq
rollout_corr/ppl_ratio
rollout_corr/log_ppl_diff, log_ppl_abs_diff, log_ppl_diff_max, log_ppl_diff_min
rollout_corr/rollout_is_catastrophic_token_fraction  ← 最关键！
rollout_corr/rollout_is_veto_fraction
rollout_corr/rollout_log_ppl, rollout_ppl
rollout_corr/training_log_ppl, training_ppl
```

**关键规则**：
- `catastrophic_token_fraction > 0` → **任何正值都表示灾难性遗忘**
- `chi2_seq > 0.5` 持续 → 严重分布不匹配
- `ppl_ratio` 偏离 1.0 → <0.9 rollout 质量差，>1.1 训练过拟合
- `rollout_ppl` 上升 + `reward` 上升 → reward hacking

## 组 E：概率分布对齐（5 tags）

```
training/rollout_actor_probs_pearson_corr  ← >0.99 健康
training/rollout_probs_diff_mean
training/rollout_probs_diff_std
training/rollout_probs_diff_max
training/rollout_probs_diff_valid
```

- `pearson_corr < 0.95` → rollout 和 actor 概率分布严重不一致

## 组 F：Response Quality（2 tags）

```
response/aborted_ratio     — >0 生成被截断
response_length/mean       — 结合 reward 分析长度偏差
```

## 通用异常检测规则（15条）

以下规则适用于所有算法：

| 异常 ID | 触发条件 | 严重度 |
|---------|---------|--------|
| REWARD_SATURATION_ALL_ONE | min==max==1.0 全程 | critical |
| REWARD_SATURATION_ALL_ZERO | min==max==0.0 全程 | critical |
| REWARD_EPOCH_DROP | epoch边界 mean 降 >50% | critical |
| ADVANTAGE_ZERO | advantages max==min==0 | critical |
| ENTROPY_COLLAPSE | entropy < 0.01 | critical |
| ENTROPY_RISING | 不降反升 (coeff=0) | warning |
| GRADIENT_SPIKE | 单步 >3x 均值 | info |
| GRADIENT_ALWAYS_CLIPPED | 持续 == clip_grad | warning |
| KL_DIVERGENCE | kl_loss > 0.1 且上升 | critical |
| KL_EPOCH_JUMP | epoch边界跳跃 >2x | warning |
| DISTRIBUTION_SHIFT | chi2_seq > 0.5 持续 | warning |
| CATASTROPHIC_FORGETTING | catastrophic_token > 0 | critical |
| REWARD_HACKING | rollout_ppl↑ + reward↑ | critical |
| PEARSON_DECOUPLING | pearson_corr < 0.95 | warning |
| ABORTED_GENERATION | aborted_ratio > 0 | warning |

# 算法配置模型

各算法的完整定义（参数集、预期 TB 模式、失效模式、调优映射）见 patches/ 目录下的独立文件：

- `patches/alg-grpo.md` — GRPO
- `patches/alg-gae-ppo.md` — GAE-PPO
- `patches/alg-dapo.md` — DAPO
- `patches/alg-gspo.md` — GSPO
- `patches/alg-gpg.md` — GPG

在进行诊断时，根据 algorithm-identification 识别的算法，加载对应的算法 patch 中的规则。

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

# 诊断工作流

完整诊断流程：

1. **数据获取**：执行 `python -m rllm_remote.accuracy_analyzer <run_id>` 获取所有数据
2. **算法识别**：从 config.json 确定当前算法
3. **通用异常检测**：运行 15 条通用规则
4. **算法特定诊断**：加载对应算法 patch，检查算法特定失效模式
5. **根因分析**：对每个诊断，分析根因并定位到具体配置参数
6. **健康评分**：计算 5 维健康评分
7. **建议生成**：每个诊断生成对应调参建议，合并冲突
8. **输出报告**：生成 analysis.json 和 accuracy_report.md

## 关键统计方法

- 趋势判断：首半段均值 vs 后半段均值对比（5% 阈值）
- 线性回归：计算 slope，判断整体趋势方向
- 方差分析：检测 batch 内 reward 是否同质化
- Epoch 边界效应：检测 epoch 切换时的指标跳变
- 突刺检测：均值 ± 3σ 阈值

# 报告生成

报告必须包含以下四个部分，缺一不可：

## 第一部分：分析过程
- 数据来源（服务器、路径）
- 分析方法（6 步方法论说明）
- 指标覆盖表
- **每个指标组的 step-by-step 数值表**，异常点标注 ← ⚠️

## 第二部分：当前配置问题诊断
- 每个问题独立的 ID
- **具体数据证据**（精确数值、step 位置、比值）
- 诊断结论（为什么这些数据说明有问题）
- **明确指出当前配置中的具体参数和值的问题**

## 第三部分：健康评分
- 5 维评分 + 每项的依据说明

## 第四部分：调优建议
- 每条包含：参数路径、当前值、建议值、优先级、数据依据、理由、预期效果、风险

报告必须**充分详实**——每个结论都绑定具体数据。参考 `docs/rllm-analyze-accuracy-design.md` 第 5.1 节的完整模板。

# 建议审批流程

分析完成后，将建议逐条展示给用户：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
建议 #N [优先级] — 算法名

  参数: xxx
  当前: xxx → 建议: xxx
  
  数据依据: 具体 TB 数据
  
  理由: 为什么这样改
  预期: 改完会怎样
  风险: 可能有什么副作用

  [Y] 批准  [V] 修改值为___  [N] 跳过  [Q] 拒绝全部
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

用户可以：
- 批准（Y）：采纳建议值
- 修改值（V）：用户自定义新值
- 跳过（N）：不采纳此条
- 拒绝全部（Q）：终止，不做任何修改

# 配置修改

## 修改范围

修改本地 `rllm_remote/output/runs/<run_id>/config.json`（RemoteTrainConfig）。

远程 AgentSDK YAML 配置在下次 `rllm-remote-run` 启动训练时自动从 config.json 重新生成。

## 修改流程

1. 收集用户批准的建议 ID 列表
2. 执行：
   ```bash
   python -m rllm_remote.accuracy_analyzer <run_id> --apply <id1> <id2> ...
   ```
   或 Python：
   ```python
   analyzer.apply_suggestions([1, 3, 5])
   ```
3. 系统自动：
   - 备份 `config.json` → `config.json.bak.{timestamp}`
   - 按 `param_path` 更新对应字段
   - 输出修改摘要

# 独立使用

```
/rllm-analyze-accuracy <run_id>
```

不带 run_id 时，自动查找 `rllm_remote/output/runs/` 下最新的 run。

执行模式：
- **auto 模式**：非交互式执行分析，输出报告但不审批建议
- **interactive 模式**（默认）：分析完成后逐条展示建议等待审批
