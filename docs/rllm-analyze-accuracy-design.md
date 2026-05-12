# rllm-analyze-accuracy 精度分析技能设计 v2

## 1. 概述

### 1.1 三层架构

```
第一层：TB 指标精度模型     →  指标本身的含义、诊断规则、健康范围
第二层：算法配置模型         →  每种算法独立的参数集、预期模式、失效模式
第三层：分析引擎 + 审批执行  →  算法识别 → 诊断 → 建议 → 审批 → 修改
```

### 1.2 设计原则

- 每个算法自包含，不假设算法间有"正交轴"关系
- 新增算法 = 新增一个算法的完整定义（参数 + 诊断规则 + 建议规则）
- TB 指标模型是公共基础层，所有算法共用

---

## 2. 第一层：TB 指标精度模型

### 2.1 指标分组

全部 83 个 TB tags 中，精度相关共 44 个，分为 6 组。

#### 组A：Reward & Score（6 tags）

```
critic/rewards/mean, max, min
critic/score/mean, max, min
```

| 指标 | 含义 | 正常范围 | 异常及原因 |
|------|------|---------|-----------|
| `rewards/mean` | batch平均reward | 随训练上升 | 不升: 任务太难/学习失效; 骤降: 遗忘 |
| `rewards/max` | batch最大reward | 通常=1.0 (二元reward) | 持续<1.0: 所有样本全错 |
| `rewards/min` | batch最小reward | 通常=0.0 (二元reward) | 持续>0: 所有样本全对(太简单) |
| `score/*` | verifier加权分 | 与rewards相同(单verifier) | 与rewards分叉: multi-verifier权重问题 |

**关键诊断规则**：
- `rewards/min == rewards/max` 全程 → batch内所有样本reward相同 → GRPO advantage=0
- `rewards/mean` 在 epoch 边界骤降 > 50% → 灾难性遗忘
- `rewards/mean` 持续 < 0.1 → 任务太难或agent/tool配置有问题

#### 组B：Advantage & Return（6 tags）

```
critic/advantages/mean, max, min
critic/returns/mean, max, min
```

| 指标 | 含义 | 正常范围 | 异常及原因 |
|------|------|---------|-----------|
| `advantages/mean` | batch平均advantage | ≈0 (GRPO组内归一化) | 非0: 组内归一化失效 |
| `advantages/max` | batch最大advantage | 0.5~2.0 | 趋于0: 策略收敛; 持续很小: reward方差不足 |
| `advantages/min` | batch最小advantage | -2.0~-0.5 | 通常=-max (GRPO组内归一化的数学性质) |
| `returns/mean` | batch平均return | 与advantages/mean接近 | GAE模式下=advantage+value |

**关键诊断规则**：
- `advantages/max` 和 `advantages/min` 绝对值相等（如 ±1.5）→ 组内reward方差充足
- `advantages/max` 从1.5收缩到0.1 → 策略收敛中，正常但可能收敛过早
- `advantages` 全为0 → 所有组内reward相同，学习完全失效

#### 组C：Actor 训练指标（8 tags）

```
actor/pg_loss           — 策略梯度损失
actor/entropy           — 策略熵
actor/grad_norm         — 梯度范数
actor/kl_loss           — KL散度（策略 vs 参考策略）
actor/ppo_kl            — PPO KL惩罚
actor/pg_clipfrac       — PPO上界裁剪比例
actor/pg_clipfrac_lower — PPO下界裁剪比例
actor/kl_coef           — 当前KL惩罚系数
```

**pg_loss**：
- 含义：`-E[A * log π(a|s)]`，负的advantage加权对数概率
- 正常：负值或零，逐渐趋近于0表示收敛
- 异常：**> 0** (实现bug)；全程=0 + advantage也为0 (无学习信号)
- 注：部分算法(如GPG) pg_loss展现出不同模式，需要结合算法判断

**entropy**：
- 含义：`H(π) = -Σ π(a|s) log π(a|s)`，策略的不确定性
- 正常：从较高值(如0.2)逐步下降至稳定值(如0.14)，下降速率缓慢
- 异常：下降到<0.01 (坍缩)；不降反升 (训练不稳定)；下降速率>0.01/step (坍缩过快)

**grad_norm**：
- 含义：所有参数梯度的L2范数
- 正常：平稳在某个范围（如2~8），每步间波动<3x
- 异常：突刺>3x均值 (某步更新异常大)；持续=clip_grad (一直触发裁剪)

**kl_loss**：
- 含义：当前策略与参考策略的KL散度。在loss中的形式取决于`kl_loss_type`
- 正常：从0附近逐步上升，幅度取决于训练步数和lr
- 异常：>0.1且持续上升 (发散风险)；epoch边界跳跃>2x (epoch间分布突变)；>0.5 (语言能力退化)

**pg_clipfrac**：
- 含义：超过PPO裁剪边界的token比例
- 正常：0.1~0.2 (标准PPO); 0 (保守更新，lr太小)
- 异常：>0.3 (裁剪过激，策略变化太大)
- 注：GPG模式全程为0是正常的

#### 组D：Rollout-Training 分布差异（15 tags）

```
rollout_corr/kl
rollout_corr/k3_kl
rollout_corr/chi2_token
rollout_corr/chi2_seq
rollout_corr/ppl_ratio
rollout_corr/log_ppl_diff
rollout_corr/log_ppl_abs_diff
rollout_corr/log_ppl_diff_max
rollout_corr/log_ppl_diff_min
rollout_corr/rollout_is_catastrophic_token_fraction
rollout_corr/rollout_is_veto_fraction
rollout_corr/rollout_log_ppl
rollout_corr/rollout_ppl
rollout_corr/training_log_ppl
rollout_corr/training_ppl
```

| 指标 | 含义 | 正常范围 | 异常及原因 |
|------|------|---------|-----------|
| `kl` | rollout-training KL | 0.0001~0.01，step1峰值后回落 | 持续上升: 分布永久分叉 |
| `chi2_token` | token级卡方距离 | <0.005 | >0.01: token级分布差异大 |
| `chi2_seq` | 序列级卡方距离 | <0.3 | >0.5持续: 严重分布不匹配 |
| `ppl_ratio` | rollout/training PPL比 | 0.9~1.1 | <0.8: rollout质量差; >1.2: training过拟合 |
| `catastrophic_token_fraction` | 灾难遗忘token比例 | **0** | **任何>0: 灾难性遗忘** |
| `veto_fraction` | 被否决rollout比例 | 0 | >0.05: generation质量问题 |
| `rollout_ppl` | rollout困惑度 | 随训练下降 | 不降反升: 模型退化 |
| `training_ppl` | 训练困惑度 | 随训练下降 | 与rollout_ppl分叉: 分布偏移 |

**关键诊断规则**：
- `kl` 峰值在step1是正常的（首次更新偏移最大），但必须随后回落
- `chi2_seq > 0.5` 且持续 → 考虑启用RolloutCorrection
- `catastrophic_token_fraction > 0` → 需要立即处理（减小lr/增大kl_coef/启用rollout_correction）
- `rollout_ppl`上升 + `reward`上升 → reward hacking（靠语言退化获得高分）

#### 组E：概率分布对齐（5 tags）

```
training/rollout_actor_probs_pearson_corr
training/rollout_probs_diff_mean
training/rollout_probs_diff_std
training/rollout_probs_diff_max
training/rollout_probs_diff_valid
```

| 指标 | 健康范围 | 异常 |
|------|---------|------|
| `pearson_corr` | >0.99 | <0.95: rollout-actor概率严重不一致 |
| `diff_mean` | 接近0，稳定 | 持续增大: 系统性偏差 |
| `diff_max` | 0.2~0.8 | 趋于1.0: 某些token完全被重学 |

#### 组F：Response Quality（2 tags）

| 指标 | 含义 | 异常 |
|------|------|------|
| `response/aborted_ratio` | 生成截断比例 | >0: max_response_length不够 |
| `response_length/mean` | 平均生成长度 | 与reward结合分析长度偏差 |

### 2.2 通用异常检测

以下检测规则适用于所有算法：

| 异常ID | 触发条件 | 严重度 |
|--------|---------|--------|
| `REWARD_SATURATION_ALL_ONE` | rewards/min == rewards/max == 1.0 持续多步 | critical |
| `REWARD_SATURATION_ALL_ZERO` | rewards/min == rewards/max == 0.0 持续多步 | critical |
| `REWARD_EPOCH_DROP` | epoch边界 rewards/mean 下降 > 50% | critical |
| `ADVANTAGE_ZERO` | advantages max==min==0 全程 | critical |
| `ENTROPY_COLLAPSE` | entropy < 0.01 | critical |
| `ENTROPY_RISING` | entropy 不降反升 (entropy_coeff=0时) | warning |
| `GRADIENT_SPIKE` | grad_norm 单步 > 3x 均值 | info |
| `GRADIENT_ALWAYS_CLIPPED` | grad_norm 持续 == clip_grad | warning |
| `KL_DIVERGENCE` | kl_loss > 0.1 且持续上升 | critical |
| `KL_EPOCH_JUMP` | kl_loss 在epoch边界跳跃 > 2x | warning |
| `DISTRIBUTION_SHIFT` | chi2_seq > 0.5 持续不回落 | warning |
| `CATASTROPHIC_FORGETTING` | catastrophic_token_fraction > 0 | critical |
| `REWARD_HACKING` | rollout_ppl上升 + reward上升 | critical |
| `PEARSON_DECOUPLING` | pearson_corr < 0.95 | warning |
| `ABORTED_GENERATION` | aborted_ratio > 0 | warning |

---

## 3. 第二层：算法配置模型

每种算法是一个自包含的配置单元，包括：参数集、TB指标模式、常见失效模式、调优策略。

### 3.1 算法：GRPO（Group Relative Policy Optimization）

**适用场景**：数学推理、代码生成等有明确二元对错的场景。当前服务器使用的算法。

**核心参数**：

| 参数 | 默认值 | 作用 | 调整方向 |
|------|--------|------|---------|
| `lr` | 1e-6 | 学习率 | ↑加速收敛 ↓防发散 |
| `kl_coef` | 0.001 | KL惩罚系数 | ↑防发散 ↓允许探索 |
| `clip_ratio` | 0.2 | PPO裁剪范围 | ↑允许更大更新 ↓更保守 |
| `entropy_coeff` | 0.0 | 熵奖励系数 | ↑防坍缩 |
| `n_samples_per_prompt` | 2~8 | 组内样本数 | ↑降advantage方差 |
| `temperature` | 1.0 | 采样温度 | ↑增多样性 ↓增确定性 |
| `top_p` | 1.0 | nucleus采样 | ↓减少低概率token |
| `total_epochs` | 1~2 | 训练轮数 | ↓防过拟合 |
| `kl_loss_type` | low_var_kl | KL loss类型 | kl/low_var_kl |
| `kl_penalty` | kl | KL惩罚估计方式 | kl/abs/mse/low_var_kl/full |

**TB指标预期模式**：
- `advantages/mean ≈ 0`，`max ≈ -min`（组内归一化数学性质）
- `critic/*` 指标存在但不需要 critic 网络
- `pg_loss` 通常很小（GRPO的advantage天然小）

**常见失效模式与调优**：

| 失效 | TB信号 | 根因 | 调整 |
|------|--------|------|------|
| 组内reward同质化 | advantages全0, min==max | reward方差为0 | ↑n_samples, ↑temperature |
| 策略坍缩 | entropy → 0, response_length缩短 | 更新太激进 | ↑temperature, ↑entropy_coeff, ↓lr |
| KL发散 | kl_loss > 0.1且持续上升 | 策略偏离参考太远 | ↑kl_coef, ↓lr |
| 灾难性遗忘 | epoch边界reward断崖 | 过拟合 | ↓epochs, ↑kl_coef, ↑n_samples |
| 训练停滞 | reward不升 + pg_loss→0 | 收敛/学习率太小 | ↓clip_ratio, ↑lr |
| Reward Hacking | reward↑ + rollout_ppl↑ | 模型学会钻空子 | ↑kl_coef, 检查reward函数 |

### 3.2 算法：GAE-PPO（Generalized Advantage Estimation + PPO）

**适用场景**：需要 Critic 网络、有过程监督的场景。

**额外参数**（比GRPO多出的）：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `gamma` | 1.0 | 折扣因子，<1.0时考虑未来reward的不确定性 |
| `lam` | 0.95 | GAE λ，trade-off bias/variance |
| `cliprange_value` | 0.5 | Critic value裁剪范围 |
| `critic_warmup` | 0 | Critic预热步数 |

**额外TB指标**（GRPO没有的）：
```
critic/vf_loss           — Value function loss
critic/vf_clipfrac       — Value裁剪比例
critic/vpred_mean        — 预测V(s)的均值
```

**额外失效模式**：

| 失效 | TB信号 | 调整 |
|------|--------|------|
| Critic不收敛 | vf_loss持续高位 | ↑critic ppo_mini_batch_size, ↑critic_warmup |
| Value预测偏差 | vpred_mean与returns差距大 | ↓lam, ↑critic网络容量 |
| 过度裁剪value | vf_clipfrac > 0.3 | ↑cliprange_value |

### 3.3 算法：DAPO（Decoupled Alignment from Policy Optimization）

**适用场景**：需要控制生成长度、防止过长回答的场景。

**额外参数**：

| 参数 | 作用 |
|------|------|
| `overlong_buffer_cfg.enable` | 启用超长惩罚 |
| `overlong_buffer_cfg.len` | 缓冲长度 |
| `overlong_buffer_cfg.penalty_factor` | 惩罚系数 |
| `filter_groups.enable` | 启用组过滤 |
| `filter_groups.metric` | 过滤指标 (acc/score/seq_reward) |
| `filter_groups.max_num_gen_batches` | 最大保留batch数 |

**额外失效模式**：

| 失效 | TB信号 | 调整 |
|------|--------|------|
| 过度惩罚 | reward整体偏低 + response_length偏短 | ↓penalty_factor, ↑buffer.len |
| 过滤过激 | 训练样本减少过多 | ↓metric阈值, ↑max_num_gen_batches |

### 3.4 算法：GSPO（Group-level Sequence Policy Optimization）

**适用场景**：需要序列级loss聚合的长序列/多轮对话场景。

**关键配置**：
- `policy_loss.loss_mode = "gspo"`
- `loss_agg_mode = "seq-mean-token-mean"`

**TB指标差异**：
- pg_loss的值域和模式与vanilla PPO不同（序列级聚合）
- pg_clipfrac的行为可能不同

**额外失效模式**：

| 失效 | TB信号 | 调整 |
|------|--------|------|
| 长序列被过度惩罚 | response_length越长reward越低 | 检查 seq-mean-token-mean 聚合 |
| 短序列优势 | response_length/mean 持续缩短 | 调整 loss_agg_mode |

### 3.5 算法：GPG（Group Policy Gradient）

**关键配置**：
- `policy_loss.loss_mode = "gpg"`
- 无显式PPO clip

**重要**：`pg_clipfrac = 0` 全程是**正常行为**，不是异常！

### 3.6 算法扩展：新增加一个算法

新增算法的模板：

```
## 算法：XXX

### 适用场景
...

### 核心参数
| 参数 | 默认值 | 作用 | 调整方向 |
...

### 额外参数（区别于通用参数）
...

### TB指标预期模式
...

### 常见失效模式与调优
| 失效 | TB信号 | 根因 | 调整 |
...

### 特殊说明
- 某指标=某值是正常行为，不要误报
- 与其他算法的关键区别
```

---

## 4. 第三层：分析引擎

### 4.1 分析流程

```
输入: run_id, ssh_password
│
├── Step 1: 数据获取（Python）
│   ├── 读取 config.json → 识别当前算法
│   ├── SSH到远程服务器抓取全部精度TB tags (44个)
│   └── 计算 step-by-step 趋势、极值、方差
│
├── Step 2: 诊断分析
│   ├── 运行通用异常检测（2.2节15条规则）
│   ├── 加载当前算法特定的诊断规则
│   ├── 生成诊断列表（含严重度、根因分析）
│   └── 计算五维健康评分
│
├── Step 3: 建议生成
│   ├── 每个诊断 → 匹配算法特定的调参建议
│   ├── 建议冲突解决（如两条建议修改同一参数）
│   └── 按优先级排序
│
├── Step 4: 输出报告
│   ├── 写入 analysis.json
│   └── 生成 accuracy_report.md
│
├── Step 5: 用户审批（交互式）
│   └── 逐条展示建议，用户选择 批准/修改/跳过
│
└── Step 6: 执行修改
    ├── 备份 config.json → config.json.bak
    ├── 按批准的建议修改参数
    └── 输出修改摘要
```

### 4.2 健康评分

| 维度 | 含义 | 基于指标 |
|------|------|---------|
| `learning_signal` | 是否有有效的学习梯度 | pg_loss, advantages, reward方差 |
| `stability` | 训练过程是否稳定 | grad_norm波动, entropy变化率 |
| `divergence_risk` | 策略偏离参考模型的风险 | kl_loss, rollout_corr/kl |
| `forgetting_risk` | 灾难性遗忘的风险 | epoch边界reward变化, catastrophic_token |
| `overall` | 综合健康度 | 上述四维加权 |

### 4.3 建议优先级

| 优先级 | 条件 | 示例 |
|--------|------|------|
| critical | 灾难性遗忘/策略发散/坍缩 | ↑kl_coef, ↓lr, ↓epochs |
| high | 学习信号缺失/训练停滞 | ↑n_samples, ↑temperature |
| medium | 梯度不稳定/轻微分布偏移 | ↓lr, ↑mini_batch_size |
| low | 单一指标轻微异常，无影响 | 观察即可 |

### 4.4 建议格式

每条建议：
```json
{
  "id": 1,
  "priority": "high",
  "param_path": "kl_coef",
  "current_value": 0.001,
  "suggested_value": 0.005,
  "reason": "KL从0.001升至0.059，epoch边界跳跃2.4x，策略漂移速度过快",
  "expected_impact": "KL增长速率减缓，策略更稳定地远离参考模型",
  "risk": "reward提升速度可能变慢",
  "algorithm_specific": true,
  "applies_to_algorithm": "grpo"
}
```

---

## 5. 输出与审批

### 5.1 分析报告 (accuracy_report.md)

报告必须包含以下四个核心部分，缺一不可。报告要**充分详实、可信**——每个结论必须有具体数据支撑。

```markdown
# 精度分析报告 — <run_id>

## 第一部分：分析过程

### 数据来源
- 远程服务器: <server-ip>, 容器 agent5.0.0_qjy
- TB 事件路径: /home/qjy/.../tensorboard_log/rllm_remote/<run_id>/
- 配置文件: rllm_remote/output/runs/<run_id>/config.json
- 训练日志: /home/qjy/.../outputs/<run_id>/training_log.txt

### 分析方法
1. 从远程 TB events 中读取全部 83 个 scalar tags，提取其中 44 个精度相关指标
2. 获取每个指标的完整 step-by-step 历史序列（共 11 步）
3. 计算每个指标的趋势（首半段 vs 后半段对比、线性回归斜率）
4. 进行异常检测：极值分析、方差分析、step间变化率分析、epoch边界效应分析
5. 将检测到的异常模式与当前算法（GRPO）的已知失效模式进行匹配
6. 基于匹配结果生成诊断和调优建议

### 分析覆盖的指标组
| 指标组 | 标签数 | 关键指标 |
|--------|--------|---------|
| Reward & Score | 6 | rewards/mean, score/mean |
| Advantage & Return | 6 | advantages/max, advantages/min |
| Actor 训练指标 | 8 | pg_loss, entropy, grad_norm, kl_loss, pg_clipfrac |
| Rollout-Training 分布差异 | 15 | kl, chi2_seq, ppl_ratio, catastrophic_token_fraction |
| 概率分布对齐 | 5 | pearson_corr, probs_diff_mean |
| Response 质量 | 2 | aborted_ratio, response_length/mean |
| 训练元信息 | 2 | epoch, global_step |

### 关键指标的 Step-by-Step 趋势
（以下列出每个关键指标在所有 step 上的具体数值，并标注异常点）

#### Rewards
```
Step:  1      2      3      4      5      6      7      8      9      10     11
mean:  0.719  0.781  0.727  0.820  0.813  0.813  0.852  0.797  0.750  0.891  0.875
max:   1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000
min:   0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
```
- rewards/mean 从 0.719 升至 0.875，整体上升趋势 (+21.7%)
- Step 7 出现明显上升（0.813→0.852），对应 epoch 1 开始
- min 始终为 0（存在答错的样本），max 始终为 1（存在答对的样本）→ reward 方差健康

#### Advantages
```
Step:   1       2       3       4       5       6       7       8       9       10      11
max:    1.500   1.500   1.500   0.866   1.500   1.500   1.500   1.500   1.500   1.500   1.500
min:   -1.500  -1.500  -1.500  -1.500  -1.500  -1.500  -1.500  -1.500  -1.500  -1.500  -1.500
mean:   0.034   0.030   0.019  -0.004  -0.000  -0.032   0.015   0.023   0.001   0.010   0.007
```
- max=1.5, min=-1.5 → GRPO 组内归一化生效，advantage 方差充足
- mean ≈ 0 → 组内零和，符合 GRPO 数学性质
- **Step 4** max 降到 0.866 ← 此处组内 reward 方差变小（该 batch 可能大部分答对或大部分答错）

#### Actor 训练指标
```
Step:        1       2       3       4       5       6       7       8       9       10      11
pg_loss:     0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
entropy:     0.171   0.210   0.183   0.161   0.169   0.151   0.159   0.164   0.163   0.140   0.145
grad_norm:   6.387   2.469   4.911   2.420   2.364   4.710   3.822   7.600   3.967   2.175   3.999
kl_loss:     0.001   0.002   0.004   0.007   0.011   0.021   0.035   0.028   0.026   0.025   0.059  ← ⚠️
```
- **pg_loss 全部为 0** ← 核心问题，见第二部分详细分析
- entropy 从 0.171 降至 0.145，下降速率 0.0024/step，属于健康收敛范围
- grad_norm 在 [2.18, 7.60] 间波动，均值 4.07，标准差 1.89
  - Step 8 突刺至 7.60（1.9σ），Step 1 突刺至 6.39（1.2σ）
- **kl_loss Step 10→11 跳跃 2.4x**（0.025→0.059），对应 epoch 1→2 边界

#### Rollout-Training 分布差异
```
Step:         1       2       3       4       5       6       7       8       9       10      11
corr_kl:      0.012   0.001   0.000   0.000   0.000   0.001   0.000   0.004   0.002   0.000   0.001  ← ⚠️Step1
chi2_seq:     0.138   0.274   0.561   0.298   0.334   0.520   0.375   0.312   0.278   0.366   0.302  ← ⚠️Step3,6
ppl_ratio:    1.014   1.001   1.000   1.000   1.000   1.002   1.000   1.005   1.002   1.000   1.001
catastrophic: 0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
veto:         0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
```
- rollout_corr/kl: Step 1 峰值 0.012（首次更新后分布偏移最大），随后回落至 ~0.001，**正常**
- chi2_seq: Step 3 峰值 0.561 (>0.5)，Step 6 峰值 0.520 (>0.5)，说明这两个 step 存在序列级分布偏移。但随后回落，**短期现象，非持续性偏移**
- ppl_ratio 始终在 [1.000, 1.014]，**非常健康**
- catastrophic_token_fraction 和 veto_fraction 全程为 0，**无灾难性遗忘和否决**

#### 概率分布对齐
```
Step:             1       2       3       4       5       6       7       8       9       10      11
pearson_corr:     0.999   0.999   0.999   0.999   0.999   0.999   0.999   0.999   0.999   0.999   0.999
probs_diff_mean:  0.003   0.003   0.003   0.003   0.003   0.003   0.003   0.003   0.003   0.003   0.003
```
- pearson_corr 始终 > 0.998，rollout 和 actor 概率分布高度一致
- probs_diff_mean 稳定在 0.003 附近，无系统性偏差


## 第二部分：当前配置问题诊断

### 问题 1：pg_loss 全程为 0 — 策略梯度未生效

**数据证据**：
- `actor/pg_loss` 在所有 11 步均为 0.000000
- `actor/pg_clipfrac` 在所有 11 步均为 0.000000
- `actor/ppo_kl` 在所有 11 步均为 0.000000
- 但 `critic/advantages/max` = 1.5, `critic/advantages/min` = -1.5（advantage 信号存在）
- `critic/rewards/min` = 0.0, `critic/rewards/max` = 1.0（reward 方差存在）

**诊断结论**：
advantage 信号存在（±1.5），reward 方差存在（min=0, max=1），但 pg_loss 始终为 0。这说明 PPO 的裁剪机制（clip_ratio=0.2）可能将所有的策略更新都裁剪掉了。结合 pg_clipfrac=0 的现象（没有 token 超过裁剪边界），更准确地说：**lr=1e-6 太小导致更新幅度在裁剪范围内但趋于零，同时 entropy_coeff=0 进一步导致无探索压力**。

**当前配置中的具体问题**：
- `lr = 1e-6` — 太小！对于 GRPO 的非零 reward 信号，这个 lr 产生的梯度更新量级太小
- `entropy_coeff = 0.0` — 关闭了探索奖励，策略没有动力保持多样性
- `clip_ratio = 0.2` — PPO 裁剪边界 0.2，但 pg_clipfrac=0 说明甚至没有 token 达到这个边界
- `kl_loss_coef = 0.001` — 此时学习主要靠 KL loss 驱动而非 reward

### 问题 2：KL 散度在 epoch 边界异常跳跃

**数据证据**：
- `actor/kl_loss` Step 1→10 逐步从 0.001 升至 0.025（正常上升速率 ~0.002/step）
- Step 10→11（epoch 1→2）：**0.025→0.059，单步跳跃 +136%**
- `training/epoch` 在 Step 7 从 0 变为 1（epoch 边界）

**诊断结论**：
KL 在 epoch 边界出现 2.4x 跳跃，说明 epoch 2 的数据分布与 epoch 1 不同，或者模型在 epoch 1 后已经过拟合到 epoch 1 的数据。在 epoch 2 面对新数据时，策略被迫大幅偏离参考模型以拟合新分布。

**当前配置中的具体问题**：
- `kl_coef = 0.001` — KL 惩罚太弱，无法有效约束策略偏移
- `kl_ctrl.type = fixed` — 固定 KL 系数无法应对 epoch 间的分布变化
- `total_epochs = 2`（推测）— 多 epoch 训练时 epoch 间数据分布可能不同

### 问题 3：梯度范数波动较大

**数据证据**：
- `actor/grad_norm` 均值 4.07，标准差 1.89（变异系数 0.46）
- Step 8 突刺至 7.60（+87% vs 均值），Step 1 突刺至 6.39（+57%）
- Step 2、4、5、10 在 2.2-2.5 之间（-40% vs 均值）

**诊断结论**：
grad_norm 波动比达 3.5x（7.60/2.18），说明不同 step 的梯度尺度差异较大。Step 1 的波动正常（首次更新），但 Step 8 的突刺出现在 epoch 内，可能与该 batch 的 reward 分布特殊有关。

**当前配置中的具体问题**：
- `lr = 1e-6` — lr 小可以部分缓解 grad_norm 波动的影响，但如果后续提高 lr，需要同步调整
- `clip_grad = 1.0`（默认值）— 所有 grad_norm 都远大于 1.0，梯度裁剪**一直在生效**，实际上每次更新幅度由 clip_grad 决定而非 lr
- `ppo_mini_batch_size = 8`（默认值）— 较小的 mini batch 可能导致单步梯度估计方差大

### 问题 4：Epoch 2 的奖励提升停滞

**数据证据**：
- Epoch 1 (Step 1-6): rewards/mean 从 0.719→0.813，上升 13%
- Epoch 2 (Step 7-11): rewards/mean 从 0.852→0.875，上升仅 2.7%
- Step 8-9: rewards/mean 出现下降（0.852→0.797→0.750）

**诊断结论**：
Epoch 2 的奖励提升幅度明显小于 Epoch 1（2.7% vs 13%），且中间出现下降。结合 pg_loss=0 的问题，epoch 2 的学习效率进一步降低。

**当前配置中的具体问题**：
- 可能与 `kl_loss` 在 epoch 2 上升过快到 0.059 有关——KL 约束变强，限制了策略改进空间
- `n_samples_per_prompt = 8`（推测）— 组内样本数可能不足以覆盖 epoch 2 的多样性


## 第三部分：健康评分

| 维度 | 分数 | 依据 |
|------|------|------|
| 学习信号 | 0.2/1.0 ⚠️ | pg_loss全程为0，学习主要依赖KL loss而非reward信号。reward虽在提升但缺乏直接的策略梯度驱动 |
| 训练稳定性 | 0.6/1.0 🟡 | grad_norm波动3.5x，Step 8突刺(+87%)。但整体均值4.07在正常范围 |
| 发散风险 | 0.7/1.0 🟡 | kl_loss从0.001升至0.059（仍在安全范围<0.1），但epoch边界跳跃2.4x需要关注 |
| 遗忘风险 | 1.0/1.0 ✅ | catastrophic_token_fraction全程为0，epoch边界reward未降反升 |
| 综合 | 0.6/1.0 ⚠️ | 核心问题是pg_loss=0导致的学习信号缺失。梯度裁剪一直生效（grad_norm min=2.18 > clip_grad=1.0？待确认）掩盖了lr偏小的问题 |

## 第四部分：调优建议
（每条建议包含：参数路径、当前值、建议值、优先级、数据依据、预期效果、风险）

### 建议 #1 [high] 增大学习率

- **参数**: `lr`
- **当前值**: 1e-6
- **建议值**: 5e-6
- **数据依据**: pg_loss全程为0，pg_clipfrac=0，说明当前lr下的更新幅度被PPO clip完全压制。同时grad_norm均值4.07 > clip_grad（当前值），说明梯度信号存在但被裁剪
- **预期效果**: pg_loss出现非零值，策略梯度开始贡献学习信号
- **风险**: 如果kl_loss同时快速上升，需要同步增大kl_coef

### 建议 #2 [medium] 增大KL惩罚系数

- **参数**: `kl_coef`
- **当前值**: 0.001
- **建议值**: 0.003
- **数据依据**: kl_loss在epoch边界跳跃2.4x（0.025→0.059），当前kl_coef=0.001不足以约束策略在epoch间的稳定漂移
- **预期效果**: 减少epoch边界KL跳跃，策略在epoch间更平滑过渡
- **风险**: 可能减缓reward提升速度。如果reward停滞，可以尝试kl_ctrl.type=adaptive

### 建议 #3 [low] 增大entropy_coeff

- **参数**: `entropy_coeff`
- **当前值**: 0.0
- **建议值**: 0.001
- **数据依据**: entropy从0.171降至0.145（-15%），虽在健康范围但配合pg_loss=0，添加轻量熵奖励可增加探索多样性
- **预期效果**: 策略保持一定探索能力，减缓entropy下降速率
- **风险**: 过大的entropy_coeff会导致策略不稳定。0.001是保守值
```

### 5.2 审批交互

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
建议 #1 [🟡 medium] — GRPO 特定

  参数: kl_coef
  当前: 0.001 → 建议: 0.002
  
  理由: kl_loss 从 0.001 升至 0.059，epoch 边界跳跃 2.4x
  
  预期: KL 增长速率减缓
  风险: reward 提升可能变慢

  [Y] 批准  [V] 修改值为___  [N] 跳过  [Q] 拒绝全部
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.3 配置修改

修改的是本地 `rllm_remote/output/runs/<run_id>/config.json`（RemoteTrainConfig）。

下次 `rllm-remote-run` 启动训练时，会自动从 `config.json` 重新生成远程 AgentSDK YAML 配置。

修改流程：
1. 备份：`config.json` → `config.json.bak.{timestamp}`
2. 修改：按批准的 `param_path` 更新对应字段
3. 输出修改摘要

---

## 6. Python 模块设计

### 6.1 accuracy_analyzer.py

```python
# rllm_remote/accuracy_analyzer.py

class AccuracyAnalyzer:
    """远程训练精度分析器"""
    
    def __init__(self, run_id: str, ssh_password: str | None = None)
    
    # Step 1: 数据获取
    def load_config(self) -> RemoteTrainConfig
    def identify_algorithm(self, config) -> str  # 返回算法名称
    def fetch_all_accuracy_tags(self) -> dict[str, list[float]]
    
    # Step 2: 诊断
    def compute_trends(self, tag_data) -> dict
    def detect_anomalies(self, tag_data, trends) -> list[Anomaly]
    def diagnose(self, tag_data, trends, algorithm) -> list[Diagnosis]
    def compute_health_scores(self, diagnoses) -> dict
    
    # Step 3: 建议
    def generate_suggestions(self, diagnoses, algorithm, config) -> list[Suggestion]
    
    # Step 4: 输出
    def write_analysis_json(self, results) -> str  # 返回路径
    def write_report(self, results) -> str  # 返回路径
    
    # Step 5-6: 审批和修改（由Skill层通过LLM交互完成）
    def apply_suggestions(self, config, approved_suggestions) -> RemoteTrainConfig
    
    # 主入口
    def analyze(self) -> str  # 执行Step 1-4, 返回报告
```

### 6.2 CLI入口

```bash
# 完整分析
python -m rllm_remote.accuracy_analyzer remote_1778504118 --ssh-password "<your-password>"

# 仅诊断（不生成报告）
python -m rllm_remote.accuracy_analyzer remote_1778504118 --diagnose-only

# 从已有 analysis.json 生成报告
python -m rllm_remote.accuracy_analyzer remote_1778504118 --from-cache
```

### 6.3 rllm_remote/__init__.py 新增

```python
def connect(run_id: str, ssh_password: str | None = None) -> tuple[RemoteTrainConfig, RemoteExecutor]:
    """加载配置并创建RemoteExecutor。规范化的远程连接入口。"""
```

---

## 7. Skill 结构

### 7.1 rllm-remote-connect

```
skill-bank/rllm/rllm-remote-connect/
├── base.md
│   ├── intro                  — SSH连接管理专家
│   ├── connection-setup       — 使用 connect() 或手动构造 RemoteExecutor
│   ├── password-management    — SSH_ASKPASS 机制、密码不落盘
│   ├── verify                 — 连接验证、容器状态检查
│   └── troubleshooting        — 连接超时、认证失败、容器未启动
└── manifest.yaml
```

### 7.2 rllm-analyze-accuracy

```
skill-bank/rllm/rllm-analyze-accuracy/
├── base.md
│   ├── intro                  — 精度分析专家，三层模型概述
│   ├── data-acquisition       — 如何获取TB数据和配置
│   ├── algorithm-identification — 从config识别当前算法
│   ├── tb-metric-model        — 6组44个精度指标的定义和诊断规则
│   ├── algorithm-models       — 各算法配置模型（引用patches）
│   ├── diagnosis-workflow     — 完整诊断流程
│   ├── report-generation      — 报告格式模板
│   ├── suggestion-workflow    — 建议生成和审批流程
│   └── config-modification    — 配置修改执行步骤
├── manifest.yaml
└── patches/
    ├── alg-grpo.md             — GRPO算法完整定义
    ├── alg-gae-ppo.md          — GAE-PPO算法完整定义
    ├── alg-dapo.md             — DAPO算法完整定义
    ├── alg-gspo.md             — GSPO算法完整定义
    └── alg-gpg.md              — GPG算法完整定义
```

每个算法 patch 格式：
```markdown
---
id: alg-grpo
target_section: algorithm-models
action: append
description: GRPO算法配置模型
status: active
---

## GRPO (Group Relative Policy Optimization)

### 参数集
[参数表格]

### TB指标预期模式
[正常/异常模式]

### 常见失效模式
[诊断 → 调优映射表]

### 特殊说明
[区别于其他算法的注意事项]
```

---

## 8. 与现有系统的关系

- `rllm-analyze` 保持不变，继续服务本地训练
- `rllm-remote-monitor` 保持不变，继续做实时监控
- `rllm-analyze-accuracy` 是新的独立 skill，面向远程训练的事后精度分析
- `rllm-analyze-performance` 性能分析（后续开发）
- 配置修改只改本地 `config.json`，远程 YAML 由 `rllm-remote-run` 在启动时生成

---

## 9. 执行计划

| # | 任务 | 文件 | 
|---|------|------|
| 1 | 创建 accuracy_analyzer.py | `rllm_remote/accuracy_analyzer.py` (new) |
| 2 | 添加 connect() 工厂 | `rllm_remote/__init__.py` (modify) |
| 3 | 创建 rllm-remote-connect skill | `skill-bank/rllm/rllm-remote-connect/` (new) |
| 4 | 创建 rllm-analyze-accuracy skill (base + 5 algorithm patches) | `skill-bank/rllm/rllm-analyze-accuracy/` (new) |
| 5 | 注册 + 编译 | `skill-bank/bank.yaml` (modify) + compile |
| 6 | 集成测试 | 使用 remote_1778504118 验证 |

---

## 10. 使用方式

### 10.1 命令行

```bash
# 完整分析（抓取TB数据 + 诊断 + 生成报告）
python -m rllm_remote.accuracy_analyzer <run_id> --ssh-password "<password>"

# 分析完成后，审批并应用特定建议
python -m rllm_remote.accuracy_analyzer <run_id> --apply 1 3 5

# 非交互式应用全部建议
python -m rllm_remote.accuracy_analyzer <run_id> --apply-all --ssh-password "<password>"
```

### 10.2 Skill 调用

```
/rllm-analyze-accuracy <run_id>
```

不带 run_id 时自动查找 `rllm_remote/output/runs/` 下最新的 run。

### 10.3 Python API

```python
from rllm_remote import connect
from rllm_remote.accuracy_analyzer import AccuracyAnalyzer

# 方式1：使用 connect() 工厂
config, executor = connect("remote_1778504118", ssh_password="...")

# 方式2：直接使用 AccuracyAnalyzer
analyzer = AccuracyAnalyzer("remote_1778504118", ssh_password="...")
report_path = analyzer.analyze()  # 执行完整分析

# 读取分析结果
import json
with open(f"rllm_remote/output/runs/remote_1778504118/analysis.json") as f:
    results = json.load(f)

# 审批建议后应用
analyzer.apply_suggestions([1, 3])  # 应用建议 #1 和 #3
```

### 10.4 输出文件

```
rllm_remote/output/runs/<run_id>/
├── analysis.json         # 结构化分析数据（程序可读）
├── accuracy_report.md    # 人类可读分析报告（4部分）
└── config.json.bak.xxx   # 配置修改前的备份
```

---

## 11. 扩展方式

### 11.1 新增算法

新增一个算法变体只需两步，不改任何已有代码：

**Step 1：创建算法 patch 文件**

在 `skill-bank/rllm/rllm-analyze-accuracy/patches/` 下新建文件，如 `alg-xxx.md`：

```yaml
---
id: alg-xxx
target_section: algorithm-models
action: append
description: XXX 算法完整定义
status: active
---

## XXX 算法

### 适用场景
...

### 识别条件
- `algorithm.adv_estimator == "xxx"` 或 `policy_loss.loss_mode == "xxx"`

### 核心参数
| 参数 | 默认值 | 作用 | 调整方向 |
...

### 额外参数
| 参数 | 默认值 | 作用 |
...

### TB 指标预期模式
- 某指标=某值是正常行为，不要误报
- 与其他算法的关键区别

### 常见失效模式与调优
| 失效 | TB 信号 | 根因 | 调整 |
...

### 特殊说明
...
```

**Step 2：注册到 manifest.yaml**

```yaml
active:
  - alg-grpo
  - alg-gae-ppo
  - alg-dapo
  - alg-gspo
  - alg-gpg
  - alg-xxx  # 新增
```

如果新算法引入了新的参数识别方式，需要同步更新 `accuracy_analyzer.py` 中的 `identify_algorithm()` 函数。

### 11.2 新增 TB 指标组

在 `accuracy_analyzer.py` 的 `TAG_GROUPS` 字典中添加新组，以及在 `detect_anomalies()` 中添加对应的检测规则。

### 11.3 新增异常检测规则

在 `accuracy_analyzer.py` 的 `detect_anomalies()` 函数中添加新规则，遵循模式：

```python
if <触发条件>:
    anomalies.append(Anomaly(
        id="ANOMALY_NAME",
        severity="critical|warning|info",
        description="人类可读描述",
        evidence=f"具体数据: {value}",
    ))
```

同时在 `generate_suggestions()` 中添加对应的调参建议。
