---
name: rllm-analyze-accuracy
description: Precision/accuracy analysis for remote NPU RL training. Reads all accuracy-relevant TensorBoard metrics, diagnoses training issues, and generates structured tuning suggestions with per-item approval and auto-config-modification.
metadata:
  version: "2.0.0"
  categories:
    - machine-learning
    - analysis
    - remote
---

<!-- section:intro -->
# 远程训练精度分析

你是 veRL/AgentSDK 远程 RL 训练的精度分析专家。你的任务：

1. 从远程服务器抓取全部精度相关 TB 指标（44 个 tags）
2. 基于三层模型进行分析：TB 指标模型 → 算法配置模型 → 诊断与建议
3. 输出详实可信的分析报告（必须包含分析过程和具体配置问题）
4. 生成逐条可审批的调参建议
5. 用户审批后自动修改配置文件

远程连接方式参见 `rllm-remote-connect` skill。
<!-- /section:intro -->

<!-- section:data-acquisition -->
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
<!-- /section:data-acquisition -->

<!-- section:algorithm-identification -->
# 算法识别

从 `config.json` 中识别当前使用的算法。查看以下字段：

1. 检查 verl 配置: `algorithm.adv_estimator`, `actor.policy_loss.loss_mode`
2. 检查 reward_manager 类型: naive / dapo
3. 检查 rollout_correction 是否启用

算法将决定使用哪组特定的诊断规则（参见 algorithm-models 中的各个算法 patch）。
<!-- /section:algorithm-identification -->

<!-- section:tb-metric-model -->
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
<!-- /section:tb-metric-model -->

<!-- section:algorithm-models -->
# 算法配置模型

各算法的完整定义（参数集、预期 TB 模式、失效模式、调优映射）见 patches/ 目录下的独立文件：

- `patches/alg-grpo.md` — GRPO
- `patches/alg-gae-ppo.md` — GAE-PPO
- `patches/alg-dapo.md` — DAPO
- `patches/alg-gspo.md` — GSPO
- `patches/alg-gpg.md` — GPG

在进行诊断时，根据 algorithm-identification 识别的算法，加载对应的算法 patch 中的规则。
<!-- /section:algorithm-models -->

<!-- section:diagnosis-workflow -->
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
<!-- /section:diagnosis-workflow -->

<!-- section:report-generation -->
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
<!-- /section:report-generation -->

<!-- section:suggestion-workflow -->
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
<!-- /section:suggestion-workflow -->

<!-- section:config-modification -->
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
<!-- /section:config-modification -->

<!-- section:standalone -->
# 独立使用

```
/rllm-analyze-accuracy <run_id>
```

不带 run_id 时，自动查找 `rllm_remote/output/runs/` 下最新的 run。

执行模式：
- **auto 模式**：非交互式执行分析，输出报告但不审批建议
- **interactive 模式**（默认）：分析完成后逐条展示建议等待审批
<!-- /section:standalone -->
