---
description: Generate or adjust TrainingConfig for rllm_train agent RL training. Supports
  initial config generation from requirements and iterative hyperparameter tuning
  based on analysis results.
metadata:
  categories:
  - machine-learning
  - hyperparameter-tuning
  version: 1.0.0
name: rllm-config
---


# rllm-config — 训练配置生成与调参

你是 rllm_train 训练配置专家。你有两个职责：根据需求生成初始配置，以及根据训练分析结果调整配置。

## 配置文件位置

- 配置定义: `rllm_train/config.py` 中的 `TrainingConfig` dataclass
- 配置输出: `rllm_train/output/runs/<run_id>/config.json`

## 模式一：初始配置生成

根据 rllm-clarify 阶段的需求摘要，生成 TrainingConfig 的 JSON 配置文件。

### 步骤

1. 读取 `rllm_train/config.py` 确认当前 TrainingConfig 的所有字段和默认值
2. 根据需求摘要设置各参数
3. 根据运行环境（Mac CPU/MPS）调整参数：
   - batch_size 不超过 4（内存限制）
   - num_generations 不超过 8
   - 优先使用小模型（0.5B）做快速迭代
4. 生成 run_id（格式: `run_<timestamp>`）
5. 将配置写入 JSON 文件

### 初始配置推荐值

| 场景 | problems | epochs | lr | batch | generations |
|---|---|---|---|---|---|
| 快速测试 | 16 | 1 | 1e-5 | 2 | 2 |
| 标准训练 | 64 | 2 | 1e-5 | 2 | 4 |
| 深度训练 | 128-256 | 3-5 | 5e-6 | 2 | 4 |

### 难度配置

difficulty 参数控制训练数据的难度分布:
- `"simple"`: 100% 简单两数运算 (适合流程验证)
- `"hard"`: 100% 多步骤应用题 (0.5B 模型几乎无法学会)
- `"mixed"`: 80% simple + 20% hard (推荐，提供学习信号的同时引入挑战)

初始配置推荐:

| 场景 | difficulty | 原因 |
|------|-----------|------|
| 流程验证 | simple | 确认 pipeline 正常 |
| 正式训练 | mixed | 80/20 比例经验证有效 |
| 能力评估 | hard | 评估模型上限，不用于训练 |

调参时的难度调整:
- reward=1.0 + loss=0 → 题目太简单，切换到 mixed 或 hard
- reward<0.1 + difficulty=hard → 太难，切换到 mixed
- mixed 下 reward 在 0.3-0.7 → 比例合适，保持不变

### Seed 随机化

当调参循环生成新配置时，如果核心超参（lr, epochs, num_problems, difficulty, num_generations, batch_size）与上一轮完全相同，自动更换 seed:

```python
import time
if config_unchanged_from_previous:
    config.seed = int(time.time()) % 100000
```

这确保即使配置相同，每轮训练也使用不同的训练数据排列，产生独立的 reward 数据点。

### 初始 difficulty 推荐

生成初始配置时，如果满足以下条件，自动提升 difficulty:

| 条件 | 推荐 difficulty |
|------|----------------|
| 0.5B 模型 + num_problems <= 32 + math task | hard |
| 0.5B 模型 + num_problems > 32 + math task | mixed |
| 1.5B+ 模型 | mixed (默认) |

如果同一模型 + 相同 difficulty 的历史训练 reward 全程 >= 0.95，自动提升:
- mixed → hard
- simple → mixed

此规则在初始配置生成时检查，不依赖调参循环。

## 模式二：调参优化

根据 rllm-analyze 阶段的分析结果，调整配置参数。

### 调参策略

读取上一轮的分析结果（`analysis.json`），根据以下规则调整：

**效果问题**:
| 症状 | 调整 | 原因 |
|---|---|---|
| reward 低 + loss 不降 | learning_rate ×2 或 num_generations +2 | 学习信号不足 |
| reward 低 + loss 降 | num_epochs +2 或 num_problems ×2 | 需要更多训练 |
| reward 震荡 | learning_rate ÷2, grad_accum_steps ×2 | 训练不稳定 |
| reward plateau | temperature ±0.1, 尝试不同 loss_type | 探索不足或过度 |
| reward 下降 | learning_rate ÷5, 回退到上一轮配置 | 过拟合或学习率过大 |

**性能问题**:
| 症状 | 调整 | 原因 |
|---|---|---|
| 训练太慢 | max_completion_length ÷2, max_agent_steps -1 | 减少生成长度 |
| 内存不足 | batch_size ÷2, num_generations ÷2 | 减少内存占用 |

### 参数联动约束（生成配置前必须验证）

1. **TRL 整除约束**:
   `(batch_size * gradient_accumulation_steps) % num_generations == 0`
   违反时: 自动调整 num_generations 为最近的合法值

2. **generation_batch_size 副作用**:
   `generation_batch_size = batch_size * gradient_accumulation_steps`
   当 grad_accum 增大时，每步生成的 trajectory 数量也增大
   影响: GRPO baseline 估计变化，训练动态改变
   建议: 调整 grad_accum 时同步说明对 generation_batch_size 的影响

3. **显存估算 (MPS)**:
   `estimated_mem = batch_size * num_generations * max_completion_length * model_params * 4`
   - 0.5B + batch=2 + gen=4 + len=256 ≈ 安全
   - 0.5B + batch=2 + gen=4 + len=512 ≈ 可能 OOM
   超出估算时: 自动降低 max_completion_length 或 num_problems

### 渐进式难度升级

调参时的难度调整增加渐进规则:

当前轮次 avg_reward >= 0.7 且 loss=0 时:
- 当前 difficulty=simple → 升级到 mixed
- 当前 difficulty=mixed (20% hard) → 升级到 mixed-hard (50% hard)
- 当前 difficulty=mixed-hard → 升级到 hard
- 同时增加 max_agent_steps: 3 → 5（给模型更多推理空间）

禁止直接从 mixed 跳到 hard:
- 轨迹证据: R3 mixed avg=0.77 → R4 hard avg=0.19（断崖下降）
- 需要中间级别 mixed-hard 作为过渡

difficulty 参数扩展:
- `"mixed-hard"`: 50% simple + 50% hard（新增，介于 mixed 和 hard 之间）

### Reward 饱和处理

当分析结果显示 reward=1.0 且 loss 接近 0 时，问题不在超参而在数据难度:

| 条件 | 建议 | 原因 |
|------|------|------|
| reward=1.0, loss=0, difficulty=simple | 切换到 mixed | 简单题已掌握，引入挑战 |
| reward=1.0, loss=0, difficulty=mixed | 增加 hard 比例或切换到 hard | mixed 中的 hard 题比例不足 |
| reward=1.0, loss=0, difficulty=hard | 训练完成，模型已达上限 | 无需继续训练 |

此规则优先级高于其他调参建议 — 当 reward 已饱和时，调整 lr/epochs 无意义。

当 difficulty=mixed 时，0.5B 模型的 num_problems 安全上限:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| num_problems | 32 | 40 | difficulty=mixed 且 model=0.5B 且 lr<=5e-6 且 epochs<=1 | Round 1-3: 40p 稳定 (reward 0.86-1.0), 64p forgetting (step 13), 128p forgetting (step 49) |

推荐配置 (0.5B + mixed):
- num_problems=40, lr=5e-6, epochs=1, batch=2, generations=4
- 预期: reward 稳定在 0.8-1.0 范围

警告: num_problems >= 64 在 mixed difficulty 下会导致 catastrophic forgetting，即使 lr=5e-6 且 epochs=1。

#### Seed 配置

- 默认使用随机 seed（`int(time.time()) % 100000`），确保每轮训练使用不同的问题集
- 如果用户明确指定 seed（如 "seed=42"），则使用用户指定值
- 调参循环中，每轮自动使用不同 seed，避免过拟合到特定问题集

#### Seed 配置

- 默认使用随机 seed（`int(time.time()) % 100000`），确保每轮训练使用不同的问题集
- 如果用户明确指定 seed（如 "seed=42"），则使用用户指定值
- 调参循环中，每轮自动使用不同 seed，避免过拟合到特定问题集

当 num_problems >= 40 且 difficulty=mixed 时，0.5B 模型的安全范围进一步收紧:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| num_problems | 32 | 24 | difficulty=mixed 且 model=0.5B | lr=5e-6 + 1 epoch + 40 problems 仍在 step 21 开始 entropy 爆增和 reward 衰退 |
| learning_rate | 5e-6 | 3e-6 | num_problems >= 40 且 model=0.5B | 5e-6 不足以防止 40 problems 时的后半段衰退 |

推荐配置 (0.5B + mixed + 快速验证):
- num_problems=24, lr=5e-6, epochs=1, batch=2, generations=4
- 预期: 6 步训练，reward 稳定不崩溃

替代方案: 保持 num_problems=40 但 lr=3e-6
- 适用于需要更多训练步数观察趋势的场景

## 配置预检（生成配置后、启动前执行）

### 必检项（不通过则拒绝启动）

1. **TRL 整除约束**:
   `(batch_size * gradient_accumulation_steps) % num_generations == 0`
   失败时: 自动调整 num_generations

2. **模型安全上限**:
   查模型级别安全配置表，检查 lr/epochs/completion_length 是否超限
   失败时: 自动降到安全值并警告

3. **difficulty 合法性**:
   `difficulty in ("simple", "hard", "mixed")`
   失败时: 默认 "mixed"

### 建议检项（不通过则警告但允许启动）

4. **显存估算 (MPS)**:
   if `batch_size * num_generations * max_completion_length > 阈值`:
   警告: "可能 OOM，建议降低 max_completion_length 或 num_problems"

5. **训练时间估算**:
   `estimated_time = num_problems * num_epochs / (batch_size * grad_accum) * avg_step_time`
   if estimated_time > 30min:
   警告: "预计训练时间 Xm，确认继续？"

## 输出

生成配置后，用 Python 代码将配置写入 JSON 文件：

```python
python -c "
from rllm_train.config import TrainingConfig
config = TrainingConfig(
    model_name='Qwen/Qwen2.5-0.5B-Instruct',
    num_problems=64,
    num_epochs=2,
    # ... 其他参数
)
config.to_json('rllm_train/output/runs/<run_id>/config.json')
print(config.summary())
"
```

## 调参时的输出格式

```
配置调整（第 N 轮 → 第 N+1 轮）：
  learning_rate:  1e-5 → 2e-5  (reward 低但 loss 在降，加大学习率)
  num_epochs:     2 → 4        (reward 趋势向上，增加训练量)
  temperature:    0.7 → 0.6    (减少随机性，稳定输出)
  其他参数保持不变
```

### Remote 模式配置生成

当 args 中包含 `backend=remote` 时，进入远程配置模式。

#### 远程配置生成流程

1. 读取 `rllm_remote/config.py` 了解 RemoteTrainConfig 参数
2. 根据需求摘要设置参数：

| 需求参数 | RemoteTrainConfig 字段 | 说明 |
|---|---|---|
| model_name | model_name_or_path | 使用服务器路径而非 HF 名称 |
| num_epochs | num_epochs | 直接映射，远程可设更大值 |
| batch_size | train_batch_size | 远程 NPU 可设更大 (默认 32) |
| num_generations | n_samples_per_prompt | 直接映射 |
| learning_rate | lr | 直接映射 |
| temperature | temperature | 直接映射 |
| max_agent_steps | max_agent_steps | 直接映射 |

3. 服务器路径默认值：

```python
model_name_or_path = "/opt/DPC/models/l00619320/code/AGENTIC_RL_WS/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct"
train_data_path = "/opt/DPC/models/l00619320/code/VERL_NPU_WS/data/gsm8k/train.parquet"
val_data_path = "/opt/DPC/models/l00619320/code/VERL_NPU_WS/data/gsm8k/test.parquet"
ssh_host = "192.168.9.142"
container_name = "agent5.0.0_qjy"
```

4. 写配置文件：

```bash
python -c "
from rllm_remote.config import RemoteTrainConfig
config = RemoteTrainConfig(
    model_name_or_path='<model_path>',
    num_epochs=<n>,
    train_batch_size=<n>,
    lr=<lr>,
    ...
)
config.to_json('rllm_remote/output/runs/<run_id>/config.json')
print('OK: config.json written')
print(config.summary())
"
```

#### NPU 参数安全范围

| 参数 | 最小值 | 最大值 | 默认值 | 说明 |
|---|---|---|---|---|
| train_batch_size | 8 | 128 | 32 | NPU 8卡可开大 batch |
| ppo_mini_batch_size | 1 | 32 | 8 | 必须能整除 train_batch_size |
| lr | 1e-7 | 1e-4 | 1e-6 | NPU 训练建议较低 lr |
| num_epochs | 10 | 500 | 100 | 远程可长时间训练 |
| n_samples_per_prompt | 2 | 16 | 8 | 受 NPU 显存限制 |
| tensor_parallel_size | 1 | 8 | 4 | 不超过 n_npus_per_node |
| max_response_length | 256 | 4096 | 2048 | 受 NPU 显存限制 |
| kl_coef | 0.0 | 0.1 | 0.001 | 防止 policy 偏离 |

#### 调参规则（远程模式）

与本地调参规则不同，远程模式有更大调整空间：

| 问题 | 远程调参策略 |
|---|---|
| Reward 低 + Loss 不收敛 | lr * 2, 增加 epochs |
| Reward 低 + Loss 下降 | 增加 epochs, 增大 n_samples_per_prompt |
| Reward 震荡 | lr / 2, kl_coef * 2 |
| NPU OOM | 减小 batch_size, max_response_length, tp_size |
| 训练太慢 | 增大 batch_size, 减小 epochs |
