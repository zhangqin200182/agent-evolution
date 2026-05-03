---
id: "001-model-safety"
target_section: "param-ranges"
action: replace
description: "按模型大小区分参数安全上限，替换原有的宽泛参数范围"
source: "2026-04-30 训练实验, run_1777516933(lr崩溃), run_1777530664(forgetting)"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### 参数安全范围

#### 模型级别安全配置（硬约束）

生成配置时，必须根据模型大小查表，参数不得超出对应上限。

| 参数 | 0.5B 上限 | 1.5B 上限 | 3B 上限 | 依据 |
|------|----------|----------|--------|------|
| learning_rate | 1e-5 | 2e-5 | 5e-5 | 0.5B 在 2e-5 时策略崩溃 |
| num_epochs | 2 | 4 | 6 | 0.5B 在 4ep 时 catastrophic forgetting |
| max_completion_length | 256 | 512 | 512 | MPS 显存限制 |
| num_problems (MPS) | 32 | 32 | 16 | 配合 num_generations=4 的显存上限 |

调参建议超出上限时，必须警告并拒绝。例如：
  建议 num_epochs: 2 → 4
  → 检查: 0.5B 模型 epochs 上限为 2
  → 拒绝，改为建议: 增加 num_problems 或换 1.5B 模型

#### 通用参数范围

| 参数 | 最小值 | 最大值 | 说明 |
|---|---|---|---|
| learning_rate | 1e-7 | 1e-3 | 超出范围大概率不收敛 |
| temperature | 0.3 | 1.5 | 太低无探索，太高太随机 |
| num_generations | 2 | 8 | GRPO 至少需要 2 |
| batch_size | 1 | 4 | Mac 内存限制 |
| num_problems | 8 | 512 | 太少不够学，太多太慢 |
| num_epochs | 1 | 20 | 过多可能过拟合 |
| max_agent_steps | 1 | 8 | 影响生成长度和速度 |
| gradient_accumulation_steps | 1 | 16 | 等效增大 batch |
