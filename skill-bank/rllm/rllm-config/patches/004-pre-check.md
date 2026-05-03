---
id: "004-pre-check"
target_section: "output"
action: prepend
description: "增加配置预检步骤，在生成配置后、启动训练前验证参数合法性"
source: "2026-04-30 训练实验, run_1777513160(整除错误), run_1777513256(OOM)"
created: "2026-04-30"

depends_on:
  - "001-model-safety"
  - "002-param-constraints"
conflicts_with: []

status: active
superseded_by: ""
---

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
