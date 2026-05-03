---
id: "002-param-constraints"
target_section: "tuning"
action: append
description: "增加参数联动约束检查，防止 num_generations 不整除和 grad_accum 副作用"
source: "2026-04-30 训练实验, run_1777513160(整除错误), run_1777521505(grad_accum副作用)"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

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
