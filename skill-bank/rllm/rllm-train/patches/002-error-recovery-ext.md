---
id: "002-error-recovery-ext"
target_section: "error-recovery"
action: replace
description: "扩展错误恢复表，覆盖 lr 崩溃、catastrophic forgetting、grad_accum 副作用、格式退化等场景"
source: "2026-04-30 训练实验, 8 轮训练暴露的各类错误场景"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

## 错误恢复策略（修订）

| 场景 | 检测方式 | 恢复策略 |
|------|---------|---------|
| OOM | "out of memory" | 自动: max_completion_length ÷2, 如仍 OOM 则 num_problems ÷2 |
| num_generations 不整除 | ValueError 启动失败 | 自动: 调整 num_generations 为最近合法值 |
| lr 过高致策略崩溃 | reward 从 >0 骤降到 0 且不恢复 | 自动: lr ÷2, 重新训练 |
| catastrophic forgetting | Epoch N+1 reward < Epoch N * 0.3 | 自动: epochs 设为当前 epoch 数 -1, 重新训练 |
| grad_accum 副作用 | 训练从第 1 步就 reward=0 | 建议: 回退 grad_accum 到上一轮值 |
| 格式退化 | tool_call 使用率后期 < 前期 50% | 建议: 减少 epochs, 或增加格式辅助 reward |
| 进程崩溃 (Traceback) | 日志含 Traceback | 读取错误信息，诊断后调整配置重试 |
| 连续 2 轮失败 | history 中连续 2 轮 reward 未提升 | 暂停，向用户报告，建议换模型或调整任务 |
