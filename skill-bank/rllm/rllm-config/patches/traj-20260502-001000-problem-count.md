---
id: traj-20260502-001000-problem-count
target_section: param-ranges
action: append
description: "0.5B + mixed 难度时 num_problems 上限从 64 降到 32，防止 forgetting"
status: proposed
source: trajectory-analysis
source_sessions: ["run_1777651915"]
---

当 difficulty=mixed 时，0.5B 模型的 num_problems 安全上限进一步收紧:

| 参数 | 原上限 | 新上限 | 条件 | 依据 |
|------|--------|--------|------|------|
| num_problems | 64 | 32 | difficulty=mixed 且 model=0.5B | lr=5e-6 + 1 epoch + 64 problems 仍在 step 9 开始 forgetting |

推荐配置 (0.5B + mixed):
- num_problems=32, lr=5e-6, epochs=1, batch=2, generations=4
- 预期: 32 步训练，reward 稳定不崩溃

替代方案: 保持 64 problems 但切换 difficulty=simple
- 适用于需要更多训练数据但不需要 hard 题目的场景
