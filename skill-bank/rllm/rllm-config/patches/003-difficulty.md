---
id: "003-difficulty"
target_section: "initial-config"
action: append
description: "增加 difficulty 配置指导，包括难度选择建议和调参时的难度调整规则"
source: "2026-04-30 训练实验, run_1777465401(太简单), run_1777512419(太难), run_1777516127(mixed最佳)"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

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
