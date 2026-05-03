---
id: "001-early-stopping"
target_section: "anomaly-detection"
action: replace
description: "调整异常阈值 + 增加 Early Stopping 机制和 Epoch 边界监控"
source: "2026-04-30 训练实验, run_1777516933(浪费46步), run_1777530664(forgetting)"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### 异常检测（修订）

| 异常 | 检测方式 | 处理 |
|---|---|---|
| Reward 归零 | 连续 5 步 reward=0 | 建议停止训练 (非仅报告) |
| Reward 崩溃 | reward 从 >0.3 降到 0 且持续 3 步 | 立即建议停止，诊断为 lr 过高或 forgetting |
| Loss 爆炸 | loss > 10 或 NaN/Inf | 立即建议停止 |
| OOM | "out of memory" | 立即建议停止 |
| 进程崩溃 | Traceback + 进程退出 | 报告错误 |
| 训练卡住 | 超过 120s 无输出 | 报告，检查进程状态 |

### Early Stopping 机制

Monitor 检测到以下条件时，向编排层发送 STOP 建议:

1. 连续 5 步 reward=0 且当前 step > total_steps * 0.2
   → "训练已崩溃，建议停止。连续 5 步 reward=0，继续训练不会恢复。"

2. Epoch 切换后 reward 断崖 (需要按 epoch 计算)
   → "进入 Epoch N 后 reward 从 X 降到 0，疑似 catastrophic forgetting，建议停止。"

3. 模型输出异常 (从 trajectory 文件检测)
   → "模型输出格式退化，不再使用 tool_call，建议停止。"

### Epoch 边界监控

计算 epoch 边界: `epoch_boundary = num_problems / (batch_size * grad_accum)`

当 step 跨越 epoch 边界时:
  读取最近 3 步的 reward
  与上一个 epoch 最后 3 步的 reward 对比
  如果下降 > 50%: 发出 catastrophic forgetting 预警
