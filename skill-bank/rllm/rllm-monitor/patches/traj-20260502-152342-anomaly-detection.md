---
id: traj-20260502-152342-anomaly-detection
target_section: anomaly-detection
action: append
description: 增加 loss=0 持续 N 步的异常检测
status: proposed
source: trajectory-analysis
source_sessions: ["65dbb67c-7794-4f87-a4ce-f6a7621eb39c"]
---

| Loss 持续为零 | 连续 10 步 loss=0 且 step > total_steps * 0.25 | 报告: "Loss 持续为 0，GRPO 可能未产生有效梯度。如果 reward 高，说明任务太简单；如果 reward 低，检查 num_generations 和 temperature" |
