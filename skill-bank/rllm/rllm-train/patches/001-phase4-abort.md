---
id: "001-phase4-abort"
target_section: "phase1-5"
action: append
description: "增加 Phase 4.5 训练中止机制，支持 early stopping、用户中止、进程崩溃场景"
source: "2026-04-30 训练实验, 多次需要手动停止训练"
created: "2026-04-30"

depends_on:
  - "rllm-monitor:001-early-stopping"
conflicts_with: []

status: active
superseded_by: ""
---

### Phase 4.5: 训练中止（新增）

触发条件 (任一):
  - Monitor 发出 STOP 建议 (early stopping)
  - 用户主动要求停止
  - 训练进程崩溃 (OOM, Traceback)

执行步骤:
  1. TaskStop 训练后台任务
  2. TaskStop Monitor 任务
  3. 等待 5s 确认进程退出
  4. 读取已生成的 trajectory 文件和日志
  5. 进入 Phase 5 分析（即使训练未完成，也分析已有数据）

中止后的分析要点:
  - 标记 analysis.json 中 `"completed": false, "abort_reason": "..."`
  - 分析崩溃前的 reward 趋势
  - 如果是 early stopping: 诊断崩溃原因并给出针对性调参建议
  - 如果是用户中止: 保存状态，支持后续恢复

### 训练机制说明

每轮训练都从 base model (如 Qwen2.5-0.5B-Instruct) 重新加载权重。
上一轮的训练结果不会影响下一轮的初始权重。
调参循环改变的是训练配置（lr, epochs, difficulty 等），不是模型起点。

在 Phase 3 启动训练时提示:
  "第 N 轮训练: 从 base model 重新开始 (不继承上一轮权重)"

在 Phase 6 最终报告中说明:
  "每轮训练独立从 base model 开始，最终模型来自第 N 轮的训练结果"
