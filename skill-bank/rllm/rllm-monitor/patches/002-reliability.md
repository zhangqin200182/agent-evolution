---
id: "002-reliability"
target_section: "monitoring-methods"
action: append
description: "增加 Monitor 可靠性设计：通用 grep 模式、双重监控策略、生命周期管理、健康检查"
source: "2026-04-30 训练实验, 多次 Monitor 静默失效导致用户投诉"
created: "2026-04-30"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### Monitor 可靠性设计

#### 1. grep 模式通用化

当前（会失效）:
```bash
tail -f ... | grep -E "/64|···|Error"
```

改为（通用）:
```bash
tail -f ... | grep -E --line-buffered "^\s*[0-9]+/[0-9]+|···|Error|Traceback|OOM|Training Report|reward="
```

#### 2. 双重监控策略

主监控: Monitor 工具，persistent=true，做实时流式通知
备用监控: 当主监控连续 2 分钟无输出时，用 tail -20 读日志尾部

切换逻辑 (由 rllm-train 编排层管理):
  if 收到 Monitor 通知: `last_notification_time = now`
  if `now - last_notification_time > 120s`:
      执行 tail -20 检查训练状态
      if 训练仍在进行: 重启 Monitor (TaskStop 旧的 + 启动新的)
      if 训练已完成: 进入 Phase 5

#### 3. Monitor 生命周期管理

rllm-train 编排层增加:
  Phase 4 启动 Monitor 后，记录 monitor_task_id
  训练完成后: TaskStop(monitor_task_id) 清理
  Monitor 超时退出且训练未完成: 自动重启新 Monitor
  用户请求停止训练: 先 TaskStop(monitor_task_id)，再 kill 训练进程

#### 4. Monitor 健康检查

启动 Monitor 后 30s 内如果无任何输出:
  检查日志文件是否存在且在增长
  检查 grep 模式是否匹配到内容
  如果日志在增长但 grep 无匹配: 报告 grep 模式可能有误，切换到宽松模式
