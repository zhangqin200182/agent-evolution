---
id: traj-20260503-091950-data-surfacing
target_section: data-surfacing
action: append
description: 增加定期 tail 检查频率，确保训练数据被 hooks 捕获
status: proposed
source: trajectory-analysis
source_sessions: ["b4d588ba-052e-4153-9c8b-5681a8850d9f"]
---

### 定期 Tail 检查频率

除 Monitor 实时流式监控外，增加定期 tail 检查:
- 每 2 分钟或每 10 步（以先到者为准）执行一次 `tail -30 training_log.txt`
- 这确保训练数据通过 Read/Bash 工具被 hooks 捕获到轨迹中
- Monitor 的 grep 输出不被 PostToolUse hook 记录，因此 tail 检查是数据表面化的关键补充
