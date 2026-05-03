---
description: CLI-2 orchestrator for the dual-CLI architecture. Reads training trajectories,
  segments, analyzes, and generates skill patches. Runs in a separate CLI session
  from rllm-train.
metadata:
  categories:
  - trajectory
  - orchestration
  version: 1.0.0
name: traj-train-optimize
---


# traj-train-optimize — 训练轨迹优化编排 (CLI-2)

你是优化 Agent 的编排者，运行在独立的 CLI 会话中（Terminal 2）。你的职责是读取训练 Agent（CLI-1）产出的轨迹数据，执行分割、分析、优化的完整流程，将优化结果写入 skill-bank。

你不执行训练。训练由 CLI-1 中的 /rllm-train 负责。两个 CLI 通过 `traj_opt/output/rounds/` 目录下的状态文件协调。

详见 `docs/trajectory-design.md` Section 18。

## 执行规则

1. 每个步骤必须通过调用对应的 skill 执行，不得内联
2. 唯一需要人工介入的环节是确认 patch（设计准则 3.4）
3. 只从 `traj_opt/output/` 读取数据，不直接访问 `rllm_train/output/`
4. 不调用 rllm-train 或任何 rllm-xx skill（训练由 CLI-1 负责）
5. 不使用 Agent 子 agent（CLI-2 本身就是独立进程，天然隔离）

## 执行步骤

### 0. 解析参数

从用户输入中提取:
- round 号（必需，或 "latest" 自动查找最新待优化轮次）

示例输入:
```
/traj-train-optimize round=1
/traj-train-optimize latest
```

如果参数为 "latest":
```python
from traj_opt.round_state import RoundState
rs = RoundState()
round_num = rs.find_pending_optimization()
if round_num is None:
    输出 "没有待优化的轮次。请先在 CLI-1 中执行 /rllm-train。"
    退出
```

### 1. 读取轮次状态

```python
from traj_opt.round_state import RoundState
rs = RoundState()
status = rs.read_status(round_num)
```

检查状态:
- `status is None` → 输出 "Round {N} 不存在。请先在 CLI-1 中执行 /rllm-train \"round={N} | ...\"" 并退出
- `status["status"] == "optimization_complete"` → 输出 "Round {N} 已优化完成" 并展示摘要，退出
- `status["status"] == "training_failed"` → 输出 "Round {N} 训练失败: {error}"，询问是否仍要分析部分数据
- `status["status"] == "training_complete"` → 继续执行

提取关键信息:
```
session_id = status["training"]["session_id"]
run_id = status["training"]["run_id"]
reward = status["training"]["reward"]
```

输出:
```
Round {N} 训练已完成:
  Run ID:    {run_id}
  Reward:    {reward}
  Session:   {session_id}
  开始优化...
```

### 2. 验证轨迹数据完整性

检查 `traj_opt/output/rllm/raw/{session_id}/events.jsonl` 是否存在且有数据:

```python
from traj_opt.store.reader import EventReader
from traj_opt.config import DEFAULT_CONFIG

reader = EventReader(DEFAULT_CONFIG)
events = reader.read_session_events(session_id)
event_count = len(events)
```

- 如果 event_count == 0: 输出警告 "Hooks 未捕获到训练数据，分析可能不完整" 但继续
- 如果 event_count > 0: 输出 "轨迹数据: {event_count} 个事件"

### 3. 分割轨迹

调用 Skill("traj-segment", args="--session {session_id}")

等待 traj-segment 完成后继续。

### 4. 分析轨迹

调用 Skill("traj-analyze-rllm", args="--session {session_id}")

等待分析完成，获取报告路径。

### 5. 生成 Patch

调用 Skill("traj-optimize", args="{report_path}")

等待用户确认 patch 并编译。

### 6. 更新轮次状态

```python
from traj_opt.round_state import RoundState
rs = RoundState()
rs.write_optimization_complete(
    round_num=round_num,
    report_path=report_path,
    patches_generated=patches_generated,
    patches_accepted=patches_accepted,
)
```

### 7. 输出轮次摘要

```
Round {N} 优化完成:
  训练:    run_id={run_id}, reward={reward}
  分析:    {suggestion_count} 条优化建议
  Patch:   {patches_generated} 生成 / {patches_accepted} 接受
  状态:    traj_opt/output/rounds/round_{N}/status.json → optimization_complete

下一步: /traj-launch-training round={N+1} | {训练描述}
```

## 上下文隔离

本 skill 运行在独立的 CLI 会话中（Terminal 2），与训练 Agent（CLI-1）物理隔离:

- CLI-2 的对话上下文中不包含任何 CLI-1 的训练细节
- 分析器只能从 `traj_opt/output/rllm/` 中的轨迹数据推断训练情况
- 不需要 Agent 子 agent 来实现隔离 — 进程边界已经提供了更强的隔离

数据边界:
- 允许读取: `traj_opt/output/rllm/` (轨迹数据)、`traj_opt/output/rounds/` (状态文件)
- 允许写入: `skill-bank/` (patch)、`.claude/skills/` (编译)、`traj_opt/output/rounds/` (状态更新)
- 禁止读取: `rllm_train/output/` (训练原始输出)
