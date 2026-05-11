---
id: "remote-backend-routing"
target_section: "phase0"
action: append
description: "Add remote NPU backend detection in Phase 0 input grading, route training to remote skills (rllm-remote-run, rllm-remote-monitor) when backend=remote"
source: "2026-05-11 Remote NPU training feature"
created: "2026-05-11"

depends_on:
  - "rllm-remote-run"
  - "rllm-remote-monitor"
conflicts_with: []

status: active
superseded_by: ""
---

### Remote Backend 检测

在 Phase 0 输入分级时，额外检查是否包含远程 NPU 训练关键词：

| 关键词 | 含义 |
|---|---|
| `NPU`, `npu`, `Ascend`, `昇腾`, `ascend` | 使用远程 NPU 后端 |
| `remote`, `远程`, `remote backend` | 使用远程后端 |
| `192.168`, `服务器`, `server` | 目标为远程服务器 |

如果检测到远程关键词，设置 `backend=remote`，后续 Phase 使用远程子 skill。

### Phase 路由（远程模式）

当 `backend=remote` 时，Phase 映射变更：

| Phase | 本地模式 (默认) | 远程模式 (backend=remote) |
|---|---|---|
| Phase 1 (需求澄清) | rllm-clarify | rllm-clarify (不变) |
| Phase 2 (配置生成) | rllm-config | rllm-config (传入 backend=remote) |
| Phase 3 (启动训练) | rllm-run | **rllm-remote-run** |
| Phase 4 (过程监控) | rllm-monitor | **rllm-remote-monitor** |
| Phase 5 (结果分析) | rllm-analyze | rllm-analyze (传入 backend=remote) |
| Phase 6 (最终报告) | 编排者执行 | 编排者执行 (标注远程模式) |

### 远程模式数据传递

```
Phase 2 → Phase 3: RemoteTrainConfig JSON (rllm_remote/output/runs/<run_id>/config.json)
Phase 3 → Phase 4: run_id + 远程日志路径 + tmux session name
Phase 4 → Phase 5: 训练完成确认 + run_id (结果已下载到本地)
Phase 5 → Phase 2: analysis.json
```

### Phase 3 (远程): 启动远程训练

**调用子 skill: rllm-remote-run**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-remote-run", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-remote-run/SKILL.md`，然后严格按其步骤执行

输入: run_id (从 Phase 2 的 config.json 路径中提取)
输出: 服务器地址 + 容器名 + 远程日志路径 + tmux session 名称
完成标志: tmux session 已创建，远程日志开始写入

### Phase 4 (远程): 远程监控

**调用子 skill: rllm-remote-monitor**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-remote-monitor", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-remote-monitor/SKILL.md`，然后严格按其步骤执行

输入: run_id
输出: 训练完成确认 + 最终 reward
完成标志: 训练进程退出或日志中出现完成标志，结果已下载到本地

### 远程 Heartbeat（双 CLI 模式）

远程模式下 heartbeat 写入与本地模式相同（编排层写入），但 Phase 3-4 期间：
- TrainingLogger 日志在远程容器内，无法直接写入本地 heartbeat
- rllm-remote-monitor 定时从远程拉取日志，解析 reward，写入本地 heartbeat

Heartbeat 写入逻辑：
```python
# rllm-remote-monitor 在每次日志拉取后更新 heartbeat
from traj_opt.round_state import RoundState

RoundState().write_heartbeat(
    round_num, run_id, phase="training",
    step=f"{current_step}/{total_steps}",
    reward=latest_reward,
    message=f"远程训练中 | server={ssh_host}"
)
```
