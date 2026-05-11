---
description: End-to-end automated agent RL training with rllm_train. Orchestrates
  requirement clarification, config generation, training execution, monitoring, result
  analysis, and iterative hyperparameter tuning until training goals are met. Supports
  auto and approve execution modes.
metadata:
  categories:
  - machine-learning
  - agent-training
  - automation
  version: 2.0.0
name: rllm-train
---


# rllm-train — 自动训练主编排

你是 rllm_train agent RL 训练的全流程编排者。你负责串联需求澄清、配置生成、训练执行、过程监控、结果分析、调参优化的完整闭环，循环直到训练目标达成。

## 执行规则（必须遵守）

1. **每个 Phase 必须通过调用对应的子 skill 来执行**，不得跳过子 skill 直接执行其内部逻辑
2. 编排者（你）只负责: Phase 间的流转控制、数据传递、停止条件判断、状态追踪、Phase 0 引导问答、Phase 6 最终报告
3. 编排者不负责: 具体的需求解析、配置生成、训练启动、日志监控、结果分析 — 这些全部委托给子 skill
4. 禁止"内联执行" — 即使你知道子 skill 的逻辑，也必须通过下面的调用方式执行，不得自己手动操作（如直接写 config.json、直接读 config.py 解析参数）
5. **Skill 调用后立即停止** — 调用 `Skill("rllm-xxx")` 后，当轮响应必须立即结束，不得在同一轮响应中跟随任何 Bash、Read、Write、Edit 等工具调用。原因: Skill 工具是异步的，系统会在下一轮消息中注入 SKILL.md 内容，只有等到注入完成后才能按 SKILL.md 的步骤执行。如果在同一轮就开始执行操作，等于绕过了 skill 的注入流程，违反了规则 1 和 4
6. **Phase 间不跳步** — 即使上一轮的 analysis.json 已经给出了明确的调参建议，调参循环仍必须经过 Phase 2 (rllm-config) → Phase 3 (rllm-run) → Phase 4 (rllm-monitor) → Phase 5 (rllm-analyze) 的完整流程。禁止在编排层直接修改 config.json 或跳过 monitor 直接读日志。
7. **调参循环中的 Phase 4 不可省略** — 每次 rllm-run 启动训练后，必须调用 rllm-monitor 监控。不得因为"上一轮已经知道训练模式"而跳过监控。Monitor 负责异常检测和 early stopping，跳过会导致 catastrophic forgetting 无法被及时发现。

## 子 skill 调用方式

每个 Phase 调用子 skill 时，按以下优先级执行：

1. **首选: Skill 工具** — 使用 `Skill("rllm-xxx", args="...")` 调用。如果该 skill 出现在可用 skill 列表中，必须用此方式。**调用后当轮响应立即结束，等待下一轮系统注入 SKILL.md 后再执行步骤**
2. **备选: Read + 执行** — 如果 Skill 工具调用失败或 skill 不在可用列表中，则:
   - 用 Read 工具读取 `.claude/skills/rllm-xxx/SKILL.md`
   - 严格按照 SKILL.md 中描述的步骤逐步执行
   - 不得省略、合并或跳过 SKILL.md 中的任何步骤

## 数据传递契约

Phase 之间通过以下方式传递数据：

```
Phase 0 → Phase 1: 组装的自然语言描述（如"用 qwen-0.5b 训练数学 agent，reward 达到 0.5"）
Phase 1 → Phase 2: 需求摘要文本（包含模型、目标、所有参数、停止条件）
Phase 2 → Phase 3: config.json 文件路径 (rllm_train/output/runs/<run_id>/config.json)
Phase 3 → Phase 4: 后台任务 ID + 日志文件路径 (rllm_train/output/runs/<run_id>/training_log.txt)
Phase 4 → Phase 5: 训练完成确认 + run_id
Phase 5 → Phase 2（循环）: analysis.json 路径 (rllm_train/output/runs/<run_id>/analysis.json)
```

## 工作目录

`/Users/kevin/code/MyProject`

## 整体流程

```
Phase 0: 输入分级与引导 (编排者自己执行)
    ↓
Phase 1: 需求澄清 → 调用 rllm-clarify
    ↓
Phase 2: 配置生成 → 调用 rllm-config
    ↓
Phase 3-5: 训练循环
    ├→ 启动训练 → 调用 rllm-run
    ├→ 过程监控 → 调用 rllm-monitor
    ├→ 结果分析 → 调用 rllm-analyze
    └→ 判断是否达成 → 未达成则调用 rllm-config 调参并重新训练
    ↓
Phase 6: 最终报告 (编排者自己执行)
```

## 执行模式

### approve 模式（默认）

在每个关键决策点暂停等待用户确认：
1. 需求摘要确认
2. 初始配置确认
3. 每轮训练结束后的调参方案确认
4. 停止训练确认

### auto 模式

全自动执行，仅在以下情况暂停：
- 训练出错需要人工干预
- 达到停止条件
- 连续 2 轮调参后 reward 无改善（可能需要人工介入）

用户可通过以下方式指定模式：
- "auto 模式" / "全自动" / "自动执行" → auto
- "approve 模式" / "人工确认" / "每步确认" → approve
- 未指定 → 默认 approve

## 详细执行步骤

### Phase 0: 输入分级与引导（编排者自己执行）

收到用户输入后，先判断信息完整度，决定走哪条路径。

#### 分级规则

检查用户输入中是否包含以下关键信息：

| 关键信息 | 识别标志 | 权重 |
|---|---|---|
| 模型 | qwen/Qwen/模型名/0.5b/1.5b/3b | 必要 |
| 训练目标 | reward/目标/达到/>=/准确率 | 必要 |
| 数据规模 | N 个问题/problems/题 | 可选 |
| 执行模式 | auto/approve/自动/确认 | 可选 |

- **充分**（含模型 + 训练目标）→ 输出"正在解析训练需求..."，直接进入 Phase 1
- **部分**（含其中之一）→ 输出"收到，还需要补充一些信息："，用 AskUserQuestion 补缺失项，然后进入 Phase 1
- **模糊**（两者都不含，如"启动训练"、"开始"、"跑一下"、空输入）→ 输出"好的，先确认几个关键参数："，用引导问答收集信息，然后进入 Phase 1

#### 模糊输入引导问答

用一次 AskUserQuestion 同时问 2 个问题：

问题 1 — 模型选择：
```
header: "模型"
question: "用哪个模型训练？"
options:
  - label: "qwen-0.5b (推荐)"
    description: "最小最快，适合快速实验和验证流程"
  - label: "qwen-1.5b"
    description: "中等大小，效果和速度的平衡点"
  - label: "qwen-3b"
    description: "最大，效果最好但训练最慢"
```

问题 2 — 训练目标：
```
header: "目标"
question: "训练到什么程度？"
options:
  - label: "快速测试 (reward >= 0.5)"
    description: "验证流程是否跑通，几分钟完成"
  - label: "标准训练 (reward >= 0.8)"
    description: "正式训练，追求较好效果"
  - label: "充分训练 (reward >= 0.95)"
    description: "追求高准确率，耗时较长"
```

收到回答后，组装成完整描述（如"用 qwen-0.5b 训练数学 agent，reward 达到 0.5"），交给 Phase 1 处理。

#### 部分输入补充

只问缺失的那一个问题，不重复问已知信息。

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

### Phase 1: 需求澄清

**调用子 skill: rllm-clarify**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-clarify", args="<Phase 0 组装的完整描述>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-clarify/SKILL.md`，然后严格按其步骤执行

输入: Phase 0 组装的自然语言描述
输出: 结构化的需求摘要（包含模型、目标、所有参数、停止条件）
完成标志: 输出了格式化的需求摘要

⚠️ 禁止跳过 rllm-clarify，直接从用户输入中提取参数。

### Phase 2: 配置生成

**调用子 skill: rllm-config**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-config", args="初始配置 | <Phase 1 的需求摘要>")`
   - 如果是调参循环（非首轮），args 改为: `"调参 | run_id=<run_id> | <Phase 5 的调参建议>"`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-config/SKILL.md`，然后严格按其步骤执行

输入: 需求摘要（首轮）或 analysis.json 中的调参建议（后续轮）
输出: config.json 文件路径
完成标志: `rllm_train/output/runs/<run_id>/config.json` 已生成

在 approve 模式下，展示配置摘要并等待用户确认后再进入 Phase 3。

⚠️ 禁止跳过 rllm-config，直接写 config.json 或直接调用 TrainingConfig。

### Phase 3: 启动训练

**调用子 skill: rllm-run**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-run", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-run/SKILL.md`，然后严格按其步骤执行

输入: run_id（从 Phase 2 的 config.json 路径中提取）
输出: 后台任务 ID + 日志文件路径
完成标志: 训练进程已启动，日志文件开始写入

⚠️ 禁止跳过 rllm-run，直接用 Bash 启动 python 训练命令。

### Phase 4: 过程监控

**调用子 skill: rllm-monitor**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-monitor", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-monitor/SKILL.md`，然后严格按其步骤执行

输入: run_id + 后台任务 ID
输出: 训练完成确认（正常完成 / 异常退出）
完成标志: 训练进程退出 或 日志中出现 "Training Report"

⚠️ 禁止跳过 rllm-monitor，直接 tail 日志或轮询进程状态。

### Phase 5: 结果分析与调参

**调用子 skill: rllm-analyze**

操作步骤：
1. 使用 Skill 工具: `Skill("rllm-analyze", args="<run_id>")`
2. 如果 Skill 工具调用失败: 用 Read 工具读取 `.claude/skills/rllm-analyze/SKILL.md`，然后严格按其步骤执行

输入: run_id
输出: 分析报告 + 调参建议（写入 analysis.json）
完成标志: `rllm_train/output/runs/<run_id>/analysis.json` 已生成

⚠️ 禁止跳过 rllm-analyze，直接读取日志文件分析 reward 趋势。

**Phase 5 后的编排决策（编排者自己执行）：**

1. 读取 analysis.json 中的 `reward.reached` 字段
2. 判断停止条件（见下方"停止条件判断"）
3. 如果达成目标 → 进入 Phase 6
4. 如果未达成:
   - 更新 training_state.json
   - 在 approve 模式下展示调参建议并等待确认
   - 回到 Phase 2，传入 analysis.json 的调参建议，调用 rllm-config 生成新配置
   - 然后继续 Phase 3 → 4 → 5 循环

### Heartbeat（双 CLI 模式）

当 `round_num` 存在时（即通过 traj-launch-training 启动），在每个 Phase 转换点写入 heartbeat，供 CLI-2 的 traj-loop 轮询脚本读取进度:

```python
# 编排层在每个 Phase 完成后执行（仅当 round_num 存在时）
from traj_opt.round_state import RoundState

# Phase 2 完成后:
RoundState().write_heartbeat(round_num, run_id, phase="config", message="配置已生成")

# Phase 3 完成后:
RoundState().write_heartbeat(round_num, run_id, phase="training", message="训练已启动")

# Phase 3-4 训练期间（自动，无需编排层操作）:
# TrainingLogger 在每个 step 完成后自动写入 heartbeat（通过 TRAJ_HEARTBEAT_PATH 环境变量）
# 格式: {"phase": "training", "step": "3/16", "reward": 0.75, ...}
# 更新频率: 每个训练 step（约 5-15 秒一次）

# Phase 4 完成后:
RoundState().write_heartbeat(round_num, run_id, phase="monitoring",
                             step=f"{current_step}/{total_steps}", reward=latest_reward,
                             message="训练完成")

# Phase 5 开始时:
RoundState().write_heartbeat(round_num, run_id, phase="analyzing", message="分析中")

# Phase 5 完成后:
RoundState().write_heartbeat(round_num, run_id, phase="analyzing", reward=final_reward,
                             message="分析完成，准备调参")

# 调参循环回到 Phase 2 时:
RoundState().write_heartbeat(round_num, new_run_id, phase="tuning",
                             message=f"第 {attempt} 次调参")
```

heartbeat.json 写入 `traj_opt/output/rounds/round_{N}/heartbeat.json`，与 status.json 同目录。两层写入机制:
1. **训练进程直接写入**（Phase 3-4 期间）: TrainingLogger 每个 step 原子写入，提供实时 step/reward 进度
2. **编排层写入**（Phase 转换时）: 提供粗粒度阶段状态（config/analyzing/tuning）

CLI-2 通过文件 mtime 变化检测活跃度，实现自适应超时。

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

### Phase 6: 最终报告（编排者自己执行）

训练目标达成（或达到停止条件）后，输出最终报告：

```
训练完成报告
============
目标:       avg reward >= <target>
结果:       avg reward = <final> ✓/✗

训练历程:
  第 1 轮: reward <start> → <end>  配置: <key params>
  第 2 轮: reward <start> → <end>  调参: <changes>
  ...

总耗时:     <time> (<N> 轮训练)
最终模型:   rllm_train/output/runs/<run_id>/final_model/
所有记录:   rllm_train/output/runs/<run_id>/
```

## 停止条件判断

每轮训练结束后（Phase 5 完成后），编排者检查以下条件（按优先级）：

1. **reward_threshold**: 最终 avg reward >= 目标值 → 成功停止
2. **max_rounds**: 已达最大轮次 → 停止（可能未达标）
3. **max_wall_time**: 总耗时超限 → 停止（可能未达标）
4. **plateau_rounds**: 连续 N 轮 reward 提升 < 5% → 停止（plateau）
5. **reward 下降**: 连续 2 轮 reward 下降 → 警告，建议停止

## 状态追踪

在训练循环中维护以下状态（编排者自己管理）：

```json
{
  "round": 1,
  "history": [
    {"round": 1, "run_id": "run_xxx", "reward_start": 0.25, "reward_end": 0.45, "config_changes": []}
  ],
  "target": {"reward_threshold": 0.8, "max_rounds": 5, "plateau_rounds": 3},
  "mode": "approve",
  "current_run_id": "run_xxx",
  "completed": false
}
```

将状态写入 `rllm_train/output/training_state.json`，以便中断后恢复。

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

## 轮次完成信号（双 CLI 模式）

当 args 中包含 `round=N` 时，在 Phase 6（最终报告）完成后执行此步骤。

### Phase 6.5: 写入轮次状态

**保底机制**: 训练进程（TrainingLogger）在 print_training_report() 时会通过 `TRAJ_ROUND_NUM` 环境变量自动写入 status.json。因此即使编排层因上下文耗尽而无法执行 Phase 6.5，CLI-2 仍能检测到训练完成。以下步骤是"优先尝试"，提供更完整的信息（多 run_id 等）。

1. 等待 hooks 刷新（sleep 2s，确保 PostToolUse hooks 完成写入）
2. 获取 session_id — 直接读取环境变量:
   ```bash
   python3 -c "
   import os
   session_id = os.environ.get('TRAJ_SESSION_ID', 'unknown')
   print(session_id)
   "
   ```
3. 收集所有 run_id（从训练循环中记录的 run_id 列表）
4. 写入轮次状态（含 session_id 和所有 run_ids）:
   ```bash
   python3 -c "
   import os
   from traj_opt.round_state import RoundState
   rs = RoundState()
   path = rs.write_training_complete(
       round_num={N},
       run_id='{final_run_id}',
       reward={final_reward},
       session_id=os.environ.get('TRAJ_SESSION_ID', 'unknown'),
       run_ids={run_ids_list},
       success=True
   )
   print(f'Round {N} 训练完成，状态已写入: {path}')
   "
   ```
5. 输出确认: "Round {N} 训练完成。在 CLI-2 中执行 /traj-train-optimize round={N} 开始优化。"

如果训练失败，改用 `write_training_failed()`。

**round 参数可选。** 独立使用 /rllm-train 时不传 round，跳过此步骤。rllm-train 的 Phase 0-6 完全不变。

### round 参数解析

在 Phase 0 中，从 args 提取 round 参数:
- `"round=1 -- 用 qwen-0.5b 训练..."` → round=1, 训练描述="用 qwen-0.5b 训练..."
- `"round=1 | 用 qwen-0.5b 训练..."` → 同上（兼容旧格式，但新代码应使用 `--`）
- `"用 qwen-0.5b 训练..."` → round=None, 跳过 Phase 6.5

### session_id 快照差分法改进

Phase 0 记录快照时，先确保目录存在:
```python
import os
raw_dir = "traj_opt/output/rllm/raw/"
os.makedirs(raw_dir, exist_ok=True)
existing = set(os.listdir(raw_dir))
```

Phase 6.5 差分时，增加容错:
```python
current = set(os.listdir(raw_dir)) if os.path.exists(raw_dir) else set()
new_sessions = current - existing
if new_sessions:
    session_id = sorted(new_sessions)[-1]
else:
    # fallback: 使用最近修改的目录
    import pathlib
    dirs = sorted(pathlib.Path(raw_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    session_id = dirs[0].name if dirs else "unknown"
```

## 使用示例

```
/rllm-train                                          ← 模糊输入，触发引导问答
/rllm-train 启动训练                                  ← 模糊输入，触发引导问答
/rllm-train 用 qwen-0.5b 训练                         ← 部分输入，只补问训练目标
/rllm-train 用 qwen-0.5b 训练数学 agent，reward 达到 0.8  ← 充分输入，直接解析
/rllm-train auto 模式，快速测试，16 个问题，reward >= 0.5
/rllm-train qwen-1.5b, 200 problems, 5 epochs, max 3 rounds
```
