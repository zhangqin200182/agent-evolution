---
name: traj-loop
description: "[DEPRECATED] Top-level orchestration skill that drives the full trajectory optimization loop. Replaced by dual-CLI architecture: use /rllm-train in CLI-1 and /traj-train-optimize in CLI-2."
metadata:
  version: "1.0.0"
  deprecated: true
  categories:
    - trajectory
    - orchestration
---

# traj-loop — 全自动轨迹优化编排

<!-- section:intro -->
> **已废弃**: 此 skill 已被双 CLI 架构替代。
> 请在 CLI-1 中使用 /rllm-train，在 CLI-2 中使用 /traj-train-optimize。
> 详见 `docs/trajectory-design.md` Section 18。

你是轨迹优化的顶层编排者。你驱动完整的闭环: 训练 → 捕获 → 分割 → 分析 → 优化，循环指定轮次。每轮使用上一轮优化后的 skill 执行训练，实现 skill 的持续自动优化。
<!-- /section:intro -->

<!-- section:rules -->
## 执行规则

1. 每个步骤必须通过调用对应的 skill 执行，不得内联
2. 唯一需要人工介入的环节是确认 patch（设计准则 3.4）
3. 每轮结束后输出本轮摘要，最终输出跨轮对比报告
4. 如果某轮训练失败，记录失败原因，继续下一轮（不中断循环）
5. **上下文隔离** — rllm-train 和 traj-analyze-rllm 必须在独立的 Agent 子 agent 中执行，确保分析器无法看到训练的执行上下文
6. **数据流隔离** — traj-xx 步骤只从 traj_opt/output/ 读数据，不直接访问 rllm_train/output/
7. **traj-segment 不可跳过** — 即使轨迹数据为空也必须调用（记录空结果供追溯）
8. **禁止直接分析训练日志** — 不在编排层 Read/tail rllm_train/ 文件做分析
9. **进度可观测** — 通过进度文件（traj_opt/output/agent_progress/）实现子 agent 对父对话的进度可见性
<!-- /section:rules -->

<!-- section:steps -->
## 执行步骤

### 0. 解析参数

从用户输入中提取:
- 训练描述（传给 rllm-train）
- 优化轮次（默认 3 轮）
- 执行模式（auto/approve，默认 approve）

示例输入:
```
/traj-loop "用 qwen-0.5b 训练数学 agent，自动优化 3 轮"
/traj-loop "qwen-0.5b, reward >= 0.8, 5 rounds, auto"
```

### 1. 循环执行

```
for round in 1..N:

    Step 1: 训练 (在独立子 agent 中)
    ─────────────────────────────────
    1.1 生成 session_id:
        session_id = f"round_{round}_{timestamp}"
        创建进度文件 traj_opt/output/agent_progress/{session_id}.json
        写入 status=init

    1.2 启动子 agent，prompt 中包含进度文件跟踪要求:
        Agent(
            prompt="读取 .claude/skills/rllm-train/SKILL.md 并按其步骤执行训练。
                    训练描述: {description}
                    工作目录: /Users/kevin/code/MyProject
                    Round {round}/{total_rounds}。

                    **进度跟踪**: 在执行过程中，定期更新进度文件:
                    traj_opt/output/agent_progress/{session_id}.json

                    每个 phase 开始/完成时更新进度。
                    训练中定期更新 step/reward。
                    训练完成后输出: run_id, 最终 reward, 是否成功。

                    **进度文件格式**:
                    {
                      "status": "running"|"completed"|"failed",
                      "phase": "phase_0_clarify"|"phase_1_config"|"phase_2_run"|"phase_3_monitor"|"phase_4_analyze",
                      "progress": {"phase_name": {"status": "completed"|"running"|"pending", "step": N, "total_steps": N, "reward": 0.5}},
                      "result": null|{"run_id": "...", "reward": 0.5, "success": true},
                      "error": null|string
                    }",
            description="rllm-train round {round}"
        )

        → 子 agent 立即返回（但实际已在后台执行）
        → 子 agent 的 Phase 1-4 完成时会更新进度文件

    1.3 轮询进度文件（每 30 秒）:
        while True:
            读取 traj_opt/output/agent_progress/{session_id}.json
            显示进度:

            [Round {round}] 训练进度:
              Phase: {phase}
              Step: {step}/{total_steps}
              Reward: {reward}
              最新: {latest_update}

            if status == "completed":
                提取 result 中的 run_id 和 reward
                跳出轮询
            elif status == "failed":
                报告错误: {error}
                询问是否重试或中止
                跳出轮询
            elif 超时 (30 分钟):
                报告超时，询问是否继续等待
            else:
                等待 30 秒后继续轮询

    1.4 清理进度文件（保留用于审计，可不清理）

    Step 1.5: 验证轨迹数据完整性
    ─────────────────────────────
    → ls traj_opt/output/raw/ 检查是否有新事件
    → 如果为空: 输出警告 "Hooks 未捕获到训练数据" 但不中断循环
    → 如果有数据: 报告事件数量

    Step 2: 分割轨迹
    ─────────────────
    调用 Skill("traj-segment")
    → 读取 traj_opt/output/raw/ → 输出到 traj_opt/output/trajectories/

    Step 3: 分析 (在独立子 agent 中)
    ─────────────────────────────────
    3.1 生成 session_id:
        analyze_session_id = f"analyze_{round}_{timestamp}"

    3.2 启动分析子 agent，prompt 中包含进度文件跟踪要求:
        Agent(
            prompt="读取 .claude/skills/traj-analyze-rllm/SKILL.md 并按其步骤执行分析。
                    工作目录: /Users/kevin/code/MyProject
                    只从 traj_opt/output/ 读取数据，不要读取 rllm_train/ 下的文件。

                    **进度跟踪**: 定期更新进度文件:
                    traj_opt/output/agent_progress/{session_id}.json

                    **进度文件格式**:
                    {
                      "status": "running"|"completed"|"failed",
                      "phase": "phase_1_load"|"phase_2_extract"|"phase_3_analyze"|"phase_4_report",
                      "progress": {...},
                      "result": null|{"report_path": "...", "suggestion_count": 3},
                      "error": null|string
                    }

                    分析完成后输出: 报告路径, 优化建议数量。",
            description="traj-analyze-rllm round {round}"
        )

        → 子 agent 在全新上下文中执行，物理上看不到 Step 1 的训练细节
        → 只能从 traj_opt/output/ 获取数据
        → 返回报告路径

    3.3 轮询分析进度（每 30 秒）:
        while True:
            读取 traj_opt/output/agent_progress/{analyze_session_id}.json
            显示进度

            [Round {round}] 分析进度:
              Phase: {phase}
              最新: {latest_update}

            if status == "completed":
                提取 result 中的 report_path 和 suggestion_count
                跳出轮询
            elif status == "failed":
                报告错误，跳出轮询

    Step 4: 生成 patch
    ──────────────────
    调用 Skill("traj-optimize", args="{report_path}")
    → 读取分析报告 → 展示 patch → 等待用户确认 → 编译

    Step 5: 本轮摘要
    ─────────────────
    输出本轮结果: round, run_id, reward, patches_generated, patches_accepted
```

### 2. Skill 调用方式

每个步骤使用 Skill 工具调用:
```
Skill("rllm-train", args="<训练描述>")
Skill("traj-segment")
Skill("traj-analyze-rllm")
Skill("traj-optimize")
```

调用 Skill 后当轮响应立即结束，等待下一轮系统注入。

### 3. 最终报告

所有轮次完成后，输出跨轮对比:

```
traj-loop 优化报告
==================
总轮次: {N}
训练描述: {description}

轮次对比:
  Round 1: reward {r1}  问题: {issues_1}  生成 patch: {patch_count_1}
  Round 2: reward {r2}  问题: {issues_2}  生成 patch: {patch_count_2}
  Round 3: reward {r3}  问题: {issues_3}  生成 patch: {patch_count_3}

优化效果:
  reward 变化: {r1} → {rN} ({improvement}%)
  累计 patch: {total_patches}
  优化的 skills: {skill_list}
```
<!-- /section:steps -->

<!-- section:state -->
## 状态管理

在 `traj_opt/output/loop_state.json` 中维护循环状态:
```json
{
  "total_rounds": 3,
  "current_round": 1,
  "description": "...",
  "mode": "approve",
  "rounds": [
    {
      "round": 1,
      "run_id": "...",
      "reward": 0.45,
      "patches_generated": 2,
      "patches_accepted": 2
    }
  ]
}
```

支持中断后恢复: 读取 loop_state.json，从上次中断的 round 继续。
<!-- /section:state -->

<!-- section:agent-isolation -->
## Agent 隔离设计

### 为什么用 Agent 而非 Skill

Skill 工具在当前对话上下文中注入 SKILL.md 并执行，所有工具调用共享同一个对话历史。
Agent 工具创建独立的子 agent，拥有全新的对话上下文。

在 traj-loop 中:
- /rllm-train 如果用 Skill 调用，其读取的 config.json、training_log.txt 等内容会留在对话上下文中，后续的 /traj-analyze-rllm 能直接看到
- 用 Agent 调用 rllm-train，训练的完整上下文在子 agent 结束后被丢弃，traj-analyze-rllm 的子 agent 物理上无法访问

### Agent 调用规范

1. rllm-train 子 agent:
   - prompt 中包含训练描述和工作目录
   - 指示子 agent 读取 rllm-train SKILL.md 并执行
   - 子 agent 返回: run_id, 最终 reward, 是否成功

2. traj-analyze-rllm 子 agent:
   - prompt 中包含工作目录和数据边界提醒
   - 指示子 agent 读取 traj-analyze-rllm SKILL.md 并执行
   - 子 agent 返回: 报告路径, 优化建议数量

### traj-segment 和 traj-optimize 不需要 Agent 隔离

- traj-segment: 只读 traj_opt/output/raw/，不涉及 rllm 内部数据，无隔离需求
- traj-optimize: 只读 traj_opt/output/reports/，生成 patch，无隔离需求
- 这两个在父对话中用 Skill 调用即可

### Hooks 在 Agent 子 agent 中的行为

Claude Code Hooks 对子 agent 中的工具调用同样生效:
- PostToolUse hook 捕获子 agent 1 中 rllm-monitor 的 Read/Bash 调用
- SubagentStop hook 在子 agent 结束时触发
- 捕获的事件写入 traj_opt/output/raw/，conversation_id 标记为子对话

这是隔离方案的关键前提: 子 agent 1 中 rllm-xx 读取的训练数据被 hooks 捕获到 traj_opt/output/，子 agent 2 才有数据可分析。

### Agent 进度可观测性

子 agent 在独立上下文中执行，对父对话完全黑盒。为实现可观测性，使用共享进度文件:

```
traj_opt/output/agent_progress/{session_id}.json
```

**子 agent 端**: 在每个 phase 完成时用 Bash/python3 更新进度文件
**父对话端**: 每 30 秒轮询进度文件并显示进度

进度文件不破坏上下文隔离:
- 父对话读取的是**进度数据**（phase/step/reward），不是**训练细节**（config.json 内容）
- 进度文件路径已在 prompt 中传递给子 agent，不涉及额外的数据泄露

详见 `docs/trajectory-design.md` Section 17。
<!-- /section:agent-isolation -->
