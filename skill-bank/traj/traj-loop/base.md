---
name: traj-loop
description: Top-level orchestration for multi-round trajectory optimization. Supports semi-auto (interactive CLI-1) and fully-auto (non-interactive CLI-1 + auto-approve patches) modes.
metadata:
  version: "2.0.0"
  categories:
    - trajectory
    - orchestration
---

# traj-loop — 多轮自动优化编排

<!-- section:intro -->
你是轨迹优化的顶层编排者，运行在 CLI-2 中。你驱动完整的闭环: 启动训练 → 等待完成 → 分割 → 分析 → 优化，循环指定轮次。每轮使用上一轮优化后的 skill 执行训练，实现 skill 的持续自动优化。

训练在独立的 CLI-1 进程中执行（交互式或非交互式），通过 `traj_opt/output/rounds/` 下的 status.json 协调。
<!-- /section:intro -->

<!-- section:params -->
## 参数

必需参数:
- 训练描述 — 传给 rllm-train 的文本（如 "用 qwen-0.5b 训练, reward >= 0.8"）
- 轮次数 — 优化轮次（默认 3 轮）

可选 flag:
- `--auto` — CLI-1 非交互式启动（`claude -p --permission-mode auto`），语义和 traj-launch-training 的 `--auto` 一致
- `--auto-approve[=LEVEL]` — patch 按置信度自动接受:
  - `--auto-approve` 或 `--auto-approve=medium` — 自动接受 high + medium，low 仍需人工
  - `--auto-approve=high` — 只自动接受 high，medium + low 仍需人工
  - `--auto-approve=all` — 全部自动接受

### 模式对应

| 参数 | CLI-1 | Patch 审核 | 模式 |
|------|-------|-----------|------|
| （默认） | 交互式 | 人工 | 半自动 |
| `--auto` | 非交互式 | 人工 | 全自动（审核保留） |
| `--auto --auto-approve` | 非交互式 | 自动（medium+） | 全自动 |
| `--auto --auto-approve=high` | 非交互式 | 仅 high 自动 | 全自动（保守） |

### 示例

```
/traj-loop 用 qwen-0.5b 训练, 3 轮
/traj-loop 用 qwen-0.5b 训练, 3 轮, --auto --auto-approve
/traj-loop 用 qwen-0.5b 训练, 5 轮, --auto --auto-approve=high
```

### auto-approve 轮次推荐

| 轮次 | 推荐 auto-approve | 原因 |
|------|-------------------|------|
| 3 轮 | `--auto-approve` (medium) | high 需要 3+ 轮证据，3 轮 loop 中前 2 轮不会有 high confidence patch |
| 5+ 轮 | `--auto-approve=high` | 第 3 轮开始可能产生 high confidence 建议 |
| 10+ 轮 | `--auto-approve=high` | 充分数据，high 阈值安全 |

<!-- /section:params -->

<!-- section:rules -->
## 执行规则

1. 每个步骤必须通过调用对应的 skill 执行，不得内联
2. `--auto` 透传给 traj-launch-training，`--auto-approve` 透传给 traj-optimize
3. 每轮结束后输出本轮摘要，最终输出跨轮对比报告
4. 如果某轮训练失败，记录失败原因，询问是否继续下一轮
5. **数据流隔离** — traj-xx 步骤只从 `traj_opt/output/` 读数据，不直接访问 `rllm_train/output/`
6. **traj-segment 不可跳过** — 即使轨迹数据为空也必须调用
7. **禁止直接分析训练日志** — 不在编排层 Read/tail `rllm_train/` 文件
8. **不使用 Agent 子 agent** — CLI-1 是独立进程，天然提供上下文隔离
<!-- /section:rules -->

<!-- section:steps -->
## 执行步骤

### 0. 解析参数

从用户输入中提取:
- 训练描述
- 优化轮次（默认 3）
- `--auto` flag（是否非交互式）
- `--auto-approve` flag 及其 level（默认无）

### 0.5 检查/恢复状态

```python
import json, os, shutil
state_path = "traj_opt/output/loop_state.json"
if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    if state.get("status") == "completed":
        # 上一次 loop 已完成
    elif state["current_round"] <= state["total_rounds"]:
        # 存在未完成的循环
```

#### 半自动模式（无 --auto）

如果存在 loop_state.json，用 AskUserQuestion 询问:
- 从断点继续（Round {current_round}）
- 重新开始（清理旧 round 目录并覆盖状态）
- 取消

#### 全自动模式（--auto）

自动决策:
- 旧 loop 已 completed → 自动清理旧 round 目录，重新开始
- 旧 loop 未完成 → 自动从断点继续

#### 清理操作

"重新开始"时执行:
```python
import shutil
for n in range(1, state["total_rounds"] + 1):
    shutil.rmtree(f"traj_opt/output/rounds/round_{n}", ignore_errors=True)
os.remove("traj_opt/output/loop_state.json")
```

### 1. 初始化状态

```python
import json, time
state = {
    "total_rounds": N,
    "current_round": 1,
    "description": "...",
    "auto": bool,
    "auto_approve": None | "high" | "medium" | "all",
    "started_at": time.time(),
    "rounds": []
}
# 写入 traj_opt/output/loop_state.json
```
<!-- PLACEHOLDER_STEPS_CONTINUE -->

### 2. 循环执行

```
for round_num in start_round..total_rounds:
    更新 loop_state.json 的 current_round

    Step 2.1: 启动训练
    ───────────────────
    构造 traj-launch-training 参数:
      args = f"round={round_num}"
      if auto: args += " --auto"
      args += f" -- {description}"

    调用 Skill("traj-launch-training", args=args)

    Step 2.2: 轮询等待训练完成
    ──────────────────────────
    详见 polling section。

    轮询结果:
    - training_complete → 继续 Step 2.3
    - training_failed:
      - 半自动: 记录失败，用 AskUserQuestion 询问是否继续下一轮
      - 全自动 (--auto): 记录失败，自动跳到下一轮
    - 超时:
      - 半自动: 报告超时，用 AskUserQuestion 询问是否继续等待
      - 全自动 (--auto): 标记本轮失败，自动跳到下一轮

    Step 2.3: 执行优化
    ───────────────────
    构造 traj-train-optimize 参数:
      args = f"round={round_num}"
      if auto_approve: args += f" --auto-approve={auto_approve_level}"

    调用 Skill("traj-train-optimize", args=args)

    等待优化完成（内部串联 segment → analyze → optimize）。

    Step 2.4: 记录本轮结果
    ──────────────────────
    从 status.json 读取本轮结果:
    ```python
    from traj_opt.round_state import RoundState
    status = RoundState().read_status(round_num)
    ```

    更新 loop_state.json:
    ```python
    state["rounds"].append({
        "round": round_num,
        "run_id": status["training"]["run_id"],
        "reward": status["training"]["reward"],
        "patches_generated": status["optimization"]["patches_generated"],
        "patches_accepted": status["optimization"]["patches_accepted"],
    })
    ```

    Step 2.5: 本轮摘要
    ───────────────────
    输出:
    ```
    Round {round_num}/{total_rounds} 完成:
      训练:  reward={reward}, run_id={run_id}
      优化:  {patches_generated} 生成 / {patches_accepted} 接受
    ```
```
<!-- PLACEHOLDER_FINAL_REPORT -->

### 3. 最终报告

所有轮次完成后，输出跨轮对比:

```
traj-loop 优化报告
==================
总轮次: {N}
训练描述: {description}
模式: {半自动|全自动}

轮次对比:
  Round 1: reward {r1}  patch: {p1} 生成 / {a1} 接受
  Round 2: reward {r2}  patch: {p2} 生成 / {a2} 接受
  Round 3: reward {r3}  patch: {p3} 生成 / {a3} 接受

优化效果:
  reward 变化: {r1} → {rN} ({improvement}%)
  累计 patch: {total_patches} 生成 / {total_accepted} 接受
```

清理 loop_state.json（标记为 completed）。
<!-- /section:steps -->

<!-- section:polling -->
## 轮询等待机制

CLI-1 训练完成后会通过 rllm-train Phase 6.5 写入 status.json。traj-loop 通过轮询此文件检测训练完成。同时读取 heartbeat.json 判断 CLI-1 是否仍在活跃工作，实现自适应超时。

### 轮询实现方式

**使用 Monitor 工具（`persistent: true`）执行 Python 轮询脚本。** 每行 stdout 输出实时推送到对话中，用户可以看到训练进度。

Monitor `persistent: true` 没有超时限制，由脚本自身的超时逻辑控制退出。脚本退出时 Monitor 自动结束。

### 轮询脚本

使用 Monitor 工具执行:

```python
Monitor(
    description="Round {round_num} training progress",
    persistent=True,
    command='python3 -c "\nimport json, os, time, sys\n\nstatus_path = \'traj_opt/output/rounds/round_{round_num}/status.json\'\nheartbeat_path = \'traj_opt/output/rounds/round_{round_num}/heartbeat.json\'\nbase_timeout = 1800\nidle_timeout = 600\nlast_heartbeat_mtime = 0\n\nstart = time.time()\n\nwhile True:\n    if os.path.exists(status_path):\n        with open(status_path) as f:\n            status = json.load(f)\n        if status[\'status\'] == \'training_complete\':\n            reward = status[\'training\'][\'reward\']\n            print(f\'COMPLETE: reward={reward}\', flush=True)\n            sys.exit(0)\n        elif status[\'status\'] == \'training_failed\':\n            error = status.get(\'training\', {}).get(\'error\', \'unknown\')\n            print(f\'FAILED: {error}\', flush=True)\n            sys.exit(1)\n\n    hb_msg = \'\'\n    if os.path.exists(heartbeat_path):\n        mtime = os.path.getmtime(heartbeat_path)\n        if mtime > last_heartbeat_mtime:\n            last_heartbeat_mtime = mtime\n            try:\n                with open(heartbeat_path) as f:\n                    hb = json.load(f)\n                hb_msg = f\' | {hb.get(\\\"phase\\\",\\\"\\\")} {hb.get(\\\"step\\\",\\\"\\\")}\'.rstrip()\n                if hb.get(\'reward\') is not None:\n                    hb_msg += f\' reward={hb[\\\"reward\\\"]}\'\n            except (json.JSONDecodeError, KeyError):\n                pass\n\n    elapsed = time.time() - start\n\n    if last_heartbeat_mtime > 0:\n        idle = time.time() - last_heartbeat_mtime\n        if idle > idle_timeout:\n            print(f\'TIMEOUT: no heartbeat update for {int(idle)}s (idle_timeout={idle_timeout}s)\', flush=True)\n            sys.exit(2)\n    else:\n        if elapsed > base_timeout:\n            print(f\'TIMEOUT: {int(elapsed)}s elapsed, no heartbeat ever received\', flush=True)\n            sys.exit(2)\n\n    print(f\'WAITING: {int(elapsed)}s elapsed{hb_msg}\', flush=True)\n    time.sleep(30)\n"'
)
```

每行输出作为 Monitor 通知实时显示在对话中。脚本退出时根据最后一行判断状态:
- `COMPLETE:` → 训练完成，继续 Step 2.3
- `FAILED:` → 训练失败，记录并询问是否继续
- `TIMEOUT:` → 超时，询问是否继续等待

### 轮询间隔与超时

- 轮询间隔: 30 秒
- 基础超时: 30 分钟（无 heartbeat 时的兜底，兼容未升级的 rllm-train）
- 空闲超时: 10 分钟（有 heartbeat 但停止更新时，说明 CLI-1 可能卡住）
- 超时后询问用户（半自动）或自动跳过（全自动 `--auto`）

### 实时进度来源

heartbeat.json 由训练进程（rllm_train 的 TrainingLogger）在每个 step 完成后直接写入，不依赖 Claude Code 工具调用。典型更新频率: 每 5-15 秒一次（取决于 step 耗时）。Monitor 的每行输出会包含最新的 step/reward 信息，用户可实时看到训练进展。
<!-- /section:polling -->

<!-- section:state -->
## 状态管理

在 `traj_opt/output/loop_state.json` 中维护循环状态:

```json
{
  "total_rounds": 3,
  "current_round": 2,
  "description": "用 qwen-0.5b 训练, reward >= 0.8",
  "auto": false,
  "auto_approve": null,
  "started_at": 1714700000,
  "rounds": [
    {
      "round": 1,
      "run_id": "run_xxx",
      "reward": 0.86,
      "patches_generated": 3,
      "patches_accepted": 3
    }
  ]
}
```

支持中断后恢复: 启动时检查 loop_state.json，如果存在未完成的循环，询问是否从断点继续。
<!-- /section:state -->
