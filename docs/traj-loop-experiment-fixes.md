# Plan: traj-loop 实验偏差修复（完整版）

## Context

`/traj-loop 用 qwen-0.5b 训练, 3 轮, --auto --auto-approve=high` 端到端测试暴露了 13 个设计预期与实际行为的偏差。部分已在实验过程中修复，本计划覆盖剩余问题。

## 问题全景

| # | 问题 | 状态 | 说明 |
|---|------|------|------|
| 1 | `\|` 被 shell 解释为管道 | 已修复 | traj-launch-training/traj-loop 已改用 `--` 分隔 |
| 2 | CLI-1 卡在 approve 模式 | 已修复 | traj-launch-training 已自动注入 "auto 模式" |
| 3 | 后台进程状态误判（`&` vs `run_in_background`） | 已修复 | traj-launch-training 已改用 `run_in_background=true` |
| 4 | cli1.log 内容不完整（Node.js block buffering） | **待修复** | 需更新说明 + 引入 heartbeat 替代 |
| 5 | 轮询超时（30 分钟不够多轮调参） | **待修复** | 需 heartbeat 自适应超时 |
| 6 | Round 1/2 训练结果完全重复（seed 固定） | 已修复 | Round 2 接受了 seed 随机化 patch |
| 7 | `--auto-approve` 未透传 | 已修复 | traj-train-optimize 已支持解析和透传 |
| 8 | 审核展示不完整（缺当前 section 对比） | 已修复 | traj-optimize 已有强制检查清单 |
| 9 | 数据提取依赖手动解析（tool_response 是 dict） | 已修复 | analyzer/base.py 已重写 extract_training_data() |
| 10 | Skill 调用链断裂 | 部分修复 | 通过 7/8 间接改善，剩余为执行遵循度问题 |
| 11 | 数据边界过严（不能读 base.md 验证 section 名） | **待修复** | traj-analyze-rllm 生成了不存在的 section 名 |
| 12 | `get_available_training_data()` 返回格式文档错误 | **待修复** | 文档写 `{trajectory_summary, ...}` 实际是 flat dict |
| 13 | cli1.log 说明不准确 | **待修复** | 同 #4 |

**本次需修复: #4, #5, #10, #11, #12, #13**（其中 #4/#13 合并，#10 通过加强 skill 约束描述修复）

## 改动清单

### Fix 1: RoundState 新增 heartbeat 读写方法 [#4, #5]

文件: `traj_opt/round_state.py`

新增两个方法:

```python
def write_heartbeat(self, round_num: int, run_id: str, phase: str,
                    step: Optional[str] = None, reward: Optional[float] = None,
                    message: Optional[str] = None) -> None:
    """CLI-1: write heartbeat during training. Called by rllm-train orchestration."""
    path = self.base_dir / f"round_{round_num}" / "heartbeat.json"
    data = {
        "run_id": run_id,
        "phase": phase,        # "config" | "training" | "monitoring" | "analyzing" | "tuning"
        "step": step,          # e.g. "12/40"
        "reward": reward,      # latest reward if available
        "message": message,    # free-text status
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    self._atomic_write(path, data)

def read_heartbeat(self, round_num: int) -> Optional[Dict[str, Any]]:
    """CLI-2: read heartbeat to monitor training progress."""
    path = self.base_dir / f"round_{round_num}" / "heartbeat.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
```

heartbeat.json 位于 `traj_opt/output/rounds/round_{N}/heartbeat.json`，与 status.json 同目录。写入用已有的 `_atomic_write()`，读取容忍 JSON 解析失败。

### Fix 2: rllm-train 写入 heartbeat [#4, #5]

文件: `skill-bank/rllm/rllm-train/base.md`

在 `<!-- section:phase1-5 -->` 中，每个 Phase 完成后增加 heartbeat 写入。仅当 `round_num is not None` 时写入（独立使用 /rllm-train 时不写）。写入逻辑在编排层，不在子 skill 中。

具体插入点:

a) Phase 2（配置生成）完成后:
```python
if round_num:
    from traj_opt.round_state import RoundState
    RoundState().write_heartbeat(round_num, run_id, phase="config", message="配置已生成")
```

b) Phase 3（启动训练）完成后:
```python
if round_num:
    RoundState().write_heartbeat(round_num, run_id, phase="training", message="训练已启动")
```

c) Phase 4（监控）— rllm-monitor 返回训练完成后:
```python
if round_num:
    RoundState().write_heartbeat(round_num, run_id, phase="monitoring",
                                 step=f"{current_step}/{total_steps}", reward=latest_reward,
                                 message="训练完成")
```

d) Phase 5（分析）开始和完成时:
```python
if round_num:
    RoundState().write_heartbeat(round_num, run_id, phase="analyzing", message="分析中")
    # ... 分析完成后 ...
    RoundState().write_heartbeat(round_num, run_id, phase="analyzing", reward=final_reward,
                                 message="分析完成，准备调参")
```

e) 调参循环回到 Phase 2 时:
```python
if round_num:
    RoundState().write_heartbeat(round_num, new_run_id, phase="tuning",
                                 message=f"第 {attempt} 次调参")
```

实现方式: 在 phase1-5 section 的每个 Phase 描述末尾，增加 "Heartbeat（双 CLI 模式）" 小节。

### Fix 3: traj-loop 轮询改为自适应超时 [#5]

文件: `skill-bank/traj/traj-loop/base.md`

替换 `<!-- section:polling -->` 的轮询脚本，改为同时检查 status.json 和 heartbeat.json:

```python
import json, os, time, sys

status_path = 'traj_opt/output/rounds/round_{round_num}/status.json'
heartbeat_path = 'traj_opt/output/rounds/round_{round_num}/heartbeat.json'
base_timeout = 1800       # 30 min base (no heartbeat ever received)
idle_timeout = 600        # 10 min since last heartbeat update
last_heartbeat_mtime = 0

start = time.time()

while True:
    # 1. Check terminal state
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)
        if status['status'] == 'training_complete':
            reward = status['training']['reward']
            print(f'COMPLETE: reward={reward}')
            sys.exit(0)
        elif status['status'] == 'training_failed':
            error = status.get('training', {}).get('error', 'unknown')
            print(f'FAILED: {error}')
            sys.exit(1)

    # 2. Check heartbeat — reset idle timer on update
    hb_msg = ''
    if os.path.exists(heartbeat_path):
        mtime = os.path.getmtime(heartbeat_path)
        if mtime > last_heartbeat_mtime:
            last_heartbeat_mtime = mtime
            try:
                with open(heartbeat_path) as f:
                    hb = json.load(f)
                hb_msg = f' | {hb.get("phase","")} {hb.get("step","")}'.rstrip()
                if hb.get('reward') is not None:
                    hb_msg += f' reward={hb["reward"]}'
            except (json.JSONDecodeError, KeyError):
                pass

    elapsed = time.time() - start

    # 3. Timeout: adaptive based on heartbeat
    if last_heartbeat_mtime > 0:
        idle = time.time() - last_heartbeat_mtime
        if idle > idle_timeout:
            print(f'TIMEOUT: no heartbeat update for {int(idle)}s (idle_timeout={idle_timeout}s)')
            sys.exit(2)
    else:
        if elapsed > base_timeout:
            print(f'TIMEOUT: {int(elapsed)}s elapsed, no heartbeat ever received')
            sys.exit(2)

    print(f'WAITING: {int(elapsed)}s elapsed{hb_msg}', flush=True)
    time.sleep(30)
```

同步更新"轮询间隔与超时"小节:
- 轮询间隔: 30 秒
- 基础超时: 30 分钟（无 heartbeat 时的兜底）
- 空闲超时: 10 分钟（有 heartbeat 但停止更新时）
- 超时后询问用户（半自动）或自动跳过（全自动 `--auto`）

### Fix 4: traj-analyze-rllm 数据边界放开 [#11]

文件: `skill-bank/traj/traj-analyze-rllm/base.md`

修改 `<!-- section:data-boundary -->`:

a) 在"允许读取的数据源"表中新增一行:

| 路径 | 内容 | 用途 |
|------|------|------|
| `skill-bank/rllm/*/base.md` | skill section 结构 | 验证 patch 目标 section 名是否存在 |

b) 从"禁止直接读取的数据源"表中移除 `skill-bank/rllm/*/base.md` 行。保留 `.claude/skills/rllm-*/SKILL.md`（编译产物仍禁止）。

c) 在"上下文隔离说明"中追加:

```
例外: 允许读取 skill-bank/rllm/*/base.md 中的 section 锚点（`<!-- section:xxx -->`），
用于验证优化建议的 target_section 是否存在。禁止读取 section 内容用于分析
（分析数据仍然只来自轨迹）。
```

### Fix 5: traj-analyze-rllm 增加 section 验证步骤 [#11]

文件: `skill-bank/traj/traj-analyze-rllm/base.md`

在 `<!-- section:execution-steps -->` 的 Step 4（生成分析报告）和 Step 5（保存报告）之间，新增 Step 4.5:

```markdown
### 4.5 验证优化建议的 target_section

对每条优化建议，验证 target_section 在目标 skill 的 base.md 中存在:

\`\`\`bash
grep -c "<!-- section:{target_section} -->" skill-bank/rllm/{skill_name}/base.md
\`\`\`

- 如果匹配数 > 0: section 存在，保留建议
- 如果匹配数 = 0: section 不存在，尝试模糊匹配:
  \`\`\`bash
  grep -o '<!-- section:[a-z0-9-]* -->' skill-bank/rllm/{skill_name}/base.md
  \`\`\`
  从匹配结果中选择最接近的 section 名，更新建议的 target_section。
  如果无法确定，在报告中标注 "[section 名待确认]"。
```

这直接解决了 Round 1 分析报告中 `param-safety`（不存在）vs `param-ranges`（实际名称）的问题。

### Fix 6: traj-analyze-rllm Step 2c 返回格式修正 [#12]

文件: `skill-bank/traj/traj-analyze-rllm/base.md`

修改 `<!-- section:execution-steps -->` 中 Step 2c（约 line 125）:

旧:
```python
# 返回: [{trajectory_summary, training_data: {config, reward_trend, perf_stats, errors, log_snippets}}]
```

新:
```python
# 返回: [{session_id, skill_name, start_time, end_time, tool_count, ..., training_data: {config, reward_trend, perf_stats, errors, log_snippets}}]
# 注意: 返回的是 flat dict（summary 字段 + training_data），不是嵌套在 trajectory_summary 下
```

### Fix 7: traj-launch-training cli1.log 说明更新 [#4, #13]

文件: `skill-bank/traj/traj-launch-training/base.md`

修改非交互式模式的 cli1.log 描述（约 line 112）:

旧:
```
- **cli1.log 记录的是 claude -p 的交互输出**，不是训练日志。训练日志在 `rllm_train/output/runs/{run_id}/training_log.txt`。
```

新:
```
- **cli1.log 是事后审计日志**，不是实时日志。由于 Node.js block buffering，`claude -p` 的 stdout 重定向到文件时只在进程退出后才 flush，训练期间 cli1.log 为空。实时训练进度通过 heartbeat.json 获取（见 traj-loop 轮询机制）。训练日志在 `rllm_train/output/runs/{run_id}/training_log.txt`。
```

### Fix 8: rllm-train 加强子 skill 调用纪律 [#10]

文件: `skill-bank/rllm/rllm-train/base.md`

在 `<!-- section:execution-rules -->` 的规则列表末尾追加:

```markdown
6. **Phase 间不跳步** — 即使上一轮的 analysis.json 已经给出了明确的调参建议，
   调参循环仍必须经过 Phase 2 (rllm-config) → Phase 3 (rllm-run) → Phase 4 (rllm-monitor) → Phase 5 (rllm-analyze) 的完整流程。
   禁止在编排层直接修改 config.json 或跳过 monitor 直接读日志。
7. **调参循环中的 Phase 4 不可省略** — 每次 rllm-run 启动训练后，必须调用 rllm-monitor 监控。
   不得因为"上一轮已经知道训练模式"而跳过监控。Monitor 负责异常检测和 early stopping，
   跳过会导致 catastrophic forgetting 无法被及时发现。
```

这解决了 Round 3 中观察到的调参循环中部分 Phase 被跳过的问题。

### Fix 9: traj-optimize 加强 section 验证 [#10, #11]

文件: `skill-bank/traj/traj-optimize/base.md`

在 `<!-- section:execution-steps -->` 的 Step 3（生成 patch 文件）之后，Step 4（展示）之前，增加验证:

```markdown
### 3.5 验证 patch 目标 section

对每个生成的 patch，验证 target_section 在目标 skill 的 base.md 中存在:

\`\`\`bash
grep -c "<!-- section:{target_section} -->" skill-bank/{group}/{skill_name}/base.md
\`\`\`

如果 section 不存在:
- 列出该 skill 的所有 section: `grep -o '<!-- section:[a-z0-9-]* -->' skill-bank/{group}/{skill_name}/base.md`
- 尝试匹配最接近的 section 名
- 如果无法确定，标记该 patch 为 "[section 待确认]"，在 Step 4 展示时高亮提示
```

这是 Fix 5 的下游防线 — 即使分析报告中的 section 名有误，optimize 阶段也能拦截。

## 文件改动总览

| 文件 | 改动量 | 修复问题 |
|------|--------|---------|
| `traj_opt/round_state.py` | 小增（~25 行） | #4 #5 heartbeat 读写 |
| `skill-bank/rllm/rllm-train/base.md` | 中改 | #4 #5 heartbeat 写入, #10 调用纪律 |
| `skill-bank/traj/traj-loop/base.md` | 中改 | #5 自适应超时轮询 |
| `skill-bank/traj/traj-analyze-rllm/base.md` | 中改 | #11 数据边界, #11 section 验证, #12 返回格式 |
| `skill-bank/traj/traj-optimize/base.md` | 小改 | #10 #11 section 验证防线 |
| `skill-bank/traj/traj-launch-training/base.md` | 小改 | #4 #13 cli1.log 说明 |

不改动: rllm-monitor, rllm-config, traj-segment, traj-train-optimize, analyzer/base.py, schema.py

## 执行顺序

### Phase 1: Python 代码
1. `traj_opt/round_state.py` — 新增 write_heartbeat / read_heartbeat

### Phase 2: Skill base.md
2. `skill-bank/rllm/rllm-train/base.md` — heartbeat 写入 + 调用纪律 (Fix 2, 8)
3. `skill-bank/traj/traj-loop/base.md` — 自适应超时轮询 (Fix 3)
4. `skill-bank/traj/traj-analyze-rllm/base.md` — 数据边界 + section 验证 + 返回格式 (Fix 4, 5, 6)
5. `skill-bank/traj/traj-optimize/base.md` — section 验证防线 (Fix 9)
6. `skill-bank/traj/traj-launch-training/base.md` — cli1.log 说明 (Fix 7)

### Phase 3: 编译 + 验证
7. `python skill-bank/compile.py --all` — 编译所有 skills
8. 验证 heartbeat: `python -c "from traj_opt.round_state import RoundState; rs = RoundState(); rs.write_heartbeat(99, 'test', 'config', message='test'); print(rs.read_heartbeat(99))"`
9. 清理: `rm -rf traj_opt/output/rounds/round_99`
10. 检查编译产物 diff: `python skill-bank/compile.py --diff rllm-train` 等
