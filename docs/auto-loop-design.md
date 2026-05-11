# traj-loop 全自动闭环: 偏差修复统一设计

## 背景

两轮 `/traj-loop 用 qwen-0.5b 训练, 3 轮, --auto --auto-approve=high` 端到端实验暴露了设计预期与实际行为的系统性偏差。本文档将所有发现的问题统一归类、定优先级、给出修复设计。

---

## 问题全景

### 已修复（无需再动）

| # | 问题 | 修复方式 |
|---|------|---------|
| 1 | `\|` 被 shell 解释为管道 | 改用 `--` 分隔 flag 和训练描述 |
| 2 | CLI-1 卡在 approve 模式 | traj-launch-training 自动注入 "auto 模式" |
| 3 | 后台进程状态误判 | 改用 `run_in_background=true` 代替 shell `&` |
| 4 | cli1.log 不实时 | 文档标注为事后审计日志 + heartbeat 替代 |
| 5 | 轮询超时不够 | 自适应超时（heartbeat mtime 驱动） |
| 6 | Round 1/2 结果重复 | seed 随机化 patch 已接受 |
| 7 | `--auto-approve` 未透传 | traj-train-optimize 已支持解析和透传 |
| 8 | 审核展示缺对比 | traj-optimize 强制检查清单 |
| 9 | 数据提取格式 | analyzer/base.py 重写 extract_training_data() |
| 10 | Skill 调用链断裂 | rllm-train 加强规则 5/6/7 |
| 11 | Section 名不存在 | traj-analyze-rllm 增加 4.5 验证步骤 |
| 12 | 返回格式文档错误 | traj-analyze-rllm Step 2c 注释修正 |
| 13 | cli1.log 说明不准确 | 同 #4 |
| 14 | Heartbeat 更新太慢 | TrainingLogger 每步写入 heartbeat.json |

### 待修复

| # | 问题 | 优先级 | 类别 |
|---|------|--------|------|
| A | Phase 6.5 不可靠 — CLI-1 上下文耗尽时无法写 status.json | P0 | 协调可靠性 |
| B | Session ID 识别错误 — CLI-2 hooks 污染 raw/ 目录 | P0 | 协调可靠性 |
| C | 轮询超时后全自动模式无恢复策略 | P1 | 容错 |
| D | 旧 round 状态阻塞新 loop 启动 | P1 | 状态管理 |
| E | Patch 生成即激活 — 拒绝后 manifest 残留 | P1 | Patch 生命周期 |
| F | yaml.dump 格式漂移 | P2 | 代码质量 |
| G | auto-approve=high 首轮永远无 patch（需 3+ 轮证据） | 设计权衡 | 置信度模型 |

---

## 修复设计

### A. Phase 6.5 可靠性 [P0]

**问题**: CLI-1 的 rllm-train 在 Phase 6.5 写入 status.json 依赖:
1. Claude Code 上下文未耗尽（能继续执行 Python 代码）
2. session_id 快照差分法正确识别本次 session
3. 训练循环中记录了所有 run_id

任何一环失败，CLI-2 的轮询脚本永远等不到 `training_complete`。

**设计**: 双写保底 — 训练进程本身在退出时写入 status.json，不依赖 Claude Code 编排层。

#### 方案: TrainingLogger 退出时写 status

在 `rllm_train/logger.py` 的 `print_training_report()` 末尾，如果检测到 `TRAJ_ROUND_NUM` 环境变量，直接写入 status.json:

```python
def _write_round_status(self, config, final_reward: float):
    """Fallback: write status.json directly from training process."""
    round_num = os.environ.get("TRAJ_ROUND_NUM")
    if not round_num:
        return
    try:
        from traj_opt.round_state import RoundState
        rs = RoundState()
        existing = rs.read_status(int(round_num))
        if existing and existing.get("status") == "training_complete":
            return  # 编排层已写入，不覆盖
        rs.write_training_complete(
            round_num=int(round_num),
            run_id=config.run_id,
            reward=final_reward,
            session_id=os.environ.get("TRAJ_SESSION_ID", "unknown"),
            run_ids=[config.run_id],
        )
    except Exception:
        pass  # best-effort, 不影响训练
```

调用点: `print_training_report()` 末尾调用 `self._write_round_status(config, final_reward)`。

#### 环境变量传递

traj-launch-training 启动 CLI-1 时注入:
```bash
TRAJ_ROUND_NUM={N} \
TRAJ_HEARTBEAT_PATH=traj_opt/output/rounds/round_{N}/heartbeat.json \
TRAJ_SESSION_ID=$(python3 -c "import uuid; print(uuid.uuid4())") \
  claude -p ...
```

`TRAJ_SESSION_ID` 由启动方预生成，解决了 session_id 识别问题（见 B）。

#### 优先级逻辑

1. 编排层 Phase 6.5 正常执行 → 写入完整 status（含多 run_id、正确 session_id）
2. 编排层失败但训练完成 → TrainingLogger 保底写入（单 run_id、预生成 session_id）
3. 训练崩溃 → heartbeat 停止更新 → CLI-2 idle_timeout 触发

#### 影响文件

- `rllm_train/logger.py` — 新增 `_write_round_status()`
- `skill-bank/traj/traj-launch-training/base.md` — 新增 `TRAJ_ROUND_NUM` 和 `TRAJ_SESSION_ID` 环境变量
- `skill-bank/rllm/rllm-train/base.md` — Phase 6.5 标注为"优先尝试，有保底"

---

### B. Session ID 识别 [P0]

**问题**: 快照差分法（Phase 0 记录 raw/ 目录，Phase 6.5 取差集）在以下情况失败:
1. CLI-2 的 hooks 也在 raw/ 下创建目录（CLI-2 执行 traj-segment 等操作时触发 PostToolUse hook）
2. 多个 CLI-1 并发时差集有多个新目录

**设计**: 预生成 session_id，通过环境变量传递。

#### 方案

1. traj-launch-training 在启动 CLI-1 前生成 UUID:
   ```python
   session_id = str(uuid.uuid4())
   ```

2. 通过 `TRAJ_SESSION_ID` 环境变量传入 CLI-1

3. PostToolUse hook 读取此环境变量作为 session 目录名:
   ```python
   # hooks/post_tool.py
   session_id = os.environ.get("TRAJ_SESSION_ID") or self._generate_session_id()
   ```

4. Phase 6.5 直接使用 `os.environ["TRAJ_SESSION_ID"]`，不再做快照差分

5. status.json 中的 session_id 字段与 raw/ 目录名一致

#### Hook 隔离

CLI-2 的 hooks 不设置 `TRAJ_SESSION_ID`，因此 CLI-2 产生的事件写入自动生成的 session 目录。CLI-1 的事件写入预生成的目录。两者天然隔离。

#### 影响文件

- `traj_opt/hooks/post_tool.py` — 优先读取 `TRAJ_SESSION_ID` 环境变量
- `skill-bank/traj/traj-launch-training/base.md` — 生成并注入 `TRAJ_SESSION_ID`
- `skill-bank/rllm/rllm-train/base.md` — Phase 6.5 简化为直接读环境变量

---

### C. 轮询超时恢复 [P1]

**问题**: 全自动模式下轮询超时后，traj-loop 询问用户是否继续等待。但 `--auto` 模式下没有用户可以回答。

**设计**: 全自动模式下超时后自动执行恢复策略。

#### 恢复策略

```python
if timeout_type == "idle_timeout":
    # heartbeat 曾经活跃但停止了 → CLI-1 可能卡住
    # 检查 CLI-1 进程是否存活
    if cli1_process_alive(task_id):
        # 进程还在但不更新 heartbeat → 可能在等用户输入（不应该在 auto 模式）
        # 再等一个 idle_timeout 周期
        extend_timeout(idle_timeout)
    else:
        # 进程已退出但没写 status → 异常退出
        mark_round_failed(round_num, "CLI-1 进程异常退出，未写入 status.json")

elif timeout_type == "base_timeout":
    # 从未收到 heartbeat → CLI-1 可能启动失败
    if cli1_process_alive(task_id):
        # 进程在但无 heartbeat → 可能还在 Phase 0/1（需求澄清）
        # 全自动模式不应该卡在这里，标记失败
        mark_round_failed(round_num, "CLI-1 启动 30 分钟未产生 heartbeat")
    else:
        mark_round_failed(round_num, "CLI-1 进程未启动或立即退出")
```

#### 失败后行为

- 全自动 (`--auto`): 记录失败，自动跳到下一轮
- 半自动: 报告超时，用 AskUserQuestion 询问

#### 影响文件

- `skill-bank/traj/traj-loop/base.md` — 轮询结果处理增加 auto 模式分支

---

### D. 旧状态清理 [P1]

**问题**: 上一次 traj-loop 的 round_1/2/3 目录残留，新 loop 启动时 `find_pending_training()` 返回 None（因为 round_3 不是 `optimization_complete`），或者 traj-launch-training 的前置检查发现 round 已存在而拒绝启动。

**设计**: traj-loop Step 0.5 增加清理逻辑。

#### 方案

```python
# traj-loop Step 0.5: 检查/恢复/清理
state_path = "traj_opt/output/loop_state.json"
if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)

    if state.get("status") == "completed":
        # 上一次 loop 已完成，清理旧状态
        # AskUserQuestion: 清理旧数据开始新 loop / 保留旧数据（手动清理）/ 取消
        pass
    elif state["current_round"] <= state["total_rounds"]:
        # 未完成的 loop
        # AskUserQuestion: 从断点继续 / 重新开始（清理）/ 取消
        pass
```

清理操作:
```python
def clean_rounds(round_nums: List[int]):
    """Remove round directories and reset loop_state."""
    for n in round_nums:
        shutil.rmtree(f"traj_opt/output/rounds/round_{n}", ignore_errors=True)
    os.remove("traj_opt/output/loop_state.json")
```

全自动模式 (`--auto`): 如果旧 loop 已 completed，自动清理并重新开始。如果旧 loop 未完成，自动从断点继续。

#### 影响文件

- `skill-bank/traj/traj-loop/base.md` — Step 0.5 扩展清理逻辑
- `traj_opt/round_state.py` — 可选: 新增 `clean_round()` 方法

---

### E. Patch 生命周期修正 [P1]

**问题**: `PatchGenerator.generate_patch()` 在生成 patch 文件后立即调用 `_activate_patch()` 将其加入 manifest.yaml 的 active 列表。如果用户随后拒绝该 patch，manifest 中仍有残留条目。下次 compile 时会尝试应用一个被拒绝的 patch。

**设计**: 分离"生成"和"激活"两个动作。

#### 方案

1. `generate_patch()` 只写文件，不修改 manifest:
   ```python
   def generate_patch(self, suggestion) -> Path:
       # ... 写文件 ...
       # 不再调用 self._activate_patch()
       return patch_path
   ```

2. 新增 `accept_patch()` 方法，在用户确认后调用:
   ```python
   def accept_patch(self, skill_name: str, patch_id: str) -> None:
       """Activate a patch after user acceptance."""
       group = self._find_group(skill_name)
       skill_dir = Path("skill-bank") / group / skill_name
       self._activate_patch(skill_dir, patch_id)
   ```

3. 新增 `reject_patch()` 方法:
   ```python
   def reject_patch(self, skill_name: str, patch_id: str) -> None:
       """Remove patch file and ensure it's not in manifest."""
       group = self._find_group(skill_name)
       skill_dir = Path("skill-bank") / group / skill_name
       patch_path = skill_dir / "patches" / f"{patch_id}.md"
       if patch_path.exists():
           patch_path.unlink()
       self._deactivate_patch(skill_dir, patch_id)

   def _deactivate_patch(self, skill_dir: Path, patch_id: str) -> None:
       """Remove patch from manifest.yaml active list."""
       manifest_path = skill_dir / "manifest.yaml"
       if not manifest_path.exists():
           return
       with open(manifest_path) as f:
           manifest = yaml.safe_load(f) or {}
       active = manifest.get("active", [])
       if patch_id in active:
           active.remove(patch_id)
           manifest["active"] = active
           with open(manifest_path, "w") as f:
               yaml.dump(manifest, f, default_flow_style=False,
                         allow_unicode=True, sort_keys=False, width=120)
   ```

4. traj-optimize skill 的 Step 5/6 调整:
   - 人工审核确认 → `accept_patch()` → compile
   - 人工拒绝 → `reject_patch()`
   - 自动审核达标 → `accept_patch()` → compile
   - 自动审核不达标 → `reject_patch()`

#### 影响文件

- `traj_opt/optimizer/patch_generator.py` — 拆分 generate/accept/reject
- `skill-bank/traj/traj-optimize/base.md` — Step 5/6 使用新 API

---

### F. yaml.dump 格式漂移 [P2]

**问题**: `yaml.dump()` 的默认行为会改变 manifest.yaml 的格式（如列表缩进、引号风格），导致 git diff 噪音。

**设计**: 固定 dump 参数。

```python
yaml.dump(manifest, f, default_flow_style=False,
          allow_unicode=True, sort_keys=False, width=120)
```

关键: `sort_keys=False` 保持字段顺序，`width=120` 避免不必要的换行。

#### 影响文件

- `traj_opt/optimizer/patch_generator.py` — yaml.dump 参数固定

---

### G. auto-approve=high 首轮无 patch [设计权衡]

**问题**: `confidence=high` 要求 3+ 轮证据。首轮实验只有 1 轮数据，所有建议都是 medium 或 low，全部被 `--auto-approve=high` 拒绝。结果: Round 1 优化阶段 0 patch 接受，Round 2 用的是和 Round 1 完全相同的 skill。

**分析**: 这不是 bug，是置信度模型的正确行为。但在 3 轮 loop 中，前 2 轮都不会有 patch 被自动接受，只有第 3 轮分析时才可能产生 high confidence 建议 — 但此时已经没有下一轮训练来验证效果。

**设计决策**: 不改变置信度模型。改为在文档中明确推荐:

| 轮次 | 推荐 auto-approve |
|------|-------------------|
| 3 轮 | `--auto-approve` (medium) |
| 5+ 轮 | `--auto-approve=high` |
| 10+ 轮 | `--auto-approve=high` |

#### 影响文件

- `skill-bank/traj/traj-loop/base.md` — 参数说明增加推荐表

---

## 实施顺序

### Phase 1: Python 代码（无 skill 改动）

1. `rllm_train/logger.py` — 新增 `_write_round_status()` 保底写入
2. `traj_opt/optimizer/patch_generator.py` — 拆分 generate/accept/reject + yaml 格式固定
3. `traj_opt/hooks/post_tool.py` — 优先读取 `TRAJ_SESSION_ID`

### Phase 2: Skill base.md

4. `skill-bank/traj/traj-launch-training/base.md` — 注入 `TRAJ_ROUND_NUM` + `TRAJ_SESSION_ID`
5. `skill-bank/rllm/rllm-train/base.md` — Phase 6.5 简化 + 标注保底机制
6. `skill-bank/traj/traj-loop/base.md` — 超时恢复 + 状态清理 + auto-approve 推荐
7. `skill-bank/traj/traj-optimize/base.md` — Step 5/6 使用 accept/reject API

### Phase 3: 编译 + 验证

8. `python skill-bank/compile.py --all`
9. 单元验证:
   - `python -c "from traj_opt.optimizer.patch_generator import PatchGenerator; ..."` — 验证 generate 不再自动 activate
   - `python -c "from rllm_train.logger import TrainingLogger; ..."` — 验证 _write_round_status 在无环境变量时 no-op
10. 集成验证: 下一次 `/traj-loop` 实验

---

## 架构决策记录

### 为什么用环境变量而不是命令行参数传递 round/session 信息？

- CLI-1 的入口是 `/rllm-train round=N -- 描述`，round 已通过 skill args 传递
- 但 TrainingLogger 是 Python 代码，不在 Claude Code 编排层，无法读取 skill args
- 环境变量是进程级别的，对 Python 代码和 Claude Code 编排层都可见
- 保底写入需要 Python 代码直接访问 round_num，环境变量是最简单的桥接

### 为什么不让 CLI-2 直接读 rllm_train/output/ 来获取训练结果？

- 数据边界隔离是核心设计准则: CLI-2 只从 `traj_opt/output/` 读数据
- 如果允许 CLI-2 读 `rllm_train/output/`，分析器就会绕过轨迹系统直接读训练日志
- 轨迹系统的价值在于: 记录的是 Claude Code 的决策过程（为什么这样调参），不只是训练结果
- status.json 是两个世界的唯一桥梁，保持这个约束

### 为什么 patch 生成和激活要分离？

- 当前: generate → activate → 展示 → 用户拒绝 → manifest 残留
- 修复后: generate → 展示 → 用户确认 → activate → compile
- 这符合"提议 → 审核 → 生效"的标准工作流
- 拒绝的 patch 文件也应该删除，避免 patches/ 目录积累垃圾

### 为什么预生成 session_id 而不是让 hook 自动生成？

- 自动生成的 session_id 基于 Claude Code 内部机制，外部无法预知
- 预生成后，启动方（traj-launch-training）、训练进程（TrainingLogger）、编排层（Phase 6.5）三方共享同一个 ID
- 消除了快照差分法的所有边界情况（CLI-2 hook 污染、并发、时序竞争）
