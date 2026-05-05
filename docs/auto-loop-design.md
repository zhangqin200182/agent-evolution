# traj-loop 自动化模式设计文档

## 1. 概述

本文档描述双 CLI 架构下三种操作模式的设计：手动、半自动、全自动。核心改动是重写 `/traj-loop` skill，增加 patch 置信度机制，改进审核展示。

## 2. 三种操作模式

| 模式 | CLI-1 启动 | 优化触发 | Patch 审核 | 使用方式 |
|------|-----------|---------|-----------|---------|
| 手动 | 交互式 | 用户手动执行 | 人工 | `/traj-launch-training` + `/traj-train-optimize` |
| 半自动 | 交互式 | CLI-2 自动 | 人工 | `/traj-loop 用 qwen-0.5b 训练, 3 轮` |
| 全自动 | 非交互式 | CLI-2 自动 | 自动（按置信度） | `/traj-loop ..., --auto --auto-approve` |

**手动模式**不涉及 traj-loop，用户分别执行两个 skill，完全控制每个步骤。

**半自动模式**由 traj-loop 编排循环，CLI-1 在新终端窗口中交互式运行（用户可以在训练过程中干预，如 rllm-train 的 approve 模式逐步确认），CLI-2 自动轮询等待训练完成后触发优化流程，patch 仍需人工审核。

**全自动模式**由 traj-loop 编排循环，CLI-1 以 `claude -p --permission-mode auto` 后台运行（无人值守），CLI-2 自动轮询等待并触发优化，patch 按置信度自动接受。

## 3. traj-loop 参数设计

两个独立 flag 控制两个维度，自由组合：

### 3.1 `--auto` — CLI-1 启动方式

- 不带 `--auto`（默认）: 交互式，osascript 打开新终端窗口
- 带 `--auto`: 非交互式，`claude -p --permission-mode auto` 后台启动

语义和 `traj-launch-training` 的 `--auto` 参数一致，traj-loop 直接透传。

### 3.2 `--auto-approve[=LEVEL]` — Patch 审核策略

- 不带此参数（默认）: 所有 patch 走 AskUserQuestion 人工审核
- `--auto-approve` 或 `--auto-approve=medium`: 自动接受 high + medium 置信度，low 仍需人工
- `--auto-approve=high`: 只自动接受 high 置信度，medium + low 仍需人工
- `--auto-approve=all`: 全部自动接受

### 3.3 参数组合矩阵

| 参数 | CLI-1 | Patch 审核 | 典型场景 |
|------|-------|-----------|---------|
| （默认） | 交互式 | 全部人工 | 首次使用，想控制训练过程 |
| `--auto` | 非交互式 | 全部人工 | 训练无需干预，但想审核每个 patch |
| `--auto-approve` | 交互式 | medium+ 自动 | 想控制训练，但信任高置信度优化 |
| `--auto --auto-approve` | 非交互式 | medium+ 自动 | 完全无人值守 |
| `--auto --auto-approve=high` | 非交互式 | 仅 high 自动 | 无人值守但保守审核 |

### 3.4 示例

```bash
# 半自动：交互式训练 + 人工审核
/traj-loop 用 qwen-0.5b 训练, 3 轮

# 全自动：非交互式训练 + 自动审核（medium+）
/traj-loop 用 qwen-0.5b 训练, 3 轮, --auto --auto-approve

# 全自动但保守：非交互式训练 + 只自动接受高置信度
/traj-loop 用 qwen-0.5b 训练, 3 轮, --auto --auto-approve=high

# 非交互式训练但 patch 仍人工审核
/traj-loop 用 qwen-0.5b 训练, 3 轮, --auto
```

## 4. traj-loop 编排流程

### 4.1 核心循环

```
解析参数 → 检查/恢复状态

for round in 1..N:
    Step 1: 启动训练
        调用 Skill("traj-launch-training", args="round={N} [--auto] | {描述}")
        → 交互式: osascript 打开新终端
        → 非交互式: claude -p 后台启动

    Step 2: 轮询等待
        每 30s 读取 traj_opt/output/rounds/round_{N}/status.json
        → training_complete: 继续
        → training_failed: 记录，询问是否继续下一轮
        → 超时 (30min): 报告，询问是否继续等待

    Step 3: 执行优化
        调用 Skill("traj-train-optimize", args="round={N}")
        → traj-train-optimize 内部调用 traj-segment → traj-analyze-rllm → traj-optimize
        → traj-optimize 根据 --auto-approve 参数决定审核方式

    Step 4: 本轮摘要
        输出 round, run_id, reward, patches_generated, patches_accepted

输出跨轮对比报告
```

### 4.2 轮询机制

```python
# 每 30s 检查一次 status.json
while True:
    status = RoundState().read_status(round_num)
    if status and status["status"] == "training_complete":
        break
    elif status and status["status"] == "training_failed":
        报告失败，询问是否继续
        break
    elif elapsed > 1800:  # 30 分钟超时
        报告超时，询问是否继续等待
    sleep 30
```

轮询使用 Bash 工具执行 Python 代码读取 status.json，不使用 Monitor（轮询间隔固定，不需要流式通知）。

### 4.3 参数透传

traj-loop 将参数透传给子 skill：
- `--auto` → 透传给 `traj-launch-training`（控制 CLI-1 启动方式）
- `--auto-approve[=LEVEL]` → 透传给 `traj-optimize`（控制审核策略）

traj-train-optimize 作为中间编排层，需要能接收并透传 `--auto-approve` 给 traj-optimize。

### 4.4 状态管理与中断恢复

`traj_opt/output/loop_state.json`:
```json
{
  "total_rounds": 3,
  "current_round": 2,
  "description": "用 qwen-0.5b 训练, reward >= 0.8",
  "auto": false,
  "auto_approve": null,
  "rounds": [
    {"round": 1, "run_id": "run_xxx", "reward": 0.86, "patches_generated": 3, "patches_accepted": 3}
  ]
}
```

启动时检查 loop_state.json，如果存在且未完成，询问是否从断点继续。

### 4.5 失败处理

| 场景 | 处理 |
|------|------|
| 训练失败 | 记录到 loop_state，询问是否继续下一轮 |
| 优化失败 | 记录到 loop_state，询问是否继续下一轮 |
| 轮询超时 | 报告超时，询问是否继续等待或跳过 |
| 用户中断 | loop_state 已保存，下次可恢复 |

## 5. Patch 置信度机制

### 5.1 置信度定义

| 级别 | 判定标准 | 含义 |
|------|---------|------|
| high | 3+ 轮轨迹的一致模式 | 高度确信，可安全自动接受 |
| medium | 1-2 轮轨迹数据 | 有依据但样本不足，建议人工审核 |
| low | 推测性解释，缺乏充分数据 | 需要更多数据验证，建议谨慎 |

### 5.2 数据流

```
traj-analyze-rllm 生成报告（含 confidence 列）
    ↓
traj-optimize 读取报告，构造 SkillOptimizationSuggestion（含 confidence 字段）
    ↓
PatchGenerator 生成 patch 文件（frontmatter 含 confidence）
    ↓
traj-optimize 展示 patch（显示置信度）
    ↓
根据 --auto-approve 阈值决定自动接受或人工审核
```

### 5.3 SkillOptimizationSuggestion 新增字段

```python
@dataclass
class SkillOptimizationSuggestion:
    # ... 现有字段 ...
    confidence: str = "medium"    # "high" / "medium" / "low"
    evidence_rounds: int = 0      # 支撑此建议的轨迹轮次数
```

### 5.4 Patch frontmatter 新增字段

```yaml
---
id: traj-{timestamp}-{section}
target_section: {section}
action: {action}
description: {description}
status: proposed
confidence: high
evidence_rounds: 3
source: trajectory-analysis
source_sessions: ["session1", "session2"]
---
```

## 6. Patch 审核展示改进

### 6.1 当前展示（不足）

```
Patch: {patch_id}
目标: {skill_name} / {target_section}
操作: {action}
优先级: {priority}
来源: {source_sessions}

--- 内容预览 ---
{patch_content 前 20 行}
---
```

缺少：置信度、优化理由、当前 section 内容对比。

### 6.2 改进后展示

```
Patch 1/3: traj-20260503-091930-param-ranges
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标:     rllm-config / param-ranges
操作:     append
优先级:   P1
置信度:   高 (基于 3 轮一致模式)
证据轮次: 3

优化理由:
  问题: Loss 始终为 0，题目太简单
  现象: Round 1-3 的 loss 均为 0，simple 题目对 0.5B 模型无学习信号
  证据: Session b4d588ba, c7b850f9, d3e... 三轮一致

--- Patch 内容 ---
{完整 patch_content}
---

--- 当前 Section 内容 (对比参考) ---
{base.md 中 target_section 的当前内容}
---
```

### 6.3 auto-approve 行为

- 自动接受时：完整展示 patch 信息 + 输出 `[自动接受] Patch: {id} (置信度: {confidence})`
- 需人工审核时：完整展示 patch 信息 + AskUserQuestion 确认
- 混合场景（部分自动、部分人工）：逐个处理，自动的直接通过，人工的等待确认

## 7. 文件改动清单

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `traj_opt/adapter/schema.py` | 小 | SkillOptimizationSuggestion 增加 confidence/evidence_rounds |
| `traj_opt/optimizer/patch_generator.py` | 小 | frontmatter 输出 confidence/evidence_rounds |
| `skill-bank/traj/traj-loop/base.md` | 重写 | 核心改动，双模式编排 |
| `skill-bank/traj/traj-optimize/base.md` | 中 | 审核展示改进 + auto-approve |
| `skill-bank/traj/traj-analyze-rllm/base.md` | 小 | 报告格式增加 confidence 列 |

不改动：`round_state.py`、`traj-train-optimize`、`traj-launch-training`、`traj-segment`、所有 `rllm/` skills。
