---
description: Generates skill-bank patches from trajectory analysis reports. Converts
  optimization suggestions into standard patch files for human review and compilation.
metadata:
  categories:
  - trajectory
  - optimization
  version: 1.0.0
name: traj-optimize
---


# traj-optimize — 生成 skill-bank patch

你是 skill 优化工具。你的职责是读取 traj-analyze-* 生成的分析报告，将其中的优化建议转化为标准的 skill-bank patch 文件，供审阅后编译生效。

支持两种审核模式:
- 人工审核（默认）: 展示详细信息，用 AskUserQuestion 等待确认
- 自动审核（`--auto-approve`）: 按置信度自动接受，低于阈值的仍需人工确认

## 执行步骤

### 1. 读取分析报告

读取 `traj_opt/output/reports/` 下最新的分析报告（或指定的报告路径）。

从报告的"优化建议"部分提取结构化建议:
- skill_name (目标 skill，必须属于 `skill-bank/rllm/` group)
- target_section (目标 section)
- action (replace/append/prepend/insert_after)
- patch_content (patch 内容)
- priority (优先级)
- confidence (置信度: high/medium/low)
- evidence_rounds (证据轮次数)
- description (描述)
- rationale (优化理由，含轨迹证据)

**Group 校验**: 如果 skill_name 不属于 `skill-bank/rllm/` group，跳过该建议并输出警告。traj-analyze-rllm 的优化目标仅限 `rllm/` group。

### 2. 解析 auto-approve 参数

从 args 中提取 `--auto-approve[=LEVEL]`:
- 无此参数: `auto_approve = None`（全部人工审核）
- `--auto-approve` 或 `--auto-approve=medium`: `auto_approve = "medium"`
- `--auto-approve=high`: `auto_approve = "high"`
- `--auto-approve=all`: `auto_approve = "all"`

置信度阈值映射:
- `"high"` → 只自动接受 confidence="high"
- `"medium"` → 自动接受 confidence="high" 或 "medium"
- `"all"` → 全部自动接受

### 3. 生成 patch 文件

```python
from traj_opt.optimizer.patch_generator import PatchGenerator
from traj_opt.adapter.schema import SkillOptimizationSuggestion

generator = PatchGenerator()

for suggestion in suggestions:
    patch_path = generator.generate_patch(suggestion)
    # patch 写入 skill-bank/{group}/{skill}/patches/traj-{timestamp}-{section}.md
```

### 3.5 验证 patch 目标 section

对每个生成的 patch，验证 target_section 在目标 skill 的 base.md 中存在:

```bash
grep -c "<!-- section:{target_section} -->" skill-bank/{group}/{skill_name}/base.md
```

如果 section 不存在:
- 列出该 skill 的所有 section: `grep -o '<!-- section:[a-z0-9-]* -->' skill-bank/{group}/{skill_name}/base.md`
- 尝试匹配最接近的 section 名
- 如果无法确定，标记该 patch 为 "[section 待确认]"，在 Step 4 展示时高亮提示

### 4. 展示 patch 供审阅

对每个生成的 patch，**必须展示以下全部信息（缺一不可）**:

```
Patch 1/3: traj-20260503-091930-param-ranges
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
目标:     {skill_name} / {target_section}
操作:     {action}
优先级:   {priority}
置信度:   {confidence_cn} (基于 {evidence_rounds} 轮数据)

优化理由:
{rationale}

--- Patch 内容 ---
{patch_content}
---

--- 当前 Section 内容 (对比参考) ---
{读取 skill-bank/{group}/{skill}/base.md 中 target_section 的当前内容}
---

{auto-approve 判定: [自动接受] 或 [需人工确认]}
```

#### 展示检查清单

每个 patch 展示时逐项确认:
1. patch_id、目标 skill/section、操作、优先级 — 从 suggestion 字段直接获取
2. 置信度 + 证据轮次 — 从 suggestion.confidence 和 evidence_rounds 获取
3. 优化理由 — 从 suggestion.rationale 获取，包含问题/现象/证据
4. 完整 patch 内容 — 不截断，展示 suggestion.patch_content 全文
5. 当前 section 内容 — **必须用 Read 工具**读取 `skill-bank/{group}/{skill}/base.md`，提取 `<!-- section:{target_section} -->` 到 `<!-- /section:{target_section} -->` 之间的内容
6. auto-approve 判定结果 — 根据置信度和阈值输出 `[自动接受]` 或 `[需人工确认]`

置信度中文映射: high→高, medium→中, low→低。

**禁止省略第 5 项**。如果 base.md 不存在或 section 未找到，输出 "(section 未找到)" 而非跳过。

### 5. 审核决策

#### 人工审核（auto_approve = None）

展示所有 patch 后，使用 AskUserQuestion 让用户选择:
- 全部接受并编译
- 逐个审阅（对每个 patch 单独确认 接受/拒绝）
- 跳过（不编译）

#### 自动审核（auto_approve != None）

对每个 patch 按置信度判断:

```python
CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}
THRESHOLD_MAP = {"high": 3, "medium": 2, "all": 0}
threshold = THRESHOLD_MAP[auto_approve_level]

for patch in patches:
    if CONFIDENCE_ORDER.get(patch.confidence, 0) >= threshold:
        输出 "[自动接受] {patch_id} (置信度: {confidence})"
        accepted.append(patch)
    else:
        输出 "[需人工确认] {patch_id} (置信度: {confidence}，低于阈值 {auto_approve_level})"
        # AskUserQuestion 单独确认此 patch
```

自动接受的 patch 仍然完整展示信息（不跳过展示），只是跳过确认步骤。

### 6. 编译

确认后（人工或自动）:
```python
from traj_opt.optimizer.compiler_bridge import CompilerBridge

bridge = CompilerBridge()
for skill_name in affected_skills:
    result = bridge.compile_skill(skill_name)
```

输出编译结果和接受/拒绝统计。

## Patch 命名规范

trajectory 生成的 patch 统一使用 `traj-` 前缀:
- `traj-20260501-120000-param-ranges.md`
- `traj-20260501-120000-anomaly-detection.md`

这样可以与手动创建的 patch 区分开来。
