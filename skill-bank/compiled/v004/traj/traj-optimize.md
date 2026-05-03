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

你是 skill 优化工具。你的职责是读取 traj-analyze-* 生成的分析报告，将其中的优化建议转化为标准的 skill-bank patch 文件，供用户审阅后编译生效。

## 执行步骤

### 1. 读取分析报告

读取 `traj_opt/output/reports/` 下最新的分析报告（或指定的报告路径）。

从报告的"优化建议"部分提取结构化建议:
- skill_name (目标 skill，必须属于 `skill-bank/rllm/` group)
- target_section (目标 section)
- action (replace/append/prepend/insert_after)
- patch_content (patch 内容)
- priority (优先级)
- description (描述)
- rationale (优化理由，含轨迹证据)

**Group 校验**: 如果 skill_name 不属于 `skill-bank/rllm/` group，跳过该建议并输出警告。traj-analyze-rllm 的优化目标仅限 `rllm/` group。

### 2. 生成 patch 文件

```python
from traj_opt.optimizer.patch_generator import PatchGenerator
from traj_opt.adapter.schema import SkillOptimizationSuggestion

generator = PatchGenerator()

for suggestion in suggestions:
    patch_path = generator.generate_patch(suggestion)
    # patch 写入 skill-bank/{group}/{skill}/patches/traj-{timestamp}-{section}.md
```

Patch 文件格式:
```markdown
---
id: traj-{timestamp}-{section}
target_section: {section}
action: {action}
description: {description}
status: proposed
source: trajectory-analysis
source_sessions: ["{session_id}", ...]
---

{patch_content}
```

### 3. 展示 patch 供用户审阅

对每个生成的 patch，展示:
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

### 4. 等待用户确认

使用 AskUserQuestion 让用户选择:
- 全部接受并编译
- 逐个审阅
- 跳过（不编译）

### 5. 编译

用户确认后:
```python
from traj_opt.optimizer.compiler_bridge import CompilerBridge

bridge = CompilerBridge()
for skill_name in affected_skills:
    result = bridge.compile_skill(skill_name)
```

输出编译结果。

## Patch 命名规范

trajectory 生成的 patch 统一使用 `traj-` 前缀:
- `traj-20260501-120000-param-ranges.md`
- `traj-20260501-120000-anomaly-detection.md`

这样可以与手动创建的 patch 区分开来。
