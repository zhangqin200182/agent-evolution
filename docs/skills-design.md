# Skills 设计

> 两组 Claude Code skill 的职责划分、调用关系和编排逻辑。Skills 是系统的使用入口。

> 命名说明：文档中 `rllm_train` 指代训练后端（代码目录 `rllm_train/`），`traj_opt` 指代优化后端（代码目录 `traj_opt/`）。

## 1. 概述

系统有两组 skill，分别驱动训练和优化：

| Group | 目录 | 驱动的后端 | 职责 |
|-------|------|-----------|------|
| rllm-xx | `skill-bank/rllm/` | rllm_train | 编排训练全流程：需求澄清 → 配置 → 训练 → 监控 → 分析 → 调参循环 |
| traj-xx | `skill-bank/traj/` | traj_opt | 编排优化全流程：分割轨迹 → 分析 → 生成 patch → 编译 |

rllm-xx skills 是 rllm_train 的自动化编排层 — rllm_train 本身可独立运行，skills 在其上加了需求澄清、异常处理、多轮调参等逻辑。

traj-xx skills 是 traj_opt 的操作指南 — traj_opt 的 Python 代码提供基础设施，skills 定义分析策略和领域知识。

## 2. rllm-xx Skills

### Skill 清单

| Skill | 类型 | 职责 |
|-------|------|------|
| rllm-train | 编排 | 全流程编排（Phase 0-6），串联子 skill |
| rllm-clarify | 工具 | 自然语言需求 → 结构化参数 |
| rllm-config | 工具 | 配置生成与调参（含参数安全范围、联动约束） |
| rllm-run | 工具 | 后台启动训练进程 |
| rllm-monitor | 工具 | 实时监控 + 异常检测 + early stopping |
| rllm-analyze | 工具 | 结果分析 + 调参建议 |

### 编排流程

```
Phase 0: 输入分级与引导 (编排者)
    ↓
Phase 1: 需求澄清 → rllm-clarify
    ↓
Phase 2: 配置生成 → rllm-config
    ↓
Phase 3-5: 训练循环
    ├→ 启动训练 → rllm-run
    ├→ 过程监控 → rllm-monitor
    ├→ 结果分析 → rllm-analyze
    └→ 未达标 → rllm-config 调参 → 重新训练
    ↓
Phase 6: 最终报告 (编排者)
Phase 6.5: 写入轮次状态 (双 CLI 模式)
```

### 数据传递契约

```
Phase 0 → 1: 自然语言描述
Phase 1 → 2: 结构化需求摘要
Phase 2 → 3: config.json 路径
Phase 3 → 4: 后台任务 ID + 日志路径
Phase 4 → 5: 训练完成确认 + run_id
Phase 5 → 2 (循环): analysis.json 路径
```

## 3. traj-xx Skills

### Skill 清单

| Skill | 类型 | 职责 |
|-------|------|------|
| traj-train-optimize | 编排 | 单轮优化编排（CLI-2），串联 segment → analyze → optimize |
| traj-launch-training | 工具 | 在 CLI-2 中启动新 CLI-1 进程执行训练 |
| traj-loop | 编排 | 全自动多轮编排（单 CLI 模式，用 Agent 子 agent 隔离） |
| traj-segment | 工具 | 将 raw events 分割为有意义的轨迹单元 |
| traj-analyze-rllm | 分析 | 分析 rllm-xx 轨迹，发现问题模式，生成优化建议 |
| traj-optimize | 工具 | 将优化建议转为 skill-bank patch，展示确认，编译 |
| traj-setup | 工具 | 一次性环境配置（hooks 安装） |
| traj-status | 工具 | 查看轮次状态和轨迹数据概览 |

### 两种编排模式

**双 CLI 模式（当前方案）**：

```
CLI-2:
  traj-launch-training → [新 CLI-1 执行 rllm-train]
  traj-train-optimize
    └─ traj-segment → traj-analyze-rllm → traj-optimize
```

**单 CLI 模式（traj-loop，实验性）**：

```
traj-loop
  └─ rllm-train (Agent 子 agent) → traj-segment → traj-analyze-rllm (Agent 子 agent) → traj-optimize
```

双 CLI 模式是当前推荐方案，用进程边界实现物理隔离。

## 4. 使用场景

### 手动流程

用户完全控制每一步：

```
/rllm-train "用 qwen-0.5b 训练数学 agent"    # 训练
/traj-segment                                  # 分割
/traj-analyze-rllm                             # 分析
/traj-optimize                                 # 生成 patch + 确认 + 编译
```

### 半自动流程

训练手动触发，分割+分析+优化一条命令：

```
/rllm-train "..."
/traj-analyze-rllm --optimize                  # 自动: 分割 → 分析 → patch → 确认
```

### 全自动流程（双 CLI）

CLI-2 中一键启动训练和优化：

```
/traj-launch-training round=1 | 用 qwen-0.5b 训练, reward >= 0.8
  → 新终端窗口打开，用户在其中交互式训练
  → 训练完成后回到 CLI-2
/traj-train-optimize round=1
  → 分割 → 分析 → 生成 patch → 确认 → 编译
/traj-launch-training round=2 | ...
  → 使用优化后的 skill 训练
```

## 5. traj-analyze-xx 通用协议

定义分析 skill 的标准接口，使新增分析器有明确模板。

### 输入协议

所有 traj-analyze-xx 的输入只来自：

| 路径 | 用途 |
|------|------|
| `traj_opt/output/trajectories/{session_id}/trajectories.jsonl` | 分割后轨迹（主要输入） |
| `traj_opt/output/raw/{session_id}/events.jsonl` | 原始事件（补充） |
| `traj_opt/output/reports/` | 历史报告（跨轮次对比） |
| `traj_opt/output/index.jsonl` | 索引（查找相关 session） |

禁止直接读取被观察 skill 的内部输出目录、源代码或 skill-bank 源文件。

### 输出协议

输出到 `traj_opt/output/reports/`，每个问题必须包含证据（引用 session_id 和轨迹数据）和置信度。

### 领域知识规范

| 允许 | 禁止 |
|------|------|
| 通用训练动态模式表 | 硬编码参数安全范围表 |
| 模式识别方法论 | 固定的 "问题 → skill" 映射 |
| 从轨迹推断的安全边界（附证据） | "经验表明" 无证据表述 |

### 创建新分析器

1. 在 `skill-bank/traj/` 下创建 `traj-analyze-{domain}/` 目录
2. 创建 `base.md`，包含 section: intro, data-boundary, analysis-framework, steps, domain-knowledge
3. 注册到 `manifest.yaml` 和 `bank.yaml`
4. 编译：`python skill-bank/compile.py traj-analyze-{domain}`

## 6. Skill 与后端的关系

| Skill Group | 后端 | 关系 |
|-------------|------|------|
| rllm-xx | rllm_train (`rllm_train/`) | rllm_train 独立可运行，skills 是自动化编排层 |
| traj-xx | traj_opt (`traj_opt/`) | traj_opt 是 skills 的 Python 后端，提供基础设施 |

rllm-xx skills 中的代码片段调用 rllm_train 的接口：
```python
from rllm_train.config import TrainingConfig
config = TrainingConfig.from_json('rllm_train/output/runs/<run_id>/config.json')
```

traj-xx skills 中的代码片段调用 traj_opt 的接口：
```python
from traj_opt.analyzer.base import AnalyzerBase
from traj_opt.optimizer.patch_generator import PatchGenerator
```

Skill 定义分析策略和领域知识（在 SKILL.md 中），Python 代码提供基础设施（读取轨迹、生成 patch、编译）。
