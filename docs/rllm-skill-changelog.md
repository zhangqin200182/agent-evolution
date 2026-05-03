# rllm Skill 系统演进记录

> 关键设计决策的时间线。详细规范见 `system-overview.md` 和各模块设计文档，skill 实现见 `skill-bank/` 各 base.md。

## 2026-04-29: v1 初始设计

建立 6 个 rllm-xx skill 的职责划分：

| Skill | 职责 |
|-------|------|
| rllm-train | 全流程编排（Phase 0-6） |
| rllm-clarify | 自然语言需求 → 结构化参数 |
| rllm-config | 配置生成与调参（含参数联动约束） |
| rllm-run | 后台启动训练进程 |
| rllm-monitor | 实时监控 + 异常检测 |
| rllm-analyze | 结果分析 + 调参建议 |

核心设计：编排者只做流转控制，每个 Phase 必须通过子 skill 执行，禁止内联。

## 2026-04-30: v2 实验优化

基于 8 轮训练实验（Qwen2.5-0.5B-Instruct, Mac MPS）的发现：

- **参数安全范围表**: 0.5B 模型 lr 上限 1e-5、epochs 上限 2，num_problems >= 64 时进一步收紧
- **Epoch 分段分析**: 按 epoch 切分 reward 序列，检测 catastrophic forgetting
- **格式退化检测**: 对比前/后 25% 步骤的 tool_call 使用率
- **Monitor 可靠性**: 通用 grep 模式、双重监控策略、健康检查
- **difficulty 参数**: simple/hard/mixed 三档，mixed 下 0.5B 的 num_problems 上限收紧到 32

## 2026-05-01: skill-bank 架构

建立 base + patch + compile 的 skill 管理系统。详见 `skill-bank-design.md`。

## 2026-05-02: 隔离设计与 trajectory 模块

- **数据表面化准则**: rllm-monitor/analyze 必须用 Read/Bash 显式读取训练数据，确保 hooks 捕获
- **数据边界**: traj-analyze-rllm 只从 traj_opt/output/ 读数据，禁止直接访问 rllm_train/
- **Agent 子 agent 方案 (Section 17)**: 尝试单 CLI + Agent 隔离，实测失败（Agent 同步阻塞）
- **双 CLI 架构 (Section 18)**: CLI-1 训练 + CLI-2 优化，通过 rounds/status.json 协调

## 2026-05-03: 端到端实测修复

Round 1 + Round 2 实测暴露 11 个问题，关键修复：

- **session_id 快照差分法**: Phase 0 记录 raw/ 目录快照，Phase 6.5 取差集
- **优化目标边界**: 用 skill-bank group 路径区分（rllm/ = 目标，traj/ = 工具），三层强制执行
- **PatchGenerator 增强**: 自动激活 patch、section 校验、group 校验
- **traj-launch-training**: osascript 交互式 / claude -p 非交互式，一键启动新 CLI-1
- **session 过滤**: get_rllm_trajectories() 支持 session_id 参数，避免历史数据污染
- **CLI-1 每次新建约束**: 保证一个 session_id 对应一次训练任务
