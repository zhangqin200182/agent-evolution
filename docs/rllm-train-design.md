# rllm_train 训练后端设计

> 自包含的 Agent RL 训练 pipeline，结合 rLLM 的 agent/environment 抽象与 TRL 的 GRPOTrainer。

> 命名说明：文档中 `rllm_train` 指代训练后端（代码目录 `rllm_train/`）。

## 1. 概述

rllm_train 是一个独立可运行的训练 pipeline，不依赖 skill 系统或优化后端。它从 rLLM 框架内联了最小的 agent/environment 抽象，避免 rLLM 的重依赖链（vllm, flash-attn, deepspeed），可在 Mac (MPS) 和 CPU 上运行。

## 2. 架构

```
train.py → GRPOTrainer → rollout_func → HFAgentExecutionEngine → agent/env loop
```

### 模块职责

| 模块 | 职责 |
|------|------|
| `train.py` | 入口。构建 dataset、model、tokenizer，组装 GRPOTrainer + 自定义 rollout function。包含 math reward function 和 tool 定义（CalculateTool, FinishTool） |
| `config.py` | `TrainingConfig` dataclass，所有超参数。`parse_natural_language()` 通过正则将自然语言描述转为配置。模型别名映射（qwen-0.5b → Qwen/Qwen2.5-0.5B-Instruct） |
| `rollout.py` | `make_rllm_rollout_func()` 返回闭包，GRPOTrainer 每步调用。创建 HFAgentExecutionEngine，运行异步 trajectory，计算 logprobs，返回 prompt/completion/mask tensors。TRL 与 rllm-style agent 执行的桥梁 |
| `hf_engine.py` | `HFAgentExecutionEngine`，使用 HuggingFace model.generate() 运行 agent-environment 循环。管理异步并行 trajectory、token 级 prompt/response 分割与 mask（1=model, 0=env）、MC return 计算 |
| `base.py` | 从 rLLM 内联的核心抽象：`BaseAgent`, `BaseEnv`, `Step`, `Action`, `Trajectory`, `ToolCall`, `ToolOutput`。BaseEnv 遵循 gym-style reset()/step() 接口 |
| `tool_agent.py` | `ToolAgent(BaseAgent)`，管理对话历史，通过 QwenToolParser 解析 tool call，格式化 observation。无法解析时 fallback 到 finish tool |
| `math_env.py` | `MathCalcEnv(BaseEnv)`，计算器环境。`generate_math_problems()` 生成算术数据集。Reward 为二值：答案匹配 1.0，否则 0.0 |
| `parsers.py` | Chat template 解析器（通用 + Qwen 专用）和 tool call 解析器。`convert_messages_to_tokens_and_masks()` 是关键函数：tokenize messages 并生成 per-token mask 区分 model/env tokens |
| `logger.py` | `TrainingLogger`，训练过程中打印实时进度表，结束时输出 summary report |
| `perf_stats.py` | `PerfTracker`，在 rollout/step/token 级别追踪耗时。分解为 LLM 推理、env 执行、logprob 计算、GRPO 训练 |
| `trajectory_writer.py` | 保存 per-step JSONL 文件，包含完整对话、metrics、decoded response text |

## 3. 关键设计决策

### Response Mask 系统

mask（1=model tokens, 0=env tokens）是 GRPO 训练正确性的核心 — 只有 model 生成的 token 才应接收梯度更新。masking 在 `hf_engine.py` 中通过 `convert_messages_to_tokens_and_masks()` 实现。

### Rollout 异步处理

rollout function 处理 asyncio event loop 边缘情况（running loop 检测、thread pool fallback），因为 TRL 的训练循环可能已有活跃的 event loop。

## 4. 数据模型

### TrainingConfig

核心超参数（`config.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_name | Qwen/Qwen2.5-0.5B-Instruct | HuggingFace 模型路径 |
| num_problems | 64 | 训练问题数 |
| num_epochs | 2 | 训练轮次 |
| learning_rate | 1e-5 | 学习率 |
| batch_size | 2 | 批大小 |
| num_generations | 4 | 每 prompt 生成数（GRPO） |
| temperature | 0.7 | 采样温度 |
| difficulty | mixed | 题目难度（simple/hard/mixed） |
| max_completion_length | 256 | 最大生成长度 |
| max_agent_steps | 3 | agent 最大交互轮次 |

### 输出目录

```
rllm_train/output/runs/<run_id>/
├── config.json          # 训练配置
├── training_log.txt     # 训练日志（reward/loss 趋势）
├── perf_stats.json      # 性能统计
├── analysis.json        # 分析结果（rllm-analyze 生成）
├── trajectories/        # per-step JSONL
└── final_model/         # 训练后的模型
```

## 5. 运行方式

```bash
# 默认配置
python -m rllm_train.train

# 自然语言配置
python -m rllm_train.train "用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch"
python -m rllm_train.train "quick test with 16 problems"

# 从配置文件（由 rllm-config skill 生成）
python -m rllm_train.run_training rllm_train/output/runs/<run_id>/config.json
```
