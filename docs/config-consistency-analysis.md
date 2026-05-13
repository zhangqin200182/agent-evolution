# rllm_remote → AgentSDK veRL YAML 配置一致性分析

## 范围

聚焦三层配置之间的参数重叠和不一致问题：
- **Layer 1**: 模型权重目录中的配置文件（`generation_config.json`, `config.json`, `tokenizer_config.json`）
- **Layer 2**: `RemoteTrainConfig → config_generator.py → AgentSDK veRL YAML`
- **Layer 3**: 代码中硬编码的参数（`rllm_common/parsers.py`, `config_generator.py`）

不涉及 MindSpeed RL（已弃用）。

## 配置层全景

```
RemoteTrainConfig (扁平 dataclass)
       │
       ▼
config_generator.py (展开逻辑 + 硬编码 BASE_CONFIG)
       │
       ▼
AgentSDK YAML (深度嵌套，Hydra 驱动)
  ├─ verl_conf           ← veRL 训练配置
  ├─ train_instances     ← 训练实例声明
  ├─ agent_instances     ← Agent 服务配置
  ├─ infer_instances     ← 推理服务配置
  └─ rollout_config      ← 仅 one_step_off，独立推理集群配置
```

---

## 〇、模型权重目录中的配置（Layer 1：最底层默认值）

### 为什么模型目录是"第零层配置源"

任何 vLLM/HuggingFace 模型权重目录下都包含一组标准配置文件：

| 文件 | 作用 | 谁读取 |
|------|------|--------|
| `config.json` | 模型架构参数（层数、维度、max_position_embeddings 等） | vLLM 加载模型时读取，决定显存分配和推理图编译 |
| `generation_config.json` | HuggingFace 默认生成参数（temperature、top_p、top_k 等） | vLLM 作为 `SamplingParams` 的 fallback 默认值 |
| `tokenizer_config.json` | Tokenizer 配置（特殊 token、chat_template、model_max_length） | `AutoTokenizer.from_pretrained()` 加载 tokenizer 时读取 |
| `tokenizer.json` | 完整 vocabulary | Tokenizer 编码/解码 |

这些文件**随模型权重一起分发**，是模型作者设定的"出厂默认值"。当我们的训练/推理配置没有显式指定某个参数时，vLLM 和 tokenizer 会**静默回退到这些文件中的值**。

### 0.1 三重配置覆盖模型

```
模型出厂默认值 (generation_config.json, config.json)
    │
    ├── 被覆盖: verl_conf.rollout.* (do_sample, n, prompt_length, response_length, ...)
    │
    ├── 被覆盖: agent_instances[].infer_service_params.* (temperature, top_p, max_tokens, ...)
    │
    └── 未被覆盖 → 静默生效: top_k, repetition_penalty, ...
```

问题不是"某个模型的默认值是什么"，而是：**我们的配置体系没有覆盖所有影响训练/推理行为的参数，导致部分参数由模型说了算——换模型行为就变**。

### 0.2 三类冲突模式

#### A. 覆盖不完整

我们的 `RemoteTrainConfig` + `config_generator.py` 只覆盖了生成参数的一个子集：

| 被覆盖的参数 | 覆盖路径 | 未覆盖的参数（静默继承模型默认） |
|------------|---------|------------------------------|
| `temperature` | `agent_temperature` → Agent 侧 | `top_k` |
| `top_p` | `agent_top_p` → Agent 侧 | `repetition_penalty` |
| `do_sample` | `rollout.do_sample` → Rollout 侧 | `min_p`, `typical_p`, `frequency_penalty`, `presence_penalty` 等 |
| `max_tokens` | `agent_max_tokens` → Agent 侧, `response_length` → Rollout 侧 | — |

**风险**：换一个模型（如 Qwen3-235B、DeepSeek），其 `generation_config.json` 中的 `top_k`、`repetition_penalty` 等默认值可能不同。训练行为会随模型变化而**无感知地改变**，导致训练结果不可复现。

#### B. 参数名不同但语义相同

| 模型文件中的字段 | 我们的等价字段 | 是否桥接 |
|----------------|--------------|:------:|
| `generation_config.temperature` | `agent_temperature` (Agent) / `temperature` (死字段) | Agent ✅ / Rollout ❌ |
| `generation_config.top_p` | `agent_top_p` (Agent) / `top_p` (死字段) | Agent ✅ / Rollout ❌ |
| `config.max_position_embeddings` | `max_model_len` | ❌ 无显式关系 |
| `tokenizer_config.model_max_length` | `max_prompt_length`, `max_response_length` | ❌ 无显式关系 |

**风险**：两套命名体系之间的对应关系靠人工理解维持。新团队成员或新模型上线时，容易遗漏桥接导致参数不一致。

#### C. 模型硬限制 vs 配置值不校验

`config.json` 中定义了模型的硬性上限：

- `max_position_embeddings` — RoPE 可支持的最大位置编码
- `vocab_size` — tokenizer 输出的 token ID 必须小于此值
- `torch_dtype` — 训练的 dtype 必须与此兼容

我们的配置值（`max_model_len`、`max_prompt_length + max_response_length`）如果超出这些硬限制，**不会在生成 YAML 时报错**，而是在运行时表现为：
- vLLM 静默截断超出部分
- 或 NPU kernel 因维度不匹配而崩溃（报错信息不指向根因）

### 0.3 派生参数的隐式依赖

`config_generator.py` 中存在硬编码的派生关系：

```python
"max_model_len": config.max_response_length * 2
```

这个关系的三个问题：

1. **假设 prompt:response = 1:1**：`* 2` 假设 prompt 和 response 各占一半上下文窗口。但 Agent 多步推理中，前几轮对话、tool call、tool response 都占用 token——实际 prompt 远大于单轮的 response。
2. **不与模型上限对比**：没有校验 `max_response_length * 2 ≤ max_position_embeddings`。如果用户配置 `max_response_length = 20000`，生成 `max_model_len = 40000`，Qwen2.5-7B 的 `max_position_embeddings = 32768`，溢出但不报错。
3. **不可追溯**：`* 2` 的系数没有任何文档或配置项暴露，未来调整只能改代码。

### 0.4 配置覆盖完整性校验原则

基于以上分析，一个合格的配置校验应该覆盖：

```
1. 覆盖完整性：模型 generation_config 中的每个参数，在我们的配置中是否有显式覆盖或显式确认"使用默认值"？

2. 硬限制校验：我们的 max_model_len / max_prompt_length / max_response_length 是否 ≤ 模型的 max_position_embeddings？

3. 语义匹配：同名（或不同名但语义相同）的参数在模型文件和我们的配置中是否一致？

4. 派生参数溯源：所有硬编码的派生关系（*2 等）是否有文档说明其假设前提？
```

---

## 一、Hybrid 模式架构与配置流

### 1.1 代码执行路径

```
start.py
  → start_direct_mode()
    → AgentManager.setup()
       读取 agent_instances，创建 AgentExecutor
    → InferManager.setup()
       读取 infer_instances，创建 InferExecutor
    → TrainRouter.train()
      → TrainExecutor.fit()
        → _run_train_and_rollout()
          start_rollout = None  ← registry: ("verl", "hybrid") → (None, verl_hybrid_train)
          start_train = verl_hybrid_train (阻塞)
            → run_ppo(verl_conf, HybridTaskRunner)
              → HybridTaskRunner.run(config=verl_conf)
                读取 config.actor_rollout_ref.model.path → 加载 tokenizer + 模型
                读取 config.data.train_files → 加载数据
                → HybridTrainer.fit()
                  同一组 GPU 上: vllm rollout + Agent 多轮对话 + reward 计算
```

### 1.2 各模块读取的配置

| 模块 | 读取的 YAML 区块 | 关键字段 |
|------|-----------------|---------|
| HybridTaskRunner | `verl_conf` 全部 | `model.path`, `data.*`, `rollout.*`, `actor.*` |
| AgentExecutor → RLLMEngineWrapper | `agent_instances[].executor_kwargs` | `tokenizer`, `max_prompt_length`, `max_model_len`, `infer_service_params.temperature/top_p/max_tokens` |
| InferExecutor | `infer_instances[].executor_kwargs` | `engine`, `model`, `tensor_parallel_size`, `gpu_memory_utilization` |

### 1.3 Hybrid 模式下的同参异位

#### A. 模型路径（2 处）

| YAML 位置 | 消费方 | 用途 |
|-----------|--------|------|
| `verl_conf.actor_rollout_ref.model.path` | HybridTaskRunner | 加载训练 tokenizer + 模型权重 |
| `agent_instances[].agent_engine_kwargs.tokenizer` | RLLMEngineWrapper | 加载 Agent 的 tokenizer |

当前 `config_generator.py` 用同一个 `model_name_or_path` 填写两处，但如果 BASE_CONFIG 的硬编码路径没更新，就会不一致。

Agent 的 tokenizer 用于将 Agent 聊天消息转为 token IDs 发送给 vllm engine。训练 tokenizer 用于将 prompt/completion 转为 token IDs 计算 loss。**两者如果用的不是同一个 tokenizer，token ID 映射不同 → 训练 loss 基于 tokenizer A 的 IDs，推理生成基于 tokenizer B 的 IDs → 训推精度不对齐。**

#### B. max_prompt_length（3 处）

| YAML 位置 | 消费方 |
|-----------|--------|
| `verl_conf.data.max_prompt_length` | verl dataset 截断 |
| `verl_conf.actor_rollout_ref.rollout.prompt_length` | vllm engine prompt 限制 |
| `agent_instances[].agent_engine_kwargs.max_prompt_length` | RLLMEngineWrapper Agent 上下文截断 |

约束：①② 都是 vllm 侧的 prompt 限制，应该一致。③ 是 Agent 发给 vllm 之前的截断。如果 ①=2048 但 ③=8192，Agent 会构造 8192 token 的 prompt 发给 vllm，但 vllm 只保留前 2048，Agent 以为自己传了完整上下文，实际被截断了。

#### C. max_response_length / max_tokens / max_model_len（4 处）

| YAML 位置 | 消费方 |
|-----------|--------|
| `verl_conf.data.max_response_length` | verl dataset 期望的 response 长度 |
| `verl_conf.actor_rollout_ref.rollout.response_length` | vllm engine 生成的最大 token 数 |
| `agent_instances[].agent_engine_kwargs.max_model_len` | Agent 侧总上下文窗口 |
| `agent_instances[].infer_service_params.max_tokens` | Agent 每步推理最大生成 token 数 |

`config_generator.py` 中 `max_model_len` 由 `max_response_length * 2` 硬编码派生（第 292 行），这个 2x 关系没有任何文档或校验。

约束：`infer_service_params.max_tokens + max_prompt_length ≤ max_model_len`。如果 Agent 的 `max_tokens=4096` 但 `max_model_len - max_prompt_length = 4096 - 2048 = 2048`，Agent 请求生成 4096 token 但 context window 只够 2048。

#### D. temperature / top_p（2 处）

| YAML 位置 | 作用 |
|-----------|------|
| `verl_conf.actor_rollout_ref.rollout.temperature` | vllm rollout 生成时的采样温度 |
| `agent_instances[].infer_service_params.temperature` | Agent 每步调用 LLM 的采样温度 |

Hybrid 模式下两者共用同一个 vllm engine，当前 rollout 设置 `do_sample: false`（贪婪解码，温度不生效），所以 Agent 侧的 temperature 是唯一生效的。但如果改为 `do_sample: true`，两者不一致会导致 rollout 生成和 Agent 后续推理使用不同的采样参数。

#### E. tensor_parallel_size（2 处）

| YAML 位置 | Hybrid 下 |
|-----------|----------|
| `verl_conf.actor_rollout_ref.rollout.tensor_model_parallel_size` | 训练 rollout vllm 的 TP |
| `infer_instances[].engine_kwargs.tensor_parallel_size` | 推理 vllm engine 的 TP |

Hybrid 下两者是同一个 vllm engine，必须一致。但写在两个不同的配置块里，没有约束。

---

## 二、One-Step-Off 模式架构与配置流

### 2.1 代码执行路径

```
start.py
  → start_direct_mode()
    → AgentManager.setup()   读取 agent_instances
    → InferManager.setup()   读取 infer_instances
    → TrainRouter.train()
      → TrainExecutor.fit()
        → _run_train_and_rollout()
          ├─ start_rollout = start_rollout()  ← 非阻塞
          │    → RolloutWorker(rollout_config.*)
          │        tokenizer_name_or_path, n_samples_per_prompt,
          │        max_prompt_length, infer_tensor_parallel_size, ...
          │    → RolloutController(rollout_config.*)
          │        tokenizer_name_or_path, weight_save_dir,
          │        train_tensor_parallel_size, infer_expert_parallel_size, ...
          │    → OneStepOffRollouter: rollout 主循环
          │
          └─ start_train = verl_full_async_train  ← 阻塞
               → run_ppo(verl_conf, FullyAsyncTaskRunner)
                 → FullyAsyncTaskRunner.run(config=verl_conf)
                   读取 config.actor_rollout_ref.model.path → 训练 tokenizer
                   读取 config.extras.* → TrainController 参数
                   → FullyAsyncTrainer.fit()
                     Actor 权重更新 + ParameterSynchronizer 同步到 rollout 集群
```

### 2.2 两套独立配置系统的并行运行

```
┌─── 训练集群 (train cluster) ───┐     ┌─── 推理集群 (rollout cluster) ───┐
│                                │     │                                  │
│  verl_conf                     │     │  rollout_config                  │
│  ├─ model.path                 │     │  ├─ tokenizer_name_or_path       │
│  ├─ data.{train_files, ...}    │     │  ├─ n_samples_per_prompt         │
│  ├─ actor.{optim, ppo_*, ...}  │     │  ├─ max_prompt_length            │
│  ├─ rollout.{n, tp, ...}       │     │  ├─ infer_tensor_parallel_size   │
│  └─ trainer.{epochs, ...}      │     │  ├─ train_tensor_parallel_size   │
│                                │     │  └─ weight_save_dir              │
│  FullyAsyncTaskRunner          │     │                                  │
│  FullyAsyncTrainer             │     │  RolloutWorker                   │
│  TrainController               │     │  RolloutController               │
│                                │     │  OneStepOffRollouter             │
│         ↓ weight sync ↓        │     │                                  │
│  ──────────────────────────────┼─────┼→ agent_instances (Agent)        │
│                                │     │  infer_instances (vllm engine)   │
└────────────────────────────────┘     └──────────────────────────────────┘
```

### 2.3 One-Step-Off 模式下的同参异位

#### A. 模型/Tokenizer 路径（3 处，最高风险）

| # | YAML 位置 | 消费方 | 用途 |
|---|----------|--------|------|
| 1 | `verl_conf.actor_rollout_ref.model.path` | FullyAsyncTaskRunner | 训练集群加载 tokenizer + 模型权重 |
| 2 | `agent_instances[].agent_engine_kwargs.tokenizer` | RLLMEngineWrapper | Agent tokenizer（推理集群上运行） |
| 3 | `rollout_config.tokenizer_name_or_path` | RolloutWorker + RolloutController | 推理集群加载 tokenizer + 权重加载路径 |

**这是最严重的风险点**：训练集群通过 `ParameterSynchronizer` 把训练后的权重同步到 ③ 的路径下，RolloutController 把新权重加载到 vllm engine。但如果 ① ≠ ③ 的 base 模型路径不一致，同步的目标路径就不对，推理集群加载的是错误的/过期的权重。同时如果 ② ≠ ③，Agent 用的 tokenizer 和 rollout vllm 用的 tokenizer 不同，token ID 映射不一致。

当前 `config_generator.py` 中：
- ①③ 都来自 `model_name_or_path`（但 ③ 只在一处覆盖）
- ② 来自 `tokenizer_name_or_path`

且 `rollout_config` 块**当前完全没有生成**——意味着 one_step_off 模式下训练无法正常启动。

#### B. max_prompt_length（4 处）

| # | YAML 位置 | 消费方 |
|---|----------|--------|
| 1 | `verl_conf.data.max_prompt_length` | 训练集群 dataset 截断 |
| 2 | `verl_conf.actor_rollout_ref.rollout.prompt_length` | 训练集群 vllm |
| 3 | `agent_instances[].agent_engine_kwargs.max_prompt_length` | Agent（推理集群） |
| 4 | `rollout_config.max_prompt_length` | RolloutWorker（推理集群） |

训练集群按 ①② 处理 prompt，推理集群按 ③④ 处理 prompt。如果 ①≠④，训练阶段生成的 trajectory 和推理阶段使用的 prompt 长度不同。

#### C. n_samples_per_prompt（3 处）

| # | YAML 位置 | 消费方 |
|---|----------|--------|
| 1 | `verl_conf.actor_rollout_ref.rollout.n` | 训练集群知道每条 prompt 生成几条 rollout |
| 2 | `verl_conf.extras.n_samples_per_prompt` | TrainController 传给 rollout 集群 |
| 3 | `rollout_config.n_samples_per_prompt` | RolloutWorker 实际生成 |

如果 ①≠③：训练期望收到 N 条轨迹，但 rollout 实际生成 M 条——维度不匹配。

#### D. 推理并行度配置（训练 vs 推理两套 TP/PP/DP）

One-Step-Off 模式下，训练和推理在**两组独立 GPU 集群**上运行，各自的 TP/PP/DP 配置分布在多个 config section 中：

**训练集群侧（verl_conf）：**

| 配置位置 | 含义 |
|---------|------|
| `verl_conf.actor_rollout_ref.actor.strategy` | fsdp 或 megatron |
| `actor.megatron.tensor_model_parallel_size` | 训练 actor TP |
| `actor.megatron.pipeline_model_parallel_size` | 训练 actor PP |
| `actor.megatron.context_parallel_size` | 训练 CP |
| `actor.megatron.expert_model_parallel_size` | 训练 EP（MoE 模型） |
| `verl_conf.actor_rollout_ref.ref.megatron.tensor_model_parallel_size` | 参考模型 TP（可与 actor 不同） |
| `verl_conf.trainer.n_gpus_per_node` / `nnodes` | 训练总 GPU 数 |

**推理集群侧（rollout_config + infer_instances）：**

| 配置位置 | 含义 | 消费方 |
|---------|------|--------|
| `rollout_config.infer_tensor_parallel_size` | 推理集群上 vLLM 的 TP | `RolloutController` 用于权重切分 |
| `rollout_config.train_tensor_parallel_size` | 训练侧的 TP（用于 RolloutController 加载训练权重） | `RolloutController` 加载权重 |
| `rollout_config.infer_expert_parallel_size` | 推理集群 EP（MoE 模型） | `RolloutController` 权重加载 |
| `infer_instances[].engine_kwargs.tensor_parallel_size` | vLLM engine 实际启动的 TP | `InferExecutor` → vLLM 启动 |
| `infer_instances[].engine_kwargs.data_parallel_size` | vLLM engine DP | `InferExecutor`（PD 分离模式） |
| `infer_instances[].engine_kwargs.enable_expert_parallel` | vLLM engine EP | `InferExecutor`（MoE 模型） |

**关键约束：**

```
约束 1: rollout_config.infer_tensor_parallel_size == infer_instances[].engine_kwargs.tensor_parallel_size
        ↓
        RolloutController 按 TP_size_A 切分权重，但 vLLM engine 按 TP_size_B 启动
        → 张量形状不匹配 → 推理崩溃

约束 2: rollout_config.train_tensor_parallel_size == actor.megatron.tensor_model_parallel_size
        ↓
        RolloutController 从训练集群同步权重时，需要知道训练侧 TP 来正确合并切分
        → TP 不匹配 → 权重加载错误 → 精度不对齐或崩溃

约束 3: infer_instances[].engine_kwargs.data_parallel_size
        ↓
        DP 仅在 infer_instances 中配置，rollout_config 中无对应字段
        → 无交叉校验 → 可能遗漏
```

**如果开启 PD 分离模式（`pd_mode: true`）：**

PD 模式下推理被拆分为 Prefill（编码）和 Decode（生成）两个独立服务，配置进一步分裂：

```
rollout_config / msrl_conf.generate_config:
  prefill_gpu_memory_utilization    ← Prefill 专用
  prefill_max_num_seqs              ← Prefill 专用
  prefill_max_num_batched_tokens    ← Prefill 专用
  prefill_max_model_len             ← Prefill 专用

infer_pd_instances:                 ← 替代 infer_instances
  - role: prefill                   ← Prefill 实例
  - role: decode                    ← Decode 实例
```

PD 模式下，Prefill 和 Decode 节点的 GPU 分配通过外部 `ranktable` 文件控制，配置文件中只声明角色和 GPU 数量，**并行度参数仍由 infer_instances/infer_pd_instances 的 engine_kwargs 控制**。

**配置分裂的核心问题：**

同一个推理集群的并行度参数（TP / PP / DP / EP）被拆分在 3 个不同的 YAML section 中（`rollout_config`、`infer_instances`、`verl_conf`），各自被不同的代码模块消费，但**没有任何跨 section 的一致性校验**。当一个值被修改时，其他位置需要手动保持同步。

---

## 三、config_generator.py 的映射缺失

### 3.1 RemoteTrainConfig 字段映射状态

| RemoteTrainConfig 字段 | YAML 目标 | 映射状态 |
|----------------------|-----------|:------:|
| `model_name_or_path` | `verl_conf.model.path` + `agent.tokenizer` | ✅ |
| `tokenizer_name_or_path` | `agent.tokenizer`（fallback 到 model） | ✅ |
| `train_data_path` | `verl_conf.data.train_files` | ✅ |
| `val_data_path` | `verl_conf.data.val_files` | ✅ |
| `num_epochs` | `verl_conf.trainer.total_epochs` | ✅ |
| `train_batch_size` | `verl_conf.data.train_batch_size` | ✅ |
| `ppo_mini_batch_size` | `verl_conf.actor.ppo_mini_batch_size` | ✅ |
| `lr` | `verl_conf.actor.optim.lr` | ✅ |
| `kl_coef` | `verl_conf.algorithm.kl_ctrl.kl_coef` + `actor.kl_loss_coef` | ✅ |
| `temperature` | **无** | ❌ 死字段 |
| `top_p` | **无** | ❌ 死字段 |
| `agent_temperature` | `agent.infer_service_params.temperature` | ✅ |
| `agent_top_p` | `agent.infer_service_params.top_p` | ✅ |
| `entropy_coeff` | `verl_conf.actor.entropy_coeff` | ✅ |
| `cluster_mode` | `train_instances.cluster_mode` | ✅ |
| `cluster_mode=one_step_off` | `rollout_config` 块 | ❌ 完全未生成 |
| `cluster_mode=one_step_off` | `defaults` 切换为 async 模板 | ❌ 未支持 |
| `cluster_mode=one_step_off` | `verl_conf.extras.*` 异步字段 | ❌ 未生成 |

### 3.2 硬编码派生

```python
# config_generator.py:292
"max_model_len": config.max_response_length * 2,
```

这个 2x 关系没有在任何地方被文档化或校验。如果 `max_response_length=2048` → `max_model_len=4096`，但 Agent 的 `max_prompt_length=2048` + `max_tokens=4096` = 6144 > 4096，溢出但没有报错。

---

## 四、跨 section 一致性约束（当前全部无校验）

### 4.1 必须相等

| 约束 | 违反后果 |
|------|---------|
| `verl_conf.model.path` == `agent.tokenizer` | 训推 token ID 映射不一致 |
| `verl_conf.data.max_prompt_length` == `rollout.prompt_length` == `agent.max_prompt_length` | prompt 截断长度不同 |
| `verl_conf.data.max_response_length` == `rollout.response_length` | response 长度不同 |
| `rollout.tensor_model_parallel_size` == `infer.tensor_parallel_size`（hybrid） | 同一 vllm engine 配置冲突 |
| `rollout_config.infer_tensor_parallel_size` == `infer_instances[].engine_kwargs.tensor_parallel_size`（one_step_off） | RolloutController 和 vLLM 使用的 TP 不一致 → 权重切分崩溃 |
| `rollout_config.train_tensor_parallel_size` == `actor.megatron.tensor_model_parallel_size`（one_step_off） | RolloutController 加载训练权重的 TP 与训练侧不一致 → 精度不对齐 |
| `rollout_config.tokenizer_name_or_path` == `agent.tokenizer`（one_step_off） | 推理集群内部 tokenizer 不一致 |
| `rollout_config.n_samples_per_prompt` == `rollout.n`（one_step_off） | 训练/推理采样数不一致 |

### 4.2 必须满足不等关系

| 约束 | 违反后果 |
|------|---------|
| `ppo_micro_batch_size` ≤ `ppo_mini_batch_size` ≤ `train_batch_size` | 训练启动报错 |
| `infer_service_params.max_tokens` + `agent.max_prompt_length` ≤ `agent.max_model_len` | context window 溢出 |
| `agent.infer_service_params.temperature` == `rollout.temperature`（若 `do_sample=true`） | 训推采样温度不一致 |

---

## Remote 服务器实地勘察

对 `<server-ip>` 服务器上 `agent5.0.0_qjy` 容器的实地分析。

### 0.1 容器环境概况

```
容器: agent5.0.0_qjy (Up 2 days)
AgentSDK 路径: /home/qjy/code/AgentSDK/master/AgentSDK/
Python 包名: agentic_rl（非 aura.aura）
模型位置: /home/qjy/models/Qwen2.5-7B-Instruct
数据位置: /home/qjy/data/processed_math_data/train_200.parquet 等
```

### 0.2 AgenticRL_Binary_Files 不存在

AgentSDK 参考配置中大量使用了 `${hydra:runtime.cwd}/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct` 这样的路径，但这个目录在服务器上**完全不存在**：

```
远程容器:
  find / -name "AgenticRL_Binary_Files" -type d  → 无结果
  
实际模型:
  /home/qjy/models/Qwen2.5-7B-Instruct
  
AgentSDK 参考配置中的路径（来自另一个部署环境）:
  /opt/DPC/models/l00619320/code/AGENTIC_RL_WS/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct
  ${hydra:runtime.cwd}/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct
```

`AgenticRL_Binary_Files` 是**另一个部署环境的约定**（DPC/ModelArts 平台），在那个环境中 `hydra:runtime.cwd` 指向 AgentSDK 目录，其下有 `AgenticRL_Binary_Files/models/` 和 `AgenticRL_Binary_Files/dataset/`。这个约定在当前服务器上不适用。

### 0.3 模型路径的三种形态

同一个模型，在配置链中有三种不同的表达：

| 形态 | 来源 | 示例 |
|------|------|------|
| Hydra 变量引用 | AgentSDK 参考配置 | `${hydra:runtime.cwd}/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct` |
| 另一环境的绝对路径 | AgentSDK 参考配置 | `/opt/DPC/models/l00619320/code/AGENTIC_RL_WS/AgenticRL_Binary_Files/models/Qwen2.5-7B-Instruct` |
| 当前服务器绝对路径 | RemoteTrainConfig | `/home/qjy/models/Qwen2.5-7B-Instruct` |

训练能运行是因为 `config_generator.py` 的 overrides 用绝对路径覆盖了 Hydra 变量。但如果有任何一个位置漏掉了覆盖，Hydra 会在运行时解析 `${hydra:runtime.cwd}` → `/home/qjy/code/AgentSDK/master/AgentSDK/`，然后拼接 `AgenticRL_Binary_Files/models/...` → **指向不存在的路径，静默失败或报错**。

### 0.4 当前生成的 YAML 中残留的 Hydra 变量

检查远程已部署的 `remote_1778504118.yaml`，仍有一个未覆盖的 Hydra 变量：

```yaml
verl_conf:
  extras:
    traj_output_path: ${hydra:runtime.cwd}/outputs   ← 未覆盖，Hydra 运行时解析
```

这个路径可以正常解析（`outputs/` 目录存在），但其他参考配置中的 Hydra 变量如果未被完整覆盖就会出问题。

### 0.5 Hydra 配置搜索路径

`run_start_in_local.sh` 的 PYTHONPATH 设置：

```bash
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/third_party/agent_engine/rllm/:${SCRIPT_DIR}/third_party/rl/mindspeed_rl/:$PYTHONPATH"
```

YAML 中的 Hydra searchpath：
```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config    ← 系统级 veRL 安装路径
    - file://AgenticRL/configs/verl_conf   ← 相对于 AgentSDK configs/ 目录
```

第二个 searchpath `file://AgenticRL/configs/verl_conf` 是相对路径，相对位置由 Hydra 的 `--config-path` 决定（即 `configs/` 目录）。实际上它指向的是 `configs/verl_conf/` 目录下的 `ppo_trainer.yaml` 或 `fully_async_ppo_trainer.yaml`。

---

## 五、总结：问题分级

```
P0 - 训练无法启动：
  - one_step_off 模式 rollout_config 块缺失
  - one_step_off 模式异步模板未切换

P1 - 静默精度不对齐（one_step_off）：
  - tokenizer 路径在 3 处独立配置，无一致性校验
  - max_prompt_length 在 4 处独立配置，无一致性校验
  - n_samples_per_prompt 在 3 处独立配置
  - 推理并行度（TP/PP/DP/EP）在 rollout_config / infer_instances / verl_conf
    三处独立配置，RolloutController 和 vLLM engine 各读各的，无交叉校验

P2 - 参数不一致但可能被容忍：
  - temperature/top_p 在训练和 Agent 侧独立配置
  - max_model_len 硬编码派生（response_length*2）无校验

P3 - 跨环境路径污染：
  - AgenticRL_Binary_Files 是 DPC 环境约定，当前服务器上不存在
  - BASE_CONFIG 中残留未覆盖的 Hydra 变量（extras.traj_output_path）
  - 若以后新增包含 ${hydra:runtime.cwd}/AgenticRL_Binary_Files 的字段
    而忘记在 overrides 中覆盖，会导致运行时路径解析失败

P4 - RemoteTrainConfig 冗余字段：
  - temperature/top_p 字段定义了但 config_generator 没用到
```

### 关键发现：模型路径的三种形态

```
┌──────────────────────────────────────────────────────────────┐
│ AgentSDK 参考配置 (DPC 环境)                                  │
│   ${hydra:runtime.cwd}/AgenticRL_Binary_Files/models/...    │
│   /opt/DPC/models/l00619320/.../AgenticRL_Binary_Files/...  │
│                                                              │
│ ↓ 不适用当前服务器（AgenticRL_Binary_Files 不存在）           │
│                                                              │
│ config_generator.py BASE_CONFIG                              │
│   /home/qjy/models/Qwen2.5-7B-Instruct (硬编码)              │
│   agent_loop_manager: aura.aura.xxx (包名错误!)              │
│                                                              │
│ ↓ overrides 覆盖为 RemoteTrainConfig 的值                    │
│                                                              │
│ 最终生成的 YAML                                               │
│   model.path: /home/qjy/models/Qwen2.5-7B-Instruct ✅        │
│   agent.tokenizer: /home/qjy/models/Qwen2.5-7B-Instruct ✅   │
│   agent_loop_manager: agentic_rl.xxx ✅ (怎么修正的?)        │
│   traj_output_path: ${hydra:runtime.cwd}/outputs ← 未覆盖    │
└──────────────────────────────────────────────────────────────┘
```

### 新增：模型层配置风险

```
P_model_A - 覆盖不完整：
  - generation_config.json 中的 top_k、repetition_penalty 等参数
    在我们的配置体系中无对应字段，静默继承模型默认值
  - 换模型 → 默认值变化 → 训练行为无感知改变，结果不可复现

P_model_B - 硬限制不校验：
  - config.json 的 max_position_embeddings 是模型硬上限
  - 我们的 max_model_len / max_prompt_length / max_response_length
    超出该上限时生成阶段不报错，运行时静默截断或崩溃

P_model_C - 派生参数不透明：
  - max_model_len = max_response_length * 2 硬编码
  - 假设 prompt:response=1:1，Agent 多步场景不成立
  - 系数无文档、无配置暴露、超限无校验
```

### 根本原因

1. `config_generator.py` 的 `BASE_CONFIG` 是从一个**与目标部署环境不同的 AgentSDK 配置**粘贴而来，再通过 overrides 打补丁。
2. 模型自带的 `generation_config.json` 是一层**隐式配置源**，大多数参数在我们的配置体系中无对应字段，静默继承模型默认值。

1. BASE_CONFIG 中残留了源环境的路径约定（`AgenticRL_Binary_Files`）
2. BASE_CONFIG 中残留了源环境的包名约定（`aura.aura` vs `agentic_rl`）
3. overrides 是增量覆盖，未覆盖的字段静默保留 BASE_CONFIG 原值
4. 没有人/机制验证"overrides 是否覆盖了所有应当覆盖的字段"
