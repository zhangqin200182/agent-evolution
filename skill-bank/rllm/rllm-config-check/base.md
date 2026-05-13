---
name: rllm-config-check
description: Validate remote NPU training config consistency across model defaults, RemoteTrainConfig, and generated AgentSDK YAML. Checks hybrid and one_step_off modes.
metadata:
  version: "1.0.0"
  categories:
    - machine-learning
    - agent-training
    - configuration-validation
---

# rllm-config-check — 远程训练配置验证

<!-- section:intro -->
在启动远程 NPU 训练前，验证配置在三层间的一致性：

- **Layer 1** (远程): 模型目录 `generation_config.json` / `config.json` 的硬限制和默认值
- **Layer 2** (本地): `RemoteTrainConfig → config_generator.py → AgentSDK YAML` 映射完整性
- **Layer 3** (本地+远程): 生成 YAML 跨 section 参数一致性

支持 **hybrid**（共卡）和 **one_step_off**（训推分离）两种模式，校验规则各自独立。

## 前置条件

- 远程训练配置已生成: `rllm_remote/output/runs/<run_id>/config.json`
- Phase 2 (远程校验) 需要 SSH 密码
- 工作目录: `/Users/kevin/code/MyProject`
<!-- /section:intro -->

<!-- section:phase1 -->
## Phase 1：本地校验（无需 SSH）

```bash
cd /Users/kevin/code/MyProject && python -m rllm_remote.config_validator \
  rllm_remote/output/runs/<run_id>/config.json \
  --phase phase1
```

检查项：

| 严重性 | 代码 | 说明 |
|--------|------|------|
| ERROR | DEAD_FIELD_TEMPERATURE / DEAD_FIELD_TOP_P | temperature/top_p 字段未映射到 YAML |
| BLOCK | MODEL_PATH_TOKENIZER_MISMATCH | 训练模型路径 ≠ Agent tokenizer 路径 |
| BLOCK | MAX_PROMPT_LENGTH_INCONSISTENT | data/rollout/agent 的 max_prompt_length 不一致 |
| BLOCK | MAX_RESPONSE_LENGTH_INCONSISTENT | data/rollout 的 max_response_length 不一致 |
| ERROR | BATCH_SIZE_HIERARCHY_VIOLATED | micro > mini 或 mini > train_batch |
| BLOCK | CONTEXT_WINDOW_OVERFLOW | max_tokens + prompt > max_model_len |
| BLOCK | TP_EXCEEDS_NPUS | tensor_parallel 超过可用 NPU 数 |
| BLOCK | UNRESOLVED_HYDRA_VARIABLE | 生成的 YAML 包含未覆盖的 ${hydra:...} |
| WARN | AGENTICRL_BINARY_FILES_REF | 引用了 DPC 环境专属路径 |
| BLOCK | HYBRID_TP_INCONSISTENCY | hybrid 模式下 rollout TP ≠ infer TP |

one_step_off 模式额外检查：

| 严重性 | 代码 | 说明 |
|--------|------|------|
| ERROR | ROLLOUT_CONFIG_MISSING | 缺少 rollout_config 块 |
| ERROR | DEFAULTS_TEMPLATE_MISMATCH | 使用同步模板而非异步模板 |
| ERROR | ONE_STEP_OFF_EXTRAS_MISSING | verl_conf.extras 缺少异步必需字段 |
| BLOCK | ROLLOUT_INFER_TP_MISMATCH | rollout_config.infer_tp ≠ infer_instances TP |
| BLOCK | TRAIN_TP_MISMATCH | rollout_config.train_tp ≠ actor TP |
| BLOCK | N_SAMPLES_INCONSISTENT | n_samples 在 rollout_config 和 verl_conf 不一致 |
<!-- /section:phase1 -->

<!-- section:phase2 -->
## Phase 2：远程校验（需 SSH 密码）

需要在 Phase 1 基础上，SSH 到服务器读取模型目录下的配置文件。

```bash
cd /Users/kevin/code/MyProject && python -m rllm_remote.config_validator \
  rllm_remote/output/runs/<run_id>/config.json \
  --ssh-password "<your-password>" \
  --phase all
```

新增检查项：

| 严重性 | 代码 | 说明 |
|--------|------|------|
| ERROR | SSH_CONNECTIVITY / CONTAINER_NOT_RUNNING | 无法连接服务器或容器 |
| ERROR | REMOTE_MODEL_CONFIG_MISSING | 模型目录中不存在 config.json |
| BLOCK | MAX_POSITION_EMBEDDINGS_EXCEEDED | prompt+response 超过模型最大位置编码 |
| BLOCK | MAX_MODEL_LEN_EXCEEDS_LIMIT | 派生的 max_model_len 超过模型上限 |
| WARN | UNCOVERED_GENERATION_PARAMS | 模型 generation_config 中未覆盖的参数 |
| WARN | TOKENIZER_MAX_LENGTH_EXCEEDED | prompt+response 超 tokenizer model_max_length |
<!-- /section:phase2 -->

<!-- section:report -->
## 报告解读

退出码: 有 ERROR 或 BLOCK 时退出 1；仅 WARN 时退出 0。

### 严重性级别

- **ERROR** — 会导致训练崩溃 → 必须修复
- **BLOCK** — 会导致静默精度不对齐 → 必须修复
- **WARN** — 未覆盖参数或潜在问题 → 检查后决定是否处理

### 输出格式

```bash
# 文本格式（默认）
python -m rllm_remote.config_validator config.json --format text

# JSON 格式
python -m rllm_remote.config_validator config.json --format json

# 保存 JSON 报告
python -m rllm_remote.config_validator config.json --output report.json
```
<!-- /section:report -->

<!-- section:error-handling -->
## 错误处理

| 场景 | 行为 |
|------|------|
| 配置文件路径不存在 | Python 异常，退出 1 |
| SSH 连接失败 | 报告 ERROR，展示 Phase 1 结果 |
| 远程模型路径不存在 | 报告 ERROR，跳过 Layer 1 检查 |
| 远程脚本执行失败 | 报告 ERROR，展示 Phase 1 结果 |
<!-- /section:error-handling -->
