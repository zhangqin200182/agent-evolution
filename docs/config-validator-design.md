# rllm-config-check — 远程训练配置验证 Skill 设计文档

## Context

远程 NPU 训练中，配置参数在三层中存在重叠和不一致（详见 `docs/config-consistency-analysis.md`）：

1. **Layer 1** — 模型目录 `generation_config.json` / `config.json`：vLLM 底层默认值，部分参数未被覆盖
2. **Layer 2** — `RemoteTrainConfig → config_generator.py → AgentSDK YAML`：映射不完整（死字段、缺失块）
3. **Layer 3** — 生成 YAML 内部跨 section 一致性：同参数多写，无校验

Hybrid 和 One-Step-Off 两种模式的配置结构差异大，需分开校验。

## 目标

创建一个独立 skill `rllm-config-check`，在执行 `rllm-remote-run` 前校验配置，阻断会导致崩溃或精度不对齐的问题。

## 设计决策

- **独立 skill**：便于调测和演进，不嵌入 rllm-remote-run 流程
- **复用 RemoteExecutor / RemoteTrainConfig**：与 rllm-remote-run 共享同一 `config.json` 输入
- **两阶段**：Phase 1（本地，无需 SSH）→ Phase 2（远程，SSH 读模型配置）
- **三档严重性**：ERROR（崩溃）→ 阻断 / BLOCK（精度风险）→ 阻断 / WARN（未覆盖参数）→ 告警但不阻断
- **Hybrid 和 One-Step-Off 分开**：各自独立的 validator 列表，避免混淆

## Implementation Plan

### Step 1: 创建 `rllm_remote/config_validator.py`

数据结构：

- `Severity` enum: ERROR, BLOCK, WARN, INFO
- `Finding` dataclass: severity, code, message, detail, paths, values, recommendation
- `Report` dataclass: findings[], passed_checks[], config_summary, block property, format_text(), to_json()

Phase 1 验证器（本地，接收 `config: RemoteTrainConfig` + `yaml_dict: dict`）：

**所有模式通用：**

| # | 函数 | 严重性 | 检查内容 |
|---|------|--------|---------|
| 1 | `check_dead_fields` | ERROR | `config.temperature`/`config.top_p` 非默认值时提示死字段 |
| 2 | `check_max_model_len_derivation` | BLOCK | `max_response_length*2` 是否足够容纳 `max_prompt_length + max_response_length` |
| 3 | `check_model_path_consistency` | BLOCK | `verl_conf.model.path` == `agent.tokenizer` |
| 4 | `check_max_prompt_length_consistency` | BLOCK | data / rollout / agent 三处 max_prompt_length 一致 |
| 5 | `check_max_response_length_consistency` | BLOCK | data / rollout 的 max_response_length 一致 |
| 6 | `check_batch_size_hierarchy` | ERROR | micro ≤ mini ≤ train_batch |
| 7 | `check_context_window_overflow` | BLOCK | `agent_max_tokens + max_prompt_length ≤ max_model_len` |
| 8 | `check_tp_vs_npus` | BLOCK | TP ≤ n_npus_per_node（TP * nnodes ≤ 总 NPU） |
| 9 | `check_hydra_variables` | BLOCK | 生成 YAML 中残留未覆盖的 `${hydra:...}` 变量 |
| 10 | `check_agentsdk_binary_files_ref` | WARN | YAML 中引用 `AgenticRL_Binary_Files` 的路径 |
| 11 | `check_rollout_temperature_gap` | WARN | agent temperature ≠ rollout temperature（若 do_sample=true 升级 BLOCK） |

**Hybrid 模式专属：**

| # | 函数 | 严重性 | 检查内容 |
|---|------|--------|---------|
| 12 | `check_hybrid_tp_consistency` | BLOCK | `rollout.tensor_model_parallel_size` == `infer_instances.tensor_parallel_size` |

**One-Step-Off 模式专属：**

| # | 函数 | 严重性 | 检查内容 |
|---|------|--------|---------|
| 13 | `check_rollout_config_exists` | ERROR | `rollout_config` 块存在且包含必要字段 |
| 14 | `check_async_template` | ERROR | `defaults` 包含 `fully_async_ppo_trainer` 而非 `ppo_trainer` |
| 15 | `check_async_extras` | ERROR | `verl_conf.extras` 包含异步所需字段 |
| 16 | `check_rollout_tp_consistency` | BLOCK | `rollout_config.infer_tp` == `infer_instances.tensor_parallel_size` |
| 17 | `check_train_tp_consistency` | BLOCK | `rollout_config.train_tp` == actor 训练 TP |
| 18 | `check_n_samples_consistency` | BLOCK | `rollout_config.n_samples_per_prompt` 与 `rollout.n` 一致 |

Phase 2 验证器（远程，接收额外的 `executor: RemoteExecutor` + `model_configs: dict`）：

| # | 函数 | 严重性 | 检查内容 |
|---|------|--------|---------|
| 19 | `check_remote_model_exists` | ERROR | 模型目录和 config.json 在远程存在 |
| 20 | `check_max_position_embeddings` | BLOCK | 总上下文 ≤ 模型 `max_position_embeddings`；派生的 `max_model_len` ≤ 上限 |
| 21 | `check_generation_config_coverage` | WARN | 模型 generation_config 中未被覆盖的参数（单条 Finding 聚合所有） |
| 22 | `check_tokenizer_max_length` | WARN | `max_prompt + max_response` ≤ tokenizer `model_max_length` |

编排函数：

- `validate_phase1(config, yaml_dict) -> Report`
- `validate_phase2(config, report, executor) -> Report`
- `validate_config(config_path, ssh_password, phase) -> Report` — 顶层入口

CLI 入口（`if __name__ == "__main__": main()`）：
- 参数：`config_json`、`--ssh-password`、`--phase {phase1,phase2,all}`、`--format {text,json}`、`--output`
- 退出码：有 ERROR 或 BLOCK 时 exit(1)

### Step 2: 创建 skill `skill-bank/rllm/rllm-config-check/`

- `base.md`：frontmatter + intro / phase1 / phase2 / report-interpretation / error-handling sections
- `manifest.yaml`：`base: base.md`，空 active/disabled

### Step 3: 注册到 `skill-bank/bank.yaml`

在 `groups.rllm.skills` 下添加：
```yaml
      rllm-config-check:
        output: .claude/skills/rllm-config-check/SKILL.md
```

### Step 4: 编译

```bash
python skill-bank/compile.py rllm-config-check
```

## Files to Create

| File | Purpose |
|------|---------|
| `rllm_remote/config_validator.py` | All validation logic + CLI |
| `skill-bank/rllm/rllm-config-check/base.md` | Skill instructions |
| `skill-bank/rllm/rllm-config-check/manifest.yaml` | Skill manifest |
| `skill-bank/rllm/rllm-config-check/patches/` | Empty dir for future patches |

## Files to Modify

| File | Change |
|------|--------|
| `skill-bank/bank.yaml` | Add `rllm-config-check` entry under `rllm` group |

## Verification

1. 用现有的 `rllm_remote/output/runs/remote_1778504118/config.json`（hybrid）运行 `--phase phase1`，验证至少发现：DEAD_FIELD_TEMPERATURE、UNRESOLVED_HYDRA_VARIABLE、TEMPERATURE_INCONSISTENCY
2. 用相同配置运行 `--phase phase2 --ssh-password ...`，验证远程读取模型配置并发现 UNCOVERED_GENERATION_PARAMS
3. 构造一个 one_step_off 配置（cluster_mode='one_step_off'），验证发现 ROLLOUT_CONFIG_MISSING、DEFAULTS_TEMPLATE_MISMATCH
4. 验证 `--format json --output /tmp/report.json` 正确输出
5. 验证存在 BLOCK/ERROR 时 exit code = 1，全 PASS 时 exit code = 0
