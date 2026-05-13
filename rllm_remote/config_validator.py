"""
Configuration validator for remote NPU agent RL training.

Validates consistency across three config layers:
  Layer 1: Model directory config files (generation_config.json, config.json)
  Layer 2: RemoteTrainConfig → generated AgentSDK YAML mapping
  Layer 3: Cross-section consistency within the generated YAML

Supports hybrid and one_step_off cluster modes with separate validation rules.

Usage:
    # Phase 1 only (local, no SSH needed)
    python -m rllm_remote.config_validator config.json --phase phase1

    # Phase 2 only (remote, needs SSH password)
    python -m rllm_remote.config_validator config.json --ssh-password xxx --phase phase2

    # Full validation
    python -m rllm_remote.config_validator config.json --ssh-password xxx --phase all
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from rllm_remote.config import RemoteTrainConfig
from rllm_remote.config_generator import BASE_CONFIG, deep_merge, generate_agentsdk_config
from rllm_remote.ssh import RemoteExecutor


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────


class Severity(Enum):
    ERROR = "ERROR"   # Will crash → block
    BLOCK = "BLOCK"   # Silent precision risk → block
    WARN = "WARN"     # Uncovered param → warn
    INFO = "INFO"     # Informational


@dataclass
class Finding:
    severity: Severity
    code: str
    message: str
    detail: str = ""
    layer: int = 0
    paths: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def format_short(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"

    def format_long(self) -> str:
        lines = [self.format_short()]
        if self.detail:
            lines.append(f"  Detail: {self.detail}")
        if self.paths:
            lines.append(f"  Paths: {', '.join(self.paths)}")
        if self.values:
            lines.append(f"  Values: {json.dumps(self.values)}")
        if self.recommendation:
            lines.append(f"  Fix: {self.recommendation}")
        return "\n".join(lines)


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    config_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def block(self) -> bool:
        return any(f.severity in (Severity.ERROR, Severity.BLOCK) for f in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(f.severity == Severity.WARN for f in self.findings)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def pass_check(self, code: str) -> None:
        self.passed_checks.append(code)

    def format_text(self) -> str:
        lines = ["=" * 60, "Configuration Validation Report", "=" * 60]
        if self.config_summary:
            lines.append(f"  Mode:       {self.config_summary.get('cluster_mode', '?')}")
            lines.append(f"  Model:      {self.config_summary.get('model_name_or_path', '?')}")
            lines.append(f"  Batch:      {self.config_summary.get('train_batch_size', '?')}")
            lines.append(f"  Prompt Len: {self.config_summary.get('max_prompt_length', '?')}")
            lines.append(f"  Response:   {self.config_summary.get('max_response_length', '?')}")
            lines.append("")

        sorted_findings = sorted(self.findings, key=lambda f: (
            0 if f.severity == Severity.ERROR else
            1 if f.severity == Severity.BLOCK else
            2 if f.severity == Severity.WARN else 3,
            f.code
        ))

        if sorted_findings:
            lines.append(f"Findings ({len(sorted_findings)}):")
            lines.append("-" * 60)
            for f in sorted_findings:
                lines.append(f.format_long())
        else:
            lines.append("No findings. All checks passed.")

        if self.passed_checks:
            lines.append("")
            lines.append(f"Passed checks ({len(self.passed_checks)}):")
            for code in self.passed_checks:
                lines.append(f"  [PASS] {code}")

        lines.append("")
        if self.block:
            lines.append("VALIDATION FAILED: Blocking issues found. Fix before proceeding.")
        else:
            lines.append("VALIDATION PASSED: No blocking issues.")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        data = {
            "block": self.block,
            "findings": [
                {**asdict(f), "severity": f.severity.value}
                for f in self.findings
            ],
            "passed_checks": self.passed_checks,
            "config_summary": self.config_summary,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_nested(d: dict | list, *keys, default=None):
    """Safely traverse a nested dict, handling list indices."""
    for k in keys:
        if isinstance(d, list) and isinstance(k, int) and 0 <= k < len(d):
            d = d[k]
        elif isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def _find_hydra_refs(obj: Any, path: str = "", results: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    """Recursively find all ${hydra:...} references in a dict/list."""
    if results is None:
        results = []
    if isinstance(obj, str):
        for m in re.finditer(r'\$\{hydra:.*?\}', obj):
            results.append((path, m.group()))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _find_hydra_refs(v, f"{path}.{k}" if path else k, results)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _find_hydra_refs(v, f"{path}[{i}]", results)
    return results


# Hydra variables considered safe. ${hydra:runtime.cwd} resolves to the AgentSDK
# working directory which always exists. Specific subdirectories are checked separately.
KNOWN_SAFE_HYDRA_REFS = {
    "${hydra:runtime.cwd}",
    "${agent_instances.0.name}",
    "${infer_instances.0.name}",
    "${train_instances.0.name}",
}


def _read_model_config_script(model_path: str) -> str:
    """Return Python code to run in remote container to read model config files."""
    import textwrap
    return textwrap.dedent(f"""\
import json, os
base = "{model_path}"
result = {{}}

config_path = os.path.join(base, "config.json")
if os.path.isfile(config_path):
    with open(config_path) as f:
        data = json.load(f)
    result["config"] = {{
        "max_position_embeddings": data.get("max_position_embeddings"),
        "vocab_size": data.get("vocab_size"),
        "torch_dtype": data.get("torch_dtype"),
    }}

gen_path = os.path.join(base, "generation_config.json")
if os.path.isfile(gen_path):
    with open(gen_path) as f:
        data = json.load(f)
    result["generation_config"] = {{
        k: v for k, v in data.items()
        if k in ("temperature", "top_p", "top_k", "repetition_penalty", "do_sample",
                 "min_p", "typical_p", "frequency_penalty", "presence_penalty",
                 "bos_token_id", "eos_token_id", "pad_token_id")
    }}

tok_path = os.path.join(base, "tokenizer_config.json")
if os.path.isfile(tok_path):
    with open(tok_path) as f:
        data = json.load(f)
    result["tokenizer_config"] = {{
        "model_max_length": data.get("model_max_length"),
    }}

print(json.dumps(result))
""")


# ──────────────────────────────────────────────
# Phase 1 validators (local)
# ──────────────────────────────────────────────

ValidatorFn = Callable[[RemoteTrainConfig, dict[str, Any], Report], None]


# ---- All modes ----

def check_dead_fields(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """temperature/top_p fields exist in RemoteTrainConfig but never written to YAML by config_generator."""
    defaults = RemoteTrainConfig()
    if config.temperature != defaults.temperature:
        report.add(Finding(
            severity=Severity.ERROR, code="DEAD_FIELD_TEMPERATURE",
            message=f"RemoteTrainConfig.temperature={config.temperature} is never written to YAML",
            detail="config_generator.py does not map temperature to any YAML field. "
                   "Use agent_temperature for agent inference sampling.",
            paths=["RemoteTrainConfig.temperature"],
            values={"temperature": config.temperature},
            recommendation="Set agent_temperature instead; if rollout sampling is needed, add mapping in config_generator.py",
        ))
    else:
        report.pass_check("DEAD_FIELD_TEMPERATURE_OK")

    if config.top_p != defaults.top_p:
        report.add(Finding(
            severity=Severity.ERROR, code="DEAD_FIELD_TOP_P",
            message=f"RemoteTrainConfig.top_p={config.top_p} is never written to YAML",
            detail="config_generator.py does not map top_p to any YAML field. "
                   "Use agent_top_p for agent inference sampling.",
            paths=["RemoteTrainConfig.top_p"],
            values={"top_p": config.top_p},
            recommendation="Set agent_top_p instead; if rollout sampling is needed, add mapping in config_generator.py",
        ))
    else:
        report.pass_check("DEAD_FIELD_TOP_P_OK")


def check_max_model_len_derivation(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """max_model_len = max_response_length * 2 must accommodate actual context."""
    derived = config.max_response_length * 2
    needed = config.max_prompt_length + config.max_response_length
    if derived < needed:
        report.add(Finding(
            severity=Severity.BLOCK, code="MAX_MODEL_LEN_UNDERFLOW",
            message=f"Derived max_model_len({derived} = response*2) < prompt+response({needed})",
            detail="Agent multi-step reasoning consumes additional tokens beyond prompt+response. "
                   "The *2 formula assumes 1:1 prompt:response ratio which may not hold.",
            paths=["RemoteTrainConfig.max_response_length → agent.max_model_len"],
            values={"derived_max_model_len": derived, "prompt_plus_response": needed},
            recommendation=f"Increase max_response_length to >= {needed // 2} or add explicit max_model_len to RemoteTrainConfig",
        ))
    else:
        report.pass_check("MAX_MODEL_LEN_DERIVATION_OK")


def check_model_path_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """verl_conf.model.path must equal agent.tokenizer for correct token ID mapping."""
    model_path = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "model", "path")
    agent_tok = _get_nested(yaml_dict, "agent_instances", 0, "executor_kwargs", "agent_engine_kwargs", "tokenizer")
    if model_path and agent_tok and model_path != agent_tok:
        report.add(Finding(
            severity=Severity.BLOCK, code="MODEL_PATH_TOKENIZER_MISMATCH",
            message="Training model path != Agent tokenizer path",
            detail="Different paths cause token ID mapping mismatch between training and inference.",
            paths=["verl_conf.actor_rollout_ref.model.path", "agent_instances[0].agent_engine_kwargs.tokenizer"],
            values={"model_path": model_path, "agent_tokenizer": agent_tok},
            recommendation="Ensure both point to the same model directory",
        ))
    else:
        report.pass_check("MODEL_PATH_CONSISTENCY_OK")


def check_max_prompt_length_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """max_prompt_length in data, rollout.prompt_length, and agent must be consistent."""
    data_pl = _get_nested(yaml_dict, "verl_conf", "data", "max_prompt_length")
    rollout_pl = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "rollout", "prompt_length")
    agent_pl = _get_nested(yaml_dict, "agent_instances", 0, "executor_kwargs", "agent_engine_kwargs", "max_prompt_length")
    vals = {"data": data_pl, "rollout": rollout_pl, "agent": agent_pl}
    unique = set(v for v in vals.values() if v is not None)
    if len(unique) > 1:
        report.add(Finding(
            severity=Severity.BLOCK, code="MAX_PROMPT_LENGTH_INCONSISTENT",
            message="max_prompt_length differs across config sections",
            paths=list(vals.keys()),
            values=vals,
            recommendation="Set all to the same value",
        ))
    else:
        report.pass_check("MAX_PROMPT_LENGTH_CONSISTENT")


def check_max_response_length_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """max_response_length in data and rollout must be consistent."""
    data_rl = _get_nested(yaml_dict, "verl_conf", "data", "max_response_length")
    rollout_rl = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "rollout", "response_length")
    if data_rl and rollout_rl and data_rl != rollout_rl:
        report.add(Finding(
            severity=Severity.BLOCK, code="MAX_RESPONSE_LENGTH_INCONSISTENT",
            message=f"data.max_response_length({data_rl}) != rollout.response_length({rollout_rl})",
            paths=["verl_conf.data.max_response_length", "verl_conf.rollout.response_length"],
            values={"data": data_rl, "rollout": rollout_rl},
            recommendation="Set both to the same value",
        ))
    else:
        report.pass_check("MAX_RESPONSE_LENGTH_CONSISTENT")


def check_batch_size_hierarchy(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """micro_batch ≤ mini_batch ≤ train_batch must hold."""
    micro = config.ppo_micro_batch_size_per_gpu
    mini = config.ppo_mini_batch_size
    train = config.train_batch_size
    if not (micro <= mini <= train):
        report.add(Finding(
            severity=Severity.ERROR, code="BATCH_SIZE_HIERARCHY_VIOLATED",
            message=f"micro({micro}) ≤ mini({mini}) ≤ train({train}) violated",
            paths=["ppo_micro_batch_size_per_gpu", "ppo_mini_batch_size", "train_batch_size"],
            values={"micro": micro, "mini": mini, "train": train},
            recommendation="Adjust batch sizes to satisfy micro ≤ mini ≤ train",
        ))
    else:
        report.pass_check("BATCH_SIZE_HIERARCHY_OK")


def check_context_window_overflow(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """agent_max_tokens + max_prompt_length must not exceed max_model_len."""
    max_tokens = config.agent_max_tokens
    max_prompt = config.max_prompt_length
    max_model = _get_nested(yaml_dict, "agent_instances", 0, "executor_kwargs", "agent_engine_kwargs", "max_model_len")
    if max_model and (max_tokens + max_prompt > max_model):
        report.add(Finding(
            severity=Severity.BLOCK, code="CONTEXT_WINDOW_OVERFLOW",
            message=f"agent_max_tokens({max_tokens}) + max_prompt_length({max_prompt}) = {max_tokens + max_prompt} > max_model_len({max_model})",
            paths=["agent_max_tokens", "max_prompt_length", "agent.max_model_len"],
            values={"max_tokens": max_tokens, "max_prompt": max_prompt, "max_model_len": max_model},
            recommendation="Increase max_model_len or reduce max_tokens/max_prompt_length",
        ))
    else:
        report.pass_check("CONTEXT_WINDOW_OK")


def check_tp_vs_npus(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """TP must fit within available NPUs."""
    tp = config.tensor_parallel_size
    total_npus = config.n_npus_per_node * config.nnodes
    if tp > total_npus:
        report.add(Finding(
            severity=Severity.ERROR, code="TP_EXCEEDS_TOTAL_NPUS",
            message=f"tensor_parallel_size({tp}) > total NPUs({total_npus})",
            paths=["tensor_parallel_size", "n_npus_per_node", "nnodes"],
            values={"tp": tp, "total_npus": total_npus},
            recommendation="Reduce tensor_parallel_size or increase n_npus_per_node/nnodes",
        ))
    elif tp > config.n_npus_per_node:
        report.add(Finding(
            severity=Severity.BLOCK, code="TP_EXCEEDS_PER_NODE",
            message=f"tensor_parallel_size({tp}) > n_npus_per_node({config.n_npus_per_node})",
            paths=["tensor_parallel_size", "n_npus_per_node"],
            values={"tp": tp, "npus_per_node": config.n_npus_per_node},
            recommendation="Reduce tensor_parallel_size or increase n_npus_per_node",
        ))
    else:
        report.pass_check("TP_VS_NPUS_OK")


def check_hydra_variables(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """Generated YAML should not contain unresolved ${hydra:...} references."""
    refs = _find_hydra_refs(yaml_dict)
    unsafe = [(path, ref) for path, ref in refs if ref not in KNOWN_SAFE_HYDRA_REFS]
    if unsafe:
        detail_lines = [f"  {p}: {r}" for p, r in unsafe]
        report.add(Finding(
            severity=Severity.BLOCK, code="UNRESOLVED_HYDRA_VARIABLE",
            message=f"{len(unsafe)} Hydra variable(s) in generated YAML not explicitly overridden",
            detail="These variables resolve at runtime. If the referenced paths don't exist "
                   "on the target server (e.g. AgenticRL_Binary_Files), training will fail silently.\n"
                   + "\n".join(detail_lines),
            paths=[p for p, _ in unsafe],
            recommendation="Add overrides in config_generator.py to replace these with absolute paths",
        ))
    else:
        report.pass_check("HYDRA_VARIABLES_OK")


def check_agentsdk_binary_files_ref(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """Warn if AgenticRL_Binary_Files paths appear (DPC convention, doesn't exist on this server)."""
    refs = _find_hydra_refs(yaml_dict)
    binary_refs = [(p, r) for p, r in refs if "AgenticRL_Binary_Files" in r]
    # Also check plain strings
    for path, ref in _find_hydra_refs(yaml_dict):
        if "AgenticRL_Binary_Files" in str(ref):
            binary_refs.append((path, ref))
    if binary_refs:
        report.add(Finding(
            severity=Severity.WARN, code="AGENTICRL_BINARY_FILES_REF",
            message=f"{len(binary_refs)} reference(s) to AgenticRL_Binary_Files (DPC-specific path)",
            detail="AgenticRL_Binary_Files is a DPC deployment convention that does not exist on this server. "
                   "These paths will resolve to non-existent directories at runtime.",
            paths=list(set(p for p, _ in binary_refs)),
            recommendation="Replace with absolute paths matching the server's model/data layout",
        ))
    else:
        report.pass_check("BINARY_FILES_REF_OK")


def check_rollout_temperature_gap(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """Agent temperature vs rollout temperature consistency."""
    agent_temp = _get_nested(yaml_dict, "agent_instances", 0, "executor_kwargs", "infer_service_params", "temperature")
    rollout_do_sample = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "rollout", "do_sample")
    # Only flag if do_sample is true (sampling mode), since temperature doesn't apply with greedy decoding
    if rollout_do_sample and agent_temp is not None:
        report.add(Finding(
            severity=Severity.BLOCK, code="ROLLOUT_TEMP_UNSET_WITH_SAMPLING",
            message=f"rollout.do_sample=true but rollout.temperature is not set; agent temperature={agent_temp}",
            detail="With do_sample=true, rollout will use model default temperature from generation_config.json, "
                   "which may differ from the agent temperature.",
            paths=["rollout.do_sample", "agent.infer_service_params.temperature"],
            values={"agent_temperature": agent_temp, "rollout_do_sample": rollout_do_sample},
            recommendation="Explicitly set rollout.temperature in config_generator overrides",
        ))
    else:
        report.pass_check("ROLLOUT_TEMPERATURE_OK")


# ---- Hybrid mode ----

def check_hybrid_tp_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """In hybrid mode, rollout TP and infer engine TP must match (same vllm instance)."""
    if config.cluster_mode != "hybrid":
        return
    rollout_tp = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "rollout", "tensor_model_parallel_size")
    infer_tp = _get_nested(yaml_dict, "infer_instances", 0, "executor_kwargs", "engine_kwargs", "tensor_parallel_size")
    if rollout_tp and infer_tp and rollout_tp != infer_tp:
        report.add(Finding(
            severity=Severity.BLOCK, code="HYBRID_TP_INCONSISTENCY",
            message=f"rollout TP({rollout_tp}) != infer TP({infer_tp}) in hybrid mode (same vllm engine)",
            paths=["verl_conf.rollout.tensor_model_parallel_size", "infer_instances[0].tensor_parallel_size"],
            values={"rollout_tp": rollout_tp, "infer_tp": infer_tp},
            recommendation="Set both to the same value",
        ))
    else:
        report.pass_check("HYBRID_TP_CONSISTENT")


# ---- One-Step-Off mode ----

ONE_STEP_OFF_ROLLOUT_REQUIRED = [
    "tokenizer_name_or_path", "n_samples_per_prompt", "max_prompt_length",
    "infer_tensor_parallel_size", "train_tensor_parallel_size", "weight_save_dir",
]

ONE_STEP_OFF_EXTRAS_REQUIRED = [
    "n_samples_per_prompt", "weight_save_dir", "global_batch_size",
]


def check_rollout_config_exists(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """one_step_off must have a rollout_config block."""
    if config.cluster_mode != "one_step_off":
        return
    rollout = yaml_dict.get("rollout_config")
    if rollout is None or (isinstance(rollout, dict) and len(rollout) == 0):
        report.add(Finding(
            severity=Severity.ERROR, code="ROLLOUT_CONFIG_MISSING",
            message="one_step_off mode requires rollout_config block, but config_generator does not generate it",
            detail="Without rollout_config, RolloutWorker and RolloutController cannot start on the inference cluster.",
            paths=["yaml.rollout_config"],
            recommendation="Add rollout_config generation to config_generator.py for one_step_off mode",
        ))
    else:
        missing = [f for f in ONE_STEP_OFF_ROLLOUT_REQUIRED if f not in rollout]
        if missing:
            report.add(Finding(
                severity=Severity.ERROR, code="ROLLOUT_CONFIG_INCOMPLETE",
                message=f"rollout_config missing required fields: {', '.join(missing)}",
                paths=[f"yaml.rollout_config.{f}" for f in missing],
                recommendation="Add these fields to the rollout_config overrides in config_generator.py",
            ))
        else:
            report.pass_check("ROLLOUT_CONFIG_OK")


def check_async_template(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """one_step_off must use fully_async_ppo_trainer template."""
    if config.cluster_mode != "one_step_off":
        return
    defaults = yaml_dict.get("defaults", [])
    has_async = any("fully_async_ppo_trainer" in d for d in defaults)
    has_sync = any("ppo_trainer" in d and "fully_async" not in d for d in defaults)
    if has_sync and not has_async:
        report.add(Finding(
            severity=Severity.ERROR, code="DEFAULTS_TEMPLATE_MISMATCH",
            message="one_step_off mode uses sync ppo_trainer template, needs fully_async_ppo_trainer",
            detail="The one_step_off async training pipeline expects the fully_async_ppo_trainer template. "
                   "Using ppo_trainer will cause incorrect behavior.",
            paths=["yaml.defaults"],
            values={"current_defaults": defaults},
            recommendation="Change defaults to use 'fully_async_ppo_trainer' instead of 'ppo_trainer'",
        ))
    else:
        report.pass_check("ASYNC_TEMPLATE_OK")


def check_async_extras(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """one_step_off requires additional fields in verl_conf.extras."""
    if config.cluster_mode != "one_step_off":
        return
    extras = yaml_dict.get("verl_conf", {}).get("extras", {})
    missing = [f for f in ONE_STEP_OFF_EXTRAS_REQUIRED if f not in extras]
    if missing:
        report.add(Finding(
            severity=Severity.ERROR, code="ONE_STEP_OFF_EXTRAS_MISSING",
            message=f"verl_conf.extras missing required fields for one_step_off: {', '.join(missing)}",
            paths=["verl_conf.extras"],
            recommendation="Add these fields to verl_conf.extras overrides in config_generator.py",
        ))
    else:
        report.pass_check("ASYNC_EXTRAS_OK")

    weight_dir = extras.get("weight_save_dir", "")
    if weight_dir and not weight_dir.startswith("/"):
        report.add(Finding(
            severity=Severity.WARN, code="WEIGHT_SAVE_DIR_NOT_ABSOLUTE",
            message=f"weight_save_dir '{weight_dir}' is not an absolute path",
            paths=["verl_conf.extras.weight_save_dir"],
            recommendation="Use an absolute path for weight_save_dir",
        ))


def check_rollout_tp_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """In one_step_off, rollout_config.infer_tp must match infer_instances TP."""
    if config.cluster_mode != "one_step_off":
        return
    rollout = yaml_dict.get("rollout_config", {})
    infer_tp_rc = rollout.get("infer_tensor_parallel_size")
    infer_tp_ii = _get_nested(yaml_dict, "infer_instances", 0, "executor_kwargs", "engine_kwargs", "tensor_parallel_size")
    if infer_tp_rc and infer_tp_ii and infer_tp_rc != infer_tp_ii:
        report.add(Finding(
            severity=Severity.BLOCK, code="ROLLOUT_INFER_TP_MISMATCH",
            message=f"rollout_config.infer_tp({infer_tp_rc}) != infer_instances TP({infer_tp_ii})",
            detail="RolloutController splits weights by infer_tp, but vLLM starts with a different TP. "
                   "Tensor shapes will not match → crash.",
            paths=["rollout_config.infer_tensor_parallel_size", "infer_instances[0].tensor_parallel_size"],
            values={"rollout_infer_tp": infer_tp_rc, "infer_instances_tp": infer_tp_ii},
            recommendation="Set both to the same value",
        ))
    else:
        report.pass_check("ROLLOUT_INFER_TP_CONSISTENT")


def check_train_tp_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """In one_step_off, rollout_config.train_tp must match actor training TP."""
    if config.cluster_mode != "one_step_off":
        return
    rollout = yaml_dict.get("rollout_config", {})
    train_tp_rc = rollout.get("train_tensor_parallel_size")
    # Try megatron strategy first
    actor_tp = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "actor", "megatron", "tensor_model_parallel_size")
    if train_tp_rc and actor_tp and train_tp_rc != actor_tp:
        report.add(Finding(
            severity=Severity.BLOCK, code="TRAIN_TP_MISMATCH",
            message=f"rollout_config.train_tp({train_tp_rc}) != actor megatron TP({actor_tp})",
            detail="RolloutController needs to know the training TP to correctly merge sharded weights. "
                   "Mismatch causes incorrect weight loading → precision misalignment.",
            paths=["rollout_config.train_tensor_parallel_size", "actor.megatron.tensor_model_parallel_size"],
            values={"rollout_train_tp": train_tp_rc, "actor_tp": actor_tp},
            recommendation="Set both to the same value",
        ))
    else:
        report.pass_check("TRAIN_TP_CONSISTENT")


def check_n_samples_consistency(config: RemoteTrainConfig, yaml_dict: dict, report: Report) -> None:
    """In one_step_off, rollout_config and verl_conf n_samples must match."""
    if config.cluster_mode != "one_step_off":
        return
    rollout = yaml_dict.get("rollout_config", {})
    rc_n = rollout.get("n_samples_per_prompt")
    verl_n = _get_nested(yaml_dict, "verl_conf", "actor_rollout_ref", "rollout", "n")
    extras_n = _get_nested(yaml_dict, "verl_conf", "extras", "n_samples_per_prompt")
    vals = {"rollout_config": rc_n, "rollout.n": verl_n, "extras": extras_n}
    unique = set(v for v in vals.values() if v is not None)
    if len(unique) > 1:
        report.add(Finding(
            severity=Severity.BLOCK, code="N_SAMPLES_INCONSISTENT",
            message=f"n_samples_per_prompt differs: {vals}",
            paths=list(vals.keys()),
            values=vals,
            recommendation="Set all n_samples_per_prompt to the same value",
        ))
    else:
        report.pass_check("N_SAMPLES_CONSISTENT")


# ──────────────────────────────────────────────
# Phase 2 validators (remote, SSH to read model)
# ──────────────────────────────────────────────

RemoteValidatorFn = Callable[[RemoteTrainConfig, dict[str, Any], Report, RemoteExecutor, dict], None]


def check_remote_model_exists(config: RemoteTrainConfig, yaml_dict: dict, report: Report,
                               executor: RemoteExecutor, model_configs: dict) -> None:
    """Model directory and config.json must exist on the remote server."""
    if not model_configs:
        report.add(Finding(
            severity=Severity.ERROR, code="REMOTE_MODEL_CONFIG_EMPTY",
            message=f"Failed to read model configs from {config.model_name_or_path}",
            paths=["RemoteTrainConfig.model_name_or_path"],
            recommendation="Verify the model path exists on the server and contains config.json",
        ))
    elif "config" not in model_configs:
        report.add(Finding(
            severity=Severity.ERROR, code="REMOTE_MODEL_CONFIG_MISSING",
            message=f"config.json not found in model directory: {config.model_name_or_path}",
            paths=["RemoteTrainConfig.model_name_or_path"],
            recommendation="Check model path; ensure config.json exists in model directory",
        ))
    else:
        report.pass_check("REMOTE_MODEL_EXISTS")


def check_max_position_embeddings(config: RemoteTrainConfig, yaml_dict: dict, report: Report,
                                   executor: RemoteExecutor, model_configs: dict) -> None:
    """Our context window must not exceed the model's max_position_embeddings."""
    model_cfg = model_configs.get("config", {})
    max_pos = model_cfg.get("max_position_embeddings")
    if max_pos is None:
        report.add(Finding(
            severity=Severity.WARN, code="MAX_POSITION_EMBEDDINGS_UNKNOWN",
            message="Could not read max_position_embeddings from model config.json",
            paths=["model config.json"],
        ))
        return

    total = config.max_prompt_length + config.max_response_length
    derived = config.max_response_length * 2

    if total > max_pos:
        report.add(Finding(
            severity=Severity.BLOCK, code="MAX_POSITION_EMBEDDINGS_EXCEEDED",
            message=f"prompt({config.max_prompt_length}) + response({config.max_response_length}) = {total} "
                    f"> model max_position_embeddings({max_pos})",
            paths=["max_prompt_length", "max_response_length", "model config.json"],
            values={"total_window": total, "max_position_embeddings": max_pos},
            recommendation=f"Reduce max_prompt_length + max_response_length to ≤ {max_pos}",
            layer=1,
        ))

    if derived > max_pos:
        report.add(Finding(
            severity=Severity.BLOCK, code="MAX_MODEL_LEN_EXCEEDS_LIMIT",
            message=f"Derived max_model_len({derived} = response*2) > model max_position_embeddings({max_pos})",
            paths=["max_response_length → max_model_len", "model config.json"],
            values={"derived_max_model_len": derived, "max_position_embeddings": max_pos},
            recommendation=f"Reduce max_response_length so that response*2 ≤ {max_pos}",
            layer=1,
        ))

    if total <= max_pos and derived <= max_pos:
        report.pass_check("MAX_POSITION_EMBEDDINGS_OK")


# Parameters covered by our config system (via agent_temperature, agent_top_p, rollout.do_sample, agent_max_tokens)
COVERED_GENERATION_PARAMS = {
    "temperature": "agent_temperature → infer_service_params.temperature",
    "top_p": "agent_top_p → infer_service_params.top_p",
    "do_sample": "rollout.do_sample (fixed in BASE_CONFIG)",
    "max_tokens": "agent_max_tokens → infer_service_params.max_tokens",
}


def check_generation_config_coverage(config: RemoteTrainConfig, yaml_dict: dict, report: Report,
                                      executor: RemoteExecutor, model_configs: dict) -> None:
    """Identify model generation_config.json params NOT covered by our config overrides."""
    gen_cfg = model_configs.get("generation_config", {})
    if not gen_cfg:
        report.pass_check("GENERATION_CONFIG_NOT_FOUND")
        return

    uncovered = {}
    for key, val in gen_cfg.items():
        if key not in COVERED_GENERATION_PARAMS and val is not None:
            # Skip token IDs (not generation params per se)
            if key.endswith("_token_id"):
                continue
            uncovered[key] = val

    if uncovered:
        report.add(Finding(
            severity=Severity.WARN, code="UNCOVERED_GENERATION_PARAMS",
            message=f"{len(uncovered)} generation_config.json param(s) not covered by our config: {list(uncovered.keys())}",
            detail="These params silently inherit model defaults. If the model changes, "
                   "these defaults may change without notice, affecting training reproducibility.",
            values=uncovered,
            recommendation="Consider adding overrides in config_generator.py or explicitly documenting "
                           "that model defaults are acceptable",
            layer=1,
        ))
    else:
        report.pass_check("ALL_GENERATION_PARAMS_COVERED")


def check_tokenizer_max_length(config: RemoteTrainConfig, yaml_dict: dict, report: Report,
                                executor: RemoteExecutor, model_configs: dict) -> None:
    """Warn if prompt+response exceeds tokenizer's model_max_length."""
    tok_cfg = model_configs.get("tokenizer_config", {})
    tok_max = tok_cfg.get("model_max_length")
    if tok_max is None:
        report.pass_check("TOKENIZER_MAX_LENGTH_UNKNOWN")
        return

    total = config.max_prompt_length + config.max_response_length
    if total > tok_max:
        report.add(Finding(
            severity=Severity.WARN, code="TOKENIZER_MAX_LENGTH_EXCEEDED",
            message=f"prompt({config.max_prompt_length}) + response({config.max_response_length}) = {total} "
                    f"> tokenizer model_max_length({tok_max})",
            paths=["max_prompt_length", "max_response_length", "tokenizer_config.json"],
            values={"total_window": total, "tokenizer_max_length": tok_max},
            recommendation="Tokenizer may silently truncate. Increase prompt/response or use a different tokenizer",
            layer=1,
        ))
    else:
        report.pass_check("TOKENIZER_MAX_LENGTH_OK")


# ──────────────────────────────────────────────
# Validator registries
# ──────────────────────────────────────────────

HYBRID_VALIDATORS: list[ValidatorFn] = [
    check_dead_fields,
    check_max_model_len_derivation,
    check_model_path_consistency,
    check_max_prompt_length_consistency,
    check_max_response_length_consistency,
    check_batch_size_hierarchy,
    check_context_window_overflow,
    check_tp_vs_npus,
    check_hydra_variables,
    check_agentsdk_binary_files_ref,
    check_rollout_temperature_gap,
    check_hybrid_tp_consistency,
]

ONE_STEP_OFF_VALIDATORS: list[ValidatorFn] = [
    check_dead_fields,
    check_max_model_len_derivation,
    check_model_path_consistency,
    check_max_prompt_length_consistency,
    check_max_response_length_consistency,
    check_batch_size_hierarchy,
    check_context_window_overflow,
    check_tp_vs_npus,
    check_hydra_variables,
    check_agentsdk_binary_files_ref,
    check_rollout_temperature_gap,
    check_rollout_config_exists,
    check_async_template,
    check_async_extras,
    check_rollout_tp_consistency,
    check_train_tp_consistency,
    check_n_samples_consistency,
]

PHASE2_VALIDATORS: list[RemoteValidatorFn] = [
    check_remote_model_exists,
    check_max_position_embeddings,
    check_generation_config_coverage,
    check_tokenizer_max_length,
]


# ──────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────


def run_phase1(config: RemoteTrainConfig) -> Report:
    """Run all local (no SSH) validators and return a Report."""
    yaml_dict = generate_agentsdk_config(config)
    report = Report()
    report.config_summary = {
        "cluster_mode": config.cluster_mode,
        "model_name_or_path": config.model_name_or_path,
        "train_batch_size": config.train_batch_size,
        "max_prompt_length": config.max_prompt_length,
        "max_response_length": config.max_response_length,
        "tensor_parallel_size": config.tensor_parallel_size,
        "n_npus_per_node": config.n_npus_per_node,
    }

    validators = ONE_STEP_OFF_VALIDATORS if config.cluster_mode == "one_step_off" else HYBRID_VALIDATORS
    for validator in validators:
        try:
            validator(config, yaml_dict, report)
        except Exception as e:
            report.add(Finding(
                severity=Severity.ERROR, code="VALIDATOR_EXCEPTION",
                message=f"Validator {validator.__name__} raised: {e}",
            ))

    return report


def run_phase2(config: RemoteTrainConfig, report: Report) -> Report:
    """Run remote validators (SSH to read model configs). Appends to existing report."""
    executor = RemoteExecutor(config)

    # Verify connectivity first
    try:
        connectivity = executor.verify_connectivity()
    except Exception as e:
        report.add(Finding(
            severity=Severity.ERROR, code="SSH_CONNECTIVITY",
            message=f"Cannot SSH to {config.ssh_host}:{config.ssh_port}",
            detail=str(e),
            recommendation="Check SSH host, port, and password",
        ))
        return report

    if not connectivity.get("ssh"):
        report.add(Finding(
            severity=Severity.ERROR, code="SSH_CONNECTIVITY",
            message=f"Cannot SSH to {config.ssh_host}:{config.ssh_port}",
            detail=str(connectivity.get("ssh_error", "")),
            recommendation="Check SSH host, port, and password",
        ))
        return report
    report.pass_check("SSH_CONNECTIVITY")

    if not connectivity.get("container"):
        report.add(Finding(
            severity=Severity.ERROR, code="CONTAINER_NOT_RUNNING",
            message=f"Container '{config.container_name}' is not running",
            recommendation=f"Run: docker start {config.container_name}",
        ))
        return report
    report.pass_check("CONTAINER_RUNNING")

    # Read remote model configs
    try:
        script = _read_model_config_script(config.model_name_or_path)
        output = executor.run_script(script, timeout=30)
        model_configs = json.loads(output)
    except Exception as e:
        report.add(Finding(
            severity=Severity.ERROR, code="REMOTE_MODEL_READ_FAILED",
            message=f"Failed to read model configs from {config.model_name_or_path}",
            detail=str(e),
            recommendation="Verify model path on remote server",
        ))
        return report

    # Generate YAML for cross-reference (reuse from phase1 if available, or regenerate)
    yaml_dict = generate_agentsdk_config(config)

    # Run Phase 2 validators
    for validator in PHASE2_VALIDATORS:
        try:
            validator(config, yaml_dict, report, executor, model_configs)
        except Exception as e:
            report.add(Finding(
                severity=Severity.ERROR, code="VALIDATOR_EXCEPTION",
                message=f"Remote validator {validator.__name__} raised: {e}",
            ))

    return report


def validate_config(config_path: str, ssh_password: str = "", phase: str = "all") -> Report:
    """Load config from JSON, run validation, return Report."""
    config = RemoteTrainConfig.from_json(config_path)
    if ssh_password:
        config.ssh_password = ssh_password

    # Always run Phase 1 first
    report = run_phase1(config)

    if phase in ("all", "phase2"):
        if not config.ssh_password:
            report.add(Finding(
                severity=Severity.ERROR, code="SSH_PASSWORD_REQUIRED",
                message="Phase 2 (remote checks) requires SSH password",
                recommendation="Provide --ssh-password to run remote validation",
            ))
        else:
            report = run_phase2(config, report)

    return report


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate remote NPU training config consistency across all layers",
    )
    parser.add_argument("config_json", help="Path to RemoteTrainConfig JSON file")
    parser.add_argument("--ssh-password", default="", help="SSH password (required for Phase 2)")
    parser.add_argument("--phase", choices=["phase1", "phase2", "all"], default="all",
                        help="phase1=local only, phase2=remote only (requires --ssh-password), all=both")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--output", default="", help="Write JSON report to this file path")
    args = parser.parse_args()

    report = validate_config(
        args.config_json,
        ssh_password=args.ssh_password,
        phase=args.phase,
    )

    # JSON output
    if args.format == "json" or args.output:
        json_data = {
            "block": report.block,
            "findings": [
                {**asdict(f), "severity": f.severity.value}
                for f in report.findings
            ],
            "passed_checks": report.passed_checks,
            "config_summary": report.config_summary,
        }
        json_text = json.dumps(json_data, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_text)
        if args.format == "json":
            print(json_text)

    # Always print text report
    if args.format == "text":
        print(report.format_text())

    sys.exit(1 if report.block else 0)


if __name__ == "__main__":
    main()
