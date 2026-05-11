"""
Remote training configuration for NPU AgentSDK backend.
Mirrors TrainingConfig and VerlTrainConfig but adds SSH/server/NPU-specific fields.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field


@dataclass
class RemoteTrainConfig:
    """Configuration for remote NPU training via AgentSDK on Ascend server."""

    # === Server connection ===
    ssh_host: str = "<server-ip>"
    ssh_port: int = 22
    ssh_user: str = "root"
    ssh_key_path: str = "~/.ssh/id_rsa"
    ssh_password: str = field(default="", repr=False)  # never serialized, use sshpass -e
    container_name: str = "agent5.0.0_qjy"

    # === Model (server-side paths) ===
    model_name_or_path: str = "/home/qjy/models/Qwen2.5-7B-Instruct"
    tokenizer_name_or_path: str = ""  # defaults to model_name_or_path

    # === Dataset (server-side paths) ===
    train_data_path: str = "/home/qjy/data/processed_math_data/train.parquet"
    val_data_path: str = "/home/qjy/data/processed_math_data/test.parquet"

    # === Training hyperparams ===
    num_epochs: int = 100
    train_batch_size: int = 32
    ppo_mini_batch_size: int = 8
    ppo_micro_batch_size_per_gpu: int = 1
    lr: float = 1e-6
    kl_coef: float = 0.001
    temperature: float = 1.0
    top_p: float = 1.0
    n_samples_per_prompt: int = 8
    max_prompt_length: int = 2048
    max_response_length: int = 2048
    entropy_coeff: float = 0.0

    # === NPU / Cluster ===
    n_npus_per_node: int = 8
    nnodes: int = 1
    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.6

    # === Framework selection ===
    train_engine: str = "verl"  # "verl" or "megatron"
    cluster_mode: str = "hybrid"  # "hybrid" or "one_step_off"

    # === Agent config ===
    agent_name: str = "math"
    max_agent_steps: int = 5
    n_parallel_agents: int = 1024
    agent_temperature: float = 1.0
    agent_top_p: float = 1.0
    agent_max_tokens: int = 4096

    # === Infer service ===
    chat_server_url: str = "http://<infer-server-ip>:8080"
    model_display_name: str = "Qwen3-235B-A22B"

    # === Output ===
    # AgentSDK installation directory inside container (contains run_start_in_local.sh, configs/, agentic_rl/)
    remote_agent_sdk_dir: str = "/home/qjy/code/AgentSDK/master/AgentSDK"
    remote_output_dir: str = "/home/qjy/code/AgentSDK/master/AgentSDK/outputs"
    local_output_dir: str = "rllm_remote/output/runs"
    run_id: str = ""

    # === Logging ===
    wandb_project: str = ""
    wandb_run_name: str = ""

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"remote_{int(time.time())}"
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path
        if not os.path.isabs(self.local_output_dir):
            self.local_output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", self.local_output_dir
            )
            self.local_output_dir = os.path.normpath(self.local_output_dir)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Remote NPU Training Configuration",
            "=" * 60,
            f"  Run ID:           {self.run_id}",
            f"  Server:           {self.ssh_user}@{self.ssh_host}:{self.ssh_port}",
            f"  Container:        {self.container_name}",
            f"  Model:            {self.model_name_or_path}",
            f"  Train engine:     {self.train_engine}",
            f"  Cluster mode:     {self.cluster_mode}",
            f"  NPUs:             {self.n_npus_per_node} (TP={self.tensor_parallel_size})",
            f"  Epochs:           {self.num_epochs}",
            f"  Batch size:       {self.train_batch_size}",
            f"  Mini batch:       {self.ppo_mini_batch_size}",
            f"  Samples/prompt:   {self.n_samples_per_prompt}",
            f"  Learning rate:    {self.lr}",
            f"  KL coef:          {self.kl_coef}",
            f"  Temperature:      {self.temperature}",
            f"  Agent:            {self.agent_name} (max_steps={self.max_agent_steps})",
            f"  Remote output:    {self.remote_output_dir}/{self.run_id}",
            f"  Local output:     {self.local_output_dir}/{self.run_id}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_json(self, path: str | None = None) -> str:
        data = {k: v for k, v in asdict(self).items() if k != "ssh_password"}
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path: str) -> "RemoteTrainConfig":
        with open(path) as f:
            data = json.load(f)
        known_fields = {f.name for f in __import__("dataclasses").fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
