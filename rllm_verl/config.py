"""
Configuration dataclass for veRL training.
Mirrors rllm_train config but adds veRL-specific fields (tensor_parallel, pipeline_parallel, etc.).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerlTrainConfig:
    """Configuration for veRL-based agent RL training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_name: str | None = None  # defaults to model_name

    # Dataset
    num_problems: int = 200
    problem_difficulty: str = "mixed"
    dataset_seed: int = 42

    # Training
    num_epochs: int = 3
    rollouts_per_problem: int = 4
    max_steps_per_episode: int = 3
    batch_size: int = 8
    mini_batch_size: int = 4
    lr: float = 1e-6
    kl_coeff: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    clip_range: float = 0.2
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # veRL-specific
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    num_gpus: int = 1
    gpu_memory_utilization: float = 0.85
    vllm_enforce_eager: bool = False

    # Logging & checkpointing
    output_dir: str = "./outputs/verl_run"
    log_interval: int = 10
    save_interval: int = 50
    wandb_project: str = ""
    wandb_run_name: str = ""

    # Misc
    seed: int = 42
    disable_thinking: bool = True

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
