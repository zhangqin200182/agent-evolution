"""
Main training entry point for veRL-based agent RL.

Usage:
    python -m rllm_verl.train --model Qwen/Qwen2.5-3B-Instruct --num-gpus 2

This script:
1. Prepares the dataset (math problems -> parquet)
2. Launches veRL training with GRPO/PPO on the agent environment
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rllm_verl.config import VerlTrainConfig
from rllm_verl.dataset import prepare_verl_dataset


def build_verl_command(config: VerlTrainConfig, data_path: str) -> list[str]:
    """Build the veRL training command."""
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"--config-path={Path(__file__).parent / 'configs'}",
        "--config-name=math_agent",
    ]

    # Hydra overrides
    overrides = [
        f"data.train_files={data_path}",
        f"data.train_batch_size={config.batch_size}",
        f"data.max_prompt_length=1024",
        f"data.max_response_length={config.max_new_tokens}",
        f"actor_rollout_ref.model.path={config.model_name}",
        f"actor_rollout_ref.actor.optim.lr={config.lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size={config.mini_batch_size}",
        f"actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={config.kl_coeff}",
        f"actor_rollout_ref.actor.clip_ratio={config.clip_range}",
        f"actor_rollout_ref.actor.entropy_coeff={config.entropy_coeff}",
        f"actor_rollout_ref.rollout.temperature={config.temperature}",
        f"actor_rollout_ref.rollout.top_p={config.top_p}",
        f"actor_rollout_ref.rollout.n={config.rollouts_per_problem}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config.tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.gpu_memory_utilization}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size={config.mini_batch_size}",
        f"algorithm.adv_estimator=grpo",
        f"trainer.total_epochs={config.num_epochs}",
        f"trainer.n_gpus_per_node={config.num_gpus}",
        f"trainer.save_freq={config.save_interval}",
        f"trainer.default_local_dir={config.output_dir}",
    ]

    if config.wandb_project:
        overrides.append(f"trainer.project_name={config.wandb_project}")
    if config.wandb_run_name:
        overrides.append(f"trainer.experiment_name={config.wandb_run_name}")

    cmd.extend(overrides)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="veRL agent RL training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--difficulty", type=str, default="mixed")
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="./outputs/verl_run")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-run", type=str, default="")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    args = parser.parse_args()

    config = VerlTrainConfig(
        model_name=args.model,
        num_gpus=args.num_gpus,
        num_problems=args.num_problems,
        problem_difficulty=args.difficulty,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        rollouts_per_problem=args.rollouts,
        lr=args.lr,
        temperature=args.temperature,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run,
    )

    # Step 1: Prepare dataset
    data_dir = os.path.join(config.output_dir, "data")
    print(f"Preparing dataset: {config.num_problems} problems ({config.problem_difficulty})")
    data_path = prepare_verl_dataset(
        num_problems=config.num_problems,
        seed=config.dataset_seed,
        difficulty=config.problem_difficulty,
        output_dir=data_dir,
    )

    # Step 2: Build and run veRL command
    cmd = build_verl_command(config, data_path)

    print("\n--- veRL Training Command ---")
    print(" \\\n  ".join(cmd))
    print("---\n")

    if args.dry_run:
        print("[dry-run] Command printed above. Exiting.")
        return

    # Launch training
    os.makedirs(config.output_dir, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(config.num_gpus))

    print(f"Launching veRL training with {config.num_gpus} GPU(s)...")
    result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).parent.parent))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
