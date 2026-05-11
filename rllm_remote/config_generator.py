"""
Generate AgentSDK-compatible YAML configuration from RemoteTrainConfig.

Strategy: embed the container's working reference config as a base dict,
then deep-merge our dynamic overrides. This avoids OmegaConf key-replacement
issues — every required key from the verl ppo_trainer template is present.
"""

import copy
from typing import Any

from rllm_remote.config import RemoteTrainConfig


# Base config extracted from the container's working reference:
# /home/qjy/code/AgentSDK/master/AgentSDK/configs/
#   base_direct_1node_qwen25_7b_train_hybrid_with_verl_fsdp.yaml
BASE_CONFIG: dict[str, Any] = {
    "agentic_ai": {
        "mode": "direct",
        "log_level": "DEBUG",
        "log_dir": "/var/log/",
    },
    "serve_conf": {
        "host": "0.0.0.0",
        "port": 8030,
    },
    "direct_conf": {
        "entrypoints": [
            {
                "job_type": "train",
                "job_name": "${train_instances.0.name}",
                "job_kwargs": {},
            }
        ]
    },
    # NOTE: defaults is at TOP LEVEL, NOT nested under hydra.
    # In Hydra 1.x, hydra.defaults is NOT the config composition directive;
    # it must be a top-level key.
    "hydra": {
        "searchpath": [
            "file:///verl/verl/trainer/config",
            "file://AgenticRL/configs/verl_conf",
        ],
    },
    "defaults": [
        "ppo_trainer",
        "ppo_trainer@verl_conf",
        "_self_",
    ],
    "verl_conf": {
        "extras": {
            "agent_service": "${agent_instances.0.name}",
            "infer_service": "${infer_instances.0.name}",
            "traj_output_path": "${hydra:runtime.cwd}/outputs",
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "kl_ctrl": {"kl_coef": 0.001},
        },
        "data": {
            "train_files": "/home/qjy/data/gsm8k/main/train-00000-of-00001.parquet",
            "val_files": "/home/qjy/data/gsm8k/main/test-00000-of-00001.parquet",
            "train_batch_size": 32,
            "max_prompt_length": 2048,
            "max_response_length": 2048,
            "filter_overlong_prompts": True,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "model": {
                "path": "/home/qjy/models/Qwen2.5-7B-Instruct",
                "use_remove_padding": False,
                "enable_gradient_checkpointing": True,
            },
            "actor": {
                "strategy": "fsdp",
                "optim": {"lr": 1e-6},
                "entropy_coeff": 0.0,
                "ppo_mini_batch_size": 8,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                },
            },
            "rollout": {
                "calculate_log_probs": True,
                "prompt_length": 2048,
                "response_length": 2048,
                "agent": {
                    "agent_loop_manager_class": (
                        "agentic_rl.trainer.train_adapter.verl.hybrid."
                        "agent_loop_manager.HybridAgentLoopManager"
                    ),
                },
                "log_prob_micro_batch_size_per_gpu": 1,
                "enable_chunked_prefill": False,
                "tensor_model_parallel_size": 4,
                "pipeline_model_parallel_size": 1,
                "name": "vllm",
                "gpu_memory_utilization": 0.6,
                "n": 8,
                "do_sample": False,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
        },
        "trainer": {
            "val_before_train": False,
            "device": "npu",
            "critic_warmup": 0,
            "project_name": "verl_grpo_example_gsm8k",
            "experiment_name": "qwen2_7b_function_rm",
            "n_gpus_per_node": 8,
            "nnodes": 1,
            "save_freq": -1,
            "test_freq": 1000,
            "total_epochs": 100,
            "logger": ["console", "tensorboard"],
        },
    },
    "train_instances": [
        {
            "name": "RL-QWEN-7B-WITH-DTN-8NPU",
            "executor_num": 1,
            "executor_kwargs": {
                "cluster_mode": "hybrid",
                "train_engine": "verl",
                "train_config": "${verl_conf}",
                "rollout_config": {},
                "agent_service": "${agent_instances.0.name}",
                "infer_service": "${infer_instances.0.name}",
            },
            "resource_info": [],
        }
    ],
    "agent_instances": [
        {
            "name": "MATH-AGENT",
            "executor_num": 1,
            "executor_kwargs": {
                "agent_engine": "rllm",
                "agent_engine_kwargs": {
                    "agent_name": "math",
                    "simplify_think_content": False,
                    "max_steps": 5,
                    "max_prompt_length": 2048,
                    "max_model_len": 4096,
                    "n_parallel_agents": 1024,
                    "tokenizer": "/home/qjy/models/Qwen2.5-7B-Instruct",
                },
                "infer_service_params": {
                    "top_p": 1,
                    "temperature": 1,
                    "max_tokens": 4096,
                    "model": "${infer_instances.0.name}",
                },
                "trajectory_save_dir": "${hydra:runtime.cwd}/outputs/trajectory.jsonl",
            },
            "resource_info": [],
        }
    ],
    "infer_instances": [
        {
            "name": "QWEN2.5-7B",
            "executor_num": 1,
            "executor_kwargs": {
                "engine": "vllm_proxy",
                "engine_kwargs": {
                    "chat_server": "http://7.246.80.80:8080",
                    "prefill_server_list": [],
                    "decode_server_list": [],
                    "model_name": "Qwen3-235B-A22B",
                },
            },
            "resource_info": [],
        }
    ],
}


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base, returning a new dict.

    Lists of dicts are merged by index (assumes same ordering).
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            merged_list = []
            for i, item in enumerate(value):
                if i < len(result[key]) and isinstance(result[key][i], dict) and isinstance(item, dict):
                    merged_list.append(deep_merge(result[key][i], item))
                else:
                    merged_list.append(copy.deepcopy(item))
            # Keep remaining base items
            merged_list.extend(copy.deepcopy(result[key][len(value):]))
            result[key] = merged_list
        else:
            result[key] = copy.deepcopy(value)
    return result


def generate_agentsdk_config(config: RemoteTrainConfig) -> dict[str, Any]:
    """Generate an AgentSDK YAML config by merging overrides into BASE_CONFIG."""

    run_name = f"RL-{config.run_id[:20]}"
    remote_run_dir = f"{config.remote_output_dir}/{config.run_id}"
    infer_name = config.container_name

    overrides = {
        "agentic_ai": {
            "log_level": "INFO",
            "log_dir": f"{remote_run_dir}/logs",
        },
        "direct_conf": {
            "entrypoints": [
                {
                    "job_type": "train",
                    "job_name": run_name,
                    "job_kwargs": {},
                }
            ]
        },
        "verl_conf": {
            "extras": {
                "agent_service": "MATH-AGENT",
                "infer_service": infer_name,
            },
            "algorithm": {
                "kl_ctrl": {"kl_coef": config.kl_coef},
            },
            "data": {
                "train_files": config.train_data_path,
                "val_files": config.val_data_path,
                "train_batch_size": config.train_batch_size,
                "max_prompt_length": config.max_prompt_length,
                "max_response_length": config.max_response_length,
            },
            "actor_rollout_ref": {
                "model": {"path": config.model_name_or_path},
                "actor": {
                    "optim": {"lr": config.lr},
                    "entropy_coeff": config.entropy_coeff,
                    "ppo_mini_batch_size": config.ppo_mini_batch_size,
                    "ppo_micro_batch_size_per_gpu": config.ppo_micro_batch_size_per_gpu,
                    "kl_loss_coef": config.kl_coef,
                },
                "rollout": {
                    "prompt_length": config.max_prompt_length,
                    "response_length": config.max_response_length,
                    "tensor_model_parallel_size": config.tensor_parallel_size,
                    "pipeline_model_parallel_size": config.pipeline_parallel_size,
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "n": config.n_samples_per_prompt,
                },
            },
            "trainer": {
                "project_name": config.wandb_project or "rllm_remote",
                "experiment_name": config.wandb_run_name or config.run_id,
                "n_gpus_per_node": config.n_npus_per_node,
                "nnodes": config.nnodes,
                "total_epochs": config.num_epochs,
            },
        },
        "train_instances": [
            {
                "name": run_name,
                "executor_kwargs": {
                    "cluster_mode": config.cluster_mode,
                    "train_engine": config.train_engine,
                    "agent_service": "MATH-AGENT",
                    "infer_service": infer_name,
                },
            }
        ],
        "agent_instances": [
            {
                "executor_kwargs": {
                    "agent_engine_kwargs": {
                        "agent_name": config.agent_name,
                        "max_steps": config.max_agent_steps,
                        "max_prompt_length": config.max_prompt_length,
                        "max_model_len": config.max_response_length * 2,
                        "n_parallel_agents": config.n_parallel_agents,
                        "tokenizer": config.tokenizer_name_or_path,
                    },
                    "infer_service_params": {
                        "top_p": config.agent_top_p,
                        "temperature": config.agent_temperature,
                        "max_tokens": config.agent_max_tokens,
                        "model": infer_name,
                    },
                    "trajectory_save_dir": f"{remote_run_dir}/trajectories.jsonl",
                },
            }
        ],
        "infer_instances": [
            {
                "name": infer_name,
                "executor_kwargs": {
                    "engine_kwargs": {
                        "chat_server": config.chat_server_url,
                        "model_name": config.model_display_name,
                    },
                },
            }
        ],
    }

    return deep_merge(BASE_CONFIG, overrides)
