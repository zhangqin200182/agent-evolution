"""
rllm_train: Self-contained Agent RL integration of rllm concepts with TRL.

Inlines the minimal rllm abstractions (agent, env, chat parser, token masking)
to avoid rllm's heavy dependency chain (vllm, flash-attn, deepspeed, etc.).
"""

from rllm_train.hf_engine import HFAgentExecutionEngine
from rllm_train.rollout import make_rllm_rollout_func
