"""Shared abstractions for agent RL training backends (rllm_train, rllm_verl)."""

from rllm_common.base import (
    Action,
    BaseAgent,
    BaseEnv,
    Step,
    ToolCall,
    ToolOutput,
    Trajectory,
)
from rllm_common.math_env import MathCalcEnv, generate_math_problems
from rllm_common.tool_agent import ToolAgent

__all__ = [
    "Action",
    "BaseAgent",
    "BaseEnv",
    "MathCalcEnv",
    "Step",
    "ToolAgent",
    "ToolCall",
    "ToolOutput",
    "Trajectory",
    "generate_math_problems",
]
