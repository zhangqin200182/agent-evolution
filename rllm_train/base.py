"""
Self-contained base abstractions inlined from rllm.
Provides BaseAgent, BaseEnv, Action, Step, Trajectory, ToolCall, ToolOutput.

Re-exports from rllm_common for backward compatibility.
"""

from rllm_common.base import (  # noqa: F401
    Action,
    BaseAgent,
    BaseEnv,
    Step,
    ToolCall,
    ToolOutput,
    Trajectory,
)
