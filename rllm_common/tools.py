"""
Shared tool definitions for agent RL training.
Extracted from rllm_train/train.py for reuse across backends.
"""

from rllm_common.base import ToolOutput


TOOL_SYSTEM_PROMPT = """You are a helpful math assistant. You can use the calculate tool to compute arithmetic expressions, and the finish tool to give your final answer.

Always use the calculate tool first, then use finish to report the result."""


class CalculateTool:
    name = "calculate"
    description = "Evaluate a mathematical expression"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '2 + 3'",
            }
        },
        "required": ["expression"],
    }

    def __init__(self, **kwargs):
        pass

    def __call__(self, expression: str = "", **kwargs) -> ToolOutput:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in str(expression)):
                return ToolOutput(output="Error: invalid expression")
            result = eval(str(expression))  # noqa: S307
            return ToolOutput(output=str(result))
        except Exception as e:
            return ToolOutput(output=f"Error: {e}")

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FinishTool:
    name = "finish"
    description = "Submit your final answer"
    parameters = {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "Your final answer",
            }
        },
        "required": ["response"],
    }

    def __init__(self, **kwargs):
        pass

    def __call__(self, response: str = "", **kwargs) -> ToolOutput:
        return ToolOutput(output=response)

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def create_simple_tool_agent_args():
    """Return agent_args dict for creating a ToolAgent with math tools."""
    return {
        "system_prompt": TOOL_SYSTEM_PROMPT,
        "parser_name": "qwen",
        "tool_map": {
            "calculate": CalculateTool,
            "finish": FinishTool,
        },
    }
