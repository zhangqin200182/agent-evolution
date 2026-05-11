"""
Reward functions for veRL agent RL training.
These are called by veRL's reward manager during rollout evaluation.
"""

import json
import re
from typing import Any


def extract_final_answer(response_text: str) -> str | None:
    """Extract the final answer from a model response containing tool calls."""
    # Look for finish tool call
    if "<tool_call>" in response_text:
        import re as _re
        # Find all tool calls
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = _re.findall(pattern, response_text, _re.DOTALL)
        for match in reversed(matches):  # check last tool call first
            try:
                call = json.loads(match)
                if call.get("name") == "finish":
                    return call.get("arguments", {}).get("response", "")
            except json.JSONDecodeError:
                continue

    # Fallback: extract last number from response
    return response_text


def compute_math_reward(response_text: str, ground_truth: str) -> float:
    """
    Compute reward for a math agent response.

    Returns 1.0 if the final numeric answer matches ground_truth, 0.0 otherwise.
    """
    answer_text = extract_final_answer(response_text)
    if answer_text is None:
        return 0.0

    # Extract numbers from the answer
    numbers = re.findall(r"-?\d+\.?\d*", str(answer_text))
    if not numbers:
        return 0.0

    predicted = float(numbers[-1])
    try:
        expected = float(ground_truth)
    except (ValueError, TypeError):
        return 0.0

    return 1.0 if abs(predicted - expected) < 1e-6 else 0.0


def reward_function(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
    """
    veRL-compatible reward function interface.

    Args:
        data_source: identifier for the data source (e.g. "math_calc")
        solution_str: the model's full response text
        ground_truth: the expected answer string
        extra_info: optional additional context

    Returns:
        float reward value
    """
    if data_source == "math_calc":
        return compute_math_reward(solution_str, ground_truth)
    return 0.0
