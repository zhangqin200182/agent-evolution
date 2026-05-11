"""
Dataset preparation for veRL training.
Converts math problems into the parquet format expected by veRL.
"""

import json
import os
from pathlib import Path

import pandas as pd

from rllm_common.math_env import generate_math_problems
from rllm_common.tools import TOOL_SYSTEM_PROMPT, CalculateTool, FinishTool, create_simple_tool_agent_args
from rllm_common.parsers import QwenToolParser


def build_prompt_for_problem(question: str) -> list[dict[str, str]]:
    """Build the initial message list for a math problem."""
    tool_parser = QwenToolParser()
    tools_json = [CalculateTool().json, FinishTool().json]
    tools_prompt = tool_parser.get_tool_prompt(json.dumps(tools_json, indent=2))

    messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT + tools_prompt},
        {"role": "user", "content": question},
    ]
    return messages


def prepare_verl_dataset(
    num_problems: int = 200,
    seed: int = 42,
    difficulty: str = "mixed",
    output_dir: str = "./data",
) -> str:
    """
    Generate math problems and save as parquet for veRL consumption.

    Returns the path to the generated parquet file.
    """
    problems = generate_math_problems(n=num_problems, seed=seed, difficulty=difficulty)

    records = []
    for i, problem in enumerate(problems):
        prompt_messages = build_prompt_for_problem(problem["question"])
        records.append({
            "data_source": "math_calc",
            "prompt": prompt_messages,
            "reward_model": {
                "style": "rule",
                "ground_truth": problem["answer"],
            },
            "extra_info": {
                "question": problem["question"],
                "answer": problem["answer"],
                "index": i,
            },
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.parquet")
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(records)} problems to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare veRL training dataset")
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["simple", "mixed", "hard"], default="mixed")
    parser.add_argument("--output-dir", type=str, default="./data")
    args = parser.parse_args()

    prepare_verl_dataset(
        num_problems=args.num_problems,
        seed=args.seed,
        difficulty=args.difficulty,
        output_dir=args.output_dir,
    )
