"""Analyzer base — infrastructure for LLM-based trajectory analysis.

The actual analysis logic lives in traj-analyze-* SKILL.md files.
This module provides utilities for reading trajectories and formatting reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from traj_opt.adapter.schema import Trajectory, TrajectoryType
from traj_opt.store.reader import TrajectoryReader
from traj_opt.config import DEFAULT_CONFIG, TrajectoryConfig


class AnalyzerBase:
    """Provides trajectory reading and summary utilities for analysis skills."""

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config
        self.reader = TrajectoryReader(config)

    def get_rllm_trajectories(self, days: Optional[int] = None, session_id: Optional[str] = None) -> List[Trajectory]:
        """Get rllm-train related trajectories from Layer 1 (rllm)."""
        if session_id:
            all_trajs = self.reader.read_session_trajectories(session_id, layer="rllm")
        else:
            all_trajs = self.reader.read_recent_trajectories(days)
        return [
            t for t in all_trajs
            if (session_id or t.layer == "rllm")
            and t.trajectory_type == TrajectoryType.SKILL
            and t.skill_name
            and t.skill_name.startswith("rllm-")
        ]

    def get_traj_trajectories(self, days: Optional[int] = None) -> List[Trajectory]:
        """Get traj-loop related trajectories from Layer 2 (traj)."""
        all_trajs = self.reader.read_recent_trajectories(days)
        return [
            t for t in all_trajs
            if t.layer == "traj"
            and t.trajectory_type == TrajectoryType.SKILL
            and t.skill_name
            and t.skill_name.startswith("traj-")
        ]

    def get_skill_trajectories(self, skill_prefix: str, days: Optional[int] = None) -> List[Trajectory]:
        """Get trajectories for skills matching a prefix."""
        all_trajs = self.reader.read_recent_trajectories(days)
        return [
            t for t in all_trajs
            if t.trajectory_type == TrajectoryType.SKILL
            and t.skill_name
            and t.skill_name.startswith(skill_prefix)
        ]

    def summarize_trajectory(self, traj: Trajectory) -> Dict[str, Any]:
        """Create a concise summary of a trajectory for LLM consumption."""
        tool_summary: Dict[str, int] = {}
        for tc in traj.tool_calls:
            tool_summary[tc.tool_name] = tool_summary.get(tc.tool_name, 0) + 1

        failed_tools = [
            {"tool": tc.tool_name, "input": tc.tool_input}
            for tc in traj.tool_calls
            if not tc.success
        ]

        return {
            "session_id": traj.session_id,
            "trajectory_id": traj.trajectory_id,
            "type": traj.trajectory_type.value,
            "skill_name": traj.skill_name,
            "skill_args": traj.skill_args,
            "duration_ms": traj.duration_ms,
            "tool_count": len(traj.tool_calls),
            "tool_summary": tool_summary,
            "files_touched": traj.files_touched,
            "intent_tags": traj.intent_tags,
            "outcome": traj.outcome,
            "failed_tools": failed_tools,
        }

    def extract_training_data(self, traj: Trajectory) -> Dict[str, Any]:
        """Extract training data surfaced by rllm-monitor/rllm-analyze via tool calls.

        Handles the actual tool_response dict structures:
        - Read: {"type": "text", "file": {"filePath": ..., "content": ...}}
        - Bash: {"stdout": ..., "stderr": ..., ...}
        """
        result: Dict[str, Any] = {
            "config": None,
            "reward_trend": None,
            "perf_stats": None,
            "errors": [],
            "log_snippets": [],
        }

        for tc in traj.tool_calls:
            resp = tc.tool_response
            if not resp:
                continue

            if tc.tool_name == "Read" and isinstance(resp, dict):
                file_path = tc.tool_input.get("file_path", "")
                content = self._extract_read_content(resp)
                if not content:
                    continue

                if "config.json" in file_path:
                    try:
                        result["config"] = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        result["config"] = {"raw": content[:2000]}

                elif "perf_stats.json" in file_path:
                    try:
                        result["perf_stats"] = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        result["perf_stats"] = {"raw": content[:2000]}

            elif tc.tool_name == "Bash" and isinstance(resp, dict):
                command = tc.tool_input.get("command", "")
                stdout = resp.get("stdout", "")

                if ("training_log" in command or "tail" in command) and stdout:
                    result["log_snippets"].append(stdout[:3000])
                    rewards = self._extract_rewards_from_log(stdout)
                    if rewards:
                        result["reward_trend"] = rewards

                if stdout and ("Error" in stdout or "Traceback" in stdout):
                    result["errors"].append({
                        "command": command,
                        "error": stdout[:2000],
                    })

        return result

    @staticmethod
    def _extract_read_content(resp: Dict[str, Any]) -> Optional[str]:
        """Extract file content from Read tool response dict."""
        if resp.get("type") == "text" and "file" in resp:
            return resp["file"].get("content")
        return None

    def get_available_training_data(self, days: Optional[int] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all rllm trajectories with extracted training data."""
        trajs = self.get_rllm_trajectories(days, session_id=session_id)
        results = []
        for traj in trajs:
            summary = self.summarize_trajectory(traj)
            training_data = self.extract_training_data(traj)
            results.append({**summary, "training_data": training_data})
        return results

    def _extract_rewards_from_log(self, log_text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract step/reward pairs from training log text."""
        import re
        pattern = r'^\s*(\d+)/(\d+)\s+\d+\s+([\d.]+)'
        rewards = []
        for line in log_text.split('\n'):
            m = re.match(pattern, line)
            if m:
                rewards.append({
                    "step": int(m.group(1)),
                    "total": int(m.group(2)),
                    "reward": float(m.group(3)),
                })
        return rewards if rewards else None

    def format_for_llm(self, trajectories: List[Trajectory]) -> str:
        """Format trajectories as JSON for LLM analysis prompt."""
        summaries = [self.summarize_trajectory(t) for t in trajectories]
        return json.dumps(summaries, indent=2, ensure_ascii=False)
