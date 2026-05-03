from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class TrajectoryConfig:
    output_dir: str = "traj_opt/output"
    layer: str = "rllm"

    capture_all_tools: bool = True
    core_tools: List[str] = field(default_factory=lambda: [
        "Bash", "Read", "Edit", "Write", "Skill", "Agent",
        "AskUserQuestion", "EnterPlanMode",
    ])

    default_segmenter: str = "default"
    file_affinity_threshold: float = 0.8

    analysis_lookback_days: int = 7
    min_trajectories_for_analysis: int = 5

    @property
    def raw_dir(self) -> Path:
        return Path(self.output_dir) / self.layer / "raw"

    @property
    def trajectories_dir(self) -> Path:
        return Path(self.output_dir) / self.layer / "trajectories"

    @property
    def reports_dir(self) -> Path:
        return Path(self.output_dir) / self.layer / "reports"

    @property
    def index_path(self) -> Path:
        return Path(self.output_dir) / "index.jsonl"

    @property
    def agent_progress_dir(self) -> Path:
        return Path(self.output_dir) / "agent_progress"

    @property
    def rounds_dir(self) -> Path:
        return Path(self.output_dir) / "rounds"


DEFAULT_CONFIG = TrajectoryConfig()
