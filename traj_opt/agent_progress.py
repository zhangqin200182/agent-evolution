"""
Agent Progress Tracker — 实现子 agent 进度的文件轮询机制

子 agent 将进度写入 traj_opt/output/agent_progress/{session_id}.json
父对话（traj-loop）定期读取并显示进度。

详见 docs/trajectory-design.md Section 17.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class AgentProgressTracker:
    """
    子 agent 进度跟踪器。子 agent 写入，父对话轮询。

    用法:
        # 子 agent 端
        tracker = AgentProgressTracker(progress_dir)
        tracker.start(agent_id, description, session_id)
        # Phase 完成后
        tracker.complete_phase("phase_0_clarify", duration_s=12)
        # 训练中定期更新
        tracker.update_step(5, 64, 0.625, speed=78.0, eta="14m")
        # 全部完成
        tracker.complete({"run_id": "run_xxx", "reward": 0.766, "success": True})

        # 父对话端
        tracker = AgentProgressTracker(progress_dir, session_id)
        data = tracker.wait_for_completion(timeout_s=3600, poll_interval=30)
        if data["status"] == "completed":
            result = data["result"]
    """

    def __init__(self, progress_dir: str, session_id: Optional[str] = None):
        self.progress_dir = Path(progress_dir)
        self._session_id = session_id
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        agent_id: str,
        description: str,
        session_id: Optional[str] = None,
    ) -> str:
        """
        初始化进度文件。返回进度文件路径。

        Args:
            agent_id: 子 agent 的 ID（来自 Agent 工具返回）
            description: 任务描述（如 "rllm-train round 1"）
            session_id: 进度文件 ID（用于文件名）。默认与 agent_id 相同。

        Returns:
            进度文件路径
        """
        self._session_id = session_id or agent_id
        path = self._path()

        data = {
            "agent_id": agent_id,
            "description": description,
            "session_id": self._session_id,
            "status": "init",
            "started_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "phase": None,
            "progress": {},
            "latest_update": "Agent 已启动",
            "result": None,
            "error": None,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def update_phase(self, phase: str, status: str = "running"):
        """
        更新当前 phase。

        Args:
            phase: phase 名称（如 "phase_2_run"）
            status: "running" | "completed" | "failed"
        """
        self._update(
            {"phase": phase, "latest_update": f"Phase: {phase} ({status})"}
        )

    def complete_phase(self, phase: str, duration_s: int):
        """
        标记 phase 完成。

        Args:
            phase: phase 名称
            duration_s: 耗时（秒）
        """
        data = self._read()
        progress = data.get("progress", {})
        progress[phase] = {"status": "completed", "duration_s": duration_s}
        self._update(
            {
                "progress": progress,
                "latest_update": f"Phase {phase} 完成 ({duration_s}s)",
            }
        )

    def update_step(
        self,
        step: int,
        total: int,
        reward: float,
        speed: Optional[float] = None,
        eta: Optional[str] = None,
    ):
        """
        更新训练步骤进度。

        Args:
            step: 当前 step
            total: 总 step 数
            reward: 当前 reward
            speed: 速度（tok/s）
            eta: 预计剩余时间
        """
        data = self._read()
        progress = data.get("progress", {})

        if "phase_2_run" not in progress:
            progress["phase_2_run"] = {}
        progress["phase_2_run"].update(
            {
                "status": "running",
                "step": step,
                "total_steps": total,
                "reward": reward,
            }
        )

        update = f"Step {step}/{total}, reward={reward:.3f}"
        if speed is not None:
            update += f", speed={speed:.1f} tok/s"
        if eta is not None:
            update += f", ETA {eta}"

        self._update(
            {
                "progress": progress,
                "phase": "phase_2_run",
                "latest_update": update,
            }
        )

    def complete(self, result: dict):
        """
        标记整个 agent 完成（成功）。

        Args:
            result: 最终结果，必须包含 run_id, reward, success
        """
        self._update(
            {
                "status": "completed",
                "result": result,
                "latest_update": "训练完成",
            }
        )

    def fail(self, error: str):
        """
        标记 agent 失败。

        Args:
            error: 错误信息
        """
        self._update(
            {
                "status": "failed",
                "error": error,
                "latest_update": f"失败: {error}",
            }
        )

    def read(self) -> dict:
        """
        读取当前进度。

        Returns:
            进度文件内容。如果文件不存在，返回 {"status": "not_found"}
        """
        return self._read()

    def wait_for_completion(
        self,
        timeout_s: int = 3600,
        poll_interval: int = 30,
    ) -> dict:
        """
        轮询直到完成或超时。

        Args:
            timeout_s: 超时时间（秒），默认 1 小时
            poll_interval: 轮询间隔（秒），默认 30 秒

        Returns:
            最终状态（completed / failed / timeout）
        """
        start = time.time()
        while time.time() - start < timeout_s:
            data = self._read()
            status = data.get("status")
            if status in ("completed", "failed"):
                return data
            time.sleep(poll_interval)
        # 超时后返回当前状态
        return self._read()

    def _path(self) -> Path:
        assert self._session_id is not None, (
            "session_id not set. Call start() first or pass session_id to constructor."
        )
        return self.progress_dir / f"{self._session_id}.json"

    def _read(self) -> dict:
        path = self._path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {"status": "not_found"}

    def _update(self, updates: dict):
        data = self._read()
        data.update(updates)
        data["updated_at"] = datetime.now().isoformat()
        with open(self._path(), "w") as f:
            json.dump(data, f, indent=2)


def get_default_progress_dir() -> Path:
    """返回默认的进度文件目录路径。"""
    return Path("traj_opt/output/agent_progress")