"""Round state coordination between CLI-1 (training) and CLI-2 (optimization).

Provides atomic read/write of per-round status files used to coordinate
the dual-CLI architecture. See docs/trajectory-design.md Section 18.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RoundState:
    """Per-round status file manager. Used by both CLI-1 and CLI-2."""

    def __init__(self, base_dir: str = "trajectory/output/rounds"):
        self.base_dir = Path(base_dir)

    def _status_path(self, round_num: int) -> Path:
        return self.base_dir / f"round_{round_num}" / "status.json"

    def _atomic_write(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.rename(tmp_path, str(path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def write_training_complete(
        self,
        round_num: int,
        run_id: str,
        reward: float,
        session_id: str,
        run_ids: Optional[List[str]] = None,
        success: bool = True,
    ) -> str:
        """CLI-1: write status after training completes. Returns status file path."""
        now = datetime.now(timezone.utc).isoformat()
        path = self._status_path(round_num)

        data = self.read_status(round_num) or {}
        data.update({
            "round": round_num,
            "status": "training_complete" if success else "training_failed",
            "training": {
                "run_id": run_id,
                "run_ids": run_ids or [run_id],
                "session_id": session_id,
                "reward": reward,
                "success": success,
                "completed_at": now,
            },
            "updated_at": now,
        })
        data.setdefault("optimization", None)
        data.setdefault("created_at", now)

        self._atomic_write(path, data)
        return str(path)

    def write_training_failed(
        self,
        round_num: int,
        run_id: str,
        error: str,
        session_id: str,
        run_ids: Optional[List[str]] = None,
    ) -> str:
        """CLI-1: write status when training fails."""
        return self.write_training_complete(
            round_num=round_num,
            run_id=run_id,
            reward=0.0,
            session_id=session_id,
            run_ids=run_ids,
            success=False,
        )

    def write_optimization_complete(
        self,
        round_num: int,
        report_path: str,
        patches_generated: int,
        patches_accepted: int,
    ) -> str:
        """CLI-2: update status after optimization completes. Returns status file path."""
        now = datetime.now(timezone.utc).isoformat()
        path = self._status_path(round_num)

        data = self.read_status(round_num)
        if not data:
            raise FileNotFoundError(
                f"Round {round_num} status not found. CLI-1 must write training status first."
            )

        data["status"] = "optimization_complete"
        data["optimization"] = {
            "report_path": report_path,
            "patches_generated": patches_generated,
            "patches_accepted": patches_accepted,
            "completed_at": now,
        }
        data["updated_at"] = now

        self._atomic_write(path, data)
        return str(path)

    def read_status(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Read status for a specific round. Returns None if not found."""
        path = self._status_path(round_num)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def find_latest_round(self) -> Optional[int]:
        """Find the highest round number that has a status file."""
        if not self.base_dir.exists():
            return None
        rounds = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("round_"):
                try:
                    rounds.append(int(d.name.split("_", 1)[1]))
                except (ValueError, IndexError):
                    continue
        return max(rounds) if rounds else None

    def find_pending_optimization(self) -> Optional[int]:
        """Find a round with status=training_complete (not yet optimized)."""
        if not self.base_dir.exists():
            return None
        for d in sorted(self.base_dir.iterdir(), reverse=True):
            if not d.is_dir() or not d.name.startswith("round_"):
                continue
            status_file = d / "status.json"
            if not status_file.exists():
                continue
            with open(status_file) as f:
                data = json.load(f)
            if data.get("status") == "training_complete":
                return data.get("round")
        return None

    def find_pending_training(self) -> Optional[int]:
        """Find the next round number after the latest optimization_complete."""
        latest = self.find_latest_round()
        if latest is None:
            return 1
        data = self.read_status(latest)
        if data and data.get("status") == "optimization_complete":
            return latest + 1
        return None

    def list_rounds(self) -> List[Dict[str, Any]]:
        """List all rounds with their status. For display purposes."""
        if not self.base_dir.exists():
            return []
        results = []
        for d in sorted(self.base_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("round_"):
                continue
            status_file = d / "status.json"
            if status_file.exists():
                with open(status_file) as f:
                    results.append(json.load(f))
        return results
