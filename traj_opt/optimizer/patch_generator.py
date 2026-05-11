"""Patch generator — creates skill-bank patch files from optimization suggestions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import yaml

from traj_opt.adapter.schema import SkillOptimizationSuggestion
from traj_opt.config import DEFAULT_CONFIG, TrajectoryConfig


class PatchGenerator:
    """Generates skill-bank patch files from SkillOptimizationSuggestions."""

    ALLOWED_TARGET_GROUPS = {"rllm"}

    def __init__(self, config: TrajectoryConfig = DEFAULT_CONFIG):
        self.config = config

    def generate_patch(self, suggestion: SkillOptimizationSuggestion) -> Path:
        """Generate a single patch file in the skill-bank directory.

        Returns the path to the created patch file.
        Does NOT activate the patch — call accept_patch() after user confirmation.
        """
        group = self._find_group(suggestion.skill_name)
        self._validate_target_group(suggestion.skill_name, group)
        skill_dir = Path("skill-bank") / group / suggestion.skill_name

        self._validate_target_section(suggestion.skill_name, group, suggestion.target_section)

        patches_dir = skill_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        patch_id = f"traj-{timestamp}-{suggestion.target_section}"
        filename = f"{patch_id}.md"
        patch_path = patches_dir / filename

        content = self._format_patch(patch_id, suggestion)
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(content)

        return patch_path

    def generate_patches(self, suggestions: List[SkillOptimizationSuggestion]) -> List[Path]:
        """Generate patch files for all suggestions."""
        return [self.generate_patch(s) for s in suggestions]

    def accept_patch(self, skill_name: str, patch_id: str) -> None:
        """Activate a patch after user acceptance."""
        group = self._find_group(skill_name)
        skill_dir = Path("skill-bank") / group / skill_name
        self._activate_patch(skill_dir, patch_id)

    def reject_patch(self, skill_name: str, patch_id: str) -> None:
        """Remove patch file and ensure it's not in manifest."""
        group = self._find_group(skill_name)
        skill_dir = Path("skill-bank") / group / skill_name
        patch_path = skill_dir / "patches" / f"{patch_id}.md"
        if patch_path.exists():
            patch_path.unlink()
        self._deactivate_patch(skill_dir, patch_id)

    def _format_patch(self, patch_id: str, suggestion: SkillOptimizationSuggestion) -> str:
        """Format a patch in skill-bank patch format."""
        desc = self._yaml_quote(suggestion.description)
        lines = [
            "---",
            f"id: {patch_id}",
            f"target_section: {suggestion.target_section}",
            f"action: {suggestion.action}",
            f"description: {desc}",
            f"status: proposed",
            f"confidence: {suggestion.confidence}",
            f"evidence_rounds: {suggestion.evidence_rounds}",
            f"source: trajectory-analysis",
            f"source_sessions: {json.dumps(suggestion.source_sessions)}",
            "---",
            "",
            suggestion.patch_content,
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _yaml_quote(value: str) -> str:
        if any(c in value for c in ":#{}[]|>&*!%@`"):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _activate_patch(self, skill_dir: Path, patch_id: str) -> None:
        """Add patch to manifest.yaml active list."""
        manifest_path = skill_dir / "manifest.yaml"
        if not manifest_path.exists():
            return

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        active = manifest.get("active", [])
        if patch_id not in active:
            active.append(patch_id)
            manifest["active"] = active

            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False, width=120)

    def _deactivate_patch(self, skill_dir: Path, patch_id: str) -> None:
        """Remove patch from manifest.yaml active list."""
        manifest_path = skill_dir / "manifest.yaml"
        if not manifest_path.exists():
            return

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        active = manifest.get("active", [])
        if patch_id in active:
            active.remove(patch_id)
            manifest["active"] = active

            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False, width=120)

    def _validate_target_section(self, skill_name: str, group: str, target_section: str) -> None:
        """Verify target_section exists in the skill's base.md."""
        base_path = Path("skill-bank") / group / skill_name / "base.md"
        if not base_path.exists():
            return
        content = base_path.read_text()
        sections = re.findall(r'<!--\s*section:([a-z0-9-]+)\s*-->', content)
        if target_section not in sections:
            available = ", ".join(sections)
            raise ValueError(
                f"Section '{target_section}' not found in {skill_name}/base.md. "
                f"Available: {available}"
            )

    def _validate_target_group(self, skill_name: str, group: str) -> None:
        """Ensure the target skill belongs to an allowed group."""
        if group not in self.ALLOWED_TARGET_GROUPS:
            raise ValueError(
                f"Skill '{skill_name}' belongs to group '{group}', "
                f"but only {self.ALLOWED_TARGET_GROUPS} groups are allowed as optimization targets. "
                f"traj/ group skills cannot be optimized by the trajectory system."
            )

    def _find_group(self, skill_name: str) -> str:
        """Determine the skill-bank group for a skill name."""
        if skill_name.startswith("rllm-"):
            return "rllm"
        if skill_name.startswith("traj-"):
            return "traj"
        return "general"
