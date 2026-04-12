"""AlphaPose tracker adapter for Autism-project.

This wraps AlphaPose's original tracker implementation without modifying the
AlphaPose source tree.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional


@dataclass
class AlphaPoseTrackerConfig:
    repo_root: str = "/home/fzliang/origin/AlphaPose"
    expected_commit: Optional[str] = None
    strict_commit: bool = False
    # Optional override for AlphaPose tracker options.
    tracker_cfg: Optional[Any] = None
    device: str = "cuda:0"
    gpus: Optional[list[int]] = None


class AlphaPoseTracker:
    def __init__(self, config: Optional[AlphaPoseTrackerConfig] = None):
        self.config = config or AlphaPoseTrackerConfig()
        self.repo_path = self._resolve_repo_path(self.config.repo_root)
        self._validate_commit(self.repo_path)
        self._ensure_import_path(self.repo_path)

        from trackers.tracker_api import Tracker
        from trackers.tracker_cfg import cfg as tcfg

        tracker_cfg = self.config.tracker_cfg or tcfg
        runtime_args = SimpleNamespace(
            device=self.config.device,
            gpus=self.config.gpus or ([0] if self.config.device.startswith("cuda") else [-1]),
        )
        self._tracker = Tracker(tracker_cfg, runtime_args)

    @staticmethod
    def _resolve_repo_path(repo_root: str) -> Path:
        repo_path = Path(repo_root).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"AlphaPose repo not found: {repo_path}")
        return repo_path

    def _validate_commit(self, repo_path: Path) -> None:
        if not self.config.expected_commit:
            return
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if self.config.strict_commit:
                raise RuntimeError(result.stderr.strip() or "Failed to read AlphaPose commit")
            return
        actual = result.stdout.strip()
        expected = self.config.expected_commit.strip()
        if actual != expected:
            message = f"AlphaPose commit mismatch: expected {expected}, got {actual}."
            if self.config.strict_commit:
                raise RuntimeError(message)
            print(f"[AlphaPoseTracker] Warning: {message}")

    @staticmethod
    def _ensure_import_path(repo_path: Path) -> None:
        repo_root = str(repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def reset(self) -> None:
        """AlphaPose's original tracker is sequence-local; rebuild it by re-init."""
        from trackers.tracker_api import Tracker
        from trackers.tracker_cfg import cfg as tcfg

        tracker_cfg = self.config.tracker_cfg or tcfg
        runtime_args = SimpleNamespace(
            device=self.config.device,
            gpus=self.config.gpus or ([0] if self.config.device.startswith("cuda") else [-1]),
        )
        self._tracker = Tracker(tracker_cfg, runtime_args)

    def update(self, *args: Any, **kwargs: Any):
        return self._tracker.update(*args, **kwargs)


def build_alphapose_tracker(config_dict: Optional[Dict[str, Any]] = None) -> AlphaPoseTracker:
    if config_dict is None:
        return AlphaPoseTracker()
    return AlphaPoseTracker(AlphaPoseTrackerConfig(**config_dict))
