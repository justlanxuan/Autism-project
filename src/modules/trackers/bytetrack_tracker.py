"""ByteTrack adapter for Autism-project.

This module wraps FoundationVision/ByteTrack as an external dependency while
exposing a stable interface for the local pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ByteTrackConfig:
	"""Configuration for initializing a ByteTrack runtime adapter."""

	# If None, adapter will import `yolox` from the active Python environment.
	repo_root: Optional[str] = "/home/fzliang/Autism-project/third-party/ByteTrack"
	expected_commit: Optional[str] = None
	strict_commit: bool = False
	frame_rate: int = 30
	track_thresh: float = 0.5
	track_buffer: int = 30
	match_thresh: float = 0.8
	mot20: bool = False


class ByteTrackTracker:
	"""Adapter around `yolox.tracker.byte_tracker.BYTETracker`.

	Input format for `update`:
	- detections: numpy array in shape [N, 5] (x1, y1, x2, y2, score)
	  or [N, 6] (x1, y1, x2, y2, objectness, class_conf).
	- frame_shape: tuple/list as (height, width).

	Output format:
	- list[dict]:
	  {"track_id": int, "bbox_xyxy": [x1, y1, x2, y2], "score": float}
	"""

	def __init__(self, config: Optional[ByteTrackConfig] = None):
		self.config = config or ByteTrackConfig()
		self._tracker = None
		self._load_runtime()

	def _load_runtime(self) -> None:
		repo_path = self._resolve_repo_path(self.config.repo_root)
		
		# Import cv2 BEFORE adding ByteTrack to path to avoid library loading issues
		import cv2  # noqa: F401
		
		if repo_path is not None:
			self._validate_commit(repo_path)
			self._ensure_import_path(repo_path)

		from yolox.tracker.byte_tracker import BYTETracker

		args = SimpleNamespace(
			track_thresh=float(self.config.track_thresh),
			track_buffer=int(self.config.track_buffer),
			match_thresh=float(self.config.match_thresh),
			mot20=bool(self.config.mot20),
		)
		self._tracker = BYTETracker(args, frame_rate=int(self.config.frame_rate))

	@staticmethod
	def _resolve_repo_path(repo_root: Optional[str]) -> Optional[Path]:
		if not repo_root:
			return None
		repo_path = Path(repo_root).expanduser().resolve()
		if not repo_path.exists():
			raise FileNotFoundError(f"ByteTrack repo not found: {repo_path}")
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
				raise RuntimeError(
					"Failed to read ByteTrack git commit: "
					f"{result.stderr.strip() or 'unknown error'}"
				)
			return

		actual = result.stdout.strip()
		expected = self.config.expected_commit.strip()
		if actual != expected:
			message = (
				f"ByteTrack commit mismatch: expected {expected}, got {actual}."
			)
			if self.config.strict_commit:
				raise RuntimeError(message)
			print(f"[ByteTrackTracker] Warning: {message}")

	@staticmethod
	def _ensure_import_path(repo_path: Path) -> None:
		repo_root = str(repo_path)
		if repo_root not in sys.path:
			sys.path.insert(0, repo_root)

	def reset(self) -> None:
		"""Reset tracker state and start a new sequence."""
		self._load_runtime()

	def update(self, detections: np.ndarray, frame_shape: Any) -> List[Dict[str, Any]]:
		"""Run one tracking update step and return normalized tracking results."""
		if self._tracker is None:
			raise RuntimeError("ByteTrack runtime is not initialized")

		det = self._normalize_detections(detections)
		height, width = self._normalize_frame_shape(frame_shape)
		online_tracks = self._tracker.update(det, (height, width), (height, width))

		results: List[Dict[str, Any]] = []
		for track in online_tracks:
			tlbr = track.tlbr.tolist() if hasattr(track.tlbr, "tolist") else list(track.tlbr)
			results.append(
				{
					"track_id": int(track.track_id),
					"bbox_xyxy": [float(v) for v in tlbr],
					"score": float(getattr(track, "score", 0.0)),
				}
			)
		return results

	@staticmethod
	def _normalize_detections(detections: np.ndarray) -> np.ndarray:
		arr = np.asarray(detections)
		if arr.ndim != 2 or arr.shape[1] not in (5, 6):
			raise ValueError(
				"detections must be shaped as [N,5] or [N,6], "
				f"got {arr.shape}"
			)
		if arr.shape[0] == 0:
			return np.zeros((0, 5), dtype=np.float32)
		return arr.astype(np.float32, copy=False)

	@staticmethod
	def _normalize_frame_shape(frame_shape: Any) -> tuple[int, int]:
		if isinstance(frame_shape, (tuple, list)) and len(frame_shape) >= 2:
			return int(frame_shape[0]), int(frame_shape[1])
		raise ValueError(
			"frame_shape must be (height, width) or [height, width], "
			f"got {frame_shape!r}"
		)


def build_bytetrack_tracker(config_dict: Optional[Dict[str, Any]] = None) -> ByteTrackTracker:
	"""Factory helper for config-driven instantiation."""
	if config_dict is None:
		return ByteTrackTracker()
	return ByteTrackTracker(ByteTrackConfig(**config_dict))
