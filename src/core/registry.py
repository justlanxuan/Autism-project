"""Global registries for pipeline components."""

from __future__ import annotations

from src.utils.factory import Registry

DETECTORS = Registry()
TRACKERS = Registry()
POSE_ESTIMATORS = Registry()
VIDEO_EXTRACTORS = Registry()
