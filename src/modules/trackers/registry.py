"""Registration helpers for trackers."""

from __future__ import annotations

from src.core.registry import TRACKER_REGISTRY

from .alphapose_tracker import build_alphapose_tracker
from .bytetrack_tracker import build_bytetrack_tracker


@TRACKER_REGISTRY.register("alphapose")
def _build_alphapose(config_dict):
    return build_alphapose_tracker(config_dict)


@TRACKER_REGISTRY.register("bytetrack")
def _build_bytetrack(config_dict):
    return build_bytetrack_tracker(config_dict)
