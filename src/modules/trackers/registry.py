"""Tracker registry."""

from src.utils.factory import Registry

TRACKER_REGISTRY = Registry()

from src.modules.trackers.bytetrack import build_bytetrack_tracker
from src.modules.trackers.alphapose import build_alphapose_tracker

TRACKER_REGISTRY.register("bytetrack")(build_bytetrack_tracker)
TRACKER_REGISTRY.register("alphapose")(build_alphapose_tracker)
