"""Tracker adapters used by Autism-project."""

from .alphapose_tracker import AlphaPoseTracker, AlphaPoseTrackerConfig, build_alphapose_tracker
from .bytetrack_tracker import ByteTrackConfig, ByteTrackTracker, build_bytetrack_tracker
from . import registry  # noqa: F401

