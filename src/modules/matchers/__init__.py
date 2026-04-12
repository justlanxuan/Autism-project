"""Matchers module for cross-modal association and alignment."""

from src.modules.matchers.imu_video_matcher import (
    IMUEncoder,
    IMUVideoMatcher,
    VideoEncoder,
    build_motionbert_backbone,
    load_despite_imu_weights,
    load_motionbert_checkpoint,
)
from src.modules.matchers.losses import SymmetricInfoNCE, retrieval_top1

__all__ = [
    # IMU-Video matching
    "IMUEncoder",
    "VideoEncoder",
    "IMUVideoMatcher",
    "build_motionbert_backbone",
    "load_motionbert_checkpoint",
    "load_despite_imu_weights",
    # Losses
    "SymmetricInfoNCE",
    "retrieval_top1",
    # Base
    "BaseMatcher",
    "HungarianMatcher",
]

# Import base classes
from src.modules.matchers.base import BaseMatcher
from src.modules.matchers.hungarian import HungarianMatcher
