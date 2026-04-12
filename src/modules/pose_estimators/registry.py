"""Registration helpers for pose estimators."""

from __future__ import annotations

from src.core.registry import POSE_ESTIMATOR_REGISTRY

from .alphapose_pose_estimator import build_alphapose_pose_estimator
from .wham_3d_estimator import build_wham_3d_estimator


@POSE_ESTIMATOR_REGISTRY.register("alphapose")
def _build_alphapose(config_dict):
    return build_alphapose_pose_estimator(config_dict)


@POSE_ESTIMATOR_REGISTRY.register("wham_3d")
def _build_wham_3d(config_dict):
    return build_wham_3d_estimator(config_dict)
