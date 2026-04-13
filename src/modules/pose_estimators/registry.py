"""Pose estimator registry."""

from src.utils.factory import Registry

POSE_ESTIMATOR_REGISTRY = Registry()

from src.modules.pose_estimators.alphapose import build_alphapose_pose_estimator
from src.modules.pose_estimators.wham_3d import build_wham_3d_estimator

POSE_ESTIMATOR_REGISTRY.register("alphapose")(build_alphapose_pose_estimator)
POSE_ESTIMATOR_REGISTRY.register("wham_3d")(build_wham_3d_estimator)
