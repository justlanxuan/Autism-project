"""Pose-estimator adapters used by Autism-project."""

from .alphapose_pose_estimator import AlphaPoseConfig, AlphaPosePoseEstimator, build_alphapose_pose_estimator
from . import registry  # noqa: F401

try:
	from .motionbert_pose_estimator import MotionBERTPoseEstimator  # type: ignore
except Exception:
	MotionBERTPoseEstimator = None

try:
	from .wham_pose_estimator import WHAMPoseEstimator  # type: ignore
except Exception:
	WHAMPoseEstimator = None


