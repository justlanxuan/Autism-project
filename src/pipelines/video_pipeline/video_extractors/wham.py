"""WHAM 3D video skeleton extractor."""

from __future__ import annotations

from typing import Any, Dict

from src.modules.pose_estimators.wham_3d import WHAM3DEstimator, build_wham_3d_estimator
from src.pipelines.video_pipeline.video_extractors.base import VideoSkeletonExtractor


class WHAMExtractor(VideoSkeletonExtractor):
    """VideoSkeletonExtractor backed by WHAM 3D estimator."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        wham_cfg = {
            "repo_root": cfg.get("pose_estimator_root", "/home/fzliang/origin/WHAM"),
            "checkpoint_file": cfg.get("pose_estimator_ckpt"),
            "device": "cuda:0" if cfg.get("gpu") is not None else "cpu",
            "run_global": cfg.get("wham_run_global", True),
            "output_dir": cfg.get("results_root", "./wham_outputs"),
        }
        self.estimator = build_wham_3d_estimator(wham_cfg)

    def extract(self, video_path: str, output_dir: str) -> str:
        """Run WHAM on a video.

        Note: WHAM outputs its own format; callers should adapt if AlphaPose JSON
        is required downstream.
        """
        results = self.estimator.process_video(video_path, output_dir=output_dir)
        # WHAM does not produce a skeleton.json directly; placeholder return.
        return f"{output_dir}/wham_results.pkl"
