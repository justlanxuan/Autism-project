"""IMU-Video fusion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.builder import build_matcher, build_pose_estimator
from src.modules.matchers import IMUEncoder, VideoEncoder, build_motionbert_backbone
from src.modules.matchers.despite import DeSPITEMatcher
from src.data.imu_processor import IMUProcessor
from src.data.loaders import WHAMSkeletonConverter
from src.data.skeleton_processor import SkeletonProcessor
from src.utils.chunk_matcher import run_chunk_trials


@dataclass
class PipelineConfig:
	window_length: int = 24
	window_stride: int = 1
	device: str = "cuda"
	chunk_windows: int = 30
	min_chunk_windows: int = 15
	num_trials: int = 50
	seed: int = 42


class IMUVideoFusionPipeline:
	"""Pipeline: IMU -> embedding, video -> skeleton -> embedding -> alignment -> matching."""

	def __init__(self, config: Dict):
		self.config = config
		mm_config = config.get("multimodal", {})
		self.pipeline_config = PipelineConfig(**mm_config.get("pipeline", {}))
		self.imu_processor = IMUProcessor(config.get("imu_columns"))
		self.skeleton_processor = SkeletonProcessor(config.get("skeleton", {}))

		# Initialize encoders directly from matchers module
		# TODO: Make this configurable via mm_config
		self.imu_encoder = IMUEncoder(input_size=48, hidden_size=512, num_layers=2, device=self.pipeline_config.device)
		backbone, _ = build_motionbert_backbone("configs/pose3d/MB_ft_h36m_global_lite.yaml")
		self.video_encoder = VideoEncoder(backbone=backbone, rep_dim=512, temporal_layers=2)
		self.alignment = DeSPITEMatcher(mm_config.get("alignment", {}))
		self.matcher = build_matcher(mm_config)
		self.pose_estimator = build_pose_estimator(config.get("video_pipeline", {}))

		wham_cfg = config.get("video_pipeline", {}).get("pose_estimator", {}).get("wham_3d", {})
		self.wham_converter = WHAMSkeletonConverter({"repo_root": wham_cfg.get("repo_root", "/home/fzliang/origin/WHAM")})

	@staticmethod
	def _build_frame_timestamps(frame_ids: np.ndarray, fps: float) -> np.ndarray:
		if fps <= 0:
			fps = 30.0
		return frame_ids.astype(np.float64) / float(fps)

	@staticmethod
	def _sample_windows(sequence: np.ndarray, window_length: int, stride: int) -> np.ndarray:
		if sequence.shape[0] < window_length:
			return np.empty((0, window_length) + sequence.shape[1:], dtype=sequence.dtype)
		windows = []
		for start in range(0, sequence.shape[0] - window_length + 1, stride):
			windows.append(sequence[start : start + window_length])
		return np.stack(windows, axis=0)

	def _encode_windows(self, encoder, windows: np.ndarray) -> np.ndarray:
		if windows.size == 0:
			return np.zeros((0, 1), dtype=np.float32)
		embeddings = encoder.encode(windows).detach().cpu().numpy()
		return embeddings

	def _load_imu_streams(self, imu_csv_paths: List[str]) -> List[Dict[str, np.ndarray]]:
		streams = []
		for path in imu_csv_paths:
			imu_raw = self.imu_processor.load_csv(path)
			imu_48d = self.imu_processor.to_48d(imu_raw)
			streams.append({
				"timestamp": imu_raw["timestamp"].astype(np.float64) / 1000.0,
				"features": imu_48d,
			})
		return streams

	def _video_to_skeletons(self, video_path: str, fps: float) -> Dict[int, Dict[str, np.ndarray]]:
		name = self.config.get("video_pipeline", {}).get("pose_estimator", {}).get("name", "alphapose")
		if name == "wham_3d":
			outputs = self.pose_estimator(video_path)
			results_3d = outputs["results_3d"]
			skeletons = self.wham_converter.convert_results(results_3d)
			for pid, info in skeletons.items():
				info["timestamps"] = self._build_frame_timestamps(info["frame_id"], fps)
				info["joints"] = self.skeleton_processor.pad_or_trim(info["joints"])
			return skeletons

		raise ValueError(
			"Unsupported pose_estimator for pipeline. "
			"Currently only 'wham_3d' is supported without external detectors."
		)

	def run(self, video_path: str, imu_csv_paths: List[str], fps: float = 30.0) -> Dict:
		imu_streams = self._load_imu_streams(imu_csv_paths)
		skeletons = self._video_to_skeletons(video_path, fps)

		imu_windows: Dict[int, np.ndarray] = {}
		video_windows: Dict[int, np.ndarray] = {}

		for idx, imu_stream in enumerate(imu_streams):
			imu_windows[idx] = self._sample_windows(
				imu_stream["features"],
				self.pipeline_config.window_length,
				self.pipeline_config.window_stride,
			)

		for person_id, info in skeletons.items():
			joints = info["joints"]
			video_windows[person_id] = self._sample_windows(
				joints,
				self.pipeline_config.window_length,
				self.pipeline_config.window_stride,
			)

		imu_embeddings = {k: self._encode_windows(self.imu_encoder, v) for k, v in imu_windows.items()}
		video_embeddings = {k: self._encode_windows(self.video_encoder, v) for k, v in video_windows.items()}

		similarity = self.alignment.similarity_matrix(imu_embeddings, video_embeddings)
		match_result = self.matcher.match(similarity, list(imu_embeddings.keys()), list(video_embeddings.keys()))

		chunk_trials = None
		if len(imu_embeddings) == 2 and len(video_embeddings) == 2:
			imu_list = [imu_embeddings[k] for k in sorted(imu_embeddings.keys())]
			video_list = [video_embeddings[k] for k in sorted(video_embeddings.keys())]
			chunk_trials = run_chunk_trials(
				imu_list,
				video_list,
				chunk_windows=self.pipeline_config.chunk_windows,
				min_chunk_windows=self.pipeline_config.min_chunk_windows,
				num_trials=self.pipeline_config.num_trials,
				seed=self.pipeline_config.seed,
			)

		return {
			"similarity": similarity,
			"matches": match_result,
			"chunk_trials": chunk_trials,
			"imu_embeddings": imu_embeddings,
			"video_embeddings": video_embeddings,
		}
