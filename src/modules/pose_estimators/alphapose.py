"""AlphaPose pose-estimation adapter for Autism-project.

This adapter does not modify AlphaPose source code. It imports AlphaPose as an
external dependency and runs bbox-to-skeleton inference directly, bypassing the
demo script and pose_nms writer path.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class AlphaPoseConfig:
    """Configuration for loading AlphaPose from an external repository."""

    repo_root: str = "/home/fzliang/origin/AlphaPose"
    cfg_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    device: str = "cuda:0"
    pose_batch_size: int = 32
    use_flip: bool = False
    image_is_bgr: bool = False
    expected_commit: Optional[str] = None
    strict_commit: bool = False


class AlphaPosePoseEstimator:
    """Run AlphaPose keypoint inference on externally provided bounding boxes."""

    def __init__(self, config: Optional[AlphaPoseConfig] = None):
        self.config = config or AlphaPoseConfig()
        self.repo_path = self._resolve_repo_path(self.config.repo_root)
        self._validate_commit(self.repo_path)
        self._ensure_import_path(self.repo_path)

        self.device = self._resolve_device(self.config.device)
        self.cfg = None
        self.pose_model = None
        self.pose_dataset = None
        self.transformation = None
        self.heatmap_to_coord = None
        self.flip = None
        self.flip_heatmap = None

        self._load_runtime()

    @staticmethod
    def _resolve_repo_path(repo_root: str) -> Path:
        repo_path = Path(repo_root).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"AlphaPose repo not found: {repo_path}")
        return repo_path

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        import torch

        if device_name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _validate_commit(self, repo_path: Path) -> None:
        if not self.config.expected_commit:
            return

        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            if self.config.strict_commit:
                raise RuntimeError(
                    "Failed to read AlphaPose git commit: "
                    f"{result.stderr.strip() or 'unknown error'}"
                )
            return

        actual = result.stdout.strip()
        expected = self.config.expected_commit.strip()
        if actual != expected:
            message = (
                f"AlphaPose commit mismatch: expected {expected}, got {actual}."
            )
            if self.config.strict_commit:
                raise RuntimeError(message)
            print(f"[AlphaPosePoseEstimator] Warning: {message}")

    @staticmethod
    def _ensure_import_path(repo_path: Path) -> None:
        repo_root = str(repo_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _load_runtime(self) -> None:
        if not self.config.cfg_file:
            raise ValueError("AlphaPoseConfig.cfg_file is required")
        if not self.config.checkpoint_file:
            raise ValueError("AlphaPoseConfig.checkpoint_file is required")

        import torch

        from alphapose.models import builder
        from alphapose.utils.config import update_config
        from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
        from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord

        self.flip = flip
        self.flip_heatmap = flip_heatmap

        self.cfg = update_config(self.config.cfg_file)
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        checkpoint = torch.load(self.config.checkpoint_file, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.pose_model.load_state_dict(state_dict, strict=False)
        self.pose_model.to(self.device)
        self.pose_model.eval()

        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.transformation = self._build_transformation(
            SimpleTransform=SimpleTransform,
            SimpleTransform3DSMPL=SimpleTransform3DSMPL,
        )
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)

    def _build_transformation(self, SimpleTransform, SimpleTransform3DSMPL):
        import torch

        input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        output_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        sigma = self.cfg.DATA_PRESET.SIGMA

        if self.cfg.DATA_PRESET.TYPE == "simple":
            return SimpleTransform(
                self.pose_dataset,
                scale_factor=0,
                input_size=input_size,
                output_size=output_size,
                rot=0,
                sigma=sigma,
                train=False,
                add_dpg=False,
                gpu_device=self.device,
            )

        if self.cfg.DATA_PRESET.TYPE == "simple_smpl":
            from easydict import EasyDict as edict

            dummy_set = edict(
                {
                    "joint_pairs_17": None,
                    "joint_pairs_24": None,
                    "joint_pairs_29": None,
                    "bbox_3d_shape": (2.2, 2.2, 2.2),
                }
            )
            return SimpleTransform3DSMPL(
                dummy_set,
                scale_factor=self.cfg.DATASET.SCALE_FACTOR,
                color_factor=self.cfg.DATASET.COLOR_FACTOR,
                occlusion=self.cfg.DATASET.OCCLUSION,
                input_size=self.cfg.MODEL.IMAGE_SIZE,
                output_size=self.cfg.MODEL.HEATMAP_SIZE,
                depth_dim=self.cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2.2, 2.2),
                rot=self.cfg.DATASET.ROT_FACTOR,
                sigma=self.cfg.MODEL.EXTRA.SIGMA,
                train=False,
                add_dpg=False,
                gpu_device=self.device,
                loss_type=self.cfg.LOSS["TYPE"],
            )

        raise ValueError(f"Unsupported AlphaPose preset type: {self.cfg.DATA_PRESET.TYPE}")

    @staticmethod
    def _normalize_boxes(boxes: Any) -> np.ndarray:
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"boxes must be shaped as [N, 4], got {arr.shape}")
        return arr

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"image must be HWC, got {arr.shape}")
        if self.config.image_is_bgr:
            return arr[:, :, ::-1].copy()
        return arr.copy()

    def reset(self) -> None:
        """Stateless adapter; provided for interface symmetry."""

    def estimate(
        self,
        image: np.ndarray,
        boxes: Any,
        track_ids: Optional[Sequence[Optional[int]]] = None,
        box_scores: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Estimate poses for a set of bounding boxes.

        Args:
            image: Input frame in HWC format.
            boxes: Bounding boxes in xyxy format, shape [N, 4].
            track_ids: Optional track ids aligned with `boxes`.
            box_scores: Optional detection scores aligned with `boxes`.

        Returns:
            A list of dictionaries, one per input box, preserving order.
        """

        orig_img = self._prepare_image(image)
        box_array = self._normalize_boxes(boxes)
        if box_array.shape[0] == 0:
            return []

        if track_ids is None:
            track_ids = [None] * box_array.shape[0]
        if box_scores is None:
            box_scores = [None] * box_array.shape[0]

        inps, cropped_boxes = self._build_pose_inputs(orig_img, box_array)
        heatmaps = self._forward_pose_model(inps)
        return self._decode_results(
            orig_img=orig_img,
            boxes=box_array,
            cropped_boxes=cropped_boxes,
            heatmaps=heatmaps,
            track_ids=track_ids,
            box_scores=box_scores,
        )

    def __call__(
        self,
        image: np.ndarray,
        boxes: Any,
        track_ids: Optional[Sequence[Optional[int]]] = None,
        box_scores: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        return self.estimate(image, boxes, track_ids=track_ids, box_scores=box_scores)

    def _build_pose_inputs(self, orig_img: np.ndarray, boxes: np.ndarray):
        input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        inps = torch.zeros(boxes.shape[0], 3, *input_size)
        cropped_boxes = torch.zeros(boxes.shape[0], 4)
        for idx, box in enumerate(boxes):
            inps[idx], cropped_box = self.transformation.test_transform(orig_img, box)
            cropped_boxes[idx] = torch.as_tensor(cropped_box, dtype=torch.float32)
        return inps, cropped_boxes

    def _forward_pose_model(self, inps: torch.Tensor) -> torch.Tensor:
        import torch

        batch_size = max(1, int(self.config.pose_batch_size))
        if self.config.use_flip:
            batch_size = max(1, batch_size // 2)

        datalen = inps.size(0)
        leftover = 1 if datalen % batch_size else 0
        num_batches = datalen // batch_size + leftover

        heatmap_chunks = []
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch = inps[batch_idx * batch_size : min((batch_idx + 1) * batch_size, datalen)]
                batch = batch.to(self.device)
                if self.config.use_flip:
                    batch = torch.cat((batch, self.flip(batch)))
                hm = self.pose_model(batch)
                if self.config.use_flip:
                    hm_flip = self.flip_heatmap(
                        hm[int(len(hm) / 2) :],
                        self.pose_dataset.joint_pairs,
                        shift=True,
                    )
                    hm = (hm[0 : int(len(hm) / 2)] + hm_flip) / 2
                heatmap_chunks.append(hm)
        return torch.cat(heatmap_chunks)

    @staticmethod
    def _select_eval_joints(num_joints: int) -> List[int]:
        if num_joints in (136, 133):
            return list(range(num_joints))
        if num_joints == 68:
            return list(range(68))
        if num_joints in (26, 21):
            return list(range(num_joints))
        return list(range(num_joints))

    def _decode_results(
        self,
        orig_img: np.ndarray,
        boxes: np.ndarray,
        cropped_boxes: torch.Tensor,
        heatmaps: torch.Tensor,
        track_ids: Sequence[Optional[int]],
        box_scores: Sequence[Optional[float]],
    ) -> List[Dict[str, Any]]:
        norm_type = self.cfg.LOSS.get("NORM_TYPE", None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        eval_joints = self._select_eval_joints(heatmaps.size(1))

        results: List[Dict[str, Any]] = []
        for idx in range(heatmaps.shape[0]):
            bbox = cropped_boxes[idx].tolist()
            if isinstance(self.heatmap_to_coord, list):
                face_hand_num = 110 if heatmaps.size(1) != 68 else 42
                pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                    heatmaps[idx][eval_joints[:-face_hand_num]],
                    bbox,
                    hm_shape=hm_size,
                    norm_type=norm_type,
                )
                pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                    heatmaps[idx][eval_joints[-face_hand_num:]],
                    bbox,
                    hm_shape=hm_size,
                    norm_type=norm_type,
                )
                pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
            else:
                pose_coord, pose_score = self.heatmap_to_coord(
                    heatmaps[idx][eval_joints],
                    bbox,
                    hm_shape=hm_size,
                    norm_type=norm_type,
                )

            results.append(
                {
                    "track_id": None if track_ids[idx] is None else int(track_ids[idx]),
                    "bbox_xyxy": [float(v) for v in boxes[idx].tolist()],
                    "bbox_score": None if box_scores[idx] is None else float(box_scores[idx]),
                    "keypoints_xy": pose_coord.tolist() if hasattr(pose_coord, "tolist") else pose_coord,
                    "keypoints_score": pose_score.tolist() if hasattr(pose_score, "tolist") else pose_score,
                    "pose_score": float(np.mean(pose_score)) if len(pose_score) else 0.0,
                }
            )
        return results


def build_alphapose_pose_estimator(
    config_dict: Optional[Dict[str, Any]] = None,
) -> AlphaPosePoseEstimator:
    """Factory helper for config-driven instantiation."""

    if config_dict is None:
        return AlphaPosePoseEstimator()
    return AlphaPosePoseEstimator(AlphaPoseConfig(**config_dict))
