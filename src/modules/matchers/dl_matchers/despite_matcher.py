"""DeSPITE-style alignment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from src.modules.matchers.base import BaseMatcher
from src.modules.matchers.losses import SymmetricInfoNCE


@dataclass
class AlignmentConfig:
    temperature: float = 0.1
    learn_temperature: bool = False
    device: str = "cuda"


class DeSPITEMatcher(BaseMatcher):
    """DeSPITE-style matcher with symmetric InfoNCE and cosine similarity."""

    def __init__(self, config_dict: Dict):
        self.config = AlignmentConfig(**config_dict)
        self._loss_fn = SymmetricInfoNCE(
            temperature=self.config.temperature,
            learn_temperature=self.config.learn_temperature,
        )

    def match(
        self,
        similarity_matrix: Any,
        imu_ids: Any = None,
        person_ids: Any = None,
    ) -> Dict[str, Any]:
        """Compute matching based on similarity matrix."""
        return {
            "similarity": similarity_matrix,
            "config": self.config,
        }

    def align(self, imu_embeddings: Any, video_embeddings: Any) -> Dict[str, Any]:
        """Compute loss and similarity for training batches."""
        if not isinstance(imu_embeddings, torch.Tensor):
            imu_embeddings = torch.as_tensor(imu_embeddings, dtype=torch.float32)
        if not isinstance(video_embeddings, torch.Tensor):
            video_embeddings = torch.as_tensor(video_embeddings, dtype=torch.float32)

        imu_embeddings = imu_embeddings.to(self.config.device)
        video_embeddings = video_embeddings.to(self.config.device)
        loss = self._loss_fn(imu_embeddings, video_embeddings)
        sim = torch.matmul(
            torch.nn.functional.normalize(imu_embeddings, dim=-1),
            torch.nn.functional.normalize(video_embeddings, dim=-1).t(),
        )
        return {
            "loss": loss,
            "similarity": sim,
        }

    @staticmethod
    def similarity_matrix(
        imu_windows: Dict[int, np.ndarray],
        video_windows: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Compute mean cosine similarity across aligned windows."""
        imu_ids = list(imu_windows.keys())
        vid_ids = list(video_windows.keys())
        sim = np.zeros((len(imu_ids), len(vid_ids)), dtype=np.float32)

        for i, imu_id in enumerate(imu_ids):
            imu_emb = imu_windows[imu_id]
            for j, vid_id in enumerate(vid_ids):
                vid_emb = video_windows[vid_id]
                n = min(len(imu_emb), len(vid_emb))
                if n == 0:
                    sim[i, j] = 0.0
                    continue
                imu_clip = imu_emb[:n]
                vid_clip = vid_emb[:n]
                denom = np.linalg.norm(imu_clip, axis=1) * np.linalg.norm(vid_clip, axis=1)
                denom = np.where(denom == 0, 1.0, denom)
                cos = (imu_clip * vid_clip).sum(axis=1) / denom
                sim[i, j] = float(np.mean(cos))
        return sim
