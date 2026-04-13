"""Skeleton preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SkeletonConfig:
    target_joints: int = 24


class SkeletonPreprocessor:
    """Normalize and reshape skeleton sequences."""

    def __init__(self, config_dict: Dict | None = None):
        self.config = SkeletonConfig(**(config_dict or {}))

    def pad_or_trim(self, skeleton: np.ndarray) -> np.ndarray:
        if skeleton.shape[-2] == self.config.target_joints:
            return skeleton
        if skeleton.shape[-2] > self.config.target_joints:
            return skeleton[..., : self.config.target_joints, :]
        pad = self.config.target_joints - skeleton.shape[-2]
        pad_shape = list(skeleton.shape)
        pad_shape[-2] = pad
        padding = np.zeros(pad_shape, dtype=skeleton.dtype)
        return np.concatenate([skeleton, padding], axis=-2)
