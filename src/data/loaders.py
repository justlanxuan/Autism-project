"""Data loaders for skeletons and WHAM outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class WHAMConverterConfig:
    repo_root: str = "/home/fzliang/origin/WHAM"
    num_joints: int = 24


class WHAMSkeletonConverter:
    """Convert WHAM outputs (verts) into skeleton joints."""

    def __init__(self, config_dict: Dict | None = None):
        self.config = WHAMConverterConfig(**(config_dict or {}))
        regressor_path = Path(self.config.repo_root) / "dataset/body_models/J_regressor_wham.npy"
        if not regressor_path.exists():
            raise FileNotFoundError(f"WHAM joint regressor not found: {regressor_path}")
        self.j_regressor = np.load(regressor_path).astype(np.float32)

    def verts_to_joints(self, verts: np.ndarray) -> np.ndarray:
        joints31 = np.einsum("jv,tvc->tjc", self.j_regressor, verts)
        return joints31[:, : self.config.num_joints, :]

    def convert_results(self, results_3d: Dict) -> Dict[int, Dict[str, np.ndarray]]:
        skeletons = {}
        for person_id, pred in results_3d.items():
            verts = pred.get("verts_cam") or pred.get("verts")
            if verts is None:
                continue
            frame_ids = np.asarray(pred.get("frame_id", np.arange(len(verts))), dtype=np.int64)
            joints = self.verts_to_joints(np.asarray(verts, dtype=np.float32))
            skeletons[int(person_id)] = {
                "frame_id": frame_ids,
                "joints": joints,
            }
        return skeletons
