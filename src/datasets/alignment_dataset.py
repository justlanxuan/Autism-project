"""Window-level IMU/Video alignment dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowAlignmentDataset(Dataset):
    """Window-level IMU/Video alignment dataset from CSV index."""

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path | None = None,
        imu_mean: Optional[np.ndarray] = None,
        imu_std: Optional[np.ndarray] = None,
        imu_sensor: Optional[str] = "R_LowArm",
        repeat_single_sensor: int = 4,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.root_dir = Path(root_dir) if root_dir is not None else self.csv_path.parent
        self.rows = self._read_rows(self.csv_path)
        self._cache: Dict[Path, Dict[str, np.ndarray]] = {}
        self.imu_mean = imu_mean.astype(np.float32) if imu_mean is not None else None
        self.imu_std = imu_std.astype(np.float32) if imu_std is not None else None
        self.imu_sensor = imu_sensor.strip() if imu_sensor else None
        self.repeat_single_sensor = int(repeat_single_sensor)

        if self.imu_sensor is not None and self.repeat_single_sensor <= 0:
            raise ValueError(f"repeat_single_sensor must be > 0, got {self.repeat_single_sensor}")

    @staticmethod
    def _read_rows(path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            raise ValueError(f"No rows found in {path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        if path not in self._cache:
            data = np.load(path, allow_pickle=True)
            self._cache[path] = {k: data[k] for k in data.files}
        return self._cache[path]

    def __getitem__(self, index: int):
        row = self.rows[index]
        npz_rel = row["npz_path"]
        npz_path = (self.root_dir / npz_rel).resolve()
        data = self._load_npz(npz_path)

        st = int(row["window_start"])
        ed = int(row["window_end"])

        imu_idx = int(row.get("imu_idx", 0))
        person_idx = int(row.get("person_idx", 0))
        skeleton_source = row.get("skeleton_source", "gt")

        # IMU: format [T, N_imu, 48] (single-person [T, 48] also handled)
        imu = data["imu"]
        if imu.ndim == 3:
            imu = imu[st:ed, imu_idx]
        else:
            imu = imu[st:ed]

        # Skeleton
        if skeleton_source == "gt":
            skel = data["gt_skeleton"][st:ed, person_idx]
        elif skeleton_source == "extract":
            pred_indices = data["gt_to_extract_map"][st:ed, person_idx]
            skel = np.zeros((ed - st, 17, 3), dtype=np.float32)
            extract_skeleton = data["extract_skeleton"]
            for i, pidx in enumerate(pred_indices):
                if pidx != -1:
                    skel[i] = extract_skeleton[st + i, pidx]
        else:
            raise ValueError(f"Unknown skeleton_source: {skeleton_source}")

        if self.imu_sensor is not None:
            imu = self._single_sensor_to_48d(imu, self.imu_sensor, self.repeat_single_sensor)

        if self.imu_mean is not None and self.imu_std is not None:
            imu = (imu - self.imu_mean) / np.maximum(self.imu_std, 1e-6)

        if imu.shape[0] != skel.shape[0]:
            raise ValueError(f"Window length mismatch in {npz_path}: {imu.shape} vs {skel.shape}")

        return {
            "imu": torch.from_numpy(imu),
            "skeleton": torch.from_numpy(skel),
            "subject": row.get("subject", ""),
            "session": row.get("session", ""),
            "split": row.get("split", ""),
        }

    @staticmethod
    def _single_sensor_to_48d(imu: np.ndarray, sensor_name: str, repeat_single_sensor: int) -> np.ndarray:
        order = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]
        if sensor_name not in order:
            raise ValueError(f"Unsupported sensor_name={sensor_name}. Must be one of {order}")

        if repeat_single_sensor != 4:
            raise ValueError(
                "Alignment IMU encoder expects 48D input. "
                f"Use repeat_single_sensor=4 for single-sensor mode; got {repeat_single_sensor}."
            )

        k = order.index(sensor_name)
        rot = imu[:, k * 9 : (k + 1) * 9]
        acc = imu[:, 36 + k * 3 : 36 + (k + 1) * 3]

        out = np.zeros((imu.shape[0], 48), dtype=np.float32)
        for i in range(4):
            out[:, i * 9 : (i + 1) * 9] = rot
            out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
        return out
