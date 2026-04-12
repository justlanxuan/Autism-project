"""IMU data loading and preprocessing utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class IMUColumnConfig:
    timestamp: str = "epoch_ms"
    acc_x: str = "加速度X(g)"
    acc_y: str = "加速度Y(g)"
    acc_z: str = "加速度Z(g)"
    quat_w: str = "四元数0()"
    quat_x: str = "四元数1()"
    quat_y: str = "四元数2()"
    quat_z: str = "四元数3()"


class IMUProcessor:
    """Load IMU CSVs and convert to 48D format."""

    def __init__(self, column_config: Optional[Dict] = None):
        self.columns = IMUColumnConfig(**(column_config or {}))

    def load_csv(self, csv_path: str | Path) -> Dict[str, np.ndarray]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"IMU CSV not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]

        if not rows:
            raise ValueError(f"Empty IMU CSV: {csv_path}")

        def col(name: str) -> np.ndarray:
            vals = []
            for r in rows:
                raw = str(r.get(name, "")).strip()
                if raw == "":
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(raw))
                    except Exception:
                        vals.append(np.nan)
            return np.asarray(vals, dtype=np.float64)

        timestamp = col(self.columns.timestamp)
        acc = np.stack([col(self.columns.acc_x), col(self.columns.acc_y), col(self.columns.acc_z)], axis=1)
        quat = np.stack([col(self.columns.quat_w), col(self.columns.quat_x), col(self.columns.quat_y), col(self.columns.quat_z)], axis=1)

        return {"timestamp": timestamp, "acc": acc, "quat": quat}

    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        r = np.zeros((len(quat), 3, 3), dtype=np.float32)
        r[:, 0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2
        r[:, 0, 1] = 2 * x * y - 2 * w * z
        r[:, 0, 2] = 2 * x * z + 2 * w * y
        r[:, 1, 0] = 2 * x * y + 2 * w * z
        r[:, 1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
        r[:, 1, 2] = 2 * y * z - 2 * w * x
        r[:, 2, 0] = 2 * x * z - 2 * w * y
        r[:, 2, 1] = 2 * y * z + 2 * w * x
        r[:, 2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2
        return r

    def to_48d(self, imu_data: Dict[str, np.ndarray]) -> np.ndarray:
        rot = self.quaternion_to_rotation_matrix(imu_data["quat"]).reshape(-1, 9)
        acc = imu_data["acc"]
        imu_features = np.concatenate([rot, rot, rot, rot, acc, acc, acc, acc], axis=1)
        return imu_features.astype(np.float32)

    @staticmethod
    def resample_to_timestamps(src_t: np.ndarray, src_x: np.ndarray, tgt_t: np.ndarray) -> np.ndarray:
        src_t = np.asarray(src_t, dtype=np.float64)
        tgt_t = np.asarray(tgt_t, dtype=np.float64)
        src_x = np.asarray(src_x)

        if src_x.ndim == 1:
            return np.interp(tgt_t, src_t, src_x)

        shape_tail = src_x.shape[1:]
        flat = src_x.reshape(src_x.shape[0], -1)
        out = np.zeros((len(tgt_t), flat.shape[1]), dtype=np.float32)
        for i in range(flat.shape[1]):
            out[:, i] = np.interp(tgt_t, src_t, flat[:, i])
        return out.reshape((len(tgt_t),) + shape_tail)
