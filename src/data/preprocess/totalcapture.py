"""Unified preprocess for TotalCapture: generate standardized NPZ + video manifest.

Example:
    python -m src.data.preprocess.totalcapture \
        --config configs/totalcapture_video_test.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.datasets.totalcapture import (
    convert_imu_to_48,
    find_sequences,
    map_totalcapture21_to_h36m17,
    normalize_skeleton,
    parse_sensor_order,
    parse_vicon_pos,
    parse_xsens_sensors,
)
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified TotalCapture preprocess")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--raw_root", type=str, default=None)
    parser.add_argument("--camera", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    return parser.parse_args()


def load_preprocess_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    data = load_config(config_path)
    preprocess = data.get("preprocess", {})
    if preprocess is None:
        return {}
    if not isinstance(preprocess, dict):
        raise ValueError(f"Invalid preprocess section in config: {config_path}")
    return preprocess


def get_video_resolution(video_path: Path) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1920, 1080
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def find_video_for_sequence(raw_root: Path, subject: str, session: str, camera: str, ext: str = ".mp4") -> Path | None:
    """Find video file matching subject, session and camera under raw_root."""
    # Primary pattern: TC_S1_acting1_cam1.mp4 under root/session/
    primary = raw_root / session / f"TC_{subject}_{session}_{camera}{ext}"
    if primary.exists():
        return primary

    # Fallback: search recursively for files containing subject, session and camera
    candidates = [
        p for p in raw_root.rglob(f"*{ext}")
        if subject in p.name and session in p.name and camera in p.name
    ]
    if candidates:
        return sorted(candidates)[0]
    return None


def main() -> None:
    args = parse_args()
    preprocess_cfg = load_preprocess_cfg(args.config)

    raw_root = Path(
        args.raw_root if args.raw_root else preprocess_cfg.get("raw_root", "/data/fzliang/totalcapture")
    ).expanduser().resolve()
    camera = args.camera if args.camera is not None else preprocess_cfg.get("camera", "cam1")

    # Derive output_dir and manifest_csv from config or CLI
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        default_manifest = Path(preprocess_cfg.get("output", "./data/interim/video_manifest.csv")).expanduser().resolve()
        output_dir = default_manifest.parent

    manifest_csv = Path(
        args.manifest_csv if args.manifest_csv else preprocess_cfg.get("output", str(output_dir / "video_manifest.csv"))
    ).expanduser().resolve()

    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    sensor_order = parse_sensor_order(preprocess_cfg.get("sensor_order"))
    sequences = find_sequences(raw_root)

    manifest_rows: list[dict[str, str]] = []

    for subject, session, vicon_path, imu_path in sequences:
        quat4, acc3 = parse_xsens_sensors(imu_path, sensor_order)
        joint_names, xyz21 = parse_vicon_pos(vicon_path)
        skel17 = map_totalcapture21_to_h36m17(joint_names, xyz21)

        tlen = min(skel17.shape[0], quat4.shape[0])
        if tlen == 0:
            print(f"Warning: empty sequence {subject}_{session}, skipping")
            continue

        skel17 = skel17[:tlen]
        quat4 = quat4[:tlen]
        acc3 = acc3[:tlen]
        imu48 = convert_imu_to_48(quat4, acc3)
        skel17 = normalize_skeleton(skel17)

        video_path = find_video_for_sequence(raw_root, subject, session, camera)
        if video_path is not None and video_path.exists():
            w, h = get_video_resolution(video_path)
            gt_bboxes = np.tile(np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32), (tlen, 1, 1))
            manifest_rows.append({"video_path": str(video_path)})
        else:
            gt_bboxes = np.zeros((tlen, 1, 4), dtype=np.float32)
            print(f"Warning: video not found for {subject}_{session} camera={camera}")

        sequence_id = f"totalcapture_{subject}_{session}_{camera}"
        frame_ids = np.arange(tlen, dtype=np.int64)
        imu = imu48[:, np.newaxis, :].astype(np.float32)          # [T, 1, 48]
        imu_ids = np.array([0], dtype=np.int64)
        gt_person_ids = np.array([0], dtype=np.int64)
        gt_skeleton = skel17[:, np.newaxis, :, :].astype(np.float32)  # [T, 1, 17, 3]
        gt_visibility = np.ones((tlen, 1), dtype=bool)

        npz_path = seq_dir / f"{sequence_id}.npz"
        np.savez_compressed(
            npz_path,
            video_path=np.array(str(video_path) if video_path else "", dtype=object),
            dataset=np.array("totalcapture", dtype=object),
            sequence_id=np.array(sequence_id, dtype=object),
            frame_ids=frame_ids,
            imu=imu,
            imu_ids=imu_ids,
            gt_person_ids=gt_person_ids,
            gt_bboxes=gt_bboxes,
            gt_visibility=gt_visibility,
            gt_skeleton=gt_skeleton,
        )

        meta = {
            "video_path": str(video_path) if video_path else "",
            "dataset": "totalcapture",
            "sequence_id": sequence_id,
            "n_frames": int(tlen),
            "n_imu": 1,
            "n_gt": 1,
            "n_pred": 0,
            "has_gt": True,
            "imu_ids": [0],
            "gt_person_ids": [0],
            "extract_person_ids": [],
        }
        (seq_dir / f"{sequence_id}.json").write_text(json.dumps(meta, indent=2))

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Preprocessed {len(sequences)} sequences -> {output_dir}")
    print(f"Manifest: {manifest_csv} ({len(manifest_rows)} videos)")


if __name__ == "__main__":
    main()
