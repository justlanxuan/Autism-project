"""Preprocess for Custom dataset: generate standardized NPZ + video manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.datasets.totalcapture import quat_to_rotmat
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom dataset preprocess")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--raw_root", type=str, default=None)
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


def parse_annotations(anno_path: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse annotation CSV.

    Returns:
        n_persons: number of persons
        frame_indices: [T] int64
        timestamps_ms: [T] float64
        bboxes: [T, N, 4] float32 in [x1, y1, x2, y2]
        visibility: [T, N] bool
    """
    with anno_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty annotation file: {anno_path}")

    cols = set(rows[0].keys())
    n_persons = 0
    while f"p{n_persons + 1}_bbox_x" in cols:
        n_persons += 1

    if n_persons == 0:
        raise ValueError(f"No person bbox columns found in {anno_path}")

    T = len(rows)
    frame_indices = np.zeros(T, dtype=np.int64)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    bboxes = np.zeros((T, n_persons, 4), dtype=np.float32)
    visibility = np.zeros((T, n_persons), dtype=bool)

    for t, row in enumerate(rows):
        frame_indices[t] = int(row["frame_index"])
        timestamps_ms[t] = float(row["timestamp_ms"])
        for p in range(n_persons):
            prefix = f"p{p + 1}_"
            x = float(row[f"{prefix}bbox_x"])
            y = float(row[f"{prefix}bbox_y"])
            w = float(row[f"{prefix}bbox_w"])
            h = float(row[f"{prefix}bbox_h"])
            bboxes[t, p] = np.array([x, y, x + w, y + h], dtype=np.float32)
            visibility[t, p] = int(row[f"{prefix}is_absent"]) == 0

    return n_persons, frame_indices, timestamps_ms, bboxes, visibility


def _find_col(candidates: List[str], row: dict) -> str:
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"Could not find any of {candidates}. Available: {list(row.keys())}")


def parse_imu_csv(imu_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse custom IMU CSV.

    Returns:
        timestamps_ms: [T] float64
        quat4: [T, 4] float32
        acc3: [T, 3] float32
    """
    with imu_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty IMU file: {imu_path}")

    ts_col = _find_col(["epoch_ms"], rows[0])
    q0_col = _find_col(["四元数0()"], rows[0])
    q1_col = _find_col(["四元数1()"], rows[0])
    q2_col = _find_col(["四元数2()"], rows[0])
    q3_col = _find_col(["四元数3()"], rows[0])
    ax_col = _find_col(["加速度X(g)"], rows[0])
    ay_col = _find_col(["加速度Y(g)"], rows[0])
    az_col = _find_col(["加速度Z(g)"], rows[0])

    T = len(rows)
    timestamps_ms = np.zeros(T, dtype=np.float64)
    quat4 = np.zeros((T, 4), dtype=np.float32)
    acc3 = np.zeros((T, 3), dtype=np.float32)

    for t, row in enumerate(rows):
        timestamps_ms[t] = float(row[ts_col])
        quat4[t] = np.array([float(row[q0_col]), float(row[q1_col]), float(row[q2_col]), float(row[q3_col])], dtype=np.float32)
        acc3[t] = np.array([float(row[ax_col]), float(row[ay_col]), float(row[az_col])], dtype=np.float32)

    return timestamps_ms, quat4, acc3


def convert_single_imu_to_48(quat4: np.ndarray, acc3: np.ndarray) -> np.ndarray:
    """Convert single-sensor IMU to 48D by repeating 12D four times."""
    T = quat4.shape[0]
    rot = quat_to_rotmat(quat4).reshape(T, 9)
    acc = acc3

    out = np.zeros((T, 48), dtype=np.float32)
    for i in range(4):
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out


def resample_imu_to_target(src_ts: np.ndarray, src_values: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Linearly interpolate IMU values to target timestamps."""
    valid_start = src_ts[0]
    valid_end = src_ts[-1]

    out = np.zeros((len(target_ts), src_values.shape[1]), dtype=np.float32)
    for d in range(src_values.shape[1]):
        out[:, d] = np.interp(target_ts, src_ts, src_values[:, d])

    out[target_ts < valid_start] = np.nan
    out[target_ts > valid_end] = np.nan
    return out


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 30.0


def main() -> None:
    args = parse_args()
    preprocess_cfg = load_preprocess_cfg(args.config)

    raw_root = Path(args.raw_root if args.raw_root else preprocess_cfg.get("raw_root", "/data/fzliang/custom")).expanduser().resolve()

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

    manifest_rows: list[dict[str, str]] = []

    for person_count_dir in sorted(raw_root.iterdir()):
        if not person_count_dir.is_dir():
            continue
        for session_dir in sorted(person_count_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # Skip metadata folders like "annotations"
            if not (session_dir / "video").exists():
                continue

            session_stem = session_dir.name
            sequence_id = f"custom_{session_stem}"
            video_path = session_dir / "video" / f"{session_stem}.mp4"
            anno_path = person_count_dir / "annotations" / f"{session_stem}.anno.csv"
            imu_dir = session_dir / "imu"

            if not video_path.exists():
                print(f"Warning: video not found for {sequence_id}, skipping")
                continue
            if not anno_path.exists():
                print(f"Warning: annotation not found for {sequence_id}, skipping")
                continue

            imu_files = sorted(imu_dir.glob("*.csv"))
            if not imu_files:
                print(f"Warning: no IMU files for {sequence_id}, skipping")
                continue

            n_persons, frame_indices, anno_ts, anno_bboxes, anno_visibility = parse_annotations(anno_path)
            if len(imu_files) != n_persons:
                print(f"Warning: IMU count ({len(imu_files)}) != person count ({n_persons}) for {sequence_id}")

            imu_data_list = []
            for imu_path in imu_files:
                imu_ts, quat4, acc3 = parse_imu_csv(imu_path)
                imu48 = convert_single_imu_to_48(quat4, acc3)
                imu_data_list.append((imu_ts, imu48))

            valid_start_ms = max(data[0][0] for data in imu_data_list)
            valid_end_ms = min(data[0][-1] for data in imu_data_list)

            crop_mask = (anno_ts >= valid_start_ms) & (anno_ts <= valid_end_ms)
            if not crop_mask.any():
                print(f"Warning: no overlap between IMU and annotation for {sequence_id}, skipping")
                continue

            crop_indices = np.where(crop_mask)[0]
            first_idx = int(crop_indices[0])
            last_idx = int(crop_indices[-1])

            target_ts = anno_ts[first_idx : last_idx + 1]
            T = len(target_ts)
            frame_ids = np.arange(T, dtype=np.int64)
            bboxes = anno_bboxes[first_idx : last_idx + 1]
            visibility = anno_visibility[first_idx : last_idx + 1]

            n_imu = len(imu_data_list)
            imu_out = np.zeros((T, n_imu, 48), dtype=np.float32)
            for i, (imu_ts, imu48) in enumerate(imu_data_list):
                resampled = resample_imu_to_target(imu_ts, imu48, target_ts)
                nan_mask = np.isnan(resampled).any(axis=1)
                resampled = np.nan_to_num(resampled, nan=0.0)
                imu_out[:, i] = resampled
                if nan_mask.any():
                    print(f"Warning: {sequence_id} IMU {i} has {nan_mask.sum()} out-of-range frames")

            person_ids = np.arange(n_persons, dtype=np.int64)
            imu_ids = np.arange(n_imu, dtype=np.int64)

            npz_path = seq_dir / f"{sequence_id}.npz"
            np.savez_compressed(
                npz_path,
                video_path=np.array(str(video_path), dtype=object),
                dataset=np.array("custom", dtype=object),
                sequence_id=np.array(sequence_id, dtype=object),
                frame_ids=frame_ids,
                imu=imu_out,
                imu_ids=imu_ids,
                gt_person_ids=person_ids,
                gt_bboxes=bboxes,
                gt_visibility=visibility,
            )

            meta = {
                "video_path": str(video_path),
                "dataset": "custom",
                "sequence_id": sequence_id,
                "n_frames": int(T),
                "n_imu": int(n_imu),
                "n_gt": int(n_persons),
                "has_gt_skeleton": False,
                "imu_ids": imu_ids.tolist(),
                "gt_person_ids": person_ids.tolist(),
                "extract_person_ids": [],
            }
            (seq_dir / f"{sequence_id}.json").write_text(json.dumps(meta, indent=2))

            manifest_rows.append({"video_path": str(video_path)})
            print(f"Processed {sequence_id}: {T} frames, {n_imu} IMUs, {n_persons} persons")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path"])
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    print(f"Preprocessed {len(manifest_rows)} sequences -> {output_dir}")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
