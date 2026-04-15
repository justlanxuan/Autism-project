"""TotalCapture dataset adapter for preprocessing."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.adapters.alphapose import load_alphapose_skeleton, find_skeleton_for_sequence


SENSOR_ORDER = ["L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm"]


@dataclass
class SequenceMeta:
    subject: str
    session: str
    split: str
    npz_path: str
    num_frames: int


def parse_subjects(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_sensor_order(spec: Sequence[str] | str | None) -> List[str]:
    if spec is None:
        return list(SENSOR_ORDER)
    if isinstance(spec, str):
        return [x.strip() for x in spec.split(",") if x.strip()]
    return [str(x).strip() for x in spec if str(x).strip()]


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    # q: [N, 4], in order [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.zeros((q.shape[0], 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def parse_vicon_pos(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid Vicon file: {path}")

    joints = [x for x in lines[0].split("\t") if x]
    rows: List[np.ndarray] = []
    for ln in lines[1:]:
        parts = [x for x in ln.split("\t") if x]
        if len(parts) < len(joints):
            continue
        coords = []
        ok = True
        for token in parts[: len(joints)]:
            vals = token.split()
            if len(vals) != 3:
                ok = False
                break
            coords.append([float(vals[0]), float(vals[1]), float(vals[2])])
        if ok:
            rows.append(np.asarray(coords, dtype=np.float32))

    if not rows:
        raise ValueError(f"No valid rows in Vicon file: {path}")
    return joints, np.stack(rows, axis=0)


def parse_xsens_sensors(path: Path, selected: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty sensors file: {path}")

    first = lines[0].split()
    if len(first) < 2:
        raise ValueError(f"Invalid sensors header: {path}")
    n_sensors = int(first[0])

    quats = []
    accs = []

    i = 1
    while i < len(lines):
        # frame index line
        _ = lines[i]
        i += 1
        if i + n_sensors > len(lines):
            break

        sensor_map: Dict[str, np.ndarray] = {}
        for _k in range(n_sensors):
            toks = lines[i].split()
            i += 1
            if len(toks) < 8:
                continue
            name = toks[0]
            vals = np.array([float(x) for x in toks[1:8]], dtype=np.float32)
            sensor_map[name] = vals

        if not all(name in sensor_map for name in selected):
            continue

        q_frame = []
        a_frame = []
        for name in selected:
            vals = sensor_map[name]
            q_frame.append(vals[:4])
            a_frame.append(vals[4:7])
        quats.append(np.stack(q_frame, axis=0))
        accs.append(np.stack(a_frame, axis=0))

    if not quats:
        raise ValueError(f"No valid IMU frames found in {path}")

    return np.stack(quats, axis=0), np.stack(accs, axis=0)


def convert_imu_to_48(quat4: np.ndarray, acc3: np.ndarray) -> np.ndarray:
    # quat4: [T, 4 sensors, 4], acc3: [T, 4 sensors, 3]
    tlen = quat4.shape[0]
    out = np.zeros((tlen, 48), dtype=np.float32)
    for i in range(4):
        rot = quat_to_rotmat(quat4[:, i, :]).reshape(tlen, 9)
        acc = acc3[:, i, :]
        out[:, i * 9 : (i + 1) * 9] = rot
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out


def map_totalcapture21_to_h36m17(joint_names: Sequence[str], xyz21: np.ndarray) -> np.ndarray:
    idx = {name: i for i, name in enumerate(joint_names)}

    def j(name: str) -> np.ndarray:
        return xyz21[:, idx[name], :]

    y = np.zeros((xyz21.shape[0], 17, 3), dtype=np.float32)
    y[:, 0, :] = j("Hips")
    y[:, 1, :] = j("RightUpLeg")
    y[:, 2, :] = j("RightLeg")
    y[:, 3, :] = j("RightFoot")
    y[:, 4, :] = j("LeftUpLeg")
    y[:, 5, :] = j("LeftLeg")
    y[:, 6, :] = j("LeftFoot")
    y[:, 7, :] = j("Spine2")
    y[:, 8, :] = j("Spine3")
    y[:, 9, :] = j("Neck")
    y[:, 10, :] = j("Head")
    y[:, 11, :] = j("LeftShoulder")
    y[:, 12, :] = j("LeftArm")
    y[:, 13, :] = j("LeftForeArm")
    y[:, 14, :] = j("RightShoulder")
    y[:, 15, :] = j("RightArm")
    y[:, 16, :] = j("RightForeArm")
    return y


def normalize_skeleton(skel: np.ndarray) -> np.ndarray:
    # Root-relative and scale-normalized to improve training stability.
    root = skel[:, 0:1, :]
    skel = skel - root
    scale = np.linalg.norm(skel[:, 8, :] - skel[:, 0, :], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    skel = skel / scale[:, None, :]
    return skel.astype(np.float32)


def find_sequences(root: Path) -> List[Tuple[str, str, Path, Path]]:
    seqs: List[Tuple[str, str, Path, Path]] = []
    for subject_dir in sorted(root.glob("S[1-5]")):
        subject = subject_dir.name
        imu_subject = subject.lower()
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name
            vicon_pos = session_dir / "gt_skel_gbl_pos.txt"
            imu_file = root / imu_subject / f"{imu_subject}_{session}_Xsens.sensors"
            if vicon_pos.exists() and imu_file.exists():
                seqs.append((subject, session, vicon_pos, imu_file))
    return seqs


def subject_to_split(subject: str, train: Sequence[str], val: Sequence[str], test: Sequence[str]) -> str:
    if subject in train:
        return "train"
    if subject in val:
        return "val"
    if subject in test:
        return "test"
    raise ValueError(f"Subject {subject} not assigned to split")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


class TotalCaptureAdapter:
    """Adapter that preprocesses TotalCapture raw data into NPZ + CSV."""

    def __init__(self, slice_cfg: dict):
        self.cfg = slice_cfg
        self.root = Path(slice_cfg.get("root", "/data/fzliang/totalcapture"))
        self.out_dir = Path(slice_cfg.get("out_dir", "data/processed/totalcapture"))
        self.window_len = int(slice_cfg.get("window_len", 24))
        self.stride = int(slice_cfg.get("stride", 16))
        self.sensor_order = parse_sensor_order(slice_cfg.get("sensor_order", SENSOR_ORDER))
        self.train_subj = parse_subjects(slice_cfg.get("train_subjects", "S1,S2,S3"))
        self.val_subj = parse_subjects(slice_cfg.get("val_subjects", "S4"))
        self.test_subj = parse_subjects(slice_cfg.get("test_subjects", "S5"))
        self.max_sequences = int(slice_cfg.get("max_sequences", 0))
        self.skeleton_source = slice_cfg.get("skeleton_source", "vicon")
        self.skeleton_root = None
        if self.skeleton_source == "alphapose":
            self.skeleton_root = Path(slice_cfg.get("skeleton_root", "/home/fzliang/MotionBERT/results_totalcapture_video"))

    def run(self) -> Path:
        seq_dir = self.out_dir / "sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)

        all_seqs = find_sequences(self.root)
        if self.max_sequences > 0:
            all_seqs = all_seqs[: self.max_sequences]

        sequence_rows: List[Dict[str, object]] = []
        window_rows: List[Dict[str, object]] = []

        for subject, session, vicon_path, imu_path in all_seqs:
            splits = []
            if subject in self.train_subj:
                splits.append("train")
            if subject in self.val_subj:
                splits.append("val")
            if subject in self.test_subj:
                splits.append("test")
            if not splits:
                print(f"Warning: Subject {subject} not assigned to any split, skipping {subject}_{session}...")
                continue

            quat4, acc3 = parse_xsens_sensors(imu_path, self.sensor_order)

            if self.skeleton_source == "alphapose":
                skeleton_file = find_skeleton_for_sequence(subject, session, self.skeleton_root)
                if skeleton_file is None:
                    print(f"Warning: No AlphaPose skeleton found for {subject}_{session}, skipping...")
                    continue
                skel17, scores = load_alphapose_skeleton(skeleton_file)
            else:
                joint_names, xyz21 = parse_vicon_pos(vicon_path)
                skel17 = map_totalcapture21_to_h36m17(joint_names, xyz21)

            tlen = min(skel17.shape[0], quat4.shape[0])
            skel17 = skel17[:tlen]
            quat4 = quat4[:tlen]
            acc3 = acc3[:tlen]

            imu48 = convert_imu_to_48(quat4, acc3)
            skel17 = normalize_skeleton(skel17)

            rel_npz = Path("sequences") / f"{subject}_{session}.npz"
            np.savez_compressed(self.out_dir / rel_npz, imu=imu48, skeleton=skel17)

            sequence_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "split": ",".join(splits),
                    "npz_path": str(rel_npz),
                    "num_frames": int(tlen),
                }
            )

            if tlen >= self.window_len:
                for st in range(0, tlen - self.window_len + 1, self.stride):
                    ed = st + self.window_len
                    for split in splits:
                        window_rows.append(
                            {
                                "subject": subject,
                                "session": session,
                                "split": split,
                                "npz_path": str(rel_npz),
                                "window_start": int(st),
                                "window_end": int(ed),
                                "window_len": int(self.window_len),
                            }
                        )

        write_csv(
            self.out_dir / "sequences.csv",
            sequence_rows,
            ["subject", "session", "split", "npz_path", "num_frames"],
        )

        write_csv(
            self.out_dir / "windows_all.csv",
            window_rows,
            ["subject", "session", "split", "npz_path", "window_start", "window_end", "window_len"],
        )

        for split in ["train", "val", "test"]:
            split_rows = [r for r in window_rows if r["split"] == split]
            write_csv(
                self.out_dir / f"windows_{split}.csv",
                split_rows,
                ["subject", "session", "split", "npz_path", "window_start", "window_end", "window_len"],
            )

        summary = {
            "num_sequences": len(sequence_rows),
            "num_windows": len(window_rows),
            "window_len": self.window_len,
            "stride": self.stride,
            "sensor_order": self.sensor_order,
            "train_subjects": self.train_subj,
            "val_subjects": self.val_subj,
            "test_subjects": self.test_subj,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("Preprocess done")
        print(f"Output dir: {self.out_dir}")
        print(f"Sequences: {len(sequence_rows)}, windows: {len(window_rows)}")
        return self.out_dir
