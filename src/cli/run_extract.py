"""Unified video skeleton extraction CLI.

Dispatches to backend-specific scripts based on detector / tracker / pose_estimator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified video skeleton extraction")

    # Core roles
    parser.add_argument("--detector", type=str, default=None, help="Detector name (optional)")
    parser.add_argument("--tracker", type=str, default="bytetrack", help="Tracker name")
    parser.add_argument("--pose_estimator", type=str, default="alphapose", help="Pose estimator name")

    # Generic CKPT / root / cfg arguments
    parser.add_argument("--detector_root", type=str, default=None)
    parser.add_argument("--detector_ckpt", type=str, default=None)
    parser.add_argument("--detector_cfg", type=str, default=None)

    parser.add_argument("--tracker_root", type=str, default=None)
    parser.add_argument("--tracker_ckpt", type=str, default=None)
    parser.add_argument("--tracker_cfg", type=str, default=None)

    parser.add_argument("--pose_estimator_root", type=str, default=None)
    parser.add_argument("--pose_estimator_ckpt", type=str, default=None)
    parser.add_argument("--pose_estimator_cfg", type=str, default=None)

    # Common pipeline args
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--manifest_csv", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--results_root", type=str, default="./data/interim/extract_outputs")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


def _append(cmd: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(key)
        return
    value_s = str(value).strip()
    if value_s:
        cmd.extend([key, value_s])


def dispatch_alphapose_bytetrack(args: argparse.Namespace) -> list[str]:
    """Build command for AlphaPose + ByteTrack backend."""
    script = str(REPO_ROOT / "src" / "cli" / "run_alphapose_bytetrack.py")
    cmd = [sys.executable, script]

    # Common args
    _append(cmd, "--video", args.video)
    _append(cmd, "--manifest_csv", args.manifest_csv)
    _append(cmd, "--limit", args.limit)
    if args.skip_existing:
        cmd.append("--skip_existing")
    _append(cmd, "--results_root", args.results_root)
    _append(cmd, "--gpu", args.gpu)
    if args.dry_run:
        cmd.append("--dry_run")

    # Pose estimator -> AlphaPose
    _append(cmd, "--alphapose_root", args.pose_estimator_root)
    _append(cmd, "--alphapose_ckpt", args.pose_estimator_ckpt)
    _append(cmd, "--alphapose_cfg", args.pose_estimator_cfg)

    # Tracker -> ByteTrack
    _append(cmd, "--bytetrack_root", args.tracker_root)
    _append(cmd, "--bytetrack_ckpt", args.tracker_ckpt)
    _append(cmd, "--bytetrack_exp_file", args.tracker_cfg)

    # Combination flags
    _append(cmd, "--tracking_mode", "bytetrack_external")
    _append(cmd, "--pose_track_backend", "bytetrack")

    return cmd


def dispatch_alphapose_only(args: argparse.Namespace) -> list[str]:
    """Build command for AlphaPose internal tracking backend."""
    script = str(REPO_ROOT / "src" / "cli" / "run_alphapose_bytetrack.py")
    cmd = [sys.executable, script]

    # Common args
    _append(cmd, "--video", args.video)
    _append(cmd, "--manifest_csv", args.manifest_csv)
    _append(cmd, "--limit", args.limit)
    if args.skip_existing:
        cmd.append("--skip_existing")
    _append(cmd, "--results_root", args.results_root)
    _append(cmd, "--gpu", args.gpu)
    if args.dry_run:
        cmd.append("--dry_run")

    # Pose estimator -> AlphaPose
    _append(cmd, "--alphapose_root", args.pose_estimator_root)
    _append(cmd, "--alphapose_ckpt", args.pose_estimator_ckpt)
    _append(cmd, "--alphapose_cfg", args.pose_estimator_cfg)

    # Combination flags
    _append(cmd, "--tracking_mode", "alphapose_internal")
    _append(cmd, "--pose_track_backend", "alphapose")

    return cmd


def main() -> None:
    args = parse_args()
    tracker = (args.tracker or "").lower()
    pose_estimator = (args.pose_estimator or "").lower()

    if pose_estimator == "alphapose" and tracker == "bytetrack":
        cmd = dispatch_alphapose_bytetrack(args)
    elif pose_estimator == "alphapose" and tracker == "alphapose":
        cmd = dispatch_alphapose_only(args)
    else:
        raise ValueError(
            f"Unsupported extractor combination: tracker={tracker}, pose_estimator={pose_estimator}. "
            f"Currently supported: (alphapose + bytetrack) or (alphapose + alphapose)."
        )

    print("[EXTRACT]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
