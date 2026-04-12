"""Wrapper to preprocess custom 2-person data using MotionBERT scripts.

Outputs go under data/processed/custom_4fold by default.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess custom 2-person dataset (4-fold LOO)")
    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--preprocessed_root", type=str, default="/data/fzliang/data/preprocess/2person")
    parser.add_argument(
        "--results_root",
        type=str,
        default="/home/fzliang/origin/MotionBERT/results_custom_2person_bytetrack_best",
    )
    parser.add_argument(
        "--matching_csv",
        type=str,
        default="/home/fzliang/origin/MotionBERT/alignment/data/custom_2person_matching_bytetrack_best/matching_confidence_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/custom_4fold)",
    )
    parser.add_argument("--window_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max_sessions", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motionbert_root = Path(args.motionbert_root).expanduser().resolve()
    script_path = motionbert_root / "alignment" / "preprocess_custom_4fold.py"
    if not script_path.exists():
        raise FileNotFoundError(f"MotionBERT preprocess script not found: {script_path}")

    if args.out_dir is None:
        out_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "custom_4fold"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()

    cmd = [
        sys.executable,
        str(script_path),
        "--preprocessed_root",
        str(Path(args.preprocessed_root).expanduser().resolve()),
        "--results_root",
        str(Path(args.results_root).expanduser().resolve()),
        "--matching_csv",
        str(Path(args.matching_csv).expanduser().resolve()),
        "--out_dir",
        str(out_dir),
        "--window_len",
        str(args.window_len),
        "--stride",
        str(args.stride),
        "--max_sessions",
        str(args.max_sessions),
    ]

    env = os.environ.copy()
    prev_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(motionbert_root) if not prev_pythonpath else f"{motionbert_root}:{prev_pythonpath}"

    subprocess.run(cmd, check=True, cwd=str(motionbert_root), env=env)


if __name__ == "__main__":
    main()
