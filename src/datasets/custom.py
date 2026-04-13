"""Custom 4-fold dataset adapter for preprocessing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class Custom4FoldAdapter:
    """Adapter that delegates custom 2-person preprocessing to MotionBERT script."""

    def __init__(self, slice_cfg: dict):
        self.cfg = slice_cfg
        self.motionbert_root = Path(self.cfg.get("motionbert_root", "/home/fzliang/origin/MotionBERT")).expanduser().resolve()
        self.preprocessed_root = Path(self.cfg.get("preprocessed_root", "/data/fzliang/data/preprocess/2person")).expanduser().resolve()
        self.results_root = Path(self.cfg.get("results_root", "/home/fzliang/origin/MotionBERT/results_custom_2person_bytetrack_best")).expanduser().resolve()
        self.matching_csv = Path(self.cfg.get("matching_csv", "/home/fzliang/origin/MotionBERT/alignment/data/custom_2person_matching_bytetrack_best/matching_confidence_summary.csv")).expanduser().resolve()
        self.out_dir = Path(self.cfg.get("out_dir", "data/processed/custom_4fold")).expanduser().resolve()
        self.window_len = int(self.cfg.get("window_len", 24))
        self.stride = int(self.cfg.get("stride", 16))
        self.max_sessions = int(self.cfg.get("max_sessions", 0))

    def run(self) -> Path:
        script_path = self.motionbert_root / "alignment" / "preprocess_custom_4fold.py"
        if not script_path.exists():
            raise FileNotFoundError(f"MotionBERT preprocess script not found: {script_path}")

        cmd = [
            sys.executable,
            str(script_path),
            "--preprocessed_root", str(self.preprocessed_root),
            "--results_root", str(self.results_root),
            "--matching_csv", str(self.matching_csv),
            "--out_dir", str(self.out_dir),
            "--window_len", str(self.window_len),
            "--stride", str(self.stride),
            "--max_sessions", str(self.max_sessions),
        ]

        env = os.environ.copy()
        prev_pythonpath = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = str(self.motionbert_root) if not prev_pythonpath else f"{self.motionbert_root}:{prev_pythonpath}"

        subprocess.run(cmd, check=True, cwd=str(self.motionbert_root), env=env)
        return self.out_dir
