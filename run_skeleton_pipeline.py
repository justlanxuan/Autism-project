#!/usr/bin/env python3
"""Wrapper to run video skeleton pipeline with config file."""

import argparse
import subprocess
import sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config


def build_args_from_config(cfg: dict) -> list:
    """Build command line args from config extract section."""
    extract = cfg.get("extract", {})
    runtime = cfg.get("runtime", {})
    
    args = []
    
    # Required args from config
    if "manifest_csv" in extract:
        args.extend(["--manifest_csv", str(extract["manifest_csv"])])
    if "limit" in extract:
        args.extend(["--limit", str(extract["limit"])])
    if "gpu" in extract and extract["gpu"] is not None:
        args.extend(["--gpu", str(extract["gpu"])])
    if "results_root" in extract:
        args.extend(["--results_root", str(extract["results_root"])])
    if "tracking_mode" in extract:
        args.extend(["--tracking_mode", str(extract["tracking_mode"])])
    if "alphapose_root" in extract:
        args.extend(["--alphapose_root", str(extract["alphapose_root"])])
    if "alphapose_python" in extract:
        args.extend(["--alphapose_python", str(extract["alphapose_python"])])
    if "alphapose_cfg" in extract:
        args.extend(["--alphapose_cfg", str(extract["alphapose_cfg"])])
    if "alphapose_ckpt" in extract:
        args.extend(["--alphapose_ckpt", str(extract["alphapose_ckpt"])])
    if "pose_track_backend" in extract:
        args.extend(["--pose_track_backend", str(extract["pose_track_backend"])])
    if "bytetrack_root" in extract:
        args.extend(["--bytetrack_root", str(extract["bytetrack_root"])])
    if "bytetrack_python" in extract:
        args.extend(["--bytetrack_python", str(extract["bytetrack_python"])])
    if "bytetrack_ckpt" in extract:
        args.extend(["--bytetrack_ckpt", str(extract["bytetrack_ckpt"])])
    if "skip_existing" in extract and extract["skip_existing"]:
        args.append("--skip_existing")
    if "dry_run" in extract and extract["dry_run"]:
        args.append("--dry_run")
        
    # ByteTrack args
    if "bytetrack_conf" in extract:
        args.extend(["--bytetrack_conf", str(extract["bytetrack_conf"])])
    if "bytetrack_nms" in extract:
        args.extend(["--bytetrack_nms", str(extract["bytetrack_nms"])])
    if "bytetrack_tsize" in extract:
        args.extend(["--bytetrack_tsize", str(extract["bytetrack_tsize"])])
    if "bytetrack_track_thresh" in extract:
        args.extend(["--bytetrack_track_thresh", str(extract["bytetrack_track_thresh"])])
    if "bytetrack_track_buffer" in extract:
        args.extend(["--bytetrack_track_buffer", str(extract["bytetrack_track_buffer"])])
    if "bytetrack_match_thresh" in extract:
        args.extend(["--bytetrack_match_thresh", str(extract["bytetrack_match_thresh"])])
    if "bytetrack_min_box_area" in extract:
        args.extend(["--bytetrack_min_box_area", str(extract["bytetrack_min_box_area"])])
    if "bytetrack_aspect_ratio_thresh" in extract:
        args.extend(["--bytetrack_aspect_ratio_thresh", str(extract["bytetrack_aspect_ratio_thresh"])])
        
    return args


def main():
    parser = argparse.ArgumentParser(description="Run skeleton pipeline from config")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    pipeline_args = build_args_from_config(cfg)
    
    script_path = Path(__file__).parent / "src" / "data" / "cli" / "run_video_skeleton_pipeline.py"
    
    cmd = [sys.executable, str(script_path)] + pipeline_args
    
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
