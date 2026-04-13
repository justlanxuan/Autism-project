"""Generate a video manifest CSV for the TotalCapture dataset.

Example:
    # Direct CLI
    python -m src.data.preprocess.preprocess_totalcapture \
        --root /data/fzliang/totalcapture \
        --camera cam1 \
        --output ./data/interim/video_manifest.csv

    # From config file
    python -m src.data.preprocess.preprocess_totalcapture \
        --config configs/test_refact.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate video manifest CSV for TotalCapture")
    parser.add_argument("--config", type=str, default=None, help="YAML config with a preprocess section")
    parser.add_argument("--root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--camera", type=str, default=None, help="Camera filter (default: cam1, use 'all' for every camera)")
    parser.add_argument("--ext", type=str, default=".mp4", help="Video extension to look for")
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


def main() -> None:
    args = parse_args()
    preprocess_cfg = load_preprocess_cfg(args.config)

    # Resolve parameters: CLI args override config values
    root = Path(args.root).expanduser().resolve() if args.root else Path(preprocess_cfg.get("raw_root", "/data/fzliang/totalcapture")).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else Path(preprocess_cfg.get("output", "./data/interim/video_manifest.csv")).expanduser().resolve()
    camera = args.camera if args.camera is not None else preprocess_cfg.get("camera", "cam1")
    ext = args.ext

    videos = sorted(root.rglob(f"*{ext}"))
    if camera != "all":
        videos = [v for v in videos if camera in v.name]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path"])
        writer.writeheader()
        for v in videos:
            writer.writerow({"video_path": str(v)})

    print(f"Generated manifest with {len(videos)} videos: {output}")


if __name__ == "__main__":
    main()
