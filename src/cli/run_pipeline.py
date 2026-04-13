#!/usr/bin/env python3
"""CLI entrypoint to run full or partial pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipelines.full_pipeline import FullPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IMU-Video alignment pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset YAML config")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages: slice,extract,train,test (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages_spec = args.stages.strip()
    if stages_spec.lower() == "all":
        stages = None
    else:
        stages = [s.strip() for s in stages_spec.split(",") if s.strip()]

    pipeline = FullPipeline(config_path=args.config, stages=stages)
    pipeline.run()


if __name__ == "__main__":
    main()
