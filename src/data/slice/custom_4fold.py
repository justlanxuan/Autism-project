"""Alignment entrypoint for custom 2-person data using MotionBERT scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config, resolve_config
from src.datasets.custom import Custom4FoldAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align custom 2-person dataset (4-fold LOO)")
    parser.add_argument("--config", type=str, default=None, help="YAML config with slice section")
    parser.add_argument("--motionbert_root", type=str, default=None)
    parser.add_argument("--preprocessed_root", type=str, default=None)
    parser.add_argument("--results_root", type=str, default=None)
    parser.add_argument("--matching_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_sessions", type=int, default=None)
    return parser.parse_args()


def load_slice_cfg(config_path: str | None) -> dict:
    if not config_path:
        return {}
    data = resolve_config(config_path)
    slice = data.get("slice", {})
    if slice is None:
        return {}
    if not isinstance(slice, dict):
        raise ValueError(f"Invalid slice section in config: {config_path}")
    return slice


def main() -> None:
    args = parse_args()
    slice_cfg = load_slice_cfg(args.config)

    if args.motionbert_root is not None:
        slice_cfg["motionbert_root"] = args.motionbert_root
    if args.preprocessed_root is not None:
        slice_cfg["preprocessed_root"] = args.preprocessed_root
    if args.results_root is not None:
        slice_cfg["results_root"] = args.results_root
    if args.matching_csv is not None:
        slice_cfg["matching_csv"] = args.matching_csv
    if args.out_dir is not None:
        slice_cfg["out_dir"] = args.out_dir
    if args.window_len is not None:
        slice_cfg["window_len"] = args.window_len
    if args.stride is not None:
        slice_cfg["stride"] = args.stride
    if args.max_sessions is not None:
        slice_cfg["max_sessions"] = args.max_sessions

    adapter = Custom4FoldAdapter(slice_cfg)
    adapter.run()


if __name__ == "__main__":
    main()
