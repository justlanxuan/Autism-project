"""Config-driven alignment train/extract entrypoint.

Usage examples:
    python experiments/train.py --config configs/totalcapture.yaml --stage all
    python experiments/train.py --config configs/custom.yaml --stage train
    python experiments/train.py --config configs/custom.yaml --stage test
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train/extract/test from dataset config")
    parser.add_argument("--config", type=str, required=True, help="Path to dataset YAML config")
    parser.add_argument("--stage", type=str, choices=["extract", "train", "test", "all"], default="all")
    return parser.parse_args()


def _append_arg(cmd: list[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(key)
        return
    value_s = str(value).strip()
    if value_s:
        cmd.extend([key, value_s])


def _run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    env = os.environ.copy()
    current_py = env.get("PYTHONPATH", "")
    root_path = str(cwd)
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = f"{root_path}{os.pathsep}{src_path}"
    if current_py:
        env["PYTHONPATH"] = f"{root_path}{os.pathsep}{src_path}{os.pathsep}{current_py}"
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def _load_cfg(path: Path) -> dict:
    from src.utils.config import load_config
    return load_config(path)


def _format_value(value: Any, fold: int | None) -> Any:
    if fold is None:
        return value
    if isinstance(value, str):
        try:
            return value.format(fold=fold)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _format_value(v, fold) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_value(v, fold) for v in value]
    return value


def _expand_cfg(cfg: dict, fold: int | None) -> dict:
    if fold is None:
        return cfg
    return _format_value(copy.deepcopy(cfg), fold)


def run_train(cfg: dict, repo_root: Path) -> None:
    model = cfg["model"]
    paths = cfg["paths"]
    out = cfg["output"]
    train = cfg["train"]

    # 使用当前 Python 环境（假设已激活 conda）
    cmd = [
        sys.executable,
        "-m",
        "src.engine.train",
        "--train_csv",
        str(paths["train_csv"]),
        "--val_csv",
        str(paths["val_csv"]),
        "--data_root",
        str(paths["data_root"]),
        "--motionbert_root",
        str(model["motionbert_root"]),
        "--motionbert_config",
        str(model["motionbert_config"]),
        "--motionbert_ckpt",
        str(model["motionbert_ckpt"]),
        "--imu_ckpt",
        str(model["imu_ckpt"]),
        "--epochs",
        str(train.get("epochs", 40)),
        "--batch_size",
        str(train.get("batch_size", 64)),
        "--num_workers",
        str(train.get("num_workers", 8)),
        "--output_root",
        str(out["output_root"]),
        "--save_dir",
        str(out["save_dir"]),
        "--run_name",
        str(out["run_name"]),
    ]
    _append_arg(cmd, "--compute_imu_stats", bool(train.get("compute_imu_stats", False)))
    _append_arg(cmd, "--imu_sensor", train.get("imu_sensor"))
    _append_arg(cmd, "--repeat_single_sensor", train.get("repeat_single_sensor"))
    _run(cmd, repo_root)


def run_test_from_config(config_path: Path, repo_root: Path) -> None:
    cfg = _load_cfg(config_path)
    python_exe = sys.executable  # 使用当前 Python 环境
    evaluate_script = (repo_root / "experiments" / "evaluate.py").resolve()
    if not evaluate_script.exists():
        raise FileNotFoundError(f"Evaluate script not found: {evaluate_script}")

    cmd = [
        python_exe,
        str(evaluate_script),
        "--config",
        str(config_path),
    ]
    _run(cmd, repo_root)


def run_extract(cfg: dict, repo_root: Path) -> None:
    extract = cfg.get("extract")
    if not isinstance(extract, dict):
        print("[INFO] No extract section found; skipping extraction stage")
        return

    python_exe = str(extract.get("python", cfg.get("runtime", {}).get("python", sys.executable)))
    script = extract.get("script", "src/data/cli/run_video_skeleton_pipeline.py")
    script_path = (repo_root / script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Extract script not found: {script_path}")

    cmd = [python_exe, str(script_path)]
    arg_map = {
        "video": "--video",
        "manifest_csv": "--manifest_csv",
        "limit": "--limit",
        "skip_existing": "--skip_existing",
        "results_root": "--results_root",
        "video_name": "--video_name",
        "run_name": "--run_name",
        "alphapose_root": "--alphapose_root",
        "alphapose_python": "--alphapose_python",
        "alphapose_cfg": "--alphapose_cfg",
        "alphapose_ckpt": "--alphapose_ckpt",
        "tracking_mode": "--tracking_mode",
        "pose_track_backend": "--pose_track_backend",
        "bt_track_thresh": "--bt_track_thresh",
        "bt_track_buffer": "--bt_track_buffer",
        "bt_match_thresh": "--bt_match_thresh",
        "bt_mot20": "--bt_mot20",
        "detbatch": "--detbatch",
        "posebatch": "--posebatch",
        "use_expandable_segments": "--use_expandable_segments",
        "bytetrack_root": "--bytetrack_root",
        "bytetrack_python": "--bytetrack_python",
        "bytetrack_exp_file": "--bytetrack_exp_file",
        "bytetrack_ckpt": "--bytetrack_ckpt",
        "bytetrack_conf": "--bytetrack_conf",
        "bytetrack_nms": "--bytetrack_nms",
        "bytetrack_tsize": "--bytetrack_tsize",
        "bytetrack_fp16": "--bytetrack_fp16",
        "bytetrack_fuse": "--bytetrack_fuse",
        "bytetrack_mot20": "--bytetrack_mot20",
        "bytetrack_track_thresh": "--bytetrack_track_thresh",
        "bytetrack_track_buffer": "--bytetrack_track_buffer",
        "bytetrack_match_thresh": "--bytetrack_match_thresh",
        "bytetrack_min_box_area": "--bytetrack_min_box_area",
        "bytetrack_aspect_ratio_thresh": "--bytetrack_aspect_ratio_thresh",
        "gpu": "--gpu",
        "dry_run": "--dry_run",
        "headless": "--headless",
    }

    for key, flag in arg_map.items():
        if key in extract:
            _append_arg(cmd, flag, extract.get(key))

    _run(cmd, repo_root)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    cfg = _load_cfg(config_path)

    print(f"[INFO] Dataset config: {cfg.get('dataset', 'unknown')}")
    print(f"[INFO] Stage: {args.stage}")

    folds = cfg.get("folds")
    if isinstance(folds, list) and folds and args.stage in {"train", "all"}:
        for fold in folds:
            fold_cfg = _expand_cfg(cfg, int(fold))
            print(f"[INFO] Running fold {fold}")
            run_train(fold_cfg, repo_root)
        if args.stage == "train":
            return

    if args.stage in {"extract", "all"}:
        run_extract(cfg, repo_root)
    if args.stage in {"train", "all"} and not (isinstance(folds, list) and folds):
        run_train(cfg, repo_root)
    if args.stage in {"test", "all"}:
        run_test_from_config(config_path, repo_root)


if __name__ == "__main__":
    main()
