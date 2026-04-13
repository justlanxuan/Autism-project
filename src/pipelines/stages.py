"""Pipeline stage implementations."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from src.pipelines.base import PipelineStage
from src.utils.config import load_config


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_with_pythonpath() -> dict:
    env = os.environ.copy()
    root = str(_repo_root())
    src = str(_repo_root() / "src")
    current = env.get("PYTHONPATH", "").strip()
    parts = [p for p in [root, src, current] if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else str(_repo_root()), env=_env_with_pythonpath())


class SliceStage(PipelineStage):
    """Run IMU-skeleton slicing according to dataset config."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = load_config(config_path)
        preprocess_cfg = cfg.get("preprocess", {})
        dataset = preprocess_cfg.get("dataset", "")

        if "custom" in dataset:
            script = "-m"
            module = "src.data.slice.custom_4fold"
            cmd = [sys.executable, script, module, "--config", str(config_path)]
        else:
            script = "-m"
            module = "src.data.slice.totalcapture"
            cmd = [sys.executable, script, module, "--config", str(config_path)]

        _run(cmd)
        # Pass through the data_root for downstream stages
        slice_cfg = cfg.get("slice", {})
        out_dir = slice_cfg.get("out_dir")
        if out_dir:
            state["data_root"] = out_dir
        return state


class ExtractStage(PipelineStage):
    """Run video skeleton extraction if config contains an extract section."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = load_config(config_path)
        extract_cfg = cfg.get("extract")
        if not isinstance(extract_cfg, dict):
            print("[INFO] No extract section in config; skipping extraction stage.")
            return state

        script_path = (_repo_root() / "src" / "cli" / "run_extract.py").resolve()
        cmd = [sys.executable, str(script_path)]

        # Automatically translate every config key into a CLI flag.
        # This keeps the stage generic and avoids hard-coding detector/tracker/pose-estimator names.
        for key, value in extract_cfg.items():
            _append_arg(cmd, f"--{key}", value)

        _run(cmd)
        return state


class TrainStage(PipelineStage):
    """Run IMU-video alignment training."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = load_config(config_path)
        model = cfg.get("model", {})
        paths = cfg.get("paths", {})
        out = cfg.get("output", {})
        train = cfg.get("train", {})
        folds = cfg.get("folds")

        if isinstance(folds, list) and folds:
            for fold in folds:
                fold_cfg = _expand_cfg(cfg, int(fold))
                print(f"[INFO] Running fold {fold}")
                self._run_train(config_path, fold_cfg)
        else:
            self._run_train(config_path, cfg)

        return state

    def _run_train(self, config_path: Path, cfg: dict) -> None:
        model = cfg.get("model", {})
        paths = cfg.get("paths", {})
        out = cfg.get("output", {})
        train = cfg.get("train", {})

        cmd = [
            sys.executable,
            "-m",
            "src.engine.train",
            "--train_csv", str(paths.get("train_csv", "")),
            "--val_csv", str(paths.get("val_csv", "")),
            "--data_root", str(paths.get("data_root", "")),
            "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
            "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
            "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
            "--imu_ckpt", str(model.get("imu_ckpt", "")),
            "--epochs", str(train.get("epochs", 40)),
            "--batch_size", str(train.get("batch_size", 64)),
            "--num_workers", str(train.get("num_workers", 8)),
            "--output_root", str(out.get("output_root", "artifacts")),
            "--save_dir", str(out.get("save_dir", ".")),
            "--run_name", str(out.get("run_name", "")),
        ]

        if train.get("compute_imu_stats"):
            cmd.append("--compute_imu_stats")
        if train.get("imu_sensor"):
            cmd.extend(["--imu_sensor", str(train["imu_sensor"])])
        if train.get("repeat_single_sensor") is not None:
            cmd.extend(["--repeat_single_sensor", str(train["repeat_single_sensor"])])

        _run(cmd)


class TestStage(PipelineStage):
    """Run standard evaluation and optional grouped evaluation."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config_path = state["config_path"]
        cfg = load_config(config_path)
        model = cfg.get("model", {})
        paths = cfg.get("paths", {})
        test = cfg.get("test", {})
        grouped = cfg.get("grouped_test", {})
        out = cfg.get("output", {})
        folds = cfg.get("folds")

        run_dir = _run_dir(cfg)
        best_ckpt = run_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

        custom_eval = test.get("mode", "").strip() == "custom_2person"
        if custom_eval:
            test_json = run_dir / "eval_results_2person.json"
            cmd = [
                sys.executable,
                "-m",
                "src.engine.eval_custom",
                "--test_csv", str(paths.get("test_csv", "")),
                "--data_root", str(paths.get("data_root", "")),
                "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
                "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
                "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
                "--checkpoint", str(best_ckpt),
                "--batch_size", str(test.get("batch_size", 64)),
                "--eval_mode", str(test.get("eval_mode", "same_time_2person")),
                "--chunk_windows", str(test.get("chunk_windows", 30)),
                "--save_json", str(test_json),
            ]
            _run(cmd)
            return state

        test_json = run_dir / "test_metrics.json"
        cmd = [
            sys.executable,
            "-m",
            "src.engine.eval",
            "--test_csv", str(paths.get("test_csv", "")),
            "--data_root", str(paths.get("data_root", "")),
            "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
            "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
            "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
            "--checkpoint", str(best_ckpt),
            "--batch_size", str(test.get("batch_size", 64)),
            "--save_json", str(test_json),
        ]
        _run(cmd)

        if grouped.get("enabled", False):
            grouped_json = run_dir / "grouped_results.json"
            grouped_csv = run_dir / "grouped_results.csv"
            cmd = [
                sys.executable,
                "-m",
                "src.engine.eval_grouped",
                "--test_csv", str(paths.get("test_csv", "")),
                "--data_root", str(paths.get("data_root", "")),
                "--motionbert_root", str(model.get("motionbert_root", "/home/fzliang/origin/MotionBERT")),
                "--motionbert_config", str(model.get("motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml")),
                "--motionbert_ckpt", str(model.get("motionbert_ckpt", "")),
                "--checkpoint", str(best_ckpt),
                "--batch_size", str(test.get("batch_size", 64)),
                "--group_sizes", str(grouped.get("group_sizes", "2,4,6,8,16")),
                "--num_trials", str(grouped.get("num_trials", 50)),
                "--chunk_windows", str(grouped.get("chunk_windows", 30)),
                "--min_chunk_windows", str(grouped.get("min_chunk_windows", 15)),
                "--seed", str(grouped.get("seed", 42)),
                "--save_json", str(grouped_json),
                "--save_csv", str(grouped_csv),
            ]
            _run(cmd)

        return state


def _run_dir(cfg: dict) -> Path:
    out = cfg.get("output", {})
    return (_repo_root() / out.get("output_root", "artifacts") / out.get("save_dir", ".") / out.get("run_name", "")).resolve()


def _expand_cfg(cfg: dict, fold: int | None) -> dict:
    import copy
    if fold is None:
        return cfg
    return _format_value(copy.deepcopy(cfg), fold)


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
