"""Config-driven alignment evaluation entrypoint.

Usage examples:
  python experiments/evaluate.py --config configs/totalcapture.yaml
  python experiments/evaluate.py --config configs/custom.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run alignment evaluation from dataset config")
	parser.add_argument("--config", type=str, required=True, help="Path to dataset yaml config")
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


def _run_dir(cfg: dict, repo_root: Path) -> Path:
	out = cfg["output"]
	return (repo_root / out["output_root"] / out["save_dir"] / out["run_name"]).resolve()


def run_test(cfg: dict, repo_root: Path) -> None:
	model = cfg["model"]
	paths = cfg["paths"]
	test = cfg.get("test", {})
	grouped = cfg.get("grouped_test", {})
	custom_eval = test.get("mode", "").strip() == "custom_2person"
	
	# 使用当前 Python 环境（假设已激活 conda）
	python_exe = sys.executable

	run_dir = _run_dir(cfg, repo_root)
	best_ckpt = run_dir / "best.pt"
	if not best_ckpt.exists():
		raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

	if custom_eval:
		test_json = run_dir / "eval_results_2person.json"
		cmd = [
			python_exe,
			"-m",
			"src.engine.eval_custom",
			"--test_csv",
			str(paths["test_csv"]),
			"--data_root",
			str(paths["data_root"]),
			"--motionbert_root",
			str(model["motionbert_root"]),
			"--motionbert_config",
			str(model["motionbert_config"]),
			"--motionbert_ckpt",
			str(model["motionbert_ckpt"]),
			"--checkpoint",
			str(best_ckpt),
			"--batch_size",
			str(test.get("batch_size", 64)),
			"--eval_mode",
			str(test.get("eval_mode", "same_time_2person")),
			"--chunk_windows",
			str(test.get("chunk_windows", 30)),
			"--save_json",
			str(test_json),
		]
		_run(cmd, repo_root)
		return

	test_json = run_dir / "test_metrics.json"
	cmd = [
		python_exe,
		"-m",
			"src.engine.eval",
		"--test_csv",
		str(paths["test_csv"]),
		"--data_root",
		str(paths["data_root"]),
		"--motionbert_root",
		str(model["motionbert_root"]),
		"--motionbert_config",
		str(model["motionbert_config"]),
		"--motionbert_ckpt",
		str(model["motionbert_ckpt"]),
		"--checkpoint",
		str(best_ckpt),
		"--batch_size",
		str(test.get("batch_size", 64)),
		"--save_json",
		str(test_json),
	]
	_run(cmd, repo_root)

	if grouped.get("enabled", False):
		grouped_json = run_dir / "grouped_results.json"
		grouped_csv = run_dir / "grouped_results.csv"
		cmd = [
			python_exe,
			"-m",
			"src.engine.eval_grouped",
			"--test_csv",
			str(paths["test_csv"]),
			"--data_root",
			str(paths["data_root"]),
			"--motionbert_root",
			str(model["motionbert_root"]),
			"--motionbert_config",
			str(model["motionbert_config"]),
			"--motionbert_ckpt",
			str(model["motionbert_ckpt"]),
			"--checkpoint",
			str(best_ckpt),
			"--batch_size",
			str(test.get("batch_size", 64)),
			"--group_sizes",
			str(grouped.get("group_sizes", "2,4,6,8,16")),
			"--num_trials",
			str(grouped.get("num_trials", 50)),
			"--chunk_windows",
			str(grouped.get("chunk_windows", 30)),
			"--min_chunk_windows",
			str(grouped.get("min_chunk_windows", 15)),
			"--seed",
			str(grouped.get("seed", 42)),
			"--save_json",
			str(grouped_json),
			"--save_csv",
			str(grouped_csv),
		]
		_run(cmd, repo_root)


def main() -> None:
	args = parse_args()
	repo_root = Path(__file__).resolve().parents[1]
	config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
	cfg = _load_cfg(config_path)

	print(f"[INFO] Dataset config: {cfg.get('dataset', 'unknown')}")
	folds = cfg.get("folds")

	if isinstance(folds, list) and folds:
		for fold in folds:
			print(f"[INFO] Evaluating fold {fold}")
			fold_cfg = _expand_cfg(cfg, int(fold))
			run_test(fold_cfg, repo_root)
		return

	run_test(cfg, repo_root)


if __name__ == "__main__":
	main()
