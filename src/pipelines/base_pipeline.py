"""Base pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AlignmentTrainConfig:
	train_csv: str
	val_csv: str
	motionbert_root: str
	motionbert_config: str
	motionbert_ckpt: str
	save_dir: str
	data_root: Optional[str] = None


class AlignmentTrainingPipeline:
	"""Thin wrapper over alignment training entrypoint."""

	def __init__(self, config: AlignmentTrainConfig):
		self.config = config

	def run(self) -> None:
		from src.engine import train

		args = [
			"--train_csv",
			self.config.train_csv,
			"--val_csv",
			self.config.val_csv,
			"--motionbert_root",
			self.config.motionbert_root,
			"--motionbert_config",
			self.config.motionbert_config,
			"--motionbert_ckpt",
			self.config.motionbert_ckpt,
			"--save_dir",
			self.config.save_dir,
		]
		if self.config.data_root:
			args += ["--data_root", self.config.data_root]

		train.main_from_argv(args)
