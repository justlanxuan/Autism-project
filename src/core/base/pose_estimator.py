"""Abstract pose-estimator interface for Autism-project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePoseEstimator(ABC):
	"""Common interface for pose-estimation backends."""

	@abstractmethod
	def reset(self) -> None:
		"""Reset any sequence-local state."""

	@abstractmethod
	def estimate(self, *args: Any, **kwargs: Any):
		"""Estimate poses and return backend-specific results."""

