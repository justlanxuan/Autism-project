"""Pipeline base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class PipelineStage(ABC):
    """Single stage in a data pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stage and return updated state dict."""
        ...
