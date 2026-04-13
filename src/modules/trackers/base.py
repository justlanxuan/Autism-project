"""Abstract tracker interface for Autism-project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):
    """Common interface for tracker backends."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state."""

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Update tracker state and return backend-specific results."""
