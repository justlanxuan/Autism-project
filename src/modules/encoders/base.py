"""Base encoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Common interface for IMU and video encoders."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into an embedding vector."""
        ...
