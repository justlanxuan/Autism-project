"""Detectors module."""

from src.modules.detectors.base import BaseDetector
from src.modules.detectors.yolox import YOLOXDetector

__all__ = ["BaseDetector", "YOLOXDetector"]
