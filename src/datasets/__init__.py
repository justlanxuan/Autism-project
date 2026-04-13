"""Datasets module."""

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.datasets.base import BaseData, BaseProcess
from src.datasets.totalcapture import TotalCaptureAdapter
from src.datasets.custom import Custom4FoldAdapter
from src.datasets.mmact import MMActDataset

__all__ = [
    "WindowAlignmentDataset",
    "BaseData",
    "BaseProcess",
    "TotalCaptureAdapter",
    "Custom4FoldAdapter",
    "MMActDataset",
]
