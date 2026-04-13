"""Pipelines module."""

from src.pipelines.base import PipelineStage
from src.pipelines.full_pipeline import FullPipeline
from src.pipelines.stages import SliceStage, ExtractStage, TrainStage, TestStage

__all__ = [
    "PipelineStage",
    "FullPipeline",
    "SliceStage",
    "ExtractStage",
    "TrainStage",
    "TestStage",
]
