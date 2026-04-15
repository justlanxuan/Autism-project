"""Base interface for video-level skeleton extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class VideoSkeletonExtractor(ABC):
    """High-level wrapper: video -> skeleton.json."""

    @abstractmethod
    def extract(self, video_path: str, output_dir: str) -> str:
        """Extract skeleton from a video and return the path to skeleton.json.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to store intermediate and final outputs.

        Returns:
            Absolute path to the generated skeleton.json.
        """
        raise NotImplementedError
