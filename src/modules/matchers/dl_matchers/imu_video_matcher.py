"""Deep-learning matcher: encoder pair wrapper."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.modules.encoders.imu import IMUEncoder
from src.modules.encoders.video import VideoEncoder


class IMUVideoMatcher(nn.Module):
    """IMU-Video cross-modal matching model."""

    def __init__(self, imu_encoder: IMUEncoder, video_encoder: VideoEncoder) -> None:
        super().__init__()
        self.imu_encoder = imu_encoder
        self.video_encoder = video_encoder

    def forward(
        self, imu: torch.Tensor, skeleton: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        z_imu = self.imu_encoder(imu)
        z_vid = self.video_encoder(skeleton)
        return {"imu": z_imu, "video": z_vid}
