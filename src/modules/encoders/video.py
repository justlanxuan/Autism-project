"""Video encoders."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.modules.encoders.base import BaseEncoder


class VideoEncoder(BaseEncoder):
    """MotionBERT-based video encoder producing window-level embeddings."""

    def __init__(
        self,
        backbone: nn.Module,
        rep_dim: int = 512,
        temporal_layers: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.joint_compress = nn.Linear(17 * rep_dim, rep_dim)
        self.temporal_lstm = nn.LSTM(
            input_size=rep_dim,
            hidden_size=rep_dim,
            num_layers=temporal_layers,
            batch_first=True,
        )

    def encode(self, skeleton_xyz: torch.Tensor) -> torch.Tensor:
        return self.forward(skeleton_xyz)

    def forward(self, skeleton_xyz: torch.Tensor) -> torch.Tensor:
        # skeleton_xyz: [B, T, 17, 3]
        rep = self.backbone(skeleton_xyz, return_rep=True)  # [B, T, 17, 512]
        bsz, tlen, joints, rep_dim = rep.shape
        frame_rep = self.joint_compress(
            rep.reshape(bsz * tlen, joints * rep_dim)
        ).reshape(bsz, tlen, rep_dim)

        h_0 = torch.zeros(self.temporal_lstm.num_layers, bsz, rep_dim, device=rep.device)
        c_0 = torch.zeros(self.temporal_lstm.num_layers, bsz, rep_dim, device=rep.device)
        out, _ = self.temporal_lstm(frame_rep, (h_0, c_0))
        return out[:, -1, :]
