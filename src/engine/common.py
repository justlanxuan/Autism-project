"""Common utilities for training and evaluation engines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.modules.encoders import IMUEncoder, VideoEncoder
from src.modules.encoders.utils import (
    build_motionbert_backbone,
    load_motionbert_checkpoint,
    load_despite_imu_weights,
)
from src.modules.matchers import IMUVideoMatcher, SymmetricInfoNCE


def build_alignment_model(
    args: Any,
    device: torch.device,
    embed_dim: int = 512,
) -> Tuple[IMUVideoMatcher, str]:
    """Build IMU-Video alignment model from CLI/config args.

    Returns:
        model: Assembled IMUVideoMatcher on device
        cfg_name: MotionBERT config name for logging
    """
    motionbert_root = Path(args.motionbert_root).expanduser().resolve()
    if str(motionbert_root) not in sys.path:
        sys.path.insert(0, str(motionbert_root))

    config_path = Path(args.motionbert_config)
    if not config_path.is_absolute():
        config_path = motionbert_root / config_path

    ckpt_path = Path(args.motionbert_ckpt) if getattr(args, "motionbert_ckpt", "") else None
    if ckpt_path is not None and not ckpt_path.is_absolute():
        ckpt_path = motionbert_root / ckpt_path

    backbone, cfg = build_motionbert_backbone(str(config_path))
    skip_motionbert_ckpt = getattr(args, "skip_motionbert_ckpt", False)
    if not skip_motionbert_ckpt:
        if ckpt_path is None:
            raise ValueError("--motionbert_ckpt is required unless --skip_motionbert_ckpt is set.")
        load_motionbert_checkpoint(backbone, str(ckpt_path), strict=True)
    else:
        print("[WARN] skip_motionbert_ckpt enabled: using randomly initialized MotionBERT backbone.")

    imu_encoder = IMUEncoder(input_size=48, hidden_size=embed_dim, num_layers=2, device=str(device))
    imu_ckpt = getattr(args, "imu_ckpt", "")
    if imu_ckpt:
        imu_ckpt_path = Path(imu_ckpt).expanduser()
        if imu_ckpt_path.exists():
            load_despite_imu_weights(imu_encoder, str(imu_ckpt_path), strict=False)
        else:
            print(f"[WARN] IMU checkpoint not found at {imu_ckpt_path}; using random init.")

    video_encoder = VideoEncoder(backbone=backbone, rep_dim=embed_dim, temporal_layers=2)
    model = IMUVideoMatcher(imu_encoder=imu_encoder, video_encoder=video_encoder).to(device)

    init_alignment_ckpt = getattr(args, "init_alignment_ckpt", "")
    if init_alignment_ckpt:
        init_path = Path(init_alignment_ckpt).expanduser()
        if not init_path.exists():
            raise FileNotFoundError(f"init_alignment_ckpt not found: {init_path}")
        raw = torch.load(str(init_path), map_location="cpu")
        init_state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(
            f"Loaded init_alignment_ckpt: {init_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    return model, getattr(cfg, "name", "unknown")


def build_optimizer(
    model: IMUVideoMatcher,
    lr_backbone: float = 1e-5,
    lr_heads: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with separate LRs for backbone and heads."""
    backbone_params = [p for p in model.video_encoder.backbone.parameters() if p.requires_grad]
    head_params = []
    head_params += [p for p in model.imu_encoder.parameters() if p.requires_grad]
    head_params += [p for p in model.video_encoder.joint_compress.parameters() if p.requires_grad]
    head_params += [p for p in model.video_encoder.temporal_lstm.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads},
        ],
        weight_decay=weight_decay,
    )


def build_loss_fn(
    temperature: float = 0.1,
    learn_temperature: bool = False,
    device: torch.device = torch.device("cpu"),
) -> SymmetricInfoNCE:
    """Build InfoNCE loss function."""
    return SymmetricInfoNCE(temperature=temperature, learn_temperature=learn_temperature).to(device)
