"""IMU-Video alignment/matching models."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import sys

import torch
import torch.nn as nn

# Import MotionBERT lib from origin (external, read-only)
MOTIONBERT_ROOT = Path(os.environ.get("MOTIONBERT_ROOT", "/home/fzliang/origin/MotionBERT"))
if str(MOTIONBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONBERT_ROOT))

try:
    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone
except ImportError as e:
    raise RuntimeError(
        f"Failed to import MotionBERT from {MOTIONBERT_ROOT}. "
        "Set MOTIONBERT_ROOT environment variable."
    ) from e


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve checkpoint path with multiple fallback strategies."""
    p = Path(checkpoint_path).expanduser()
    candidates = [p]

    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(_repo_root() / p)

    for c in candidates:
        if c.exists():
            return c.resolve()

    discovered = []
    for pat in ["checkpoint/**/*.bin", "checkpoint/**/*.pt", "checkpoint/**/*.pth"]:
        discovered.extend(glob.glob(str(_repo_root() / pat), recursive=True))
    discovered = sorted(discovered)[:12]
    discovered_txt = "\n".join(discovered) if discovered else "(none found under checkpoint/)"

    raise FileNotFoundError(
        "MotionBERT checkpoint not found.\n"
        f"Provided path: {checkpoint_path}\n"
        "Tried locations:\n"
        + "\n".join(str(c) for c in candidates)
        + "\n\nDiscovered checkpoint-like files:\n"
        + discovered_txt
    )


class IMUEncoder(nn.Module):
    """DeSPITE IMU encoder structure (2-layer LSTM, last-step output)."""

    def __init__(
        self,
        input_size: int = 48,
        hidden_size: int = 512,
        num_layers: int = 2,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        h_0 = torch.zeros(self.num_layers, bsz, self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, bsz, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        return out[:, -1, :].reshape(-1, self.hidden_size)


class VideoEncoder(nn.Module):
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


def build_motionbert_backbone(
    config_path: str | Path,
) -> Tuple[nn.Module, SimpleNamespace]:
    """Build MotionBERT backbone from config."""
    args = get_config(str(config_path))
    backbone = load_backbone(args)
    return backbone, args


def load_motionbert_checkpoint(
    backbone: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> None:
    """Load MotionBERT checkpoint with automatic state_dict extraction."""
    ckpt_path = resolve_checkpoint_path(checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "model_pos" in ckpt:
        state = ckpt["model_pos"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        clean_state[nk] = v

    backbone.load_state_dict(clean_state, strict=strict)


def load_despite_imu_weights(
    imu_encoder: IMUEncoder,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> None:
    """Load DeSPITE IMU encoder weights."""
    ckpt_path = resolve_checkpoint_path(checkpoint_path)
    raw = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]

    if not isinstance(raw, dict):
        raise ValueError("Unsupported IMU checkpoint format.")

    cleaned = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("imu_encoder."):
            nk = nk[len("imu_encoder.") :]
        if nk.startswith("lstm."):
            cleaned[nk] = v

    if not cleaned:
        if all(k in raw for k in imu_encoder.state_dict().keys()):
            cleaned = {k: raw[k] for k in imu_encoder.state_dict().keys()}
        else:
            raise ValueError("No IMU encoder keys found in checkpoint.")

    imu_encoder.load_state_dict(cleaned, strict=strict)
