"""Encoder utilities: backbone builders and checkpoint loaders."""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch

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


def build_motionbert_backbone(
    config_path: str | Path,
) -> Tuple[torch.nn.Module, SimpleNamespace]:
    """Build MotionBERT backbone from config."""
    args = get_config(str(config_path))
    backbone = load_backbone(args)
    return backbone, args


def load_motionbert_checkpoint(
    backbone: torch.nn.Module,
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
    imu_encoder: torch.nn.Module,
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
            nk = nk[len("imu_encoder."):]
        if nk.startswith("lstm."):
            cleaned[nk] = v

    if not cleaned:
        if all(k in raw for k in imu_encoder.state_dict().keys()):
            cleaned = {k: raw[k] for k in imu_encoder.state_dict().keys()}
        else:
            raise ValueError("No IMU encoder keys found in checkpoint.")

    imu_encoder.load_state_dict(cleaned, strict=strict)
