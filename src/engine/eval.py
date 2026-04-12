"""Evaluate IMU-video alignment model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.modules.matchers import (
    IMUEncoder,
    IMUVideoMatcher,
    VideoEncoder,
    SymmetricInfoNCE,
    retrieval_top1,
    build_motionbert_backbone,
    load_motionbert_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IMU-video alignment model")
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imu_sensor = args.imu_sensor.strip() if args.imu_sensor else None
    ds = WindowAlignmentDataset(
        args.test_csv,
        root_dir=args.data_root,
        imu_sensor=imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    motionbert_root = Path(args.motionbert_root).expanduser().resolve()
    if str(motionbert_root) not in sys.path:
        sys.path.insert(0, str(motionbert_root))

    # Models already imported from src.modules.matchers

    config_path = Path(args.motionbert_config)
    if not config_path.is_absolute():
        config_path = motionbert_root / config_path

    ckpt_path = Path(args.motionbert_ckpt) if args.motionbert_ckpt else None
    if ckpt_path is not None and not ckpt_path.is_absolute():
        ckpt_path = motionbert_root / ckpt_path

    backbone, _ = build_motionbert_backbone(str(config_path))
    if not args.skip_motionbert_ckpt:
        if ckpt_path is None:
            raise ValueError("--motionbert_ckpt is required unless --skip_motionbert_ckpt is set.")
        load_motionbert_checkpoint(backbone, str(ckpt_path), strict=True)
    else:
        print("[WARN] skip_motionbert_ckpt enabled: using randomly initialized MotionBERT backbone.")

    model = IMUVideoMatcher(
        imu_encoder=IMUEncoder(input_size=48, hidden_size=512, num_layers=2, device=str(device)),
        video_encoder=VideoEncoder(backbone=backbone, rep_dim=512, temporal_layers=2),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    loss_fn = SymmetricInfoNCE(temperature=args.temperature).to(device)

    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            imu = batch["imu"].to(device)
            skeleton = batch["skeleton"].to(device)
            out = model(imu=imu, skeleton=skeleton)
            total_loss += float(loss_fn(out["imu"], out["video"]).item())
            total_top1 += retrieval_top1(out["imu"], out["video"]) 
            n += 1

    metrics = {
        "loss": total_loss / max(n, 1),
        "top1": total_top1 / max(n, 1),
        "num_batches": n,
        "num_windows": len(ds),
    }

    print(json.dumps(metrics, indent=2))
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
