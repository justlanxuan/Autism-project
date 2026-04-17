"""Train IMU-video alignment model (MotionBERT-style)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.engine.common import build_alignment_model, build_optimizer, build_loss_fn
from src.modules.matchers import retrieval_top1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IMU-Video alignment")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None, help="Root for npz relative paths")

    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--imu_ckpt", type=str, default="")
    parser.add_argument("--init_alignment_ckpt", type=str, default="")

    parser.add_argument("--embed_dim", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_heads", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--learn_temperature", action="store_true")
    parser.add_argument("--compute_imu_stats", action="store_true")
    parser.add_argument("--imu_stats_json", type=str, default="")
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)

    parser.add_argument("--imu_noise_std", type=float, default=0.01)
    parser.add_argument("--imu_dropout_prob", type=float, default=0.05)
    parser.add_argument("--skel_noise_std", type=float, default=0.005)
    parser.add_argument("--joint_dropout_prob", type=float, default=0.05)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_root",
        type=str,
        default="artifacts",
        help="Root folder to store all training artifacts and checkpoints.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".",
        help="Run subdirectory under output_root; use '.' to place runs directly under output_root.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name; if omitted, timestamp is used.",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


def resolve_save_dir(args: argparse.Namespace) -> Path:
    """Resolve output directory and force all artifacts under output_root."""
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name.strip() if args.run_name else ""
    if not run_name:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")

    save_dir_token = str(args.save_dir).strip().replace("\\", "/")
    if not save_dir_token:
        save_dir_token = "."

    return (output_root / save_dir_token / run_name).resolve()


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "imu": batch["imu"].to(device),
        "skeleton": batch["skeleton"].to(device),
    }


def read_csv_rows(csv_path: str) -> list[dict[str, str]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_imu_stats_from_train_csv(train_csv: str, data_root: str | None) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(train_csv)
    base = Path(data_root) if data_root else Path(train_csv).resolve().parent

    # Aggregate stats per (npz_path, imu_idx)
    from collections import defaultdict
    per_source: dict[tuple[str, int], list] = defaultdict(lambda: [None, None, 0])

    for row in rows:
        rel = row["npz_path"]
        imu_idx = int(row.get("imu_idx", 0))
        key = (rel, imu_idx)
        if per_source[key][2] > 0:
            continue  # already accumulated for this source

        data = np.load((base / rel).resolve(), allow_pickle=True)
        imu = data["imu"].astype(np.float64)
        if imu.ndim == 3:
            imu = imu[:, imu_idx, :]

        per_source[key][0] = imu.sum(axis=0)
        per_source[key][1] = (imu * imu).sum(axis=0)
        per_source[key][2] = imu.shape[0]

    total_count = sum(v[2] for v in per_source.values())
    if total_count == 0:
        raise ValueError("No IMU frames found while computing stats.")

    sums = np.zeros_like(list(per_source.values())[0][0])
    sq_sums = np.zeros_like(list(per_source.values())[0][1])
    for s, sq, c in per_source.values():
        sums += s
        sq_sums += sq

    mean = sums / total_count
    var = np.maximum(sq_sums / total_count - mean * mean, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def maybe_augment_inputs(imu: torch.Tensor, skeleton: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    if args.imu_noise_std > 0:
        imu = imu + torch.randn_like(imu) * args.imu_noise_std

    if args.imu_dropout_prob > 0:
        feat_keep = (torch.rand(imu.shape[0], 1, imu.shape[2], device=imu.device) > args.imu_dropout_prob).float()
        imu = imu * feat_keep

    if args.skel_noise_std > 0:
        skeleton = skeleton + torch.randn_like(skeleton) * args.skel_noise_std

    if args.joint_dropout_prob > 0:
        joint_keep = (
            torch.rand(skeleton.shape[0], 1, skeleton.shape[2], 1, device=skeleton.device) > args.joint_dropout_prob
        ).float()
        skeleton = skeleton * joint_keep

    return imu, skeleton


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_epoch(model, data_loader, loss_fn, device) -> Dict[str, float]:
    if data_loader is None:
        return {"loss": 0.0, "top1": 0.0}

    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            b = move_to_device(batch, device)
            out = model(imu=b["imu"], skeleton=b["skeleton"])
            loss = loss_fn(out["imu"], out["video"])
            acc = retrieval_top1(out["imu"], out["video"])
            total_loss += float(loss.item())
            total_acc += acc
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "top1": 0.0}
    return {
        "loss": total_loss / total_batches,
        "top1": total_acc / total_batches,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    imu_mean = None
    imu_std = None
    if args.imu_stats_json:
        stats = json.loads(Path(args.imu_stats_json).read_text())
        imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
        imu_std = np.asarray(stats["imu_std"], dtype=np.float32)
    elif args.compute_imu_stats:
        imu_mean, imu_std = compute_imu_stats_from_train_csv(args.train_csv, args.data_root)

    imu_sensor = args.imu_sensor.strip() if args.imu_sensor else None
    train_ds = WindowAlignmentDataset(
        args.train_csv,
        root_dir=args.data_root,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
    )
    try:
        val_ds = WindowAlignmentDataset(
            args.val_csv,
            root_dir=args.data_root,
            imu_mean=imu_mean,
            imu_std=imu_std,
            imu_sensor=imu_sensor,
            repeat_single_sensor=args.repeat_single_sensor,
        )
    except ValueError as e:
        if "No rows found" in str(e):
            print(f"[WARN] Validation CSV is empty: {args.val_csv}. Validation will be skipped.")
            val_ds = None
        else:
            raise

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    model, cfg_name = build_alignment_model(args, device, embed_dim=args.embed_dim)
    optimizer = build_optimizer(
        model,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
    )
    loss_fn = build_loss_fn(
        temperature=args.temperature,
        learn_temperature=args.learn_temperature,
        device=device,
    )

    save_dir = resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)

    if imu_mean is not None and imu_std is not None:
        (save_dir / "imu_stats.json").write_text(
            json.dumps({"imu_mean": imu_mean.tolist(), "imu_std": imu_std.tolist()}, indent=2)
        )

    val_count = len(val_ds) if val_ds is not None else 0
    print(f"Train windows: {len(train_ds)}, Val windows: {val_count}")
    print(f"Trainable params: {count_trainable_params(model):,}")
    print(f"Backbone cfg name: {cfg_name}")
    print(f"Artifacts directory: {save_dir}")

    epoch_logs = []

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        for p in model.video_encoder.backbone.parameters():
            p.requires_grad = not freeze_backbone

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            b = move_to_device(batch, device)
            b["imu"], b["skeleton"] = maybe_augment_inputs(b["imu"], b["skeleton"], args)
            out = model(imu=b["imu"], skeleton=b["skeleton"])
            loss = loss_fn(out["imu"], out["video"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            acc = retrieval_top1(out["imu"], out["video"])
            running_loss += float(loss.item())
            running_acc += acc
            steps += 1

            if step % args.log_interval == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] step {step}/{len(train_loader)} "
                    f"loss={running_loss / steps:.4f} top1={running_acc / steps:.4f}"
                )

        val_metrics = evaluate_epoch(model, val_loader, loss_fn, device)
        train_loss = running_loss / max(steps, 1)
        train_top1 = running_acc / max(steps, 1)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f}"
        )

        epoch_logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
            }
        )
        with (save_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_logs[-1], ensure_ascii=True) + "\n")

        payload = {
            "epoch": epoch,
            "args": vars(args),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_top1": val_metrics["top1"],
        }
        torch.save(payload, save_dir / "last.pt")

        score_for_best = val_metrics["top1"] if val_loader is not None else train_top1
        if score_for_best > best_val:
            best_val = score_for_best
            torch.save(payload, save_dir / "best.pt")

    metrics = {"best_val_top1": best_val, "save_dir": str(save_dir)}
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Training complete. Best val top1={best_val:.4f}")


if __name__ == "__main__":
    main()
