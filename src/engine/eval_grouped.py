"""Grouped IMU-video matching evaluation (adapts from MotionBERT)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
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
    p = argparse.ArgumentParser(description="Grouped IMU-video matching evaluation")
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    p.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    p.add_argument("--motionbert_ckpt", type=str, default="")
    p.add_argument("--skip_motionbert_ckpt", action="store_true")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--imu_sensor", type=str, default="R_LowArm")
    p.add_argument("--repeat_single_sensor", type=int, default=4)
    p.add_argument("--chunk_windows", type=int, default=30)
    p.add_argument("--min_chunk_windows", type=int, default=15)
    p.add_argument("--group_sizes", type=str, default="2,4,6,8,16")
    p.add_argument("--num_trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_csv", type=str, default="")
    return p.parse_args()


def parse_group_sizes(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def pair_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean cosine similarity between two sequences of embeddings."""
    n = min(len(a), len(b))
    if n <= 0:
        return -1.0
    sims = [1.0 - cosine(a[t], b[t]) for t in range(n)]
    return float(np.mean(sims))


def build_chunk_units(seq_embeddings: List[Dict], chunk_windows: int = 30, min_chunk_windows: int = 15) -> List[Dict]:
    """Build chunk units from sequences for grouped evaluation."""
    units = []
    for seq in seq_embeddings:
        imu_emb = seq["imu_emb"]
        vid_emb = seq["vid_emb"]
        seq_name = seq["seq_name"]

        n = min(len(imu_emb), len(vid_emb))
        if n < min_chunk_windows:
            continue

        start = 0
        cid = 0
        while start < n:
            end = min(start + chunk_windows, n)
            if end - start >= min_chunk_windows:
                units.append(
                    {
                        "unit_id": f"{seq_name}_c{cid:03d}",
                        "seq_name": seq_name,
                        "imu_emb": imu_emb[start:end],
                        "vid_emb": vid_emb[start:end],
                    }
                )
                cid += 1
            start += chunk_windows
    return units


def evaluate_grouped(units: List[Dict], group_size: int, num_trials: int = 50, seed: int = 42) -> Dict:
    """Evaluate grouped matching accuracy for a given group size."""
    if len(units) < group_size:
        return {
            "group_size": group_size,
            "num_units": len(units),
            "num_trials": 0,
            "mean_acc": None,
            "std_acc": None,
            "mean_diag_sim": None,
            "mean_offdiag_sim": None,
            "note": f"insufficient units ({len(units)} < {group_size})",
        }

    rng = np.random.default_rng(seed)
    trial_acc = []
    trial_diag = []
    trial_offdiag = []

    for _ in range(num_trials):
        idx = rng.choice(len(units), size=group_size, replace=False)
        sel = [units[i] for i in idx]

        sim = np.zeros((group_size, group_size), dtype=np.float32)
        for i in range(group_size):
            for j in range(group_size):
                sim[i, j] = pair_similarity(sel[i]["imu_emb"], sel[j]["vid_emb"])

        row_ind, col_ind = linear_sum_assignment(-sim)
        correct = np.sum(row_ind == col_ind)
        trial_acc.append(float(correct) / float(group_size))

        trial_diag.append(float(np.mean(np.diag(sim))))
        if group_size > 1:
            mask = ~np.eye(group_size, dtype=bool)
            trial_offdiag.append(float(np.mean(sim[mask])))
        else:
            trial_offdiag.append(float("nan"))

    return {
        "group_size": group_size,
        "num_units": len(units),
        "num_trials": num_trials,
        "mean_acc": float(np.mean(trial_acc)),
        "std_acc": float(np.std(trial_acc)),
        "mean_diag_sim": float(np.mean(trial_diag)),
        "mean_offdiag_sim": float(np.mean(trial_offdiag)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = WindowAlignmentDataset(
        args.test_csv,
        root_dir=args.data_root,
        imu_sensor=args.imu_sensor,
        repeat_single_sensor=args.repeat_single_sensor,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = []
    with open(args.test_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

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
    model.eval()

    print(f"Computing embeddings for {len(ds)} windows...")
    imu_all = []
    vid_all = []
    with torch.no_grad():
        for batch in loader:
            imu = batch["imu"].to(device)
            skel = batch["skeleton"].to(device)
            out = model(imu=imu, skeleton=skel)
            imu_all.append(out["imu"].detach().cpu().numpy())
            vid_all.append(out["video"].detach().cpu().numpy())

    imu_all = np.concatenate(imu_all, axis=0)
    vid_all = np.concatenate(vid_all, axis=0)

    # Group embeddings by sequence (subject_session)
    seq_map = {}
    for i, row in enumerate(rows):
        seq_name = f"{row['subject']}_{row['session']}"
        if seq_name not in seq_map:
            seq_map[seq_name] = {"imu": [], "vid": []}
        seq_map[seq_name]["imu"].append(imu_all[i])
        seq_map[seq_name]["vid"].append(vid_all[i])

    seq_embeddings = []
    for seq_name, d in seq_map.items():
        seq_embeddings.append(
            {
                "seq_name": seq_name,
                "imu_emb": np.stack(d["imu"], axis=0),
                "vid_emb": np.stack(d["vid"], axis=0),
            }
        )

    print(f"Created {len(seq_embeddings)} sequences from {len(rows)} windows")

    # Build chunk units for grouped evaluation
    units = build_chunk_units(
        seq_embeddings,
        chunk_windows=args.chunk_windows,
        min_chunk_windows=args.min_chunk_windows,
    )
    print(f"Built {len(units)} chunk units for grouped evaluation")

    # Evaluate for each group size
    grouped_results = []
    for g in parse_group_sizes(args.group_sizes):
        print(f"Evaluating group size {g}...")
        result = evaluate_grouped(units, g, num_trials=args.num_trials, seed=args.seed)
        grouped_results.append(result)
        if result["mean_acc"] is not None:
            print(f"  mean_acc={result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
        else:
            print(f"  {result.get('note', 'skipped')}")

    payload = {
        "args": vars(args),
        "num_sequences": len(seq_embeddings),
        "num_units": len(units),
        "grouped_results": grouped_results,
    }

    print("\n" + json.dumps(payload, indent=2))

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved grouped results to {out_json}")

    if args.save_csv:
        import pandas as pd

        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(grouped_results).to_csv(out_csv, index=False)
        print(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    main()
