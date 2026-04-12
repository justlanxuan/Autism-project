#!/usr/bin/env python3
"""Extract AlphaPose skeleton JSON for a video (optional ByteTrack)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def run_cmd(cmd: list[str], cwd: Path | None, env: dict | None, dry_run: bool) -> None:
    cmd_str = " ".join(cmd)
    print(f"\n[CMD] {cmd_str}")
    if cwd is not None:
        print(f"[CWD] {cwd}")
    if dry_run:
        print("[DRY_RUN] skip execution")
        return
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def apply_headless_env(env: dict[str, str]) -> dict[str, str]:
    """Apply environment settings for servers without an X display."""
    out = dict(env)
    out.setdefault("MPLBACKEND", "Agg")
    out.setdefault("QT_QPA_PLATFORM", "offscreen")
    out.setdefault("SDL_VIDEODRIVER", "dummy")
    out.setdefault("DISPLAY", "")
    out.setdefault("HEADLESS", "1")
    return out


def extract_video_frames(video_path: Path, frame_dir: Path, dry_run: bool) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frame_dir / f"{idx}.jpg"), frame)
        idx += 1
    cap.release()

    if idx == 0:
        raise RuntimeError("No frames extracted from input video.")


def pick_latest_txt(track_vis_dir: Path) -> Path:
    txts = sorted(track_vis_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    if not txts:
        raise FileNotFoundError(f"No tracking txt found under: {track_vis_dir}")
    return txts[-1]


def convert_bytetrack_txt_to_alphapose_detfile(
    track_txt: Path, frame_dir: Path, out_json: Path, dry_run: bool
) -> None:
    if dry_run:
        return

    rows = []
    with track_txt.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            score = float(parts[6])
            rows.append((frame_id, track_id, x, y, w, h, score))

    if not rows:
        raise RuntimeError(f"No valid rows parsed from {track_txt}")

    min_frame = min(r[0] for r in rows)
    dets = []
    for frame_id, track_id, x, y, w, h, score in rows:
        frame_idx = frame_id - min_frame
        image_path = frame_dir / f"{frame_idx}.jpg"
        if not image_path.exists():
            continue
        dets.append(
            {
                "image_id": str(image_path),
                "bbox": [x, y, w, h],
                "score": score,
                "idx": track_id,
            }
        )

    if not dets:
        raise RuntimeError("Converted detfile is empty. Check ByteTrack output and extracted frames.")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(dets, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video -> AlphaPose skeleton JSON pipeline")

    parser.add_argument("--video", type=str, default=None, help="Single input video path")
    parser.add_argument(
        "--manifest_csv",
        type=str,
        default=None,
        help="Optional manifest with a video_path column; if set, process all rows",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process from manifest (0 = all)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if skeleton.json already exists")

    parser.add_argument("--results_root", type=str, default="src/data/interim/video_pipeline_outputs")
    parser.add_argument("--video_name", type=str, default=None, help="Output folder name for single video")
    parser.add_argument("--run_name", type=str, default=None, help="Legacy alias for --video_name")

    parser.add_argument("--alphapose_root", type=str, default="/home/fzliang/origin/AlphaPose")
    parser.add_argument(
        "--alphapose_python",
        type=str,
        default=sys.executable,
        help="Python executable for AlphaPose environment",
    )
    parser.add_argument(
        "--alphapose_cfg",
        type=str,
        default="configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
    )
    parser.add_argument(
        "--alphapose_ckpt",
        type=str,
        default="/home/fzliang/AlphaPose/pretrained_models/fast_res50_256x192.pth",
    )

    parser.add_argument(
        "--tracking_mode",
        type=str,
        default="alphapose_internal",
        choices=["alphapose_internal", "bytetrack_external"],
    )

    parser.add_argument(
        "--pose_track_backend",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "alphapose"],
    )
    parser.add_argument("--bt_track_thresh", type=float, default=0.5)
    parser.add_argument("--bt_track_buffer", type=int, default=30)
    parser.add_argument("--bt_match_thresh", type=float, default=0.8)
    parser.add_argument("--bt_mot20", action="store_true")

    parser.add_argument("--detbatch", type=int, default=None)
    parser.add_argument("--posebatch", type=int, default=None)
    parser.add_argument("--use_expandable_segments", action="store_true")

    parser.add_argument("--bytetrack_root", type=str, default="/home/fzliang/origin/ByteTrack")
    parser.add_argument("--bytetrack_python", type=str, default="python3")
    parser.add_argument("--bytetrack_exp_file", type=str, default="exps/example/mot/yolox_x_mix_det.py")
    parser.add_argument("--bytetrack_ckpt", type=str, default="pretrained/bytetrack_x_mot17.pth.tar")
    parser.add_argument("--bytetrack_conf", type=float, default=0.1)
    parser.add_argument("--bytetrack_nms", type=float, default=0.7)
    parser.add_argument("--bytetrack_tsize", type=int, default=640)
    parser.add_argument("--bytetrack_fp16", action="store_true")
    parser.add_argument("--bytetrack_fuse", action="store_true")
    parser.add_argument("--bytetrack_mot20", action="store_true")
    parser.add_argument("--bytetrack_track_thresh", type=float, default=0.5)
    parser.add_argument("--bytetrack_track_buffer", type=int, default=30)
    parser.add_argument("--bytetrack_match_thresh", type=float, default=0.8)
    parser.add_argument("--bytetrack_min_box_area", type=float, default=10.0)
    parser.add_argument("--bytetrack_aspect_ratio_thresh", type=float, default=1.6)

    parser.add_argument("--gpu", type=int, default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--headless",
        dest="headless",
        action="store_true",
        help="Run subprocesses in headless mode (default).",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Disable headless environment overrides.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only")
    parser.set_defaults(headless=True)

    return parser.parse_args()


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def torch_cuda_available() -> bool:
    """Best-effort check to decide whether GPU execution is actually possible."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def iter_videos(args: argparse.Namespace) -> Iterable[Path]:
    if args.manifest_csv:
        manifest_path = Path(args.manifest_csv).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
        with manifest_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                video_path = row.get("video_path", "").strip()
                if not video_path:
                    continue
                count += 1
                if args.limit and count > args.limit:
                    break
                yield Path(video_path).expanduser().resolve()
    else:
        if not args.video:
            raise ValueError("Provide --video or --manifest_csv")
        yield Path(args.video).expanduser().resolve()


def run_single(video_path: Path, args: argparse.Namespace, results_root: Path) -> Path:
    if not video_path.exists() and not args.dry_run:
        raise FileNotFoundError(f"Video not found: {video_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    default_video_name = video_path.stem
    video_name = args.video_name or args.run_name or default_video_name

    video_result_dir = results_root / video_name
    video_result_dir.mkdir(parents=True, exist_ok=True)

    skeleton_json = video_result_dir / "skeleton.json"
    if args.skip_existing and skeleton_json.exists():
        print(f"[SKIP] {skeleton_json} already exists")
        return video_result_dir

    alphapose_root = Path(args.alphapose_root).expanduser().resolve()
    if not alphapose_root.exists():
        raise FileNotFoundError(f"AlphaPose root not found: {alphapose_root}")

    ap_cfg = resolve_path(alphapose_root, args.alphapose_cfg)
    ap_ckpt = resolve_path(alphapose_root, args.alphapose_ckpt)

    ap_outdir = (video_result_dir / "alphapose_raw").resolve()
    ap_outdir.mkdir(parents=True, exist_ok=True)
    json_path = ap_outdir / "alphapose-results.json"

    use_gpu = args.gpu is not None and torch_cuda_available()
    if args.gpu is not None and not use_gpu:
        print("[WARN] GPU requested but CUDA is unavailable in current environment; falling back to CPU.")

    env_ap = os.environ.copy()
    env_ap["PYTHONPATH"] = str(alphapose_root)
    if use_gpu:
        env_ap["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        env_ap.pop("CUDA_VISIBLE_DEVICES", None)
    if args.use_expandable_segments:
        env_ap["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if args.headless:
        env_ap = apply_headless_env(env_ap)

    if args.tracking_mode == "bytetrack_external":
        bytetrack_root = Path(args.bytetrack_root).expanduser().resolve()
        if not bytetrack_root.exists():
            raise FileNotFoundError(f"ByteTrack root not found: {bytetrack_root}")

        bt_outdir = video_result_dir / "bytetrack_raw"
        bt_outdir.mkdir(parents=True, exist_ok=True)
        bt_frame_dir = bt_outdir / "frames"
        bt_track_txt = bt_outdir / "bytetrack_tracks.txt"
        bt_detfile_json = bt_outdir / "bytetrack_detfile.json"

        bt_exp_file = resolve_path(bytetrack_root, args.bytetrack_exp_file)
        bt_ckpt = resolve_path(bytetrack_root, args.bytetrack_ckpt)
        bt_exp_name = bt_exp_file.stem
        bt_track_vis_dir = bytetrack_root / "YOLOX_outputs" / bt_exp_name / "track_vis"

        env_bt = os.environ.copy()
        existing_py = env_bt.get("PYTHONPATH", "")
        env_bt["PYTHONPATH"] = str(bytetrack_root)
        if existing_py:
            env_bt["PYTHONPATH"] = f"{bytetrack_root}{os.pathsep}{existing_py}"
        if use_gpu:
            env_bt["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        else:
            env_bt.pop("CUDA_VISIBLE_DEVICES", None)
        if args.headless:
            env_bt = apply_headless_env(env_bt)

        cmd_bt = [
            args.bytetrack_python,
            "tools/demo_track.py",
            "video",
            "--path",
            str(video_path),
            "--save_result",
            "-f",
            str(bt_exp_file),
            "-c",
            str(bt_ckpt),
            "--device",
            "gpu" if use_gpu else "cpu",
            "--conf",
            str(args.bytetrack_conf),
            "--nms",
            str(args.bytetrack_nms),
            "--tsize",
            str(args.bytetrack_tsize),
            "--track_thresh",
            str(args.bytetrack_track_thresh),
            "--track_buffer",
            str(args.bytetrack_track_buffer),
            "--match_thresh",
            str(args.bytetrack_match_thresh),
            "--min_box_area",
            str(args.bytetrack_min_box_area),
            "--aspect_ratio_thresh",
            str(args.bytetrack_aspect_ratio_thresh),
        ]
        if args.bytetrack_fp16:
            cmd_bt.append("--fp16")
        if args.bytetrack_fuse:
            cmd_bt.append("--fuse")
        if args.bytetrack_mot20:
            cmd_bt.append("--mot20")

        run_cmd(cmd_bt, cwd=bytetrack_root, env=env_bt, dry_run=args.dry_run)

        if not args.dry_run:
            latest_txt = pick_latest_txt(bt_track_vis_dir)
            shutil.copy2(latest_txt, bt_track_txt)
            extract_video_frames(video_path, bt_frame_dir, dry_run=False)
            convert_bytetrack_txt_to_alphapose_detfile(bt_track_txt, bt_frame_dir, bt_detfile_json, dry_run=False)

        cmd_ap = [
            args.alphapose_python,
            "scripts/demo_inference.py",
            "--cfg",
            str(ap_cfg),
            "--checkpoint",
            str(ap_ckpt),
            "--detfile",
            str(bt_detfile_json),
            "--outdir",
            str(ap_outdir),
        ]
        if args.posebatch is not None:
            cmd_ap.extend(["--posebatch", str(args.posebatch)])

        run_cmd(cmd_ap, cwd=alphapose_root, env=env_ap, dry_run=args.dry_run)
    else:
        cmd_ap = [
            args.alphapose_python,
            "scripts/demo_inference.py",
            "--cfg",
            str(ap_cfg),
            "--checkpoint",
            str(ap_ckpt),
            "--video",
            str(video_path),
            "--outdir",
            str(ap_outdir),
            "--pose_track",
            "--showbox",
            "--save_img",
            "--save_video",
        ]

        if args.detbatch is not None:
            cmd_ap.extend(["--detbatch", str(args.detbatch)])
        if args.posebatch is not None:
            cmd_ap.extend(["--posebatch", str(args.posebatch)])

        run_cmd(cmd_ap, cwd=alphapose_root, env=env_ap, dry_run=args.dry_run)

    if not args.dry_run and not json_path.exists():
        raise FileNotFoundError(f"AlphaPose JSON not found: {json_path}")
    if not args.dry_run:
        shutil.copy2(json_path, skeleton_json)

    summary = {
        "video": str(video_path),
        "video_name": str(video_name),
        "video_result_dir": str(video_result_dir),
        "tracking_mode": args.tracking_mode,
        "bytetrack_root": str(args.bytetrack_root),
        "alphapose_outdir": str(ap_outdir),
        "alphapose_json": str(json_path),
        "skeleton_json": str(skeleton_json),
        "pose_track_backend": args.pose_track_backend,
        "timestamp": ts,
    }

    summary_path = video_result_dir / "pipeline_run_summary.json"
    if not args.dry_run:
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

    print("\nPipeline finished.")
    print(f"Video result dir: {video_result_dir}")
    print(f"Skeleton JSON: {skeleton_json}")
    if not args.dry_run:
        print(f"Run summary: {summary_path}")

    return video_result_dir


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    for video_path in iter_videos(args):
        run_single(video_path, args, results_root)


if __name__ == "__main__":
    main()
