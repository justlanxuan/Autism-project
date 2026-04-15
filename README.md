# Autism Reid Project

IMU-video alignment and re-identification

## Quick Start

### 1. Environment Setup

```bash
# Create base environment
conda create -n autism_test python=3.10 -y
conda activate autism_test

# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install numpy==1.26.0 scipy==1.11.4 pandas matplotlib opencv-python \
    tensorboard timm einops scikit-learn scikit-image pycocotools lap \
    cython-bbox thop easydict loguru natsort tabulate
```


### 2. Prepare External Repositories & Checkpoints

#### 2.1 Clone submodules (AlphaPose + ByteTrack)

This repository uses Git submodules for third-party pose estimation and tracking tools. After cloning, initialize them:

```bash
git submodule update --init --recursive
```

Submodules configured:
- **AlphaPose** → `third-party/AlphaPose` (`https://github.com/L-Ark/AlphaPose.git`)
- **ByteTrack** → `third-party/ByteTrack` (`https://github.com/L-Ark/ByteTrack.git`)

#### 2.2 Clone additional dependencies (MotionBERT + despite)

These repositories are expected to exist as **sibling directories** next to this project (or update the paths in your config accordingly):

```bash
# In the parent directory of this repository
git clone https://github.com/Walter0807/MotionBERT.git
git clone https://github.com/thkreutz/despite.git
```

Expected layout:
```
/workspace/
├── Autism-project/          # this repository
├── MotionBERT/              # pose backbone
└── despite/                 # optional IMU encoder pretraining
```

#### 2.3 Download checkpoints

| Checkpoint | Expected Path | Download Source | Purpose |
|------------|---------------|-----------------|---------|
| **MotionBERT-Lite** | `MotionBERT/checkpoint/pretrain/MB_lite_models.bin` | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS27Ydcbpxlkl0ng?e=rq2Btn) | Pose backbone |
| **AlphaPose** | `Autism-project/third-party/AlphaPose/pretrained_models/fast_res50_256x192.pth` | [Google Drive](https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn) | 2D pose detection |
| **ByteTrack** | `Autism-project/third-party/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar` | [Google Drive](https://drive.google.com/file/d/1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5/view?usp=sharing) | Person tracking |
| **IMU encoder** | `despite/pretrained_models/v2/SIE_v2.pth` | [TUDataLib](https://tudatalib.ulb.tu-darmstadt.de/items/17c47531-5e6d-4c86-a685-740d8f94f398) (download `OpenAccess_models.zip`) | DeSPITE pre-trained Skeleton+IMU encoder |

> **Note:** If neither IMU checkpoint is available, the matcher will still train from random initialization.

### 3. Preprocess

If you are using a **video-based workflow** (e.g. `configs/totalcapture_video_test.yaml`), generate the video manifest first. The preprocess parameters are defined in the `preprocess` section of your config:

```yaml
preprocess:
  dataset: totalcapture        # dataset type: totalcapture or custom
  raw_root: /data/fzliang/totalcapture
  camera: cam1
  output: ./data/interim/video_manifest.csv
```

Run it directly from the config:

```bash
python -m src.data.preprocess.preprocess_totalcapture --config configs/totalcapture_video_test.yaml
```

Or run manually with CLI overrides:

```bash
python -m src.data.preprocess.preprocess_totalcapture \
    --root /data/fzliang/totalcapture \
    --camera cam1 \
    --output ./data/interim/video_manifest.csv
```

### 4. Run the Full Pipeline

The unified pipeline supports four stages: `extract` (video skeleton), `slice` (IMU-skeleton slicing into windows), `train`, `test`.


```bash
# Run everything
./run.sh configs/totalcapture_video_test.yaml all

# Or run individual stages
./run.sh configs/totalcapture_video_test.yaml extract      # video -> skeleton (for video workflows)
./run.sh configs/totalcapture_video_test.yaml slice        # IMU + skeleton -> npz + csv
./run.sh configs/totalcapture_video_test.yaml train        # train matcher
./run.sh configs/totalcapture_video_test.yaml test         # evaluate matcher
```

> **Note:** The default stage order for `all` is `extract -> slice -> train -> test`. For Vicon-based configs (no `extract` section), the extract stage is automatically skipped.

You can also call the Python CLI directly:

```bash
python -m src.cli.run_pipeline --config configs/totalcapture_video_test.yaml --stages all
python -m src.cli.run_pipeline --config configs/totalcapture_video_test.yaml --stages extract,slice
```

---

## Available Configs

| Config | Description |
|--------|-------------|
| `configs/totalcapture_vicon_test.yaml` | Quick Vicon test: S1 only, 2 epochs |
| `configs/totalcapture_vicon.yaml` | Full Vicon training: S1-S5, 50 epochs |
| `configs/totalcapture_video_test.yaml` | Video workflow quick test: 1 video, S1, 2 epochs |
| `configs/totalcapture_video.yaml` | Full video workflow training: all videos, S1-S5, 50 epochs |
| `configs/custom.yaml` | Custom 4-fold cross-validation |

---

## Project Structure

```
Autism-project/
├── configs/                      # YAML configuration files
├── src/
│   ├── cli/                      # Command-line entrypoints
│   │   ├── run_pipeline.py       # Unified pipeline driver
│   │   ├── run_extract.py        # Unified skeleton extraction dispatcher
│   │   └── run_alphapose_bytetrack.py  # Backend implementation: AlphaPose + ByteTrack
│   │
│   ├── pipelines/                # High-level workflow orchestration
│   │   ├── base.py               # PipelineStage base class
│   │   ├── stages.py             # Extract / Slice / Train / Test stages
│   │   └── full_pipeline.py      # Compose and run stages
│   │
│   ├── engine/                   # Training & evaluation engines
│   │   ├── common.py             # Shared model-building utilities
│   │   ├── train.py              # Training script
│   │   ├── eval.py               # Standard evaluation
│   │   ├── eval_custom.py        # 2-person matching evaluation
│   │   └── eval_grouped.py       # Grouped evaluation
│   │
│   ├── datasets/                 # Dataset adapters + PyTorch Datasets
│   │   ├── alignment_dataset.py  # WindowAlignmentDataset
│   │   ├── totalcapture.py       # TotalCapture preprocessing adapter
│   │   ├── custom.py             # Custom 4-fold adapter
│   │   ├── mmact.py              # MMAct adapter (future)
│   │   └── base.py               # BaseData / BaseProcess abstractions
│   │
│   ├── data/                     # Data preparation & slicing tools
│   │   ├── slice/                  # IMU-skeleton slicing entrypoints
│   │   │   ├── totalcapture.py
│   │   │   └── custom_4fold.py
│   │   ├── preprocess/             # Preprocessing helpers
│   │   │   └── preprocess_totalcapture.py  # Generate video manifest for TotalCapture
│   │   ├── preprocessors/
│   │   │   ├── imu.py
│   │   │   ├── skeleton.py
│   │   │   └── wham.py
│   │   └── adapters/
│   │       └── alphapose.py
│   │
│   ├── modules/                  # Core algorithm modules
│   │   ├── encoders/             # IMU / Video encoders
│   │   │   ├── imu.py            # IMUEncoder
│   │   │   ├── video.py          # VideoEncoder
│   │   │   └── utils.py          # MotionBERT backbone builders
│   │   │
│   │   ├── matchers/
│   │   │   ├── base.py
│   │   │   ├── hungarian.py      # Pure-algorithm matcher
│   │   │   ├── dl_matchers/      # Deep-learning matchers
│   │   │   │   ├── imu_video_matcher.py
│   │   │   │   └── despite_matcher.py
│   │   │   ├── physics_matchers/ # Future: physics-based matchers
│   │   │   └── losses.py
│   │   │
│   │   ├── trackers/             # Tracking adapters
│   │   │   ├── base.py
│   │   │   ├── bytetrack.py
│   │   │   └── alphapose.py
│   │   │
│   │   └── pose_estimators/      # Pose-estimation adapters
│   │       ├── base.py
│   │       ├── alphapose.py
│   │       └── wham_3d.py
│   │
│   └── utils/                    # Utilities
│       ├── config.py
│       ├── factory.py            # Lightweight registry
│       ├── chunk_matcher.py
│       └── merge_tracklets.py
│
├── data/
│   ├── processed/                # Preprocessed NPZ + CSV outputs
│   └── interim/                  # Video skeleton extraction outputs (reusable)
├── artifacts/                    # Training checkpoints & logs
├── third-party/                  # AlphaPose, ByteTrack
└── run.sh                        # Bash wrapper around the pipeline
```

---

## Data Preparation & Slicing

### Step 1: Preprocess (Video manifest generation)

For **video-based workflows** (e.g. `configs/totalcapture_video_test.yaml` or `configs/totalcapture_video.yaml`), you first need to generate a `video_manifest.csv` that lists all videos to be processed. This is a one-time preprocessing step.

```bash
# Generate manifest from config
python -m src.data.preprocess.preprocess_totalcapture --config configs/totalcapture_video_test.yaml

# Or specify manually
python -m src.data.preprocess.preprocess_totalcapture \
    --root /data/fzliang/totalcapture \
    --camera cam1 \
    --output ./data/interim/video_manifest.csv

# Generate manifest for ALL cameras
python -m src.data.preprocess.preprocess_totalcapture \
    --root /data/fzliang/totalcapture \
    --camera all \
    --output ./data/interim/video_manifest_all.csv
```

> **Note:** For Vicon-based configs (e.g. `configs/totalcapture_vicon_test.yaml`), this step is **not needed** because skeleton data already exists as ground-truth motion capture.

---

### Step 2: Slice

The `slice` stage turns raw IMU + skeleton data into windowed training examples. It produces two kinds of outputs:

1. **Per-sequence `.npz` files** — Each sequence gets one compressed NPZ containing the full synchronized IMU and skeleton arrays.
2. **CSV metadata tables** — Sliding-window indices that point into the per-sequence NPZs.

#### Running Slice

```bash
# As part of the full pipeline
./run.sh configs/totalcapture_vicon_test.yaml all

# Or run only the slice stage
./run.sh configs/totalcapture_vicon_test.yaml slice
```

Under the hood this executes `src.datasets.totalcapture.TotalCaptureAdapter` (or `src.data.slice.custom_4fold` for custom configs).

#### Output Layout

After slicing you will see:

```
data/processed/<dataset_name>/
├── sequences/
│   ├── S1_walking1.npz
│   ├── S1_walking2.npz
│   └── ...
├── sequences.csv
├── windows_all.csv
├── windows_train.csv
├── windows_val.csv
└── windows_test.csv
```

The exact `<dataset_name>` is taken from `paths.output_dir` in your config (e.g. `totalcapture` or `custom_4fold`).

### What's Inside the NPZ Files?

Every `.npz` under `sequences/` contains exactly two arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `imu` | `(T, 48)` | 48-dimensional IMU feature per frame |
| `skeleton` | `(T, 17, 3)` | 17-joint 3D skeleton per frame (Human3.6M format) |

`T` is the number of frames in that specific sequence. These arrays are **not** pre-cut into windows; the window slicing happens at training time via the CSV metadata.

### CSV Format

The slice stage writes four CSV files. The most important ones for training are `windows_<split>.csv` (e.g. `windows_train.csv`). Each row represents one training/validation/test window:

| Column | Meaning |
|--------|---------|
| `subject` | Subject identifier (e.g. `S1`, `S2`) |
| `session` | Session / action name (e.g. `walking1`, `acting2`) |
| `split` | `train`, `val`, or `test` |
| `npz_path` | Relative path to the per-sequence NPZ (e.g. `sequences/S1_walking1.npz`) |
| `window_start` | Starting frame index inside the sequence NPZ |
| `window_end` | Ending frame index inside the sequence NPZ |
| `window_len` | Length of the window (usually `config.slice.window_len`) |

During training, `WindowAlignmentDataset` reads these CSVs, loads the referenced NPZ on demand, and extracts `imu[window_start:window_end]` and `skeleton[window_start:window_end]` as a single training sample.

### Key Config Options

Inside the `slice` section of your YAML:

```yaml
slice:
  train_subjects: [S1, S2, S3]   # subjects assigned to train split
  val_subjects: [S4]             # subjects assigned to val split
  test_subjects: [S5]            # subjects assigned to test split
  window_len: 60                 # frames per window
  stride: 30                     # sliding-window step
  skeleton_source: vicon         # or "alphapose" / "wham"
  max_sequences: null            # limit sequences per subject (for quick tests)
```

> **Tip:** The adapter gracefully skips subjects that are not assigned to any split, so you can safely point `data_root` at a folder containing more subjects than you actually want to use.

---

## Workflows

### Vicon Skeleton (Ground Truth)

Uses high-precision motion capture data. **Faster and more accurate.**

Example: `configs/totalcapture_vicon_test.yaml`

```bash
./run.sh configs/totalcapture_vicon_test.yaml all
```

### Video Skeleton Extraction

Extracts skeleton from videos using the detector / tracker / pose-estimator combination specified in your config.

Example: `configs/totalcapture_video_test.yaml`

```yaml
extract:
  tracker: bytetrack               # or "alphapose"
  pose_estimator: alphapose
  manifest_csv: ./data/interim/video_manifest.csv
  results_root: ./data/interim/totalcapture_video_test_skeletons
  skip_existing: true
  gpu: 0
  # Generic checkpoint / root / cfg keys (not hard-coded to specific model names)
  tracker_root: ./third-party/ByteTrack
  tracker_ckpt: /home/fzliang/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar
  pose_estimator_root: ./third-party/AlphaPose
  pose_estimator_cfg: configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml
  pose_estimator_ckpt: pretrained_models/fast_res50_256x192.pth
```

```bash
./run.sh configs/totalcapture_video_test.yaml all
```

The pipeline will:
1. **Preprocess** — generate `video_manifest.csv` from the dataset root
2. **Extract** skeletons from the videos listed in the manifest
3. **Slice** the extracted skeletons with IMU into `npz` + `csv`
4. **Train** the IMU-Video matcher
5. **Test** the trained checkpoint

---

## Notes

- First run computes IMU statistics and saves to `imu_stats.json` for reuse.
- Use `configs/totalcapture_vicon_test.yaml` or `configs/totalcapture_video_test.yaml` for quick testing (~5 minutes).
- Use `configs/totalcapture_vicon.yaml` or `configs/totalcapture_video.yaml` for full training (~2 hours).
- Video extraction is slow (~5-10 min per video) but only needed once; intermediate results are cached under `data/interim/`.
