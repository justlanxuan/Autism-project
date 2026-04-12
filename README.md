# Autism Reid Project

IMU-video alignment and re-identification using MotionBERT backbone. Supports two data workflows:
- **Vicon Skeleton**: High-precision motion capture (ground truth)
- **Video Skeleton**: Extracted from videos using AlphaPose + ByteTrack

## Quick Start

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n autism_reid python=3.10 -y
conda activate autism_reid

# Install PyTorch with CUDA
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install core dependencies
conda install numpy=1.26 scipy=1.11 pandas matplotlib opencv -y

# Install Python packages
pip install tqdm pyyaml pillow tensorboard timm einops scikit-learn scikit-image
pip install pycocotools lap cython-bbox thop easydict loguru
pip install natsort tabulate  # For video processing
```

### 2. Set Environment Variables

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
conda activate autism_reid
export MKL_THREADING_LAYER=GNU
export PYTHONPATH="${PWD}:${PWD}/src"
```

---

## Workflow A: Vicon Skeleton (Recommended)

Use high-precision Vicon motion capture data (ground truth). **Faster and more accurate.**

### Configurations

| Config | Description |
|--------|-------------|
| `configs/totalcapture_test.yaml` | Quick test: S1 only, 2 epochs |
| `configs/totalcapture.yaml` | Full training: S1-S5, 50 epochs |
| `configs/custom.yaml` | Custom 4-fold cross-validation |

### Run Complete Workflow

```bash
# One-command workflow
./run.sh configs/totalcapture_test.yaml all

# Or step by step:
./run.sh configs/totalcapture_test.yaml preprocess
./run.sh configs/totalcapture_test.yaml train
./run.sh configs/totalcapture_test.yaml test
```

### Manual Execution

```bash
# Step 1: Preprocess (IMU + Vicon skeleton → NPZ)
python src/data/preprocess_totalcapture.py --config configs/totalcapture_test.yaml

# Step 2: Train
python -m src.engine.train \
  --train_csv data/processed/totalcapture_test/windows_train.csv \
  --val_csv data/processed/totalcapture_test/windows_train.csv \
  --data_root data/processed/totalcapture_test \
  --motionbert_root /home/fzliang/origin/MotionBERT \
  --motionbert_config configs/pose3d/MB_ft_h36m_global_lite.yaml \
  --epochs 2 --batch_size 32 \
  --compute_imu_stats --imu_sensor R_LowArm --repeat_single_sensor 4 \
  --output_root artifacts --run_name totalcapture_test

# Step 3: Test
python -m src.engine.eval \
  --test_csv data/processed/totalcapture_test/windows_train.csv \
  --data_root data/processed/totalcapture_test \
  --motionbert_root /home/fzliang/origin/MotionBERT \
  --checkpoint artifacts/totalcapture_test/best.pt
```

---

## Workflow B: Video Skeleton Extraction

Extract skeleton from videos using AlphaPose + ByteTrack. **For datasets without Vicon.**

### Step 1: Create Video Manifest

Create a CSV file listing your videos:

```csv
video_path,video_name,subject,session
/data/videos/session1_cam1.mp4,session1_cam1,subject1,session1
/data/videos/session1_cam2.mp4,session1_cam2,subject1,session1
/data/videos/session2_cam1.mp4,session2_cam1,subject2,session2
```

Save as: `data/interim/video_manifest.csv`

### Step 2: Extract Skeleton from Videos

```bash
python src/data/cli/run_video_skeleton_pipeline.py \
  --manifest_csv data/interim/video_manifest.csv \
  --limit 1 \
  --gpu 0 \
  --results_root data/interim/video_skeleton_outputs \
  --alphapose_root third-party/AlphaPose \
  --alphapose_ckpt pretrained_models/fast_res50_256x192.pth \
  --tracking_mode bytetrack_external \
  --bytetrack_root third-party/ByteTrack \
  --bytetrack_ckpt /home/fzliang/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar \
  --bytetrack_python $(which python) \
  --alphapose_python $(which python)
```

**Outputs:**
- `data/interim/video_skeleton_outputs/<video_name>/alphapose_raw/`
- `data/interim/video_skeleton_outputs/<video_name>/tracking_results.txt`

### Step 3: Preprocess Extracted Skeleton

```bash
python src/data/preprocess_totalcapture.py \
  --config configs/totalcapture_video.yaml
```

Or manually:
```bash
python src/data/preprocess_totalcapture.py \
  --root /data/fzliang/totalcapture \
  --out_dir data/processed/totalcapture_video \
  --skeleton_source alphapose \
  --skeleton_root data/interim/video_skeleton_outputs \
  --window_len 24 --stride 16 \
  --train_subjects S1 --val_subjects S1 --test_subjects S1
```

### Step 4: Train & Test

Same as Workflow A, using the preprocessed data from video skeleton.

---

## Repository Structure

```
Autism-project/
├── configs/                      # Configuration files
│   ├── totalcapture.yaml         # Full Vicon training (S1-S5, 50 epochs)
│   ├── totalcapture_test.yaml    # Quick test (S1, 2 epochs)
│   ├── totalcapture_video.yaml   # Video skeleton workflow
│   └── custom.yaml               # Custom 4-fold dataset
├── run.sh                        # Main workflow script
├── src/
│   ├── data/
│   │   ├── preprocess_totalcapture.py    # Preprocessing
│   │   └── cli/run_video_skeleton_pipeline.py  # Video extraction
│   ├── engine/
│   │   ├── train.py              # Training
│   │   ├── eval.py               # Standard evaluation
│   │   ├── eval_custom.py        # 2-person matching
│   │   └── eval_grouped.py       # Grouped evaluation
│   └── modules/matchers/         # IMU-Video matching models
├── data/
│   ├── processed/                # Preprocessed NPZ + CSV
│   └── interim/                  # Video extraction outputs
├── artifacts/                    # Training outputs
└── third-party/                  # AlphaPose, ByteTrack
```

---

## External Dependencies & Checkpoints

### Required Checkpoints

| Checkpoint | Expected Location | Purpose |
|------------|-------------------|---------|
| MotionBERT | `/home/fzliang/origin/MotionBERT/checkpoint/pretrain/MB_lite_models.bin` | Pose backbone |
| IMU encoder | `/home/fzliang/origin/despite/totalcapture/checkpoints/si_totalcapture_best.pth` | IMU encoder (optional) |
| ByteTrack | `/home/fzliang/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar` | Person tracking |
| AlphaPose | `third-party/AlphaPose/pretrained_models/fast_res50_256x192.pth` | 2D pose detection |

**Note:** If checkpoints are missing, models will train from random initialization (slower convergence).

---

## Configuration Guide

### Skeleton Source Options

**Vicon (Ground Truth):**
```yaml
preprocess:
  skeleton_source: vicon  # Use Vicon motion capture
```

**AlphaPose (Video-based):**
```yaml
preprocess:
  skeleton_source: alphapose
  skeleton_root: data/interim/video_skeleton_outputs
```

### Key Training Parameters

```yaml
train:
  epochs: 50
  batch_size: 64
  imu_sensor: R_LowArm        # Which IMU sensor to use
  repeat_single_sensor: 4     # Repeat sensor 4x = 48-dim input
  compute_imu_stats: true     # Compute mean/std from training data
```

---

## Troubleshooting

### MKL Threading Error
```bash
export MKL_THREADING_LAYER=GNU
```

### CUDA Out of Memory
```bash
# Reduce batch size
python -m src.engine.train ... --batch_size 16

# Or use CPU
python -m src.engine.train ... --device cpu
```

### Missing Module
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PWD}:${PWD}/src"

# Install missing packages
pip install <package_name>
```

---

## Notes

- First run computes IMU statistics and saves to `imu_stats.json` for reuse
- Use `configs/totalcapture_test.yaml` for quick testing (~5 minutes)
- Use `configs/totalcapture.yaml` for full training (~2 hours)
- Video extraction is slow (~5-10 min per video) but only needed once
