#!/bin/bash
# 简化版运行脚本 - 使用统一 conda 环境

set -e

# 激活环境
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate autism_reid

echo "=== Environment: $CONDA_DEFAULT_ENV ==="
echo "Python: $(which python)"

export PYTHONPATH="${PWD}:${PWD}/src"

CONFIG="${1:-configs/totalcapture_test.yaml}"
STAGE="${2:-all}"

echo "Config: $CONFIG"
echo "Stage: $STAGE"

case $STAGE in
  preprocess)
    echo "=== Stage 1: Preprocessing ==="
    python src/data/preprocess_totalcapture.py --config "$CONFIG"
    ;;
  
  train)
    echo "=== Stage 2: Training ==="
    python experiments/train.py --config "$CONFIG" --stage train
    ;;
  
  test)
    echo "=== Stage 3: Testing ==="
    python experiments/evaluate.py --config "$CONFIG"
    ;;
  
  all)
    echo "=== Full Pipeline ==="
    python src/data/preprocess_totalcapture.py --config "$CONFIG"
    python experiments/train.py --config "$CONFIG" --stage train
    python experiments/evaluate.py --config "$CONFIG"
    ;;
  
  *)
    echo "Usage: $0 [config.yaml] [preprocess|train|test|all]"
    echo ""
    echo "Configs:"
    echo "  configs/totalcapture.yaml       - Full training (S1-S5, 50 epochs)"
    echo "  configs/totalcapture_test.yaml  - Quick test (S1 only, 2 epochs)"
    echo "  configs/custom.yaml             - Custom 4-fold dataset"
    exit 1
    ;;
esac

echo "=== Done ==="
