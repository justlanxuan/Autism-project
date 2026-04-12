#!/bin/bash
###############################################################################
# Custom 2-Person 4-Fold Cross-Validation Pipeline
# 
# Follows MotionBERT/alignment logic for custom 2-person dataset
# Tests matches 2-person groups (the only meaningful group size for this dataset)
###############################################################################

set -e  # Exit on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOTIONBERT_ROOT="/home/fzliang/origin/MotionBERT"
VENV="${MOTIONBERT_ROOT}/venv_alphapose/bin/python"

# Data paths
PREPROCESSED_ROOT="/data/fzliang/data/preprocess/2person"
RESULTS_ROOT="/home/fzliang/origin/MotionBERT/results_custom_2person_bytetrack_best"
MATCHING_CSV="/home/fzliang/origin/MotionBERT/alignment/data/custom_2person_matching_bytetrack_best/matching_confidence_summary.csv"

# Output directories
PROCESSED_DIR="${PROJECT_ROOT}/data/processed/custom_4fold"
OUTPUT_ROOT="${PROJECT_ROOT}/artifacts"
CHECKPOINT_DIR="${OUTPUT_ROOT}"

# Dataset parameters
WINDOW_LEN=24
STRIDE=16
EPOCHS=150
BATCH_SIZE=64

# IMU checkpoint
IMU_CKPT="/home/fzliang/origin/despite/totalcapture/checkpoints/si_totalcapture_best.pth"

# MotionBERT parameters
MOTIONBERT_CONFIG="configs/pose3d/MB_ft_h36m_global_lite.yaml"
MOTIONBERT_CKPT="checkpoint/pretrain/MB_lite_models.bin"

###############################################################################
# STEP 1: Preprocess Custom 4-Fold Dataset (if not already done)
###############################################################################
echo "=========================================="
echo "STEP 1: Preprocessing custom 4-fold dataset"
echo "=========================================="

if [ ! -d "${PROCESSED_DIR}/folds" ]; then
    echo "Running preprocessing..."
    ${VENV} -m src.data.preprocess_custom_4fold \
        --motionbert_root "${MOTIONBERT_ROOT}" \
        --preprocessed_root "${PREPROCESSED_ROOT}" \
        --results_root "${RESULTS_ROOT}" \
        --matching_csv "${MATCHING_CSV}" \
        --out_dir "${PROCESSED_DIR}" \
        --window_len "${WINDOW_LEN}" \
        --stride "${STRIDE}"
    echo "✅ Preprocessing complete"
else
    echo "✅ Processed data already exists at ${PROCESSED_DIR}"
fi

# Verify folds exist
for fold in 1 2 3 4; do
    fold_dir="${PROCESSED_DIR}/folds/fold_${fold}"
    if [ ! -d "${fold_dir}" ]; then
        echo "❌ ERROR: Fold ${fold} not found at ${fold_dir}"
        exit 1
    fi
    echo "✅ fold_${fold} exists"
done

###############################################################################
# STEP 2: Train Each Fold
###############################################################################
echo ""
echo "=========================================="
echo "STEP 2: Training each fold"
echo "=========================================="

FOLD_CHECKPOINTS=()

for fold in 1 2 3 4; do
    fold_dir="${PROCESSED_DIR}/folds/fold_${fold}"
    train_csv="${fold_dir}/windows_train.csv"
    val_csv="${fold_dir}/windows_val.csv"
    data_root="${PROCESSED_DIR}"
    
    echo ""
    echo "--- Training Fold ${fold} ---"
    
    if [ ! -f "${train_csv}" ] || [ ! -f "${val_csv}" ]; then
        echo "❌ ERROR: Missing CSV files for fold ${fold}"
        exit 1
    fi
    
    run_name="custom4fold_fold${fold}_r_lowarm_4x_${EPOCHS}ep"
    checkpoint_path="${CHECKPOINT_DIR}/${run_name}/best.pt"
    
    if [ -f "${checkpoint_path}" ]; then
        echo "✅ Checkpoint already exists: ${checkpoint_path}"
        echo "    Skipping training..."
    else
        echo "🔄 Training fold ${fold}..."
        
        cd "${PROJECT_ROOT}"
        ${VENV} -m src.engine.train \
            --train_csv "${train_csv}" \
            --val_csv "${val_csv}" \
            --data_root "${data_root}" \
            --motionbert_root "${MOTIONBERT_ROOT}" \
            --motionbert_config "${MOTIONBERT_CONFIG}" \
            --motionbert_ckpt "${MOTIONBERT_CKPT}" \
            --imu_ckpt "${IMU_CKPT}" \
            --compute_imu_stats \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --save_dir "." \
            --run_name "${run_name}" \
            --device cuda
        
        if [ $? -eq 0 ]; then
            echo "✅ Training fold ${fold} successful"
        else
            echo "❌ Training fold ${fold} failed"
            exit 1
        fi
    fi
    
    FOLD_CHECKPOINTS+=("${checkpoint_path}")
done

###############################################################################
# STEP 3: Evaluate Each Fold (2-person matching)
###############################################################################
echo ""
echo "=========================================="
echo "STEP 3: Evaluating each fold (2-person groups)"
echo "=========================================="

for fold in 1 2 3 4; do
    fold_dir="${PROCESSED_DIR}/folds/fold_${fold}"
    test_csv="${fold_dir}/windows_test.csv"
    data_root="${PROCESSED_DIR}"
    
    echo ""
    echo "--- Evaluating Fold ${fold} ---"
    
    if [ ! -f "${test_csv}" ]; then
        echo "❌ ERROR: Missing test CSV for fold ${fold}"
        exit 1
    fi
    
    run_name="custom4fold_fold${fold}_r_lowarm_4x_${EPOCHS}ep"
    checkpoint="${CHECKPOINT_DIR}/${run_name}/best.pt"
    eval_results="${CHECKPOINT_DIR}/${run_name}/eval_results_2person.json"
    
    if [ ! -f "${checkpoint}" ]; then
        echo "❌ ERROR: Checkpoint not found: ${checkpoint}"
        exit 1
    fi
    
    echo "🔄 Evaluating fold ${fold}..."
    
    cd "${PROJECT_ROOT}"
    ${VENV} -m src.engine.eval_custom \
        --test_csv "${test_csv}" \
        --data_root "${data_root}" \
        --motionbert_root "${MOTIONBERT_ROOT}" \
        --motionbert_config "${MOTIONBERT_CONFIG}" \
        --motionbert_ckpt "${MOTIONBERT_CKPT}" \
        --checkpoint "${checkpoint}" \
        --batch_size "${BATCH_SIZE}" \
        --save_json "${eval_results}" \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation fold ${fold} successful"
        echo "   Results saved to: ${eval_results}"
    else
        echo "⚠️  Evaluation fold ${fold} had issues (may not have script yet)"
    fi
done

###############################################################################
# STEP 4: Summary & Cross-Validation Results
###############################################################################
echo ""
echo "=========================================="
echo "STEP 4: Cross-Validation Summary"
echo "=========================================="

echo ""
echo "Training completed for all folds:"
for fold in 1 2 3 4; do
    run_name="custom4fold_fold${fold}_r_lowarm_4x_${EPOCHS}ep"
    checkpoint="${CHECKPOINT_DIR}/${run_name}/best.pt"
    if [ -f "${checkpoint}" ]; then
        echo "  ✅ Fold ${fold}: ${checkpoint}"
    else
        echo "  ❌ Fold ${fold}: Missing checkpoint"
    fi
done

echo ""
echo "Evaluation results:"
for fold in 1 2 3 4; do
    run_name="custom4fold_fold${fold}_r_lowarm_4x_${EPOCHS}ep"
    eval_results="${CHECKPOINT_DIR}/${run_name}/eval_results_2person.json"
    if [ -f "${eval_results}" ]; then
        echo "  ✅ Fold ${fold}: ${eval_results}"
        # Show top-1 accuracy if available
        if command -v jq &> /dev/null; then
            acc=$(jq '.top1_accuracy // .average_top1_accuracy // "N/A"' "${eval_results}" 2>/dev/null || echo "N/A")
            echo "     Top-1 Accuracy: ${acc}"
        fi
    else
        echo "  ⏳ Fold ${fold}: Evaluation pending"
    fi
done

echo ""
echo "=========================================="
echo "✅ Custom 4-Fold Pipeline Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review evaluation results in:"
echo "     ${CHECKPOINT_DIR}/"
echo ""
echo "  2. Compare 2-person matching performance across folds"
echo ""
echo "  3. For detailed results, run:"
echo "     cat ${CHECKPOINT_DIR}/custom4fold_fold{1,2,3,4}*/eval_results_2person.json | jq ."
echo ""
