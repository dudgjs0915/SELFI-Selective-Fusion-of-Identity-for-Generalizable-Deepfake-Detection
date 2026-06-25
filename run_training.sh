#!/bin/bash

# Single-Phase Training and Testing Script
# Usage: bash run_training.sh [DETECTOR_CONFIG] [GPU_ID] [TASK_TARGET] [DATASET_JSON_FOLDER]
# Example: bash run_training.sh training/config/detector/xception.yaml 0 "CLIP_Variability" "./preprocessing/dataset_json_variability_top_30"

set -e  # Exit on error

# Configuration
DETECTOR_PATH=${1:-"training/config/detector_clip/clip.yaml"}
GPU_ID=${2:-0}  # Default GPU 0
TASK_TARGET=${3:-"CLIP_Variability"}
DATASET_JSON_FOLDER=${4:-'./preprocessing/dataset_json_variability_top_30'}
TEST_DATASETS=("FaceForensics++" "Celeb-DF-v2" "Celeb-DF-v1" "DeepFakeDetection" "DFDC" "DFDCP" "FaceShifter")

echo "========================================"
echo "Single-Phase Training Pipeline"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "Detector Config: ${DETECTOR_PATH}"
echo "Task Target: ${TASK_TARGET}"
echo "Dataset JSON Folder: ${DATASET_JSON_FOLDER}"
echo "========================================"
echo ""

# Step 1: Training
echo "[Step 1/2] Starting Training..."
echo "----------------------------------------"

# Capture the output and extract log directory
TRAINING_OUTPUT=$(CUDA_VISIBLE_DEVICES=${GPU_ID} python training/train.py \
    --detector_path ${DETECTOR_PATH} \
    --task_target "${TASK_TARGET}" \
    --dataset_json_folder "${DATASET_JSON_FOLDER}" \
    --wandb 2>&1)

TRAINING_EXIT_CODE=$?
echo "$TRAINING_OUTPUT"

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Extract log directory from training output
LOG_DIR=$(echo "$TRAINING_OUTPUT" | grep "TRAINING_LOG_DIR=" | tail -n 1 | cut -d'=' -f2)

if [ -z "$LOG_DIR" ]; then
    echo "⚠️  Could not extract log directory from training output. Trying to find latest..."
    # Extract model name from config
    MODEL_NAME=$(grep "model_name:" ${DETECTOR_PATH} | head -n 1 | awk '{print $2}')
    LOG_DIR=$(ls -dt logs/training/${MODEL_NAME}_* 2>/dev/null | head -n 1)
    
    if [ -z "$LOG_DIR" ]; then
        echo "❌ No training log directory found!"
        exit 1
    fi
fi

echo "✅ Training completed!"
echo "Training results: ${LOG_DIR}"
echo ""

# Step 2: Testing
echo "[Step 2/2] Testing on all datasets..."
echo "----------------------------------------"

# Find checkpoint
CKPT_PATH="${LOG_DIR}/test/avg/ckpt_best.pth"

if [ ! -f "$CKPT_PATH" ]; then
    echo "⚠️  Best checkpoint not found at ${CKPT_PATH}"
    # Try alternative locations
    CKPT_PATH=$(find ${LOG_DIR} -name "ckpt_best.pth" -type f | head -n 1)
    
    if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
        echo "❌ No checkpoint found!"
        exit 1
    fi
fi

echo "Checkpoint: ${CKPT_PATH}"
echo ""

# Test
echo "📊 Testing model..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python training/test.py \
    --detector_path ${DETECTOR_PATH} \
    --weights_path ${CKPT_PATH} \
    --test_dataset "${TEST_DATASETS[@]}" \
    --excel

if [ $? -ne 0 ]; then
    echo "❌ Testing failed!"
    exit 1
fi
echo "✅ Testing completed!"

# Summary
echo ""
echo "========================================"
echo "🎉 All tasks completed successfully!"
echo "========================================"
echo "Training logs: ${LOG_DIR}"
echo "Test results: ${LOG_DIR}/test/avg/"
echo "========================================"
