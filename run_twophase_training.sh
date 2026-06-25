#!/bin/bash

# Two-Phase Training and Testing Script
# Usage: bash run_twophase_training.sh [GPU_ID]

set -e  # Exit on error

# Configuration
GPU_ID=${1:-0}  # Default GPU 0, or use first argument
DETECTOR_PATH="training/config/detector_clip/clip_lora_twophase.yaml"
TEST_DATASETS=("FaceForensics++" "Celeb-DF-v2" "Celeb-DF-v1" "DeepFakeDetection" "DFDC" "DFDCP" "FaceShifter")

echo "========================================"
echo "Two-Phase Training Pipeline"
echo "========================================"
echo "GPU ID: ${GPU_ID}"
echo "Detector Config: ${DETECTOR_PATH}"
echo "========================================"
echo ""

# Step 1: Training
echo "[Step 1/3] Starting Two-Phase Training..."
echo "----------------------------------------"

# Capture the output and extract log directory
TRAINING_OUTPUT=$(CUDA_VISIBLE_DEVICES=${GPU_ID} python training/train_two_phase.py \
    --detector_path ${DETECTOR_PATH} \
    --wandb 2>&1)

TRAINING_EXIT_CODE=$?
echo "$TRAINING_OUTPUT"

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Extract log directory from training output
LATEST_LOG=$(echo "$TRAINING_OUTPUT" | grep "TWOPHASE_LOG_DIR=" | tail -n 1 | cut -d'=' -f2)

if [ -z "$LATEST_LOG" ]; then
    echo "⚠️  Could not extract log directory from training output. Trying to find latest..."
    LATEST_LOG=$(ls -dt logs/training/clip_lora_twophase_* 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_LOG" ]; then
        echo "❌ No training log directory found!"
        exit 1
    fi
fi

echo "✅ Training completed!"
echo "Training results: ${LATEST_LOG}"
echo ""

# Step 2: Verify checkpoints
echo "[Step 2/3] Verifying checkpoints..."
echo "----------------------------------------"
PHASE0_CKPT="${LATEST_LOG}/phase0/test/avg/ckpt_best.pth"
PHASE1_CKPT="${LATEST_LOG}/phase1/test/avg/ckpt_best.pth"

if [ ! -f "$PHASE0_CKPT" ]; then
    echo "❌ Phase 0 checkpoint not found: ${PHASE0_CKPT}"
    exit 1
fi

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "❌ Phase 1 checkpoint not found: ${PHASE1_CKPT}"
    exit 1
fi

echo "Phase 0 checkpoint: ${PHASE0_CKPT}"
echo "Phase 1 checkpoint: ${PHASE1_CKPT}"
echo ""

# Step 3: Testing
echo "[Step 3/3] Testing on all datasets..."
echo "----------------------------------------"

# Test Phase 0
echo ""
echo "📊 Testing Phase 0 model..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python training/test.py \
    --detector_path ${DETECTOR_PATH} \
    --weights_path ${PHASE0_CKPT} \
    --test_dataset "${TEST_DATASETS[@]}" \
    --excel

if [ $? -ne 0 ]; then
    echo "❌ Phase 0 testing failed!"
    exit 1
fi
echo "✅ Phase 0 testing completed!"

# Test Phase 1
echo ""
echo "📊 Testing Phase 1 model (with LoRA)..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python training/test.py \
    --detector_path ${DETECTOR_PATH} \
    --weights_path ${PHASE1_CKPT} \
    --test_dataset "${TEST_DATASETS[@]}" \
    --excel

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 testing failed!"
    exit 1
fi
echo "✅ Phase 1 testing completed!"

# Summary
echo ""
echo "========================================"
echo "🎉 All tasks completed successfully!"
echo "========================================"
echo "Training logs: ${LATEST_LOG}"
echo "Phase 0 results: ${LATEST_LOG}/phase0/test/avg/"
echo "Phase 1 results: ${LATEST_LOG}/phase1/test/avg/"
echo "========================================"
