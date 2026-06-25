#!/bin/bash

# Test Only Script (for existing trained models)
# Usage: bash run_test_only.sh [LOG_DIR] [GPU_ID]
# Example: bash run_test_only.sh logs/training/clip_lora_twophase_2026-02-18-20-54-49 0

set -e  # Exit on error

# Configuration
LOG_DIR=${1}
GPU_ID=${2:-0}  # Default GPU 0
DETECTOR_PATH="training/config/detector_clip/clip_lora_twophase.yaml"
TEST_DATASETS=("FaceForensics++" "Celeb-DF-v2" "Celeb-DF-v1" "DeepFakeDetection" "DFDC" "DFDCP" "FaceShifter")

# If no log directory provided, find the latest one
if [ -z "$LOG_DIR" ]; then
    echo "No log directory provided. Finding latest..."
    LOG_DIR=$(ls -dt logs/training/clip_lora_twophase_* 2>/dev/null | head -n 1)
    
    if [ -z "$LOG_DIR" ]; then
        echo "❌ No training log directory found!"
        echo "Usage: bash run_test_only.sh [LOG_DIR] [GPU_ID]"
        exit 1
    fi
fi

echo "========================================"
echo "Testing Two-Phase Models"
echo "========================================"
echo "Log Directory: ${LOG_DIR}"
echo "GPU ID: ${GPU_ID}"
echo "Detector Config: ${DETECTOR_PATH}"
echo "========================================"
echo ""

# Check if log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: ${LOG_DIR}"
    exit 1
fi

PHASE0_CKPT="${LOG_DIR}/phase0/test/avg/ckpt_best.pth"
PHASE1_CKPT="${LOG_DIR}/phase1/test/avg/ckpt_best.pth"

echo "Checking checkpoints..."
if [ ! -f "$PHASE0_CKPT" ]; then
    echo "⚠️  Phase 0 checkpoint not found: ${PHASE0_CKPT}"
    PHASE0_CKPT=""
else
    echo "✓ Phase 0: ${PHASE0_CKPT}"
fi

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "⚠️  Phase 1 checkpoint not found: ${PHASE1_CKPT}"
    PHASE1_CKPT=""
else
    echo "✓ Phase 1: ${PHASE1_CKPT}"
fi

if [ -z "$PHASE0_CKPT" ] && [ -z "$PHASE1_CKPT" ]; then
    echo "❌ No checkpoints found!"
    exit 1
fi
echo ""

# Test Phase 0
if [ -n "$PHASE0_CKPT" ]; then
    echo "========================================"
    echo "📊 Testing Phase 0 model..."
    echo "========================================"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python training/test.py \
        --detector_path ${DETECTOR_PATH} \
        --weights_path ${PHASE0_CKPT} \
        --test_dataset "${TEST_DATASETS[@]}" \
        --excel
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase 0 testing failed!"
    else
        echo "✅ Phase 0 testing completed!"
    fi
    echo ""
fi

# Test Phase 1
if [ -n "$PHASE1_CKPT" ]; then
    echo "========================================"
    echo "📊 Testing Phase 1 model (with LoRA)..."
    echo "========================================"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python training/test.py \
        --detector_path ${DETECTOR_PATH} \
        --weights_path ${PHASE1_CKPT} \
        --test_dataset "${TEST_DATASETS[@]}" \
        --excel
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase 1 testing failed!"
    else
        echo "✅ Phase 1 testing completed!"
    fi
    echo ""
fi

# Summary
echo "========================================"
echo "🎉 Testing completed!"
echo "========================================"
echo "Results saved in: ${LOG_DIR}"
if [ -n "$PHASE0_CKPT" ]; then
    echo "Phase 0 results: ${LOG_DIR}/phase0/test/avg/"
fi
if [ -n "$PHASE1_CKPT" ]; then
    echo "Phase 1 results: ${LOG_DIR}/phase1/test/avg/"
fi
echo "========================================"
