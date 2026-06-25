# Two-Phase Training with Variability-Based Continual Learning

## Overview

This implementation adds a two-phase training approach to DeepfakeBench where:

- **Phase 0**: Train model on full dataset and collect training dynamics (variability, confidence, correctness)
- **Phase 1**: Continual learning on high-variability samples (top 20% from both real and fake classes)

The system automatically filters hard-to-learn examples based on variability metrics and performs continual learning to improve model performance on challenging cases.

## Files Modified/Created

### New Files
1. **`training/train_two_phase.py`** - Main two-phase training script
2. **`training/config/detector/xception_twophase.yaml`** - Example configuration with two-phase training enabled

### Modified Files
1. **`training/trainer/trainer.py`** - Added `phase` parameter for phase-aware logging and checkpoint saving
2. **`training/dataset/abstract_dataset.py`** - Added frame filtering support for Phase 1
3. **`training/config/detector/xception.yaml`** - Added Phase 1 configuration parameters

## Configuration Parameters

Add these parameters to your detector config file (e.g., `xception.yaml`):

```yaml
# two-phase training config
phase1_enabled: true   # Enable two-phase training (required)
phase1_nEpochs: 10     # Number of epochs for Phase 1
phase1_lr: 0.00002     # Learning rate for Phase 1 (typically 1/10 of Phase 0)
phase1_variability_top_percent: 20   # Top % of high-variability samples (real + fake)
phase1_optimizer_reset: true   # Reinitialize optimizer for Phase 1
```

## Usage

### Basic Usage

```bash
# Run two-phase training
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml
```

### Debug Mode (Fast Testing)

```bash
# Quick test with reduced data and epochs
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml \
    --debug
```

Debug mode settings:
- Phase 0: 3 epochs, 1% of training data
- Phase 1: 2 epochs, 1% of training data  
- Test data: 10% of test data

### Advanced Options

```bash
# Skip Phase 0 (requires existing checkpoint and dynamics)
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml \
    --skip_phase0

# Specify custom dataset
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml \
    --train_dataset FaceForensics++ Celeb-DF-v2 \
    --test_dataset FaceForensics++ DeepFakeDetection

# Use with WandB logging
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml \
    --wandb
```

## How It Works

### Phase 0: Full Dataset Training
1. Trains model on complete dataset
2. Collects training dynamics for each sample across all epochs:
   - **Variability**: Standard deviation of predicted probabilities
   - **Confidence**: Mean probability of true class
   - **Correctness**: Fraction of epochs with correct prediction
3. Saves best checkpoint to `{log_dir}/phase0/{dataset}/ckpt_best.pth`
4. Saves training dynamics to `{log_dir}/phase0/training_dynamics.json`

### Filtering
1. Loads training dynamics from Phase 0
2. Separates samples by label (real vs fake)
3. Sorts each group by variability (descending)
4. Selects top N% from each group independently
5. Creates set of allowed frame paths for Phase 1

### Phase 1: Continual Learning
1. Loads Phase 0 best checkpoint
2. Filters training dataset to only include high-variability samples
3. Reinitializes optimizer with Phase 1 learning rate (default: 1/10 of Phase 0)
4. Trains for additional epochs on filtered dataset
5. Saves best checkpoint to `{log_dir}/phase1/{dataset}/ckpt_best.pth`
6. Saves Phase 1 training dynamics to `{log_dir}/phase1/training_dynamics.json`

## Output Structure

```
logs/training/
└── {model_name}_twophase_{timestamp}/
    ├── training.log                      # Combined training log
    ├── phase0/
    │   ├── training_dynamics.json        # Phase 0 metrics
    │   └── {dataset}/
    │       ├── ckpt_best.pth            # Best checkpoint from Phase 0
    │       └── auc/
    │           └── metric_board/        # TensorBoard logs
    └── phase1/
        ├── training_dynamics.json        # Phase 1 metrics
        └── {dataset}/
            ├── ckpt_best.pth            # Best checkpoint from Phase 1
            └── auc/
                └── metric_board/        # TensorBoard logs
```

## Expected Behavior

### Sample Counts
- Phase 0: Full dataset (e.g., 10,000 samples)
- Phase 1: ~20% of Phase 0 (e.g., ~2,000 samples)
  - ~10% from real samples (top 20% variability)
  - ~10% from fake samples (top 20% variability)

### Training Time
- Phase 0: Same as regular training
- Phase 1: ~20% of Phase 0 time (fewer samples)
- Total: ~120% of regular training time

### Performance
- Phase 1 focuses training on hard examples
- Expected improvements on:
  - Hard-to-classify samples
  - Cross-dataset generalization
  - Edge cases and boundary regions

## Validation

The implementation includes several validation checks:

1. **Checkpoint Verification**: Ensures Phase 0 checkpoint exists before Phase 1
2. **Dynamics Validation**: Checks training dynamics file exists and is non-empty
3. **Sample Count**: Warns if filtered dataset is too small (< 100 samples)
4. **Label Balance**: Logs distribution of real/fake samples after filtering

## Troubleshooting

### Error: "phase1_enabled is not set to True"
- Set `phase1_enabled: true` in your detector config file
- Or use the provided `xception_twophase.yaml` config

### Error: "Phase 0 checkpoint not found"
- Ensure Phase 0 completed successfully
- Check `{log_dir}/phase0/{dataset}/ckpt_best.pth` exists
- Don't use `--skip_phase0` on first run

### Error: "Training dynamics file not found"
- Phase 0 must complete fully before Phase 1
- Check `{log_dir}/phase0/training_dynamics.json` exists

### Warning: "Very few samples selected"
- Increase `phase1_variability_top_percent` (e.g., from 20 to 30)
- Or reduce filtering if dataset is small

## Comparison with Regular Training

To compare with single-phase training:

```bash
# Regular training
python training/train.py \
    --detector_path training/config/detector/xception.yaml

# Two-phase training
python training/train_two_phase.py \
    --detector_path training/config/detector/xception_twophase.yaml
```

## Notes

- Compatible with all existing detectors (xception, efficientnet, etc.)
- Works with both image-level and video-level datasets
- Supports DDP (distributed training) - use `--ddp` flag
- Phase 1 uses the same test datasets as Phase 0 for fair comparison
- Training dynamics are saved for both phases separately for analysis

## Future Enhancements

Potential improvements:
1. Multi-phase training (Phase 2, 3, etc.)
2. Different selection criteria (confidence, correctness)
3. Adaptive variability thresholds
4. Combined variability + confidence filtering
5. Dynamic sample reweighting instead of filtering
