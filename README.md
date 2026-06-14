# 🛰️ Aerial GCP Pose Estimation

Multi-task deep learning pipeline for **Ground Control Point (GCP)** marker localisation and shape classification from aerial drone imagery.

## Architecture

```
Input Image (2048×1365)
        │
   Resize 512×512
        │
  EfficientNet-B2 (pretrained ImageNet)
        │
   Global Avg Pool → Feature Vector (1408-d)
        │
   ┌────┴────┐
   │         │
Keypoint   Classification
  Head       Head
   │         │
(x, y)    Cross / L-Shaped / Square
[0, 1]     3-class softmax
```

**Why EfficientNet-B2?**
- Best accuracy-per-FLOP via compound scaling — critical on Kaggle's limited GPU
- 9.1M parameters — large enough for 512×512 inputs, small enough for batch_size=16
- Strong ImageNet features transfer well to aerial imagery (edges, textures, spatial patterns)

## Training Strategy

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Optimizer** | AdamW (lr=1e-4, wd=1e-4) | Better generalisation than vanilla Adam |
| **Scheduler** | CosineAnnealingWarmRestarts | Periodic restarts help escape local minima |
| **Precision** | FP16 (AMP) | 2× speedup, 50% less VRAM |
| **Batch size** | 16 (effective 32 via grad accum) | Fits Kaggle P100/T4 16GB |
| **Backbone freeze** | First 5 epochs | Stabilises head training before fine-tuning |
| **Backbone LR** | 10× lower than heads | Prevents catastrophic forgetting |
| **Loss** | 5.0 × SmoothL1 + 1.0 × CrossEntropy | Keypoint task is harder, needs more gradient |
| **Early stopping** | Patience 7 on val_loss | Prevents overfitting |
| **Grad clipping** | max_norm=1.0 | Prevents exploding gradients from regression |

### Data Augmentations (keypoint-aware via Albumentations)
- Horizontal/Vertical flip, Random 90° rotation
- ShiftScaleRotate (±8% shift, ±15% scale, ±15° rotation)
- ColorJitter / BrightnessContrast
- GaussNoise / GaussianBlur
- CoarseDropout

All geometric augmentations are applied to keypoints simultaneously using `albumentations.KeypointParams`.

## Dataset Handling

### Challenges Identified
1. **Real-world data**: Not a clean academic dataset — some images may be corrupt or have unusual dimensions
2. **Class imbalance**: Stratified train/val split ensures proportional representation
3. **Missing files**: Pipeline gracefully skips missing images with logging
4. **Keypoint at edges**: After augmentation, keypoints may fall outside the image — handled via fallback to original normalised coordinates

### Keypoint Normalisation Flow
```
Original coords (e.g., 1024.5, 850.2 in 2048×1365)
    → Albumentations resize+augment (handles transform automatically)
    → Normalise to [0, 1] by dividing by IMG_SIZE (512)
    → Model predicts [0, 1] via Sigmoid
    → Inference: multiply by original image dimensions to get pixel coords
```

## Project Structure

```
gcp_pose_estimation/
├── config.py              # All hyperparameters and paths
├── dataset.py             # Dataset class + Albumentations pipelines
├── model.py               # Multi-task EfficientNet architecture
├── engine.py              # Train/val step functions + metrics
├── train.py               # Training orchestrator (for local use)
├── inference.py           # Inference + predictions.json generation
├── utils.py               # Seeding, early stopping, logging
├── kaggle_notebook.py     # Self-contained Kaggle notebook (all-in-one)
└── README.md              # This file
```

## How to Run on Kaggle

### Step 1: Upload Dataset
Upload your `train_dataset` and `test_dataset` folders as a **Kaggle Dataset**.
Name it something like `datasets`.

### Step 2: Create Notebook
1. Create a new Kaggle Notebook
2. Add your dataset (it will mount at `/kaggle/input/datasets/`)
3. Enable **GPU** accelerator (P100 or T4)

### Step 3: Run Training + Inference
Copy the contents of `kaggle_notebook.py` into your notebook cells and run.

**Important**: Update the paths in the `Config` class if your Kaggle dataset name differs:
```python
TRAIN_DATA_DIR = "/kaggle/input/YOUR-DATASET-NAME/train_dataset"
TEST_DATA_DIR  = "/kaggle/input/YOUR-DATASET-NAME/test_dataset"
TRAIN_LABELS   = "/kaggle/input/YOUR-DATASET-NAME/train_dataset/curated_gcp_marks.json"
```

### Step 4: Download Results
After training completes, download from `/kaggle/working/`:
- `best_model.pth` — trained model weights
- `predictions.json` — test set predictions

## Evaluation Metrics

- **Keypoint**: PCK@10px, PCK@25px, PCK@50px (Percentage of Correct Keypoints)
- **Classification**: Macro F1-Score across 3 shape classes
- Both metrics are computed and logged every epoch during validation

## Requirements

```
torch>=2.0
timm>=0.9
albumentations>=1.3.0
opencv-python
scikit-learn
numpy
```

All dependencies are pre-installed on Kaggle except `timm` and `albumentations`, which are installed automatically by the notebook.
