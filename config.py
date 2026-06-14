"""
Configuration for GCP Pose Estimation Pipeline.
All hyperparameters and paths are centralized here.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Paths (adjust for your Kaggle dataset name) ──────────────────────
    TRAIN_DATA_DIR: str = "/kaggle/input/datasets/akshaysriram06/land-images/Dataset/train_dataset"
    TEST_DATA_DIR: str = "/kaggle/input/datasets/akshaysriram06/land-images/Dataset/test_dataset"
    TRAIN_LABELS: str = "/kaggle/input/datasets/akshaysriram06/land-images/Dataset/train_dataset/gcp_marks.json"
    OUTPUT_DIR: str = "/kaggle/working"

    # ── Model ────────────────────────────────────────────────────────────
    BACKBONE: str = "efficientnet_b2"
    IMG_SIZE: int = 768
    NUM_CLASSES: int = 3
    PRETRAINED: bool = True
    DROP_RATE: float = 0.3

    # ── Training ─────────────────────────────────────────────────────────
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 40
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_ACCUM_STEPS: int = 2
    GRAD_CLIP_NORM: float = 1.0

    # ── Loss weights ─────────────────────────────────────────────────────
    KP_LOSS_WEIGHT: float = 5.0
    CLS_LOSS_WEIGHT: float = 1.0

    # ── Scheduler (CosineAnnealingWarmRestarts) ──────────────────────────
    T_0: int = 10
    T_MULT: int = 2

    # ── Early stopping ───────────────────────────────────────────────────
    PATIENCE: int = 7

    # ── Backbone freeze schedule ─────────────────────────────────────────
    FREEZE_EPOCHS: int = 5

    # ── Data split ───────────────────────────────────────────────────────
    VAL_SPLIT: float = 0.2
    NUM_WORKERS: int = 2
    SEED: int = 42

    # ── Shape classes (alphabetical order) ───────────────────────────────
    SHAPE_CLASSES: List[str] = field(
        default_factory=lambda: ["Cross", "L-Shaped", "Square"]
    )

    # ── Test-Time Augmentation ───────────────────────────────────────────
    USE_TTA: bool = False

    @property
    def CHECKPOINT_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "best_model.pth")

    @property
    def PREDICTIONS_PATH(self):
        return os.path.join(self.OUTPUT_DIR, "predictions.json")
