"""
Main training orchestrator.
Run locally:  python train.py
On Kaggle:    executed from the Kaggle notebook (kaggle_train.py)
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from config import Config
from dataset import (
    GCPDataset, load_train_data,
    get_train_transforms, get_val_transforms,
)
from model import GCPModel
from engine import train_one_epoch, validate_one_epoch
from utils import seed_everything, EarlyStopping, MetricLogger


def main(config: Config = None):
    if config is None:
        config = Config()

    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    image_paths, labels = load_train_data(config)

    # Stratified split by shape class
    shape_classes = [l["verified_shape"] for l in labels]
    class_dist = Counter(shape_classes)
    print(f"[TRAIN] Class distribution: {dict(class_dist)}")

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.VAL_SPLIT, random_state=config.SEED
    )
    train_idx, val_idx = next(sss.split(image_paths, shape_classes))

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(f"[TRAIN] Train: {len(train_paths)}, Val: {len(val_paths)}")

    # ── Datasets & Loaders ───────────────────────────────────────────────
    train_ds = GCPDataset(
        train_paths, train_labels, config,
        transform=get_train_transforms(config),
    )
    val_ds = GCPDataset(
        val_paths, val_labels, config,
        transform=get_val_transforms(config),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = GCPModel(config).to(device)
    print(f"[TRAIN] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer, Scheduler, Scaler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_MULT
    )
    scaler = GradScaler()

    # ── Training loop ────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    logger = MetricLogger()
    best_val_loss = float("inf")

    # Freeze backbone for initial epochs
    if config.FREEZE_EPOCHS > 0:
        model.freeze_backbone()
        print(f"[TRAIN] Backbone frozen for first {config.FREEZE_EPOCHS} epochs")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Unfreeze backbone after FREEZE_EPOCHS
        if epoch == config.FREEZE_EPOCHS + 1:
            model.unfreeze_backbone()
            print("[TRAIN] Backbone unfrozen ─ full fine-tuning")
            # Reset optimizer to include backbone params with lower lr
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": config.LR * 0.1},
                    {"params": model.keypoint_head.parameters()},
                    {"params": model.classification_head.parameters()},
                ],
                lr=config.LR,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.T_0, T_mult=config.T_MULT
            )

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, config, device
        )

        # Validate
        val_metrics = validate_one_epoch(model, val_loader, config, device)

        # Step scheduler
        scheduler.step(epoch)

        # Log
        all_metrics = {**train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        logger.log(epoch, all_metrics)

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_f1": val_metrics["val_f1"],
                    "config": config.__dict__,
                },
                config.CHECKPOINT_PATH,
            )
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")

        # Early stopping
        if early_stopping(val_metrics["val_loss"]):
            print(f"\n[TRAIN] Early stopping at epoch {epoch}")
            break

    print(f"\n[TRAIN] Training complete. Best val_loss: {best_val_loss:.4f}")
    print(f"[TRAIN] Checkpoint saved to: {config.CHECKPOINT_PATH}")
    return logger


if __name__ == "__main__":
    main()
