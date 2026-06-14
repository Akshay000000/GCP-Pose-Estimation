"""
Training and validation engine with AMP, gradient accumulation,
and comprehensive metric tracking.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import f1_score


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_pck(pred_kps, gt_kps, thresholds=(10, 25, 50),
                img_w=2048.0, img_h=1365.0):
    """
    Percentage of Correct Keypoints at various pixel thresholds.

    Args:
        pred_kps: (N, 2) tensor, normalised [0, 1]
        gt_kps:   (N, 2) tensor, normalised [0, 1]
        thresholds: pixel thresholds to evaluate
        img_w, img_h: original image dimensions for scaling
    """
    scale = torch.tensor([[img_w, img_h]], device=pred_kps.device)
    pred_px = pred_kps * scale
    gt_px = gt_kps * scale
    dists = torch.norm(pred_px - gt_px, dim=1)

    results = {}
    for t in thresholds:
        results[f"PCK@{t}"] = (dists <= t).float().mean().item()
    results["mean_dist_px"] = dists.mean().item()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, scaler, config, device):
    """Run one training epoch with AMP and gradient accumulation."""
    model.train()

    kp_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_kp = 0.0
    running_cls = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, keypoints, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            out = model(images)
            kp_loss = kp_criterion(out["keypoints"], keypoints)
            cls_loss = cls_criterion(out["logits"], labels)
            loss = (config.KP_LOSS_WEIGHT * kp_loss +
                    config.CLS_LOSS_WEIGHT * cls_loss)
            loss = loss / config.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * config.GRAD_ACCUM_STEPS
        running_kp += kp_loss.item()
        running_cls += cls_loss.item()

        _, preds = out["logits"].max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    n = len(dataloader)
    return {
        "train_loss": running_loss / n,
        "train_kp_loss": running_kp / n,
        "train_cls_loss": running_cls / n,
        "train_acc": correct / max(total, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_one_epoch(model, dataloader, config, device):
    """Run one validation epoch and compute all metrics."""
    model.eval()

    kp_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_kp = 0.0
    running_cls = 0.0

    all_pred_kps = []
    all_gt_kps = []
    all_pred_cls = []
    all_gt_cls = []

    for images, keypoints, labels in dataloader:
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            out = model(images)
            kp_loss = kp_criterion(out["keypoints"], keypoints)
            cls_loss = cls_criterion(out["logits"], labels)
            loss = (config.KP_LOSS_WEIGHT * kp_loss +
                    config.CLS_LOSS_WEIGHT * cls_loss)

        running_loss += loss.item()
        running_kp += kp_loss.item()
        running_cls += cls_loss.item()

        all_pred_kps.append(out["keypoints"].cpu())
        all_gt_kps.append(keypoints.cpu())

        _, preds = out["logits"].max(1)
        all_pred_cls.extend(preds.cpu().numpy())
        all_gt_cls.extend(labels.cpu().numpy())

    n = len(dataloader)

    # Classification metrics
    macro_f1 = f1_score(all_gt_cls, all_pred_cls, average="macro", zero_division=0)
    accuracy = np.mean(np.array(all_pred_cls) == np.array(all_gt_cls))

    # Keypoint metrics (PCK)
    all_pred_kps = torch.cat(all_pred_kps, dim=0)
    all_gt_kps = torch.cat(all_gt_kps, dim=0)
    pck = compute_pck(all_pred_kps, all_gt_kps)

    metrics = {
        "val_loss": running_loss / n,
        "val_kp_loss": running_kp / n,
        "val_cls_loss": running_cls / n,
        "val_acc": accuracy,
        "val_f1": macro_f1,
    }
    metrics.update(pck)
    return metrics
