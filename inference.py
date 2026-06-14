"""
Inference script: loads trained model and generates predictions.json
for the test dataset.
"""
import os
import json
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from config import Config
from dataset import GCPDataset, load_test_data, get_test_transforms
from model import GCPModel


def run_inference(config: Config = None):
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFERENCE] Using device: {device}")

    # ── Load model ───────────────────────────────────────────────────────
    model = GCPModel(config).to(device)
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFERENCE] Loaded checkpoint from epoch {checkpoint['epoch']}")

    # ── Load test data ───────────────────────────────────────────────────
    test_paths = load_test_data(config)
    test_ds = GCPDataset(
        test_paths,
        labels=None,
        config=config,
        transform=get_test_transforms(config),
        is_test=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # ── Predict ──────────────────────────────────────────────────────────
    predictions = {}

    with torch.no_grad():
        for images, paths, orig_ws, orig_hs in test_loader:
            images = images.to(device, non_blocking=True)

            with autocast():
                out = model(images)

            pred_kps = out["keypoints"].cpu()    # (B, 2) in [0, 1]
            pred_cls = out["logits"].argmax(1).cpu()

            for i in range(len(paths)):
                path = paths[i]
                ow = float(orig_ws[i])
                oh = float(orig_hs[i])

                # Scale normalised keypoints back to original pixel coords
                px = float(pred_kps[i, 0]) * ow
                py = float(pred_kps[i, 1]) * oh
                shape = config.SHAPE_CLASSES[int(pred_cls[i])]

                # Derive relative path from test data dir
                rel_path = os.path.relpath(path, config.TEST_DATA_DIR)
                rel_path = rel_path.replace("\\", "/")   # Windows compat

                predictions[rel_path] = {
                    "mark": {"x": round(px, 2), "y": round(py, 2)},
                    "verified_shape": shape,
                }

    # ── TTA (optional) ───────────────────────────────────────────────────
    if config.USE_TTA:
        predictions = run_tta(model, test_paths, config, device, predictions)

    # ── Save ─────────────────────────────────────────────────────────────
    with open(config.PREDICTIONS_PATH, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"[INFERENCE] Saved {len(predictions)} predictions → {config.PREDICTIONS_PATH}")
    return predictions


def run_tta(model, test_paths, config, device, base_predictions):
    """
    Test-Time Augmentation: average keypoint predictions over
    original + horizontal flip + vertical flip + both flips.
    Classification uses majority vote.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    import numpy as np
    from collections import Counter

    tta_transforms = [
        # Original (already computed in base_predictions)
        None,
        # Horizontal flip
        A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # Vertical flip
        A.Compose([
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]

    print("[TTA] Running test-time augmentation...")

    for aug_idx, aug in enumerate(tta_transforms):
        if aug is None:
            continue

        for path in test_paths:
            image = cv2.imread(path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

            transformed = aug(image=image)
            img_tensor = transformed["image"].unsqueeze(0).to(device)

            with torch.no_grad(), autocast():
                out = model(img_tensor)

            kp = out["keypoints"].cpu().squeeze()
            cls_idx = out["logits"].argmax(1).cpu().item()

            # Reverse the flip on keypoints
            kp_x, kp_y = float(kp[0]), float(kp[1])
            if aug_idx == 1:   # horizontal flip
                kp_x = 1.0 - kp_x
            elif aug_idx == 2: # vertical flip
                kp_y = 1.0 - kp_y

            rel_path = os.path.relpath(path, config.TEST_DATA_DIR).replace("\\", "/")

            if rel_path in base_predictions:
                entry = base_predictions[rel_path]
                # Accumulate for averaging
                if "_tta_kps" not in entry:
                    entry["_tta_kps"] = [(entry["mark"]["x"] / orig_w, entry["mark"]["y"] / orig_h)]
                    entry["_tta_cls"] = [config.SHAPE_CLASSES.index(entry["verified_shape"])]
                entry["_tta_kps"].append((kp_x, kp_y))
                entry["_tta_cls"].append(cls_idx)

    # Average keypoints, majority vote classification
    for rel_path, entry in base_predictions.items():
        if "_tta_kps" in entry:
            avg_x = np.mean([k[0] for k in entry["_tta_kps"]])
            avg_y = np.mean([k[1] for k in entry["_tta_kps"]])
            # Need orig size — extract from first prediction
            orig_w_approx = entry["mark"]["x"] / entry["_tta_kps"][0][0] if entry["_tta_kps"][0][0] > 0 else 2048
            orig_h_approx = entry["mark"]["y"] / entry["_tta_kps"][0][1] if entry["_tta_kps"][0][1] > 0 else 1365
            entry["mark"]["x"] = round(float(avg_x * orig_w_approx), 2)
            entry["mark"]["y"] = round(float(avg_y * orig_h_approx), 2)

            cls_vote = Counter(entry["_tta_cls"]).most_common(1)[0][0]
            entry["verified_shape"] = config.SHAPE_CLASSES[cls_vote]

            del entry["_tta_kps"]
            del entry["_tta_cls"]

    print("[TTA] Done.")
    return base_predictions


if __name__ == "__main__":
    run_inference()
