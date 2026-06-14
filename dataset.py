"""
Dataset class and Albumentations augmentation pipelines.
Handles keypoint-aware transforms for the GCP dataset.
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class GCPDataset(Dataset):
    """
    PyTorch Dataset for GCP aerial images.

    For training:  returns (image_tensor, keypoint_tensor, class_index)
    For testing:   returns (image_tensor, relative_path, orig_width, orig_height)
    """

    def __init__(self, image_paths, labels, config, transform=None, is_test=False):
        """
        Args:
            image_paths: list of absolute paths to images
            labels:      list of label dicts (None for test)
            config:      Config dataclass
            transform:   albumentations Compose object
            is_test:     if True, returns metadata instead of labels
        """
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image ──────────────────────────────────────────────────────
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for corrupt / unreadable images
            image = np.zeros(
                (self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=np.uint8
            )
            orig_h, orig_w = self.config.IMG_SIZE, self.config.IMG_SIZE
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

        # ── Test mode ────────────────────────────────────────────────────
        if self.is_test:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return image, img_path, orig_w, orig_h

        # ── Training / validation mode ───────────────────────────────────
        label = self.labels[idx]
        kp_x = float(label["mark"]["x"])   # absolute pixel coords
        kp_y = float(label["mark"]["y"])
        shape_class = label["verified_shape"]
        class_idx = self.config.SHAPE_CLASSES.index(shape_class)

        # Pass absolute-pixel keypoints; albumentations handles resize
        keypoints = [(kp_x, kp_y)]

        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed["image"]
            keypoints = transformed["keypoints"]

        # Normalise keypoints to [0, 1] in the resized coordinate space
        if len(keypoints) > 0:
            kp = keypoints[0]
            kp_tensor = torch.tensor(
                [kp[0] / self.config.IMG_SIZE, kp[1] / self.config.IMG_SIZE],
                dtype=torch.float32,
            )
        else:
            # Keypoint fell outside image after augmentation → use original
            kp_tensor = torch.tensor(
                [kp_x / orig_w, kp_y / orig_h], dtype=torch.float32
            )

        kp_tensor = kp_tensor.clamp(0.0, 1.0)
        return image, kp_tensor, class_idx


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation pipelines
# ═══════════════════════════════════════════════════════════════════════════

def get_train_transforms(config):
    """Heavy augmentations for training (keypoint-aware)."""
    return A.Compose(
        [
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2,
                        saturation=0.2, hue=0.1, p=1.0,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                p=0.3,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=True
        ),
    )


def get_val_transforms(config):
    """Minimal transforms for validation (keypoint-aware)."""
    return A.Compose(
        [
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy", remove_invisible=True
        ),
    )


def get_test_transforms(config):
    """Transforms for test / inference (no keypoint handling)."""
    return A.Compose(
        [
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_train_data(config):
    """
    Parse curated_gcp_marks.json and return parallel lists of
    (image_paths, labels) that actually exist on disk.
    """
    with open(config.TRAIN_LABELS, "r") as f:
        annotations = json.load(f)

    image_paths = []
    labels = []
    skipped = 0

    for rel_path, label in annotations.items():
        abs_path = os.path.join(config.TRAIN_DATA_DIR, rel_path)
        if not os.path.isfile(abs_path):
            skipped += 1
            continue
        # Validate shape class
        if label.get("verified_shape") not in config.SHAPE_CLASSES:
            skipped += 1
            continue
        image_paths.append(abs_path)
        labels.append(label)

    print(f"[DATA] Loaded {len(image_paths)} images, skipped {skipped}")
    return image_paths, labels


def load_test_data(config):
    """Walk test directory and collect all image paths."""
    image_paths = []
    valid_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    for root, _, files in os.walk(config.TEST_DATA_DIR):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                image_paths.append(os.path.join(root, fname))

    print(f"[DATA] Found {len(image_paths)} test images")
    return image_paths
