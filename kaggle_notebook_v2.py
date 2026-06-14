# %% [markdown]
# # GCP Pose Estimation V2 — Heatmap Keypoint Regression
# Fixes: L-Shape class, heatmap-based localization, larger input

# %%
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm", "albumentations>=1.3.0"])

# %%
import os, json, random, cv2, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from collections import Counter
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
class Config:
    TRAIN_DATA_DIR  = "/kaggle/input/datasets/akshaysriram06/land-images/Dataset/train_dataset"
    TEST_DATA_DIR   = "/kaggle/input/datasets/akshaysriram06/land-images/Dataset/test_dataset"
    TRAIN_LABELS    = "/kaggle/input/datasets/akshaysriram06/land-images/Dataset/train_dataset/gcp_marks.json"
    OUTPUT_DIR      = "/kaggle/working"
    CHECKPOINT_PATH = "/kaggle/working/best_model_v2.pth"
    PREDICTIONS_PATH= "/kaggle/working/predictions.json"

    BACKBONE    = "efficientnet_b2"
    IMG_SIZE    = 768
    HEATMAP_SIGMA = 2.0
    NUM_CLASSES = 3
    PRETRAINED  = True
    DROP_RATE   = 0.3

    BATCH_SIZE       = 8
    GRAD_ACCUM_STEPS = 4
    NUM_EPOCHS       = 50
    LR               = 3e-4
    WEIGHT_DECAY     = 1e-4
    GRAD_CLIP_NORM   = 1.0

    KP_LOSS_WEIGHT  = 5.0
    CLS_LOSS_WEIGHT = 1.0
    COORD_LOSS_WEIGHT = 3.0

    T_0    = 10
    T_MULT = 2
    PATIENCE      = 10
    FREEZE_EPOCHS = 3
    VAL_SPLIT   = 0.2
    NUM_WORKERS = 2
    SEED        = 42

    # FIXED: match actual JSON values
    SHAPE_CLASSES = ["Cross", "L-Shape", "Square"]

cfg = Config()

def seed_everything(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════
# %%
def create_heatmap(kp_x_norm, kp_y_norm, H, W, sigma=2.0):
    """Gaussian heatmap target centered at normalized keypoint, normalized to [0, 1]."""
    y_grid = torch.arange(H).float()
    x_grid = torch.arange(W).float()
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    cx = kp_x_norm * (W - 1)
    cy = kp_y_norm * (H - 1)
    hm = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    hm = hm / (hm.max() + 1e-8)  # normalize peak to 1.0
    return hm

def spatial_soft_argmax(heatmap, temp=50.0):
    """Differentiable argmax: heatmap (B,1,H,W) -> coords (B,2) in [0,1]."""
    B, C, H, W = heatmap.shape
    flat = heatmap.view(B, -1)
    weights = F.softmax(flat * temp, dim=-1).view(B, 1, H, W)
    x_coords = torch.linspace(0, 1, W, device=heatmap.device).view(1, 1, 1, W)
    y_coords = torch.linspace(0, 1, H, device=heatmap.device).view(1, 1, H, 1)
    x = (weights * x_coords).sum(dim=[2, 3]).squeeze(1)
    y = (weights * y_coords).sum(dim=[2, 3]).squeeze(1)
    return torch.stack([x, y], dim=1)

def compute_pck(pred, gt, orig_ws, orig_hs, thresholds=(10, 25, 50)):
    """PCK using actual per-image dimensions."""
    # Since we use LongestMaxSize+PadIfNeeded, the padded IMG_SIZE square
    # corresponds to max(orig_w, orig_h) in the original image scale.
    max_dims = torch.max(orig_ws, orig_hs).float().unsqueeze(1)
    pred_px = pred * max_dims
    gt_px = gt * max_dims
    dists = torch.norm(pred_px - gt_px, dim=1)
    res = {f"PCK@{t}": (dists <= t).float().mean().item() for t in thresholds}
    res["mean_dist_px"] = dists.mean().item()
    return res

# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
# %%
def get_train_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=cfg.IMG_SIZE),
        A.PadIfNeeded(min_height=cfg.IMG_SIZE, min_width=cfg.IMG_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08),
                 rotate=(-15, 15), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.5),
        A.OneOf([A.GaussNoise(p=1.0), A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def get_val_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=cfg.IMG_SIZE),
        A.PadIfNeeded(min_height=cfg.IMG_SIZE, min_width=cfg.IMG_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

def get_test_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=cfg.IMG_SIZE),
        A.PadIfNeeded(min_height=cfg.IMG_SIZE, min_width=cfg.IMG_SIZE,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class GCPDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8)
            orig_h, orig_w = cfg.IMG_SIZE, cfg.IMG_SIZE
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

        if self.is_test:
            if self.transform:
                image = self.transform(image=image)["image"]
            return image, img_path, torch.tensor(orig_w), torch.tensor(orig_h)

        label = self.labels[idx]
        kp_x, kp_y = float(label["mark"]["x"]), float(label["mark"]["y"])
        class_idx = cfg.SHAPE_CLASSES.index(label["verified_shape"])
        keypoints = [(kp_x, kp_y)]

        if self.transform:
            t = self.transform(image=image, keypoints=keypoints)
            image = t["image"]
            keypoints = t["keypoints"]

        if len(keypoints) > 0:
            kp = keypoints[0]
            kp_norm = torch.tensor([kp[0] / cfg.IMG_SIZE, kp[1] / cfg.IMG_SIZE], dtype=torch.float32)
        else:
            kp_norm = torch.tensor([kp_x / orig_w, kp_y / orig_h], dtype=torch.float32)

        kp_norm = kp_norm.clamp(0.0, 1.0)

        # Create heatmap target
        # Heatmap size = feature map size after backbone (IMG_SIZE/32) * 8 (after 3x upsample)
        hm_size = cfg.IMG_SIZE // 4
        heatmap_target = create_heatmap(kp_norm[0].item(), kp_norm[1].item(),
                                        hm_size, hm_size, cfg.HEATMAP_SIGMA)

        return image, heatmap_target, kp_norm, class_idx, torch.tensor(orig_w), torch.tensor(orig_h)

# ══════════════════════════════════════════════════════════════════════
# MODEL — Heatmap-based keypoint + classification
# ══════════════════════════════════════════════════════════════════════
# %%
class HeatmapHead(nn.Module):
    """Upsample feature map -> single-channel heatmap."""
    def __init__(self, in_channels):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(True))
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final(x)

class GCPModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.BACKBONE, pretrained=cfg.PRETRAINED,
            features_only=True, out_indices=[4])
        feat_dim = self.backbone.feature_info[-1]['num_chs']

        self.heatmap_head = HeatmapHead(feat_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(cfg.DROP_RATE),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(cfg.DROP_RATE/2),
            nn.Linear(128, cfg.NUM_CLASSES))

    def forward(self, x):
        feats = self.backbone(x)[-1]
        heatmap = self.heatmap_head(feats)
        keypoints = spatial_soft_argmax(heatmap)
        pooled = self.gap(feats).flatten(1)
        logits = self.cls_head(pooled)
        return {"keypoints": keypoints, "logits": logits, "heatmap": heatmap}

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True

# ══════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════
# %%
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    hm_fn = nn.MSELoss()
    coord_fn = nn.MSELoss()
    cls_fn = nn.CrossEntropyLoss()
    tot_loss = tot_hm = tot_coord = tot_cls = correct = total = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (imgs, hm_targets, kps, cls, ow, oh) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        hm_targets = hm_targets.to(device, non_blocking=True).unsqueeze(1)
        kps = kps.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        with autocast('cuda'):
            out = model(imgs)
            hm_loss = hm_fn(out["heatmap"], hm_targets)
            coord_loss = coord_fn(out["keypoints"], kps)
            cls_loss = cls_fn(out["logits"], cls)
            loss = (cfg.KP_LOSS_WEIGHT * hm_loss +
                    cfg.COORD_LOSS_WEIGHT * coord_loss +
                    cfg.CLS_LOSS_WEIGHT * cls_loss) / cfg.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)

        tot_loss += loss.item() * cfg.GRAD_ACCUM_STEPS
        tot_hm += hm_loss.item(); tot_coord += coord_loss.item(); tot_cls += cls_loss.item()
        _, pred = out["logits"].max(1)
        total += cls.size(0); correct += pred.eq(cls).sum().item()

    n = len(loader)
    return {"train_loss": tot_loss/n, "train_hm": tot_hm/n,
            "train_coord": tot_coord/n, "train_cls": tot_cls/n,
            "train_acc": correct/max(total,1)}

@torch.no_grad()
def validate(model, loader):
    model.eval()
    hm_fn = nn.MSELoss(); coord_fn = nn.MSELoss(); cls_fn = nn.CrossEntropyLoss()
    tot_loss = tot_hm = tot_coord = tot_cls = 0
    all_pkp, all_gkp, all_pc, all_gc = [], [], [], []
    all_ow, all_oh = [], []

    for imgs, hm_targets, kps, cls, ow, oh in loader:
        imgs = imgs.to(device, non_blocking=True)
        hm_targets = hm_targets.to(device, non_blocking=True).unsqueeze(1)
        kps = kps.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        with autocast('cuda'):
            out = model(imgs)
            hm_loss = hm_fn(out["heatmap"], hm_targets)
            coord_loss = coord_fn(out["keypoints"], kps)
            cls_loss = cls_fn(out["logits"], cls)
            loss = (cfg.KP_LOSS_WEIGHT * hm_loss +
                    cfg.COORD_LOSS_WEIGHT * coord_loss +
                    cfg.CLS_LOSS_WEIGHT * cls_loss)

        tot_loss += loss.item(); tot_hm += hm_loss.item()
        tot_coord += coord_loss.item(); tot_cls += cls_loss.item()
        all_pkp.append(out["keypoints"].cpu()); all_gkp.append(kps.cpu())
        _, pred = out["logits"].max(1)
        all_pc.extend(pred.cpu().numpy()); all_gc.extend(cls.cpu().numpy())
        all_ow.append(ow); all_oh.append(oh)

    n = len(loader)
    f1 = f1_score(all_gc, all_pc, average="macro", zero_division=0)
    acc = np.mean(np.array(all_pc) == np.array(all_gc))
    pck = compute_pck(torch.cat(all_pkp), torch.cat(all_gkp),
                      torch.cat(all_ow), torch.cat(all_oh))

    return {"val_loss": tot_loss/n, "val_hm": tot_hm/n,
            "val_coord": tot_coord/n, "val_cls": tot_cls/n,
            "val_acc": acc, "val_f1": f1} | pck

# ══════════════════════════════════════════════════════════════════════
# EDA + DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════
# %%
with open(cfg.TRAIN_LABELS, "r") as f:
    annotations = json.load(f)

print(f"Total annotated samples: {len(annotations)}")
shapes = [v.get("verified_shape", "Unknown") for v in annotations.values()]
shape_counts = Counter(shapes)
print(f"\nClass distribution:")
for cls, cnt in sorted(shape_counts.items()):
    print(f"  {cls:>10s}: {cnt:>4d}  ({100*cnt/len(shapes):.1f}%)")

# Load valid samples
image_paths, labels_list = [], []
skipped = 0
for rel_path, label in annotations.items():
    abs_path = os.path.join(cfg.TRAIN_DATA_DIR, rel_path)
    if not os.path.isfile(abs_path):
        skipped += 1; continue
    shape = label.get("verified_shape")
    if shape not in cfg.SHAPE_CLASSES:
        skipped += 1; continue
    image_paths.append(abs_path)
    labels_list.append(label)

print(f"\nLoaded {len(image_paths)} samples, skipped {skipped}")

# Stratified split
shape_labels = [l["verified_shape"] for l in labels_list]
sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED)
train_idx, val_idx = next(sss.split(image_paths, shape_labels))

train_paths  = [image_paths[i] for i in train_idx]
train_labels = [labels_list[i] for i in train_idx]
val_paths    = [image_paths[i] for i in val_idx]
val_labels   = [labels_list[i] for i in val_idx]

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
print(f"Train dist: {Counter([l['verified_shape'] for l in train_labels])}")
print(f"Val   dist: {Counter([l['verified_shape'] for l in val_labels])}")

# %%
train_ds = GCPDataset(train_paths, train_labels, transform=get_train_transforms())
val_ds   = GCPDataset(val_paths,   val_labels,   transform=get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True)

# Sanity check
batch = next(iter(train_loader))
print(f"Images: {batch[0].shape}, Heatmap: {batch[1].shape}, "
      f"KP: {batch[2].shape}, Class: {batch[3].shape}")

# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════
# %%
model = GCPModelV2().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.FREEZE_EPOCHS, eta_min=1e-6)
scaler = GradScaler('cuda')

best_val_loss = float("inf")
patience_counter = 0

if cfg.FREEZE_EPOCHS > 0:
    model.freeze_backbone()
    print(f"Backbone FROZEN for first {cfg.FREEZE_EPOCHS} epochs")

# %%
for epoch in range(1, cfg.NUM_EPOCHS + 1):
    if epoch == cfg.FREEZE_EPOCHS + 1:
        model.unfreeze_backbone()
        print("═══ Backbone UNFROZEN ═══")
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": cfg.LR * 0.1},
            {"params": model.heatmap_head.parameters()},
            {"params": model.cls_head.parameters()},
        ], lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.NUM_EPOCHS - cfg.FREEZE_EPOCHS, eta_min=1e-6)

    trn = train_one_epoch(model, train_loader, optimizer, scaler)
    val = validate(model, val_loader)
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"Ep {epoch:>2d}/{cfg.NUM_EPOCHS} | "
          f"loss: {trn['train_loss']:.4f}/{val['val_loss']:.4f} | "
          f"hm: {trn['train_hm']:.5f}/{val['val_hm']:.5f} | "
          f"coord: {trn['train_coord']:.5f}/{val['val_coord']:.5f} | "
          f"acc: {trn['train_acc']:.3f}/{val['val_acc']:.3f} | "
          f"F1: {val['val_f1']:.3f} | "
          f"PCK@10/25/50: {val['PCK@10']:.3f}/{val['PCK@25']:.3f}/{val['PCK@50']:.3f} | "
          f"dist: {val['mean_dist_px']:.1f}px | lr: {lr:.2e}")

    if val["val_loss"] < best_val_loss:
        best_val_loss = val["val_loss"]
        patience_counter = 0
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "val_loss": best_val_loss, "val_f1": val["val_f1"]},
                   cfg.CHECKPOINT_PATH)
        print(f"  ✓ Best model (val_loss={best_val_loss:.4f}, "
              f"PCK@25={val['PCK@25']:.3f}, dist={val['mean_dist_px']:.1f}px)")
    else:
        patience_counter += 1
        if patience_counter >= cfg.PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch}"); break

print(f"\n✅ Training complete. Best val_loss: {best_val_loss:.4f}")

# ══════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════
# %%
ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded best model from epoch {ckpt['epoch']}")

test_paths = []
for root, _, files in os.walk(cfg.TEST_DATA_DIR):
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            test_paths.append(os.path.join(root, f))
print(f"Test images: {len(test_paths)}")

test_ds = GCPDataset(test_paths, labels=None, transform=get_test_transforms(), is_test=True)
test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.NUM_WORKERS, pin_memory=True)

predictions = {}
with torch.no_grad():
    for imgs, paths, ows, ohs in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast('cuda'):
            out = model(imgs)
        pkp = out["keypoints"].cpu()
        pcls = out["logits"].argmax(1).cpu()
        for i in range(len(paths)):
            ow, oh = float(ows[i]), float(ohs[i])
            # Reverse aspect-ratio-preserving resize + padding
            scale = cfg.IMG_SIZE / max(ow, oh)
            scaled_w = ow * scale
            scaled_h = oh * scale
            pad_x = (cfg.IMG_SIZE - scaled_w) / 2.0
            pad_y = (cfg.IMG_SIZE - scaled_h) / 2.0
            # Model predicts in [0, 1] of padded IMG_SIZE space
            kp_x_padded = float(pkp[i, 0]) * cfg.IMG_SIZE
            kp_y_padded = float(pkp[i, 1]) * cfg.IMG_SIZE
            # Remove padding and un-scale
            px = round((kp_x_padded - pad_x) / scale, 2)
            py = round((kp_y_padded - pad_y) / scale, 2)
            # Clamp to image bounds
            px = max(0, min(px, ow))
            py = max(0, min(py, oh))
            shape = cfg.SHAPE_CLASSES[int(pcls[i])]
            rel = os.path.relpath(paths[i], cfg.TEST_DATA_DIR).replace("\\", "/")
            predictions[rel] = {"mark": {"x": px, "y": py}, "verified_shape": shape}

with open(cfg.PREDICTIONS_PATH, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\n✅ Saved {len(predictions)} predictions → {cfg.PREDICTIONS_PATH}")
for k, v in list(predictions.items())[:5]:
    print(f"  {k}: ({v['mark']['x']}, {v['mark']['y']}), {v['verified_shape']}")
