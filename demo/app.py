"""
GCP Pose Estimation — Interactive Demo
Heatmap-based keypoint localization + shape classification for aerial GCP markers.
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
IMG_SIZE = 640
SHAPE_CLASSES = ["Cross", "L-Shape", "Square"]
NUM_CLASSES = 3
BACKBONE = "efficientnet_b2"
DROP_RATE = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_v2.pth")

# Color palette for each class
CLASS_COLORS = {
    "Cross":   (255, 59, 48),    # Red
    "L-Shape": (52, 199, 89),    # Green
    "Square":  (0, 122, 255),    # Blue
}

# ══════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (mirrors kaggle_notebook_v2.py exactly)
# ══════════════════════════════════════════════════════════════════════
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
            BACKBONE, pretrained=False,
            features_only=True, out_indices=[4])
        feat_dim = self.backbone.feature_info[-1]['num_chs']

        self.heatmap_head = HeatmapHead(feat_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(DROP_RATE),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(DROP_RATE / 2),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        feats = self.backbone(x)[-1]
        heatmap = self.heatmap_head(feats)
        keypoints = spatial_soft_argmax(heatmap)
        pooled = self.gap(feats).flatten(1)
        logits = self.cls_head(pooled)
        return {"keypoints": keypoints, "logits": logits, "heatmap": heatmap}


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
model = None


def load_model():
    global model
    if model is not None:
        return model

    net = GCPModelV2().to(DEVICE)
    net.eval()  # Set eval mode FIRST to avoid BatchNorm issues with batch_size=1

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        net.load_state_dict(state, strict=False)
        print(f"[OK] Loaded model weights from {MODEL_PATH}")
    else:
        print(f"[WARN] No weights found at {MODEL_PATH} -- running with random weights (demo mode)")

    model = net
    return model


# ══════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_np):
    """Resize, normalize, convert to tensor."""
    img = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0).to(DEVICE)


def draw_prediction(image_np, x_px, y_px, shape, confidence):
    """Draw keypoint marker and label on the image."""
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    h, w = image_np.shape[:2]

    color = CLASS_COLORS.get(shape, (255, 255, 255))
    r = max(12, int(min(w, h) * 0.012))  # adaptive radius

    # Outer ring
    draw.ellipse(
        [x_px - r * 2, y_px - r * 2, x_px + r * 2, y_px + r * 2],
        outline=color, width=max(3, r // 3))

    # Crosshair
    line_len = r * 3
    lw = max(2, r // 4)
    draw.line([(x_px - line_len, y_px), (x_px + line_len, y_px)], fill=color, width=lw)
    draw.line([(x_px, y_px - line_len), (x_px, y_px + line_len)], fill=color, width=lw)

    # Center dot
    draw.ellipse(
        [x_px - r // 2, y_px - r // 2, x_px + r // 2, y_px + r // 2],
        fill=color)

    # Label background
    try:
        font = ImageFont.truetype("arial.ttf", max(20, int(min(w, h) * 0.022)))
    except OSError:
        font = ImageFont.load_default()

    label = f"{shape} ({confidence:.1%})"
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    lx = max(10, min(x_px - tw // 2, w - tw - 10))
    ly = max(10, y_px - r * 3 - th - 10)

    # Pill-shaped label bg
    pad = 8
    draw.rounded_rectangle(
        [lx - pad, ly - pad, lx + tw + pad, ly + th + pad],
        radius=12, fill=(*color, 200))
    draw.text((lx, ly), label, fill=(255, 255, 255), font=font)

    # Coordinate tag below crosshair
    coord_text = f"({int(x_px)}, {int(y_px)})"
    cbbox = draw.textbbox((0, 0), coord_text, font=font)
    ctw = cbbox[2] - cbbox[0]
    cth = cbbox[3] - cbbox[1]
    cx = max(10, min(x_px - ctw // 2, w - ctw - 10))
    cy = min(h - cth - 10, y_px + r * 3 + 10)
    draw.rounded_rectangle(
        [cx - pad, cy - pad, cx + ctw + pad, cy + cth + pad],
        radius=10, fill=(0, 0, 0, 180))
    draw.text((cx, cy), coord_text, fill=(255, 255, 255), font=font)

    return np.array(img)


@torch.no_grad()
def predict(image):
    """Run inference on a single image."""
    if image is None:
        return None, "No image provided"

    net = load_model()

    # Convert PIL → numpy RGB
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image

    orig_h, orig_w = image_np.shape[:2]
    inp = preprocess(image_np)

    with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
        out = net(inp)

    # Extract predictions
    kp = out["keypoints"].cpu().numpy()[0]  # [x_norm, y_norm]
    logits = out["logits"].cpu()
    probs = F.softmax(logits, dim=1).numpy()[0]
    cls_idx = int(logits.argmax(1)[0])

    x_px = float(kp[0]) * orig_w
    y_px = float(kp[1]) * orig_h
    shape = SHAPE_CLASSES[cls_idx]
    confidence = float(probs[cls_idx])

    # Draw result on image
    result_img = draw_prediction(image_np.copy(), x_px, y_px, shape, confidence)

    # Build info text
    info = (
        f"### Prediction Results\n\n"
        f"| Property | Value |\n"
        f"|---|---|\n"
        f"| **Keypoint X** | {x_px:.1f} px |\n"
        f"| **Keypoint Y** | {y_px:.1f} px |\n"
        f"| **Shape** | {shape} |\n"
        f"| **Confidence** | {confidence:.1%} |\n"
        f"| **Image Size** | {orig_w} × {orig_h} |\n"
    )

    return result_img, info


# ══════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")
sample_images = sorted([
    os.path.join(SAMPLE_DIR, f)
    for f in os.listdir(SAMPLE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]) if os.path.isdir(SAMPLE_DIR) else []

CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.gr-button-primary {
    background-color: #0f172a !important; /* Sleek dark slate */
    color: white !important;
    border: none !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    background-color: #334155 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}
footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="GCP Pose Estimation") as demo:
    gr.Markdown(
        """
        # GCP Pose Estimation
        Upload an aerial image to detect the **center keypoint** and **shape** of a Ground Control Point marker.

        *Built with EfficientNet-B2 backbone + heatmap-based keypoint regression.*
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Aerial Image",
                height=400,
            )
            predict_btn = gr.Button(
                "Detect GCP",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Prediction",
                height=400,
            )
            output_info = gr.Markdown(
                value="*Upload an image and click Detect to see results.*"
            )

    if sample_images:
        gr.Markdown("### Sample Images")
        gr.Examples(
            examples=[[img] for img in sample_images],
            inputs=input_image,
            outputs=[output_image, output_info],
            fn=predict,
            cache_examples=False,
            examples_per_page=8,
        )

    gr.Markdown(
        """
        ---
        <center>

        **How it works:** The model uses an EfficientNet-B2 backbone with a heatmap decoder head
        to localize the GCP center, and a classification head to identify the marker shape
        (Cross, L-Shape, or Square).

        </center>
        """,
    )

    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_image, output_info],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=THEME)

