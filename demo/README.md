---
title: GCP Pose Estimation
emoji: 📐
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
---

# GCP Pose Estimation

Aerial Ground Control Point (GCP) keypoint localization and shape classification.

**Model:** EfficientNet-B2 + Heatmap Decoder  
**Tasks:** Keypoint regression (x, y) + Shape classification (Cross / L-Shape / Square)

Upload an aerial image containing a GCP marker to detect its center coordinates and shape.
