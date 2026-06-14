# Architecture & Engineering Decision Log

This document tracks the major technical decisions made during the development of the Aerial GCP Pose Estimation pipeline.

## 1. Keypoint Localization Approach
* **Context:** The core requirement is predicting the exact $(x, y)$ coordinate of the GCP center.
* **Options Considered:** 
  1. *Direct Regression:* Using Fully Connected (Linear) layers to output 2 continuous values $(x, y)$.
  2. *Heatmap-based Decoder:* Upsampling features to generate a 2D probability heatmap representing the keypoint location.
* **Decision:** **Heatmap-based Decoder with Differentiable Spatial Soft-Argmax.**
* **Rationale:** Direct regression destroys spatial information and struggles to map high-resolution image features to precise coordinates. Heatmaps maintain spatial correspondence. By applying a Spatial Soft-Argmax over the heatmap, we calculate the expected value of the coordinates. This achieves sub-pixel accuracy while remaining fully differentiable for end-to-end training.

## 2. Backbone Selection
* **Context:** High-resolution aerial imagery (2048x1365) requires strong feature extraction, but computational constraints exist.
* **Options Considered:** ResNet-50/101, Vision Transformers (ViT), EfficientNet.
* **Decision:** **EfficientNet-B2.**
* **Rationale:** EfficientNet provides the best accuracy-per-parameter via compound scaling. B2 is lightweight enough (~9.1M parameters) to allow reasonable batch sizes (e.g., 16) with high-resolution input crops on standard GPUs, while still offering the receptive field necessary to understand aerial textures and GCP marker structures.

## 3. Handling Out-of-Bounds Augmented Keypoints
* **Context:** To ensure robustness, heavy geometric augmentations (affine transforms, rotations) were applied. However, this occasionally pushed the center keypoint out of the crop boundaries.
* **Issue Encountered:** The default Albumentations behavior (`remove_invisible=True`) drops out-of-bounds keypoints. The data loader would then fall back to returning the *original, pre-augmented* coordinates alongside the *augmented* image, resulting in wildly incorrect supervision and catastrophic model divergence.
* **Decision:** **Set `remove_invisible=False` and manually clamp coordinates.**
* **Rationale:** By keeping the keypoint metadata even if it shifts slightly outside the frame, we can manually clamp the coordinates to `[0, 1]`. This ensures the supervision signal remains geometrically aligned with the augmented image.

## 4. Heatmap Hyperparameter Tuning (Sigma & Temperature)
* **Context:** The initial model converged to a trivial solution (predicting all zeros for the heatmap) and predicted the exact center of the image `(0.5, 0.5)` for every sample.
* **Issue Encountered:** A Gaussian Sigma of `4.0` on a 160x160 heatmap resulted in ~60 positive pixels vs. 25,540 background pixels. The MSE loss was easily minimized by predicting a flat zero matrix. Furthermore, a Softmax temperature of `10.0` over 25,600 elements created a uniform distribution, causing the expected value (soft-argmax) to default to the image center.
* **Decision:** **Increase Gaussian Sigma to `10.0` and Softmax Temperature to `50.0`.**
* **Rationale:** A wider sigma (`10.0`) provided enough gradient signal to prevent the model from collapsing to zero. A higher temperature (`50.0`) sharpened the spatial softmax distribution, allowing the model to confidently isolate the peak activation and output varied, highly localized coordinate predictions.

## 5. Deployment Architecture
* **Context:** The assignment requested reproducible inference. 
* **Options Considered:** Providing only a Jupyter Notebook vs. building a web API (Vercel/FastAPI) vs. Gradio.
* **Decision:** **Self-contained Gradio application deployed on Hugging Face Spaces.**
* **Rationale:** Hugging Face Spaces natively provisions the heavy Linux environments required for PyTorch inference (unlike Vercel, which has a 250MB serverless limit). Gradio allows for an intuitive UI (upload image -> see crosshair prediction) while generating an accessible REST API automatically.
