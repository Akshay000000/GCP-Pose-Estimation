# Aerial GCP Pose Estimation

## 1. Network Architecture Choice and Rationale
The pipeline utilizes a shared-backbone, multi-task architecture built on **EfficientNet-B2**. The model splits into two parallel heads to simultaneously predict the Ground Control Point (GCP) keypoint coordinates and shape class.

**Rationale:**
* **Backbone (EfficientNet-B2):** Chosen for its excellent trade-off between parameter efficiency and feature extraction capability. High-resolution aerial imagery requires strong spatial understanding, but deploying overly large models (like ResNet-101 or ViT) introduces unnecessary computational overhead.
* **Keypoint Localization (Heatmap + Spatial Soft-Argmax):** Direct coordinate regression (e.g., a Linear layer predicting X, Y) often struggles with highly non-linear spatial mappings. Instead, we use a custom **Heatmap Decoder** (transposed convolutions) that upsamples the backbone features into a spatial heatmap. We then apply **Differentiable Spatial Soft-Argmax** to convert the 2D heatmap into precise $(x, y)$ coordinates. This provides the accuracy of heatmap localization while remaining fully differentiable for end-to-end training.
* **Shape Classification:** A simple MLP head applied after Global Average Pooling (GAP) on the backbone features handles the 3-class shape classification (Cross, L-Shape, Square).

## 2. Training Strategy
* **Data Augmentation:** Because GCP markers can appear at any orientation, scale, or lighting condition, heavy geometric augmentations were applied using Albumentations: `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, and `Affine` (scaling, rotation, translation). Color jitter and Gaussian noise were added to simulate different drone cameras and weather conditions.
* **Loss Functions:** 
  * *Heatmap Loss:* Mean Squared Error (MSE) between the predicted heatmap and a generated 2D Gaussian target centered at the ground-truth keypoint.
  * *Coordinate Loss:* MSE between the soft-argmax predicted $(x, y)$ coordinates and the normalized ground-truth $(x, y)$.
  * *Classification Loss:* Cross Entropy Loss.
  * *Total Loss:* A weighted sum of the above, heavily prioritizing the keypoint localization gradients (`15.0 * HM_Loss + 8.0 * Coord_Loss + 1.0 * Cls_Loss`).
* **Optimization & Scheduling:** Trained using AdamW optimizer with mixed precision (AMP) for speed. A `CosineAnnealingLR` scheduler was used to ensure smooth convergence without disruptive learning rate spikes. The backbone was frozen for the first 3 epochs to allow the randomly initialized heads to stabilize before fine-tuning the entire network.

## 3. Challenges Mitigated
* **Keypoint Dropping during Augmentation:** Albumentations defaults to dropping keypoints that fall outside the image boundaries after affine transformations. Because our keypoints represent single center points, an augmentation pushing the marker 1px out of bounds would silently drop the keypoint, causing the dataset loader to fall back to the pre-augmented coordinates on a transformed image. This was mitigated by setting `remove_invisible=False` and manually clamping coordinates to `[0, 1]`.
* **Sparse Gradients on High-Res Heatmaps:** The model initially failed to converge (predicting all zeros) because the Gaussian sigma (`4.0`) was too small for the 160x160 target heatmaps, creating an overwhelming class imbalance of background pixels. Increasing `HEATMAP_SIGMA` to `10.0` provided a wider Gaussian spread, giving the network a stronger gradient signal to learn localization.
* **Soft-Argmax Temperature Collapse:** A standard spatial soft-argmax across 25,600 pixels resulted in overly uniform weights, pulling all predictions toward the center of the image `(0.5, 0.5)`. The softmax `temp` parameter was aggressively increased from `10.0` to `50.0`, sharpening the distribution to accurately isolate the peak heatmap activation.

## 4. Inference & Reproducibility
The easiest way to test the model is via the live interactive web demo deployed on Hugging Face Spaces. It runs the exact same inference pipeline as the testing script.

**Live Demo Link:** 
[Insert your Hugging Face Space URL here, e.g., https://huggingface.co/spaces/AkshaySriram/gcp-pose-estimation]

### Generating `predictions.json` Locally
To run the inference script locally and reproduce the `predictions.json` file:

1. Ensure your directory structure matches the assignment (`train_dataset/` and `test_dataset/` in the root).
2. Download the `best_model_v2.pth` weights and place them in the root directory.
3. Install dependencies:
   ```bash
   pip install torch torchvision timm albumentations opencv-python
   ```
4. Run the inference block in the provided `kaggle_notebook_v2.py` script. The script automatically iterates over `test_dataset`, performs inference with mixed-precision, denormalizes the predicted coordinates back to the original image dimensions, maps the class indices to string names, and saves the output formatted exactly to specification in `predictions.json`.
