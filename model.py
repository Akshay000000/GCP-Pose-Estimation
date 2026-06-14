"""
Multi-task EfficientNet model for GCP keypoint regression + shape classification.
"""
import torch
import torch.nn as nn
import timm


class GCPModel(nn.Module):
    """
    Shared-backbone multi-task model.
    - Backbone: EfficientNet-B2 (pretrained on ImageNet)
    - Head 1: Keypoint regression → (x, y) normalised to [0, 1]
    - Head 2: Shape classification → 3 classes
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=0,          # strip default head → returns feature vector
            drop_rate=config.DROP_RATE,
        )
        feat_dim = self.backbone.num_features  # 1408 for efficientnet_b2

        # ── Keypoint regression head ─────────────────────────────────────
        self.keypoint_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROP_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROP_RATE / 2),
            nn.Linear(128, 2),
            nn.Sigmoid(),           # constrain output to [0, 1]
        )

        # ── Classification head ──────────────────────────────────────────
        self.classification_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROP_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROP_RATE / 2),
            nn.Linear(128, config.NUM_CLASSES),
        )

    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.keypoint_head(features)
        logits = self.classification_head(features)
        return {"keypoints": keypoints, "logits": logits}

    # ── Freeze / unfreeze backbone for staged training ───────────────────
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
