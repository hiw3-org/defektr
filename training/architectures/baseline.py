"""
Baseline miner architecture: MobileNetV2 classifier-only.

A pretrained MobileNetV2 backbone with its classifier replaced by a single
sigmoid output node.  No mask head — the model can only answer "defective or
not" and therefore scores at most α_cls × α_acc = 0.6 × 0.5 = 0.30 of the
total validator reward.

Input:  float32 tensor  [B, 3, 256, 256]  (normalised, ImageNet stats)
Output: float32 tensor  [B]               (sigmoid confidence, 1 = defective)

ONNX export input name:  "image"
ONNX export output name: "confidence"
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class BaselineMiner(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Keep the feature extractor, drop the original classifier
        self.features = backbone.features  # output: [B, 1280, 8, 8] for 256×256 input

        self.pool = nn.AdaptiveAvgPool2d(1)  # → [B, 1280, 1, 1]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 256, 256] float32

        Returns:
            [B] float32  — raw sigmoid output (confidence of being defective)
        """
        feat = self.features(x)            # [B, 1280, 8, 8]
        feat = self.pool(feat)             # [B, 1280, 1, 1]
        feat = feat.flatten(1)             # [B, 1280]
        logit = self.classifier(feat)      # [B, 1]
        return torch.sigmoid(logit).squeeze(1)  # [B]
