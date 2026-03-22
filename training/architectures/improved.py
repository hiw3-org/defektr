"""
Improved miner architecture: MobileNetV2 encoder + lightweight U-Net decoder.

Same MobileNetV2 backbone as the baseline (identical speed profile) with an
added upsampling decoder that produces a 256×256 binary defect mask.  Two
output heads:

  - confidence: float32 [B]          (sigmoid, 1 = defective)  — same as baseline
  - mask:       float32 [B, 1, 256, 256]  (sigmoid, 1 = defect pixel)

Having a mask head unlocks the full localisation component of the validator
reward and, when trained with augmented data, also boosts the robustness score.

Input:  float32 tensor [B, 3, 256, 256]  (normalised, ImageNet stats)
ONNX export input name:   "image"
ONNX export output names: "confidence", "mask"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DecoderBlock(nn.Module):
    """Upsample by 2× then refine with two conv layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ImprovedMiner(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # ── Encoder: reuse MobileNetV2 feature stages ────────────────────────
        # For a 256×256 input the spatial sizes at each stage are:
        #   features[0..1]  → stride 2  → 128×128  (ch=16)
        #   features[2..3]  → stride 4  → 64×64    (ch=24)
        #   features[4..6]  → stride 8  → 32×32    (ch=32)
        #   features[7..13] → stride 16 → 16×16    (ch=96)
        #   features[14..18]→ stride 32 → 8×8      (ch=1280)
        self.enc1 = backbone.features[0:2]   # → 128×128, ch=16
        self.enc2 = backbone.features[2:4]   # → 64×64,   ch=24
        self.enc3 = backbone.features[4:7]   # → 32×32,   ch=32
        self.enc4 = backbone.features[7:14]  # → 16×16,   ch=96
        self.enc5 = backbone.features[14:]   # → 8×8,     ch=1280

        if freeze_backbone:
            for stage in (self.enc1, self.enc2, self.enc3, self.enc4, self.enc5):
                for p in stage.parameters():
                    p.requires_grad = False

        # ── Classification head (identical to baseline) ───────────────────────
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1),
        )

        # ── Decoder: upsample 8×8 → 256×256 ─────────────────────────────────
        # Each block doubles spatial resolution; channel counts shrink toward 1.
        self.dec4 = _DecoderBlock(1280, 256)  # 8   → 16
        self.dec3 = _DecoderBlock(256,   64)  # 16  → 32
        self.dec2 = _DecoderBlock(64,    32)  # 32  → 64
        self.dec1 = _DecoderBlock(32,    16)  # 64  → 128
        self.dec0 = _DecoderBlock(16,     8)  # 128 → 256

        self.mask_head = nn.Conv2d(8, 1, kernel_size=1)  # → [B, 1, 256, 256]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, 256, 256] float32

        Returns:
            confidence: [B]             float32  sigmoid
            mask:       [B, 1, 256, 256] float32  sigmoid
        """
        # Encoder
        e1 = self.enc1(x)   # [B, 16,   128, 128]
        e2 = self.enc2(e1)  # [B, 24,    64,  64]
        e3 = self.enc3(e2)  # [B, 32,    32,  32]
        e4 = self.enc4(e3)  # [B, 96,    16,  16]
        e5 = self.enc5(e4)  # [B, 1280,   8,   8]

        # Classification head
        feat       = self.pool(e5).flatten(1)       # [B, 1280]
        confidence = torch.sigmoid(self.classifier(feat)).squeeze(1)  # [B]

        # Decoder (no skip connections — keeps the decoder tiny)
        d = self.dec4(e5)   # [B, 256, 16, 16]
        d = self.dec3(d)    # [B,  64, 32, 32]
        d = self.dec2(d)    # [B,  32, 64, 64]
        d = self.dec1(d)    # [B,  16, 128, 128]
        d = self.dec0(d)    # [B,   8, 256, 256]
        mask = torch.sigmoid(self.mask_head(d))     # [B, 1, 256, 256]

        return confidence, mask
