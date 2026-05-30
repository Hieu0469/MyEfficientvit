"""
EfficientViT-DeepLabV3+ Segmentation Model
===========================================
Kiến trúc:
  - Backbone  : EfficientViTBackbone / EfficientViTLargeBackbone
  - Decoder   : DeepLabV3+ style
                  • ASPP  trên stage4 (stride 32)  → context features
                  • Low-level projection từ stage1 (stride 4)
                  • Concat + Conv → full resolution
  - Head      : Conv1×1 → n_classes

DeepLabV3+ decoder path:
  stage4 (stride 32) ──► ASPP ──────────────────────────────────────┐
                                                                     ▼
  stage1 (stride  4) ──► LowLevelProj ──► cat ──► FusionConv ──► up×4 ──► SegHead

So với DeepLabV3+ gốc (dùng ResNet/Xception):
  - ASPP giữ nguyên (rates 6/12/18 + global avg pool)
  - Low-level feature lấy từ stage1 (stride 4) thay vì layer1 của ResNet
  - Backbone thay bằng EfficientViT (lightweight transformer)

Đặt learning rate riêng:
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": model.decoder.parameters(),  "lr": 1e-3},
        {"params": model.seg_head.parameters(), "lr": 1e-3},
    ])
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Cần cài: pip install efficientvit ─────────────────────────────────────────
from efficientvit.models.efficientvit.backbone import (
    EfficientViTBackbone,
    EfficientViTLargeBackbone,
    efficientvit_backbone_b0,
    efficientvit_backbone_b1,
    efficientvit_backbone_b2,
    efficientvit_backbone_b3,
    efficientvit_backbone_l1,
    efficientvit_backbone_l2,
)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3,
                  stride: int = 1, padding: int = 1,
                  dilation: int = 1, bias: bool = False) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding * dilation if dilation > 1 else padding,
                  dilation=dilation, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ASPPPool(nn.Module):
    """
    Global Average Pooling branch của ASPP.
    Dùng GroupNorm thay BatchNorm vì sau AdaptiveAvgPool2d(1) tensor có
    spatial size 1×1 — BN sẽ báo lỗi khi batch_size=1 ở training mode.
    """

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 32):
        super().__init__()
        # Đảm bảo num_groups chia hết out_ch; fallback về 1 nếu cần
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups //= 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.proj(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    5 nhánh song song: Conv1×1 + Conv3×3 rate 6/12/18 + GlobalAvgPool
    → concat → Conv1×1 project
    """

    def __init__(self, in_ch: int, out_ch: int = 256,
                 rates: tuple[int, ...] = (6, 12, 18), dropout: float = 0.5):
        super().__init__()
        self.branches = nn.ModuleList([
            _conv_bn_relu(in_ch, out_ch, kernel=1, padding=0),          # 1×1
            *[_conv_bn_relu(in_ch, out_ch, dilation=r) for r in rates], # dilated
            ASPPPool(in_ch, out_ch),                                     # global
        ])
        self.project = nn.Sequential(
            _conv_bn_relu(out_ch * (1 + len(rates) + 1), out_ch, kernel=1, padding=0),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.project(out)


# ──────────────────────────────────────────────────────────────────────────────
# DeepLabV3+ Decoder
# ──────────────────────────────────────────────────────────────────────────────

class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ decoder gồm:
      1. ASPP   : xử lý high-level feature (stage4, stride 32)
      2. LowLevelProj : giảm channel low-level feature (stage1, stride 4)
      3. FusionConv   : concat ASPP↑ + low-level → refine
      4. Upsample ×4  : đưa về stride 1 (full resolution)

    Args:
        high_in_ch    : channel của stage4 (bottleneck)
        low_in_ch     : channel của stage1 (low-level skip)
        aspp_out_ch   : số channel output của ASPP (mặc định 256)
        low_proj_ch   : số channel sau khi project low-level (mặc định 48)
        fusion_ch     : số channel của FusionConv (mặc định 256)
        aspp_rates    : dilation rates cho ASPP
        dropout       : dropout trong ASPP
    """

    def __init__(
        self,
        high_in_ch: int,
        low_in_ch: int,
        aspp_out_ch: int = 256,
        low_proj_ch: int = 48,
        fusion_ch: int = 256,
        aspp_rates: tuple[int, ...] = (6, 12, 18),
        dropout: float = 0.5,
    ):
        super().__init__()

        # 1. ASPP trên high-level feature (stride 32)
        self.aspp = ASPP(high_in_ch, aspp_out_ch, rates=aspp_rates, dropout=dropout)

        # 2. Project low-level feature (stride 4) xuống ít channel
        self.low_proj = _conv_bn_relu(low_in_ch, low_proj_ch, kernel=1, padding=0)

        # 3. Fusion: concat(ASPP↑, low_proj) → refine
        self.fusion = nn.Sequential(
            _conv_bn_relu(aspp_out_ch + low_proj_ch, fusion_ch),
            _conv_bn_relu(fusion_ch, fusion_ch),
        )

        # 4. Upsample ×4 đưa từ stride 4 về full resolution
        self.final_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self.out_channels = fusion_ch   # dùng khi xây seg_head

    def forward(
        self,
        high: torch.Tensor,   # stage4, stride 32
        low: torch.Tensor,    # stage1, stride 4
    ) -> torch.Tensor:
        # ── ASPP branch ──────────────────────────────────────────────────────
        aspp_feat = self.aspp(high)   # (B, aspp_out_ch, H/32, W/32)

        # ── Upsample ASPP lên stride 4 (×8) ─────────────────────────────────
        aspp_up = F.interpolate(
            aspp_feat, size=low.shape[-2:], mode="bilinear", align_corners=False
        )

        # ── Low-level projection ──────────────────────────────────────────────
        low_feat = self.low_proj(low)   # (B, low_proj_ch, H/4, W/4)

        # ── Concat + Fusion ───────────────────────────────────────────────────
        fused = torch.cat([aspp_up, low_feat], dim=1)
        fused = self.fusion(fused)      # (B, fusion_ch, H/4, W/4)

        # ── Final upsample ×4 → full resolution ──────────────────────────────
        return self.final_up(fused)     # (B, fusion_ch, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class EfficientViTDeepLabV3Plus(nn.Module):
    """
    Segmentation model:
      backbone  = EfficientViT encoder
      decoder   = DeepLabV3PlusDecoder  (ASPP + low-level fusion)
      seg_head  = Conv1×1 → n_classes

    Args:
        backbone        : EfficientViTBackbone hoặc EfficientViTLargeBackbone
        high_in_ch      : channel của stage4 (high-level feature)
        low_in_ch       : channel của stage1 (low-level feature)
        aspp_out_ch     : channel output ASPP (mặc định 256)
        low_proj_ch     : channel sau project low-level (mặc định 48)
        fusion_ch       : channel FusionConv (mặc định 256)
        n_classes       : số lớp segmentation
        dropout         : dropout trong ASPP
    """

    def __init__(
        self,
        backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone],
        high_in_ch: int,
        low_in_ch: int,
        aspp_out_ch: int = 256,
        low_proj_ch: int = 48,
        fusion_ch: int = 256,
        n_classes: int = 19,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = DeepLabV3PlusDecoder(
            high_in_ch=high_in_ch,
            low_in_ch=low_in_ch,
            aspp_out_ch=aspp_out_ch,
            low_proj_ch=low_proj_ch,
            fusion_ch=fusion_ch,
            dropout=dropout,
        )
        self.seg_head = nn.Conv2d(self.decoder.out_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────────────────
        feat = self.backbone(x)
        low  = feat["stage1"]   # stride 4  – low-level detail
        high = feat["stage4"]   # stride 32 – high-level context (bottleneck)

        # ── Decoder ──────────────────────────────────────────────────────────
        d = self.decoder(high, low)   # (B, fusion_ch, H, W)

        # ── Seg Head ─────────────────────────────────────────────────────────
        return self.seg_head(d)       # (B, n_classes, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────────────────────────────────────

def _make_deeplabv3plus(
    backbone,
    high_in_ch: int,
    low_in_ch: int,
    n_classes: int,
    aspp_out_ch: int = 256,
    low_proj_ch: int = 48,
    fusion_ch: int = 256,
    dropout: float = 0.5,
) -> EfficientViTDeepLabV3Plus:
    return EfficientViTDeepLabV3Plus(
        backbone=backbone,
        high_in_ch=high_in_ch,
        low_in_ch=low_in_ch,
        aspp_out_ch=aspp_out_ch,
        low_proj_ch=low_proj_ch,
        fusion_ch=fusion_ch,
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_deeplabv3plus_b0(n_classes: int = 19, **kw) -> EfficientViTDeepLabV3Plus:
    """EfficientViT-B0 + DeepLabV3+ decoder"""
    # B0 widths: [8, 16, 32, 64, 128]  → stage1=16, stage4=128
    return _make_deeplabv3plus(efficientvit_backbone_b0(**kw),
                               high_in_ch=128, low_in_ch=16,
                               aspp_out_ch=128, low_proj_ch=24, fusion_ch=128,
                               n_classes=n_classes)


def efficientvit_deeplabv3plus_b1(n_classes: int = 19, **kw) -> EfficientViTDeepLabV3Plus:
    """EfficientViT-B1 + DeepLabV3+ decoder"""
    # B1 widths: [16, 32, 64, 128, 256]  → stage1=32, stage4=256
    return _make_deeplabv3plus(efficientvit_backbone_b1(**kw),
                               high_in_ch=256, low_in_ch=32,
                               n_classes=n_classes)


def efficientvit_deeplabv3plus_b2(n_classes: int = 19, **kw) -> EfficientViTDeepLabV3Plus:
    """EfficientViT-B2 + DeepLabV3+ decoder"""
    # B2 widths: [24, 48, 96, 192, 384]  → stage1=48, stage4=384
    return _make_deeplabv3plus(efficientvit_backbone_b2(**kw),
                               high_in_ch=384, low_in_ch=48,
                               n_classes=n_classes)


def efficientvit_deeplabv3plus_b3(n_classes: int = 19, **kw) -> EfficientViTDeepLabV3Plus:
    """EfficientViT-B3 + DeepLabV3+ decoder"""
    # B3 widths: [32, 64, 128, 256, 512]  → stage1=64, stage4=512
    return _make_deeplabv3plus(efficientvit_backbone_b3(**kw),
                               high_in_ch=512, low_in_ch=64,
                               n_classes=n_classes)


def efficientvit_deeplabv3plus_l1(n_classes: int = 19, **kw) -> EfficientViTDeepLabV3Plus:
    """EfficientViT-L1 + DeepLabV3+ decoder"""
    return _make_deeplabv3plus(efficientvit_backbone_l1(**kw),
                               high_in_ch=512, low_in_ch=64,
                               n_classes=n_classes)


def efficientvit_deeplabv3plus_l2(n_classes: int = 19, **kw) -> EfficientViTDeepLabV3Plus:
    """EfficientViT-L2 + DeepLabV3+ decoder"""
    return _make_deeplabv3plus(efficientvit_backbone_l2(**kw),
                               high_in_ch=512, low_in_ch=64,
                               n_classes=n_classes)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 512, 1024   # Cityscapes resolution

    configs = {
        "b0": efficientvit_deeplabv3plus_b0,
        "b1": efficientvit_deeplabv3plus_b1,
        "b2": efficientvit_deeplabv3plus_b2,
        "b3": efficientvit_deeplabv3plus_b3,
        "l1": efficientvit_deeplabv3plus_l1,
        "l2": efficientvit_deeplabv3plus_l2,
    }

    variant = sys.argv[1] if len(sys.argv) > 1 else "b1"
    if variant not in configs:
        print(f"Unknown variant '{variant}'. Choose from: {list(configs)}")
        sys.exit(1)

    print(f"\n=== EfficientViT-DeepLabV3+-{variant.upper()} ===")
    model = configs[variant](n_classes=19).to(device)
    model.eval()

    x = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        out = model(x)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(out.shape)}   (expected: [1, 19, {H}, {W}])")
    print(f"Params : {n_params:.2f} M")
    assert out.shape == (1, 19, H, W), "Shape mismatch!"
    print("✅ Shape check passed.")

    # ── Ví dụ đặt learning rate riêng từng phần ──────────────────────────────
    print("\n── Param groups ──────────────────────────────────────────────────")
    groups = [
        ("backbone", model.backbone.parameters()),
        ("decoder",  model.decoder.parameters()),
        ("seg_head", model.seg_head.parameters()),
    ]
    for name, params in groups:
        n = sum(p.numel() for p in params) / 1e6
        print(f"  {name:<10}: {n:.2f} M params")
