"""
EfficientViT-UNet Segmentation Model
=====================================
Kiến trúc:
  - Encoder  : EfficientViTBackbone / EfficientViTLargeBackbone  (từ MIT-HAN-Lab)
  - Decoder  : UNet-style (skip connections + double-conv upsample)
  - Head     : SegHead (ConvLayer 1×1 → n_classes, giống efficientvit_seg)

Feature map từ backbone (B-series, input 1024×512):
  stage0  → stride 2   (H/2)
  stage1  → stride 4   (H/4)
  stage2  → stride 8   (H/8)
  stage3  → stride 16  (H/16)
  stage4  → stride 32  (H/32)   ← bottleneck

UNet decoder path:
  stage4 → up → cat(stage3) → DoubleConv
         → up → cat(stage2) → DoubleConv
         → up → cat(stage1) → DoubleConv
         → up → cat(stage0) → DoubleConv
         → up (×2 về full res nếu cần, hoặc giữ stride-2)
         → SegHead 1×1
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

class DoubleConv(nn.Module):
    """Conv-BN-ReLU × 2  (standard UNet block)."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    """
    Upsample (bilinear) → concat skip → DoubleConv.
    Tương đương một bước decoder của UNet.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """
        Args:
            in_ch   : số channel của feature map từ cấp sâu hơn (trước upsample)
            skip_ch : số channel của skip connection từ encoder
            out_ch  : số channel output của block này
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Padding nếu kích thước lẻ
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# UNet Decoder (gom toàn bộ các bước upsample thành một module riêng)
# ──────────────────────────────────────────────────────────────────────────────

class UNetDecoder(nn.Module):
    """
    Toàn bộ phần decoder UNet: up4 → up3 → up2 → up1 → final_up.

    Được tách thành module độc lập để dễ đặt learning rate riêng, ví dụ:

        optimizer = AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {"params": model.decoder.parameters(),  "lr": 1e-3},
            {"params": model.seg_head.parameters(), "lr": 1e-3},
        ])

    Args:
        backbone_widths  : [w0, w1, w2, w3, w4] – width_list của backbone
        decoder_channels : [d0, d1, d2, d3]      – out_ch tại mỗi bước up
    """

    def __init__(self, backbone_widths: list[int], decoder_channels: list[int]):
        super().__init__()
        w0, w1, w2, w3, w4 = backbone_widths
        d0, d1, d2, d3 = decoder_channels

        self.up4 = UNetUpBlock(in_ch=w4, skip_ch=w3, out_ch=d0)   # stride 32 → 16
        self.up3 = UNetUpBlock(in_ch=d0, skip_ch=w2, out_ch=d1)   # stride 16 → 8
        self.up2 = UNetUpBlock(in_ch=d1, skip_ch=w1, out_ch=d2)   # stride  8 → 4
        self.up1 = UNetUpBlock(in_ch=d2, skip_ch=w0, out_ch=d3)   # stride  4 → 2
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.out_channels = d3   # tiện tham chiếu khi xây seg_head

    def forward(
        self,
        s4: torch.Tensor,
        s3: torch.Tensor,
        s2: torch.Tensor,
        s1: torch.Tensor,
        s0: torch.Tensor,
    ) -> torch.Tensor:
        d = self.up4(s4, s3)   # stride 16
        d = self.up3(d,  s2)   # stride 8
        d = self.up2(d,  s1)   # stride 4
        d = self.up1(d,  s0)   # stride 2
        d = self.final_up(d)   # stride 1 (full resolution)
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class EfficientViTUNetSeg(nn.Module):
    """
    Segmentation model:
      backbone  = EfficientViT encoder
      decoder   = UNetDecoder  (up4 → up3 → up2 → up1 → final_up)
      seg_head  = 1×1 Conv → n_classes

    Đặt learning rate riêng cho từng phần:

        optimizer = AdamW([
            {"params": model.backbone.parameters(), "lr": 1e-4},
            {"params": model.decoder.parameters(),  "lr": 1e-3},
            {"params": model.seg_head.parameters(), "lr": 1e-3},
        ])

    Args:
        backbone        : EfficientViTBackbone hoặc EfficientViTLargeBackbone
        backbone_widths : list[int] – width_list của backbone (5 phần tử)
        decoder_channels: list[int] – số channel tại mỗi bước decode (4 phần tử)
        n_classes       : số lớp segmentation
        dropout         : dropout trước lớp cuối
    """

    def __init__(
        self,
        backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone],
        backbone_widths: list[int],
        decoder_channels: list[int],
        n_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder  = UNetDecoder(backbone_widths, decoder_channels)
        self.seg_head = nn.Sequential(
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(self.decoder.out_channels, n_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────────────────
        feat = self.backbone(x)
        s0 = feat["stage0"]   # stride 2
        s1 = feat["stage1"]   # stride 4
        s2 = feat["stage2"]   # stride 8
        s3 = feat["stage3"]   # stride 16
        s4 = feat["stage4"]   # stride 32  (bottleneck)

        # ── Decoder ──────────────────────────────────────────────────────────
        d = self.decoder(s4, s3, s2, s1, s0)

        # ── Seg Head ─────────────────────────────────────────────────────────
        return self.seg_head(d)


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions  (B-series & L-series)
# ──────────────────────────────────────────────────────────────────────────────

def _make_model(
    backbone,
    backbone_widths: list[int],
    decoder_channels: list[int],
    n_classes: int,
    dropout: float = 0.0,
) -> EfficientViTUNetSeg:
    return EfficientViTUNetSeg(
        backbone=backbone,
        backbone_widths=backbone_widths,
        decoder_channels=decoder_channels,
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_unet_b0(n_classes: int = 19, dropout: float = 0.0, **kwargs) -> EfficientViTUNetSeg:
    """EfficientViT-B0 + UNet decoder  (~light weight)"""
    backbone = efficientvit_backbone_b0(**kwargs)
    return _make_model(
        backbone,
        backbone_widths=[8, 16, 32, 64, 128],
        decoder_channels=[64, 48, 32, 16],
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_unet_b1(n_classes: int = 19, dropout: float = 0.0, **kwargs) -> EfficientViTUNetSeg:
    """EfficientViT-B1 + UNet decoder"""
    backbone = efficientvit_backbone_b1(**kwargs)
    return _make_model(
        backbone,
        backbone_widths=[16, 32, 64, 128, 256],
        decoder_channels=[128, 96, 64, 32],
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_unet_b2(n_classes: int = 19, dropout: float = 0.0, **kwargs) -> EfficientViTUNetSeg:
    """EfficientViT-B2 + UNet decoder"""
    backbone = efficientvit_backbone_b2(**kwargs)
    return _make_model(
        backbone,
        backbone_widths=[24, 48, 96, 192, 384],
        decoder_channels=[192, 128, 96, 48],
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_unet_b3(n_classes: int = 19, dropout: float = 0.0, **kwargs) -> EfficientViTUNetSeg:
    """EfficientViT-B3 + UNet decoder  (~heaviest B-series)"""
    backbone = efficientvit_backbone_b3(**kwargs)
    return _make_model(
        backbone,
        backbone_widths=[32, 64, 128, 256, 512],
        decoder_channels=[256, 192, 128, 64],
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_unet_l1(n_classes: int = 19, dropout: float = 0.0, **kwargs) -> EfficientViTUNetSeg:
    """EfficientViT-L1 + UNet decoder"""
    backbone = efficientvit_backbone_l1(**kwargs)
    return _make_model(
        backbone,
        backbone_widths=[32, 64, 128, 256, 512],
        decoder_channels=[256, 192, 128, 64],
        n_classes=n_classes,
        dropout=dropout,
    )


def efficientvit_unet_l2(n_classes: int = 19, dropout: float = 0.0, **kwargs) -> EfficientViTUNetSeg:
    """EfficientViT-L2 + UNet decoder"""
    backbone = efficientvit_backbone_l2(**kwargs)
    return _make_model(
        backbone,
        backbone_widths=[32, 64, 128, 256, 512],
        decoder_channels=[256, 192, 128, 64],
        n_classes=n_classes,
        dropout=dropout,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = 512, 1024   # Cityscapes resolution

    configs = {
        "b0": efficientvit_unet_b0,
        "b1": efficientvit_unet_b1,
        "b2": efficientvit_unet_b2,
        "b3": efficientvit_unet_b3,
        "l1": efficientvit_unet_l1,
        "l2": efficientvit_unet_l2,
    }

    variant = sys.argv[1] if len(sys.argv) > 1 else "b1"
    if variant not in configs:
        print(f"Unknown variant '{variant}'. Choose from: {list(configs)}")
        sys.exit(1)

    print(f"\n=== EfficientViT-UNet-{variant.upper()} ===")
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
