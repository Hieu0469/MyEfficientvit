"""
efficientvit_seg_standalone.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Toàn bộ EfficientViT-Seg gộp vào 1 file duy nhất.
Nguồn: https://github.com/mit-han-lab/efficientvit
  - efficientvit/models/nn/ops.py
  - efficientvit/models/efficientvit/backbone.py
  - efficientvit/models/efficientvit/seg.py

Tương thích: Python 3.8+  |  PyTorch >= 1.11
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cách dùng:
    from efficientvit_seg_standalone import efficientvit_seg_b0
    model = efficientvit_seg_b0("cityscapes")
    model.load_state_dict(torch.load("weights.pt", map_location="cpu"))
    model.eval().cuda()
    out = model(torch.randn(1, 3, 512, 1024).cuda())  # → (1, 19, 64, 128)
"""

from __future__ import annotations  # Python 3.8 compat cho type hints

import inspect
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 0.  UTILITY  (từ efficientvit/models/utils/)
# ══════════════════════════════════════════════════════════════════════════════

def get_same_padding(kernel_size: int) -> int:
    assert kernel_size % 2 != 0, "kernel_size phải là số lẻ"
    return kernel_size // 2


def val2list(x: Any, repeat_time: int = 1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * repeat_time


def val2tuple(x: Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat]] * max(0, min_len - len(x))
    return tuple(x)


def list_sum(x: list) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[float] = None,
    mode: str = "bicubic",
    align_corners: bool = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor,
                             mode=mode, align_corners=align_corners)
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)


def build_kwargs_from_config(config: dict, target_func) -> dict:
    """Lọc kwargs chỉ giữ key mà target_func chấp nhận."""
    valid = set(inspect.signature(target_func).parameters.keys())
    return {k: v for k, v in config.items() if k in valid}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NORMALISATION  (từ efficientvit/models/nn/norm.py)
# ══════════════════════════════════════════════════════════════════════════════

REGISTERED_NORM_DICT: Dict[str, Any] = {
    "bn2d": nn.BatchNorm2d,
    "ln":   nn.LayerNorm,
    "ln2d": nn.LayerNorm,   # alias – áp dụng trên (B,C,H,W) bằng cách permute
}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm hoạt động trên tensor (B,C,H,W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.permute(0, 2, 3, 1)
        out = super().forward(out)
        return out.permute(0, 3, 1, 2)


def build_norm(name: Optional[str] = "bn2d", num_features: int = None) -> Optional[nn.Module]:
    if name is None:
        return None
    if name == "ln2d":
        return LayerNorm2d(num_features)
    if name == "ln":
        return nn.LayerNorm(num_features)
    if name in REGISTERED_NORM_DICT:
        return REGISTERED_NORM_DICT[name](num_features)
    raise ValueError(f"Norm '{name}' không hỗ trợ")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ACTIVATION  (từ efficientvit/models/nn/act.py)
# ══════════════════════════════════════════════════════════════════════════════

REGISTERED_ACT_DICT: Dict[str, Any] = {
    "relu":   nn.ReLU,
    "relu6":  nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu":   nn.SiLU,
    "gelu":   partial(nn.GELU, approximate="tanh"),
}


def build_act(name: Optional[str] = "relu", inplace: bool = True) -> Optional[nn.Module]:
    if name is None:
        return None
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        # một số act không nhận tham số inplace
        try:
            return act_cls(inplace=inplace)
        except TypeError:
            return act_cls()
    raise ValueError(f"Activation '{name}' không hỗ trợ")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  BASIC LAYERS  (từ ops.py – phần "Basic Layers")
# ══════════════════════════════════════════════════════════════════════════════

class ConvLayer(nn.Module):
    """Conv2d + Norm + Act — building block cơ bản nhất."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
    ):
        super().__init__()
        padding = get_same_padding(kernel_size) * dilation
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, out_channels)
        self.act  = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode: str = "bicubic",
        size: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        factor: int = 2,
        align_corners: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == tuple(self.size)) \
                or self.factor == 1:
            return x
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dropout: float = 0,
        norm: Optional[str] = None,
        act_func: Optional[str] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear  = nn.Linear(in_features, out_features, use_bias)
        self.norm    = build_norm(norm, out_features)
        self.act     = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BASIC BLOCKS  (từ ops.py – phần "Basic Blocks")
# ══════════════════════════════════════════════════════════════════════════════

class DSConv(nn.Module):
    """Depthwise-Separable Conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: Any = False,
        norm: Any = ("bn2d", "bn2d"),
        act_func: Any = ("relu6", None),
    ):
        super().__init__()
        use_bias  = val2tuple(use_bias, 2)
        norm      = val2tuple(norm, 2)
        act_func  = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(in_channels, in_channels, kernel_size, stride,
                                    groups=in_channels, norm=norm[0], act_func=act_func[0],
                                    use_bias=use_bias[0])
        self.point_conv = ConvLayer(in_channels, out_channels, 1,
                                    norm=norm[1], act_func=act_func[1],
                                    use_bias=use_bias[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (expand → depthwise → project)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        use_bias: Any = False,
        norm: Any = ("bn2d", "bn2d", "bn2d"),
        act_func: Any = ("relu6", "relu6", None),
    ):
        super().__init__()
        use_bias  = val2tuple(use_bias, 3)
        norm      = val2tuple(norm, 3)
        act_func  = val2tuple(act_func, 3)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.inverted_conv = ConvLayer(in_channels, mid_channels, 1,
                                       norm=norm[0], act_func=act_func[0], use_bias=use_bias[0])
        self.depth_conv    = ConvLayer(mid_channels, mid_channels, kernel_size, stride,
                                       groups=mid_channels, norm=norm[1], act_func=act_func[1],
                                       use_bias=use_bias[1])
        self.point_conv    = ConvLayer(mid_channels, out_channels, 1,
                                       norm=norm[2], act_func=act_func[2], use_bias=use_bias[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(self.inverted_conv(x)))


class FusedMBConv(nn.Module):
    """Fused MBConv: spatial conv + point conv (bỏ bước depthwise riêng)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        groups: int = 1,
        use_bias: Any = False,
        norm: Any = ("bn2d", "bn2d"),
        act_func: Any = ("relu6", None),
    ):
        super().__init__()
        use_bias  = val2tuple(use_bias, 2)
        norm      = val2tuple(norm, 2)
        act_func  = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.spatial_conv = ConvLayer(in_channels, mid_channels, kernel_size, stride,
                                      groups=groups, use_bias=use_bias[0],
                                      norm=norm[0], act_func=act_func[0])
        self.point_conv   = ConvLayer(mid_channels, out_channels, 1,
                                      use_bias=use_bias[1], norm=norm[1], act_func=act_func[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.spatial_conv(x))


class GLUMBConv(nn.Module):
    """Gated Linear Unit MBConv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        use_bias: Any = False,
        norm: Any = (None, None, "ln2d"),
        act_func: Any = ("silu", "silu", None),
    ):
        super().__init__()
        use_bias  = val2tuple(use_bias, 3)
        norm      = val2tuple(norm, 3)
        act_func  = val2tuple(act_func, 3)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.glu_act      = build_act(act_func[1], inplace=False)
        self.inverted_conv = ConvLayer(in_channels, mid_channels * 2, 1,
                                       use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.depth_conv    = ConvLayer(mid_channels * 2, mid_channels * 2, kernel_size, stride,
                                       groups=mid_channels * 2, use_bias=use_bias[1],
                                       norm=norm[1], act_func=None)
        self.point_conv    = ConvLayer(mid_channels, out_channels, 1,
                                       use_bias=use_bias[2], norm=norm[2], act_func=act_func[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate
        return self.point_conv(x)


class ResBlock(nn.Module):
    """Plain residual block 3×3 → 3×3 (dùng trong dòng Large)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: Any = False,
        norm: Any = ("bn2d", "bn2d"),
        act_func: Any = ("relu6", None),
    ):
        super().__init__()
        use_bias  = val2tuple(use_bias, 2)
        norm      = val2tuple(norm, 2)
        act_func  = val2tuple(act_func, 2)
        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer(in_channels, mid_channels, kernel_size, stride,
                               use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.conv2 = ConvLayer(mid_channels, out_channels, kernel_size, 1,
                               use_bias=use_bias[1], norm=norm[1], act_func=act_func[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LITE-MLA  (Lightweight Multi-scale Linear Attention)
# ══════════════════════════════════════════════════════════════════════════════

class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention (từ ops.py – LiteMLA)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim: int = 8,
        use_bias: Any = False,
        norm: Any = (None, "bn2d"),
        act_func: Any = (None, None),
        kernel_func: str = "relu",
        scales: Tuple[int, ...] = (5,),
        eps: float = 1e-15,
    ):
        super().__init__()
        self.eps  = eps
        heads     = int(in_channels // dim * heads_ratio) if heads is None else heads
        total_dim = heads * dim
        use_bias  = val2tuple(use_bias, 2)
        norm      = val2tuple(norm, 2)
        act_func  = val2tuple(act_func, 2)
        self.dim  = dim

        self.qkv = ConvLayer(in_channels, 3 * total_dim, 1,
                             use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3 * total_dim, 3 * total_dim, scale,
                          padding=get_same_padding(scale),
                          groups=3 * total_dim, bias=use_bias[0]),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1,
                          groups=3 * heads, bias=use_bias[0]),
            )
            for scale in scales
        ])
        self.kernel_func = build_act(kernel_func, inplace=False)
        self.proj = ConvLayer(total_dim * (1 + len(scales)), out_channels, 1,
                              use_bias=use_bias[1], norm=norm[1], act_func=act_func[1])

    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = qkv.size()
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        qkv = qkv.reshape(B, -1, 3 * self.dim, H * W)
        q = qkv[:, :, :self.dim]
        k = qkv[:, :, self.dim: 2 * self.dim]
        v = qkv[:, :, 2 * self.dim:]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk  = torch.matmul(v, k.transpose(-1, -2))
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)
        return out.reshape(B, -1, H, W)

    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = qkv.size()
        qkv = qkv.reshape(B, -1, 3 * self.dim, H * W)
        q = qkv[:, :, :self.dim]
        k = qkv[:, :, self.dim: 2 * self.dim]
        v = qkv[:, :, 2 * self.dim:]
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        att = torch.matmul(k.transpose(-1, -2), q)
        orig_dtype = att.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            att = att.float()
        att = att / (att.sum(dim=2, keepdim=True) + self.eps)
        att = att.to(orig_dtype)
        out = torch.matmul(v, att)
        return out.reshape(B, -1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv] + [op(qkv) for op in self.aggreg]
        qkv = torch.cat(multi_scale_qkv, dim=1)
        H, W = qkv.shape[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv).to(qkv.dtype)
        else:
            out = self.relu_quadratic_att(qkv)
        return self.proj(out)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EfficientViTBlock  (LiteMLA + MBConv/GLUMBConv FFN)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act: Optional[str] = None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.pre_norm  = pre_norm
        self.main      = main
        self.shortcut  = shortcut
        self.post_act  = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
        if self.post_act:
            res = self.post_act(res)
        return res


class EfficientViTBlock(nn.Module):
    """EfficientViT Transformer Block = LiteMLA context + MBConv/GLUMBConv local."""

    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim: int = 32,
        expand_ratio: float = 4,
        scales: Tuple[int, ...] = (5,),
        norm: str = "bn2d",
        act_func: str = "hswish",
        context_module: str = "LiteMLA",
        local_module: str = "MBConv",
    ):
        super().__init__()

        # Context module
        if context_module == "LiteMLA":
            self.context_module = ResidualBlock(
                LiteMLA(in_channels=in_channels, out_channels=in_channels,
                        heads_ratio=heads_ratio, dim=dim, norm=(None, norm), scales=scales),
                IdentityLayer(),
            )
        else:
            raise ValueError(f"context_module '{context_module}' không hỗ trợ")

        # Local module
        if local_module == "MBConv":
            self.local_module = ResidualBlock(
                MBConv(in_channels=in_channels, out_channels=in_channels,
                       expand_ratio=expand_ratio,
                       use_bias=(True, True, False), norm=(None, None, norm),
                       act_func=(act_func, act_func, None)),
                IdentityLayer(),
            )
        elif local_module == "GLUMBConv":
            self.local_module = ResidualBlock(
                GLUMBConv(in_channels=in_channels, out_channels=in_channels,
                          expand_ratio=expand_ratio,
                          use_bias=(True, True, False), norm=(None, None, norm),
                          act_func=(act_func, act_func, None)),
                IdentityLayer(),
            )
        else:
            raise NotImplementedError(f"local_module '{local_module}' không hỗ trợ")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FUNCTIONAL BLOCKS  (DAGBlock, OpSequential)
# ══════════════════════════════════════════════════════════════════════════════

class DAGBlock(nn.Module):
    """Directed-Acyclic-Graph block: nhiều input → merge → middle → nhiều output."""

    def __init__(
        self,
        inputs:     Dict[str, nn.Module],
        merge:      str,
        post_input: Optional[nn.Module],
        middle:     nn.Module,
        outputs:    Dict[str, nn.Module],
    ):
        super().__init__()
        self.input_keys  = list(inputs.keys())
        self.input_ops   = nn.ModuleList(list(inputs.values()))
        self.merge       = merge
        self.post_input  = post_input
        self.middle      = middle
        self.output_keys = list(outputs.keys())
        self.output_ops  = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = [op(feature_dict[k]) for k, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.cat(feat, dim=1)
        else:
            raise NotImplementedError(f"merge='{self.merge}' không hỗ trợ")
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for k, op in zip(self.output_keys, self.output_ops):
            feature_dict[k] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    """Sequential bỏ qua phần tử None trong danh sách."""

    def __init__(self, op_list: List[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 8.  BACKBONE – dòng B  (từ backbone.py – EfficientViTBackbone)
# ══════════════════════════════════════════════════════════════════════════════

class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        in_channels: int = 3,
        dim: int = 32,
        expand_ratio: float = 4,
        norm: str = "bn2d",
        act_func: str = "hswish",
    ):
        super().__init__()
        self.width_list = []

        # ── Input stem ──
        input_stem: List[nn.Module] = [
            ConvLayer(in_channels, width_list[0], stride=2, norm=norm, act_func=act_func)
        ]
        for _ in range(depth_list[0]):
            block = self._build_local_block(width_list[0], width_list[0], 1, 1, norm, act_func)
            input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_ch = width_list[0]
        self.input_stem = OpSequential(input_stem)
        self.width_list.append(in_ch)

        # ── Stages ──
        stages: List[nn.Module] = []

        # Stage 1–2 (local only, stride=2 đầu mỗi stage)
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage: List[nn.Module] = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block  = self._build_local_block(in_ch, w, stride, expand_ratio, norm, act_func)
                stage.append(ResidualBlock(block, IdentityLayer() if stride == 1 else None))
                in_ch = w
            stages.append(OpSequential(stage))
            self.width_list.append(in_ch)

        # Stage 3–4 (local down + EfficientViTBlock)
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self._build_local_block(in_ch, w, 2, expand_ratio, norm, act_func,
                                            fewer_norm=True)
            stage.append(ResidualBlock(block, None))
            in_ch = w
            for _ in range(d):
                stage.append(EfficientViTBlock(in_ch, dim=dim, expand_ratio=expand_ratio,
                                                norm=norm, act_func=act_func))
            stages.append(OpSequential(stage))
            self.width_list.append(in_ch)

        self.stages = nn.ModuleList(stages)

    @staticmethod
    def _build_local_block(
        in_channels: int, out_channels: int, stride: int,
        expand_ratio: float, norm: str, act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            return DSConv(in_channels, out_channels, stride=stride,
                          use_bias=(True, False) if fewer_norm else False,
                          norm=(None, norm) if fewer_norm else norm,
                          act_func=(act_func, None))
        return MBConv(in_channels, out_channels, stride=stride,
                      expand_ratio=expand_ratio,
                      use_bias=(True, True, False) if fewer_norm else False,
                      norm=(None, None, norm) if fewer_norm else norm,
                      act_func=(act_func, act_func, None))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {"input": x}
        out["stage0"] = x = self.input_stem(x)
        for i, stage in enumerate(self.stages, 1):
            out[f"stage{i}"] = x = stage(x)
        out["stage_final"] = x
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 9.  BACKBONE – dòng L  (từ backbone.py – EfficientViTLargeBackbone)
# ══════════════════════════════════════════════════════════════════════════════

class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list:       List[int],
        depth_list:       List[int],
        block_list:       Optional[List[str]] = None,
        expand_list:      Optional[List[float]] = None,
        fewer_norm_list:  Optional[List[bool]] = None,
        in_channels:      int = 3,
        qkv_dim:          int = 32,
        norm:             str = "bn2d",
        act_func:         str = "gelu",
    ):
        super().__init__()
        block_list      = block_list      or ["res", "fmb", "fmb", "mb", "att"]
        expand_list     = expand_list     or [1, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True]
        self.width_list = []
        stages: List[nn.Module] = []

        # ── Stage 0 (stem) ──
        in_ch = width_list[0]
        stage0: List[nn.Module] = [
            ConvLayer(in_channels, in_ch, stride=2, norm=norm, act_func=act_func)
        ]
        for _ in range(depth_list[0]):
            block = self._build_local_block(block_list[0], in_ch, in_ch, 1,
                                             expand_list[0], norm, act_func, fewer_norm_list[0])
            stage0.append(ResidualBlock(block, IdentityLayer()))
        stages.append(OpSequential(stage0))
        self.width_list.append(in_ch)

        # ── Stage 1 ~ N ──
        for sid, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage: List[nn.Module] = []
            # Downsampling block
            down_type = "mb" if block_list[sid] not in ("mb", "fmb") else block_list[sid]
            block = self._build_local_block(down_type, in_ch, w, 2,
                                             expand_list[sid] * 4, norm, act_func,
                                             fewer_norm_list[sid])
            stage.append(ResidualBlock(block, None))
            in_ch = w
            for _ in range(d):
                if block_list[sid].startswith("att"):
                    sc = (3,) if block_list[sid] == "att@3" else (5,)
                    stage.append(EfficientViTBlock(in_ch, dim=qkv_dim,
                                                   expand_ratio=expand_list[sid],
                                                   scales=sc, norm=norm, act_func=act_func))
                else:
                    block = self._build_local_block(block_list[sid], in_ch, in_ch, 1,
                                                     expand_list[sid], norm, act_func,
                                                     fewer_norm_list[sid])
                    stage.append(ResidualBlock(block, IdentityLayer()))
            stages.append(OpSequential(stage))
            self.width_list.append(in_ch)

        self.stages = nn.ModuleList(stages)

    @staticmethod
    def _build_local_block(
        block: str, in_channels: int, out_channels: int,
        stride: int, expand_ratio: float, norm: str, act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        ub2 = (True, False) if fewer_norm else False
        n2  = (None, norm) if fewer_norm else norm
        ub3 = (True, True, False) if fewer_norm else False
        n3  = (None, None, norm) if fewer_norm else norm
        if block == "res":
            return ResBlock(in_channels, out_channels, stride=stride,
                            use_bias=ub2, norm=n2, act_func=(act_func, None))
        elif block == "fmb":
            return FusedMBConv(in_channels, out_channels, stride=stride,
                               expand_ratio=expand_ratio,
                               use_bias=ub2, norm=n2, act_func=(act_func, None))
        elif block == "mb":
            return MBConv(in_channels, out_channels, stride=stride,
                          expand_ratio=expand_ratio,
                          use_bias=ub3, norm=n3, act_func=(act_func, act_func, None))
        raise ValueError(f"block='{block}' không hỗ trợ")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {"input": x}
        for i, stage in enumerate(self.stages):
            out[f"stage{i}"] = x = stage(x)
        out["stage_final"] = x
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 10.  BACKBONE FACTORIES  (từ backbone.py)
# ══════════════════════════════════════════════════════════════════════════════

def efficientvit_backbone_b0(**kw) -> EfficientViTBackbone:
    return EfficientViTBackbone(width_list=[8,16,32,64,128], depth_list=[1,2,2,2,2],
                                dim=16, **build_kwargs_from_config(kw, EfficientViTBackbone))

def efficientvit_backbone_b1(**kw) -> EfficientViTBackbone:
    return EfficientViTBackbone(width_list=[16,32,64,128,256], depth_list=[1,2,3,3,4],
                                dim=16, **build_kwargs_from_config(kw, EfficientViTBackbone))

def efficientvit_backbone_b2(**kw) -> EfficientViTBackbone:
    return EfficientViTBackbone(width_list=[24,48,96,192,384], depth_list=[1,3,4,4,6],
                                dim=32, **build_kwargs_from_config(kw, EfficientViTBackbone))

def efficientvit_backbone_b3(**kw) -> EfficientViTBackbone:
    return EfficientViTBackbone(width_list=[32,64,128,256,512], depth_list=[1,4,6,6,9],
                                dim=32, **build_kwargs_from_config(kw, EfficientViTBackbone))

def efficientvit_backbone_l0(**kw) -> EfficientViTLargeBackbone:
    return EfficientViTLargeBackbone(width_list=[32,64,128,256,512], depth_list=[1,1,1,4,4],
                                     **build_kwargs_from_config(kw, EfficientViTLargeBackbone))

def efficientvit_backbone_l1(**kw) -> EfficientViTLargeBackbone:
    return EfficientViTLargeBackbone(width_list=[32,64,128,256,512], depth_list=[1,1,1,6,6],
                                     **build_kwargs_from_config(kw, EfficientViTLargeBackbone))

def efficientvit_backbone_l2(**kw) -> EfficientViTLargeBackbone:
    return EfficientViTLargeBackbone(width_list=[32,64,128,256,512], depth_list=[1,2,2,8,8],
                                     **build_kwargs_from_config(kw, EfficientViTLargeBackbone))

def efficientvit_backbone_l3(**kw) -> EfficientViTLargeBackbone:
    return EfficientViTLargeBackbone(width_list=[64,128,256,512,1024], depth_list=[1,2,2,8,8],
                                     **build_kwargs_from_config(kw, EfficientViTLargeBackbone))


# ══════════════════════════════════════════════════════════════════════════════
# 11.  SEG HEAD  (từ seg.py – SegHead)
# ══════════════════════════════════════════════════════════════════════════════

class SegHead(DAGBlock):
    def __init__(
        self,
        fid_list:        List[str],
        in_channel_list: List[int],
        stride_list:     List[int],
        head_stride:     int,
        head_width:      int,
        head_depth:      int,
        expand_ratio:    float,
        middle_op:       str,
        final_expand:    Optional[float],
        n_classes:       int,
        dropout:         float = 0,
        norm:            str = "bn2d",
        act_func:        str = "hswish",
    ):
        # ── Input branches: căn chỉnh mỗi scale về head_stride ──
        inputs: Dict[str, nn.Module] = {}
        for fid, in_ch, stride in zip(fid_list, in_channel_list, stride_list):
            factor = stride // head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(in_ch, head_width, 1, norm=norm, act_func=None)
            else:
                inputs[fid] = OpSequential([
                    ConvLayer(in_ch, head_width, 1, norm=norm, act_func=None),
                    UpSampleLayer(factor=factor),
                ])

        # ── Middle: stack residual blocks ──
        middle_blocks: List[nn.Module] = []
        for _ in range(head_depth):
            if middle_op == "mbconv":
                block = MBConv(head_width, head_width, expand_ratio=expand_ratio,
                               norm=norm, act_func=(act_func, act_func, None))
            elif middle_op == "fmbconv":
                block = FusedMBConv(head_width, head_width, expand_ratio=expand_ratio,
                                    norm=norm, act_func=(act_func, None))
            else:
                raise NotImplementedError(f"middle_op='{middle_op}' không hỗ trợ")
            middle_blocks.append(ResidualBlock(block, IdentityLayer()))
        middle = OpSequential(middle_blocks)

        # ── Output: classifier 1×1 ──
        if final_expand is not None:
            exp_ch  = int(head_width * final_expand)
            out_ops: List[Optional[nn.Module]] = [
                ConvLayer(head_width, exp_ch, 1, norm=norm, act_func=act_func),
                ConvLayer(exp_ch, n_classes, 1, use_bias=True, dropout=dropout,
                          norm=None, act_func=None),
            ]
        else:
            out_ops = [
                ConvLayer(head_width, n_classes, 1, use_bias=True, dropout=dropout,
                          norm=None, act_func=None),
            ]
        outputs: Dict[str, nn.Module] = {"segout": OpSequential(out_ops)}

        super().__init__(inputs, "add", None, middle=middle, outputs=outputs)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  EfficientViTSeg  (từ seg.py)
# ══════════════════════════════════════════════════════════════════════════════

class EfficientViTSeg(nn.Module):
    def __init__(
        self,
        backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone],
        head:     SegHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        feed_dict = self.head(feed_dict)
        return feed_dict["segout"]


# ══════════════════════════════════════════════════════════════════════════════
# 13.  SEG MODEL FACTORIES  (từ seg.py – khớp hoàn toàn API gốc)
# ══════════════════════════════════════════════════════════════════════════════

def _seg_head_b(dataset, in_ch_list, head_width, head_depth, **kw) -> SegHead:
    if dataset == "cityscapes":
        return SegHead(["stage4","stage3","stage2"], in_ch_list, [32,16,8],
                       8, head_width, head_depth, 4, "mbconv", 4, 19,
                       **build_kwargs_from_config(kw, SegHead))
    if dataset == "ade20k":
        return SegHead(["stage4","stage3","stage2"], in_ch_list, [32,16,8],
                       8, head_width, head_depth, 4, "mbconv", None, 150,
                       **build_kwargs_from_config(kw, SegHead))
    raise NotImplementedError(dataset)


def efficientvit_seg_b0(dataset: str, **kw) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b0(**kw)
    if dataset == "cityscapes":
        head = SegHead(["stage4","stage3","stage2"], [128,64,32], [32,16,8],
                       8, 32, 1, 4, "mbconv", 4, 19,
                       **build_kwargs_from_config(kw, SegHead))
    else:
        raise NotImplementedError(dataset)
    return EfficientViTSeg(backbone, head)


def efficientvit_seg_b1(dataset: str, **kw) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b1(**kw)
    head = _seg_head_b(dataset, [256,128,64], 64, 3, **kw)
    return EfficientViTSeg(backbone, head)


def efficientvit_seg_b2(dataset: str, **kw) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b2(**kw)
    head = _seg_head_b(dataset, [384,192,96], 96, 3, **kw)
    return EfficientViTSeg(backbone, head)


def efficientvit_seg_b3(dataset: str, **kw) -> EfficientViTSeg:
    backbone = efficientvit_backbone_b3(**kw)
    head = _seg_head_b(dataset, [512,256,128], 128, 3, **kw)
    return EfficientViTSeg(backbone, head)


def efficientvit_seg_l1(dataset: str, **kw) -> EfficientViTSeg:
    backbone = efficientvit_backbone_l1(**kw)
    if dataset == "cityscapes":
        head = SegHead(["stage4","stage3","stage2"], [512,256,128], [32,16,8],
                       8, 256, 3, 1, "fmbconv", None, 19, act_func="gelu",
                       **build_kwargs_from_config(kw, SegHead))
    elif dataset == "ade20k":
        head = SegHead(["stage4","stage3","stage2"], [512,256,128], [32,16,8],
                       8, 128, 3, 4, "fmbconv", 8, 150, act_func="gelu",
                       **build_kwargs_from_config(kw, SegHead))
    else:
        raise NotImplementedError(dataset)
    return EfficientViTSeg(backbone, head)


def efficientvit_seg_l2(dataset: str, **kw) -> EfficientViTSeg:
    backbone = efficientvit_backbone_l2(**kw)
    if dataset == "cityscapes":
        head = SegHead(["stage4","stage3","stage2"], [512,256,128], [32,16,8],
                       8, 256, 5, 1, "fmbconv", None, 19, act_func="gelu",
                       **build_kwargs_from_config(kw, SegHead))
    elif dataset == "ade20k":
        head = SegHead(["stage4","stage3","stage2"], [512,256,128], [32,16,8],
                       8, 128, 3, 4, "fmbconv", 8, 150, act_func="gelu",
                       **build_kwargs_from_config(kw, SegHead))
    else:
        raise NotImplementedError(dataset)
    return EfficientViTSeg(backbone, head)


# ══════════════════════════════════════════════════════════════════════════════
# 14.  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n{'─'*55}")

    cases = [
        ("b0-cityscapes", lambda: efficientvit_seg_b0("cityscapes")),
        ("b1-cityscapes", lambda: efficientvit_seg_b1("cityscapes")),
        ("b2-cityscapes", lambda: efficientvit_seg_b2("cityscapes")),
        ("b3-cityscapes", lambda: efficientvit_seg_b3("cityscapes")),
        ("l1-cityscapes", lambda: efficientvit_seg_l1("cityscapes")),
        ("l2-cityscapes", lambda: efficientvit_seg_l2("cityscapes")),
    ]

    for name, fn in cases:
        model = fn().eval().to(device)
        dummy = torch.randn(1, 3, 512, 1024, device=device)
        with torch.no_grad():
            _ = model(dummy)                    # warmup
            t0  = time.perf_counter()
            out = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1  = time.perf_counter()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  {name:20s}  out={tuple(out.shape)}  "
              f"params={params:.1f}M  lat={(t1-t0)*1000:.1f}ms")

    print(f"{'─'*55}\n✅  Tất cả model chạy thành công!\n")
