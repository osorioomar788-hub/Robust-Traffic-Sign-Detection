"""
Custom PyTorch modules for YOLOv8 architecture enhancement.

Phase 3 — Architecture Engineering | MICAI 2025
Project: Robust Traffic Sign Detection

This module implements two custom nn.Module classes designed to be injected
into the Ultralytics YOLOv8 model via runtime registry injection (see
docs/PHASE3_ARCHITECTURE_PLAN.md §4 for the integration strategy).

Modules:
    SubPixelConv: Learnable upsampling via sub-pixel convolution (ESPCN).
    CoordAtt:     Coordinate Attention for positional-aware channel recalibration.

References:
    [1] Shi et al., "Real-Time Single Image and Video Super-Resolution Using
        an Efficient Sub-Pixel Convolutional Neural Network", CVPR 2016.
    [2] Hou et al., "Coordinate Attention for Efficient Mobile Network Design",
        CVPR 2021.

NOTE: This file is pure PyTorch — no Ultralytics imports. The registry
injection is handled externally in the entry-point script.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SubPixelConv(nn.Module):
    """Learnable upsampling via sub-pixel convolution (ESPCN) [1].

    Replaces ``nn.Upsample`` in the YOLOv8 neck for improved spatial detail
    recovery on small traffic signs (< 32×32 px).

    Pipeline (for upscale factor *r*):
        1. ``Conv2d(c1, c2·r², k=3, pad=1)`` — expand channels to carry
           sub-pixel information.
        2. ``PixelShuffle(r)`` — rearrange ``(B, c2·r², H, W)`` →
           ``(B, c2, H·r, W·r)`` by interleaving channel values into the
           spatial grid (see [1] Eq. 4).
        3. ``BatchNorm2d(c2) + SiLU()`` — normalize and activate.

    Shape:
        - Input:  ``(B, c1, H, W)``
        - Output: ``(B, c2, H·r, W·r)``

    Example (YOLOv8n P4 → P3 upsample)::

        >>> m = SubPixelConv(256, 128, scale=2)
        >>> x = torch.randn(2, 256, 20, 20)
        >>> m(x).shape
        torch.Size([2, 128, 40, 40])

    Args:
        c1: Input channels  (auto-injected by ``parse_model`` from prev layer).
        c2: Output channels (specified in the YAML ``args`` list).
        scale: Upscale factor (default ``2``, matching YOLOv8 upsample stride).

    Reference:
        [1] Shi et al., CVPR 2016 — ESPCN.
    """

    def __init__(self, c1: int, c2: int, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale

        # Step 1: Expand channels to c2 * r² so PixelShuffle can rearrange
        # them into spatial dimensions.  kernel=3, pad=1 preserves H×W.
        # bias=False because BatchNorm absorbs the bias term.
        self.conv = nn.Conv2d(
            c1,
            c2 * scale ** 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        # Step 2: Rearrange channels → spatial grid.
        # (B, c2·r², H, W) → (B, c2, H·r, W·r)
        # Implements the periodic shuffling from [1] Eq. 4:
        #   Y[b, c, h·r + δh, w·r + δw] = X[b, c·r² + δh·r + δw, h, w]
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Step 3: Normalize across the c2 output channels, then activate.
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, c1, H, W)``.

        Returns:
            Upsampled tensor of shape ``(B, c2, H * scale, W * scale)``.
        """
        # (B, c1, H, W) → (B, c2·r², H, W)
        out = self.conv(x)
        # (B, c2·r², H, W) → (B, c2, H·r, W·r)
        out = self.pixel_shuffle(out)
        # Normalize + activate (no shape change)
        out = self.act(self.bn(out))
        return out


class CoordAtt(nn.Module):
    """Coordinate Attention block [2].

    Encodes both channel information and long-range spatial dependencies
    via directional (horizontal + vertical) average pooling, enabling
    the network to attend to sign regions under adverse conditions
    (rain, fog, night-time).

    This is a **shape-preserving** attention module: ``output.shape == input.shape``.
    In the YAML architecture, ``c1 == c2`` is required.

    Pipeline:
        **Stage 1 — Coordinate Information Embedding:**
            - Pool across width  → ``(B, C, H, 1)``  (captures vertical position)
            - Pool across height → ``(B, C, 1, W)``  (captures horizontal position)

        **Stage 2 — Shared Transform:**
            - Permute width-pool to ``(B, C, W, 1)``
            - Concat along dim=2  → ``(B, C, H+W, 1)``
            - 1×1 Conv  → ``(B, C_r, H+W, 1)``  where ``C_r = max(8, C // reduction)``
            - BatchNorm + ReLU

        **Stage 3 — Split + Independent Attention Maps:**
            - Split → ``(B, C_r, H, 1)`` + ``(B, C_r, W, 1)``
            - Re-permute width → ``(B, C_r, 1, W)``
            - Independent 1×1 Conv + Sigmoid → attention maps ``A_h``, ``A_w``

        **Stage 4 — Apply Attention:**
            - ``Out = X * A_h * A_w``  (element-wise with broadcasting)

    Shape:
        - Input:  ``(B, c1, H, W)``
        - Output: ``(B, c1, H, W)``  *(same shape — attention is a refinement)*

    Example::

        >>> m = CoordAtt(128, 128, reduction=32)
        >>> x = torch.randn(2, 128, 40, 40)
        >>> m(x).shape
        torch.Size([2, 128, 40, 40])

    Args:
        c1: Input channels (auto-injected by ``parse_model``).
        c2: Output channels (must equal ``c1`` — accepted for ``parse_model`` compat).
        reduction: Channel reduction ratio for the shared 1×1 conv (default ``32``).

    Reference:
        [2] Hou et al., CVPR 2021 — Coordinate Attention.
    """

    def __init__(self, c1: int, c2: int, reduction: int = 32) -> None:
        super().__init__()
        # c2 is accepted for Ultralytics parse_model compatibility but must
        # equal c1 because attention does not change channel dimensionality.
        assert c1 == c2, (
            f"CoordAtt requires c1 == c2 (shape-preserving), got c1={c1}, c2={c2}"
        )

        # Reduced channel count for the shared bottleneck, clamped to min=8
        # to avoid degenerate single-channel representations.
        mid_channels: int = max(8, c1 // reduction)

        # ── Stage 1: Directional pooling ──────────────────────────────
        # These are parameterless; output sizes are dynamically set in forward().
        # pool_h: (B, C, H, W) → (B, C, H, 1)  — pool across width
        # pool_w: (B, C, H, W) → (B, C, 1, W)  — pool across height
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # preserves H, collapses W
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # collapses H, preserves W

        # ── Stage 2: Shared 1×1 conv + BN + ReLU (channel reduction) ─
        # Operates on the concatenated (H+W) spatial dimension.
        self.conv_shared = nn.Conv2d(c1, mid_channels, kernel_size=1, bias=False)
        self.bn_shared = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)  # ReLU per [2] (not SiLU)

        # ── Stage 3: Independent 1×1 convs (expand back to c1) ───────
        # Each branch produces a directional attention map ∈ [0, 1] via Sigmoid.
        self.conv_h = nn.Conv2d(mid_channels, c1, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, c1, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Attention-refined tensor of shape ``(B, C, H, W)`` (same as input).
        """
        _, _, h, w = x.shape

        # ── Stage 1: Coordinate Information Embedding ─────────────────
        # Pool across width  → captures "where vertically" the signal is.
        x_h: Tensor = self.pool_h(x)                # (B, C, H, 1)
        # Pool across height → captures "where horizontally" the signal is.
        x_w: Tensor = self.pool_w(x)                # (B, C, 1, W)

        # ── Stage 2: Concat + Shared Transform ───────────────────────
        # Permute x_w so both tensors share dim=3 == 1, enabling concat on dim=2.
        x_w_T: Tensor = x_w.permute(0, 1, 3, 2)    # (B, C, W, 1)

        # Concatenate along the spatial dimension (dim=2).
        y: Tensor = torch.cat([x_h, x_w_T], dim=2)  # (B, C, H+W, 1)

        # Shared bottleneck: reduce channels C → C_r.
        y = self.act(self.bn_shared(self.conv_shared(y)))  # (B, C_r, H+W, 1)

        # ── Stage 3: Split + Independent Attention Maps ──────────────
        # Split back into height and width components.
        y_h, y_w = y.split([h, w], dim=2)
        # y_h: (B, C_r, H, 1)
        # y_w: (B, C_r, W, 1)

        # Re-permute width component back to spatial convention.
        y_w = y_w.permute(0, 1, 3, 2)               # (B, C_r, 1, W)

        # Independent 1×1 convolutions → expand back to C channels + Sigmoid.
        a_h: Tensor = torch.sigmoid(self.conv_h(y_h))  # (B, C, H, 1)
        a_w: Tensor = torch.sigmoid(self.conv_w(y_w))  # (B, C, 1, W)

        # ── Stage 4: Apply Attention ─────────────────────────────────
        # Element-wise multiplication with broadcasting:
        #   (B, C, H, W) * (B, C, H, 1) * (B, C, 1, W) → (B, C, H, W)
        return x * a_h * a_w
