# src/models/attention.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.
    Applies channel-wise attention: global pooling -> bottleneck MLP -> sigmoid.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < reduction:
            reduction = max(1, channels // 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    """
    Channel attention used in CBAM: uses avg + max pooling over spatial dims.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < reduction:
            reduction = max(1, channels // 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention used in CBAM: uses concatenated avg & max over channels.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Combines channel and spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        # Spatial attention
        sa = self.spatial_att(x)
        x = x * sa
        return x


def build_attention(
    channels: int,
    attention_type: str = "none",
    reduction: int = 16,
    spatial_kernel_size: int = 7,
) -> Optional[nn.Module]:
    """
    Small factory: returns an attention module or None.
    """
    at = (attention_type or "none").lower()
    if at == "none":
        return None
    if at == "se":
        return SEBlock(channels=channels, reduction=reduction)
    if at == "cbam":
        return CBAM(channels=channels, reduction=reduction, spatial_kernel_size=spatial_kernel_size)
    raise ValueError(f"Unknown attention_type: {attention_type}")
