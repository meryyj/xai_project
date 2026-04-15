# src/models/main_cnn.py

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .attention import build_attention


def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution: depthwise (per-channel) conv + 1x1 pointwise.

    Great for reducing parameters while keeping representational power.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    """
    A single feature block:
      (Conv/DepthwiseSeparable -> BN -> Act) x 2 -> [Attention] -> [Residual] -> MaxPool -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_depthwise: bool = False,
        activation: str = "elu",
        dropout: float = 0.0,
        use_residual: bool = True,
        attention_type: str = "none",
    ):
        super().__init__()

        self.use_residual = use_residual
        self.attention_type = attention_type

        ConvLayer = DepthwiseSeparableConv2d if use_depthwise else nn.Conv2d

        self.conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = get_activation(activation)

        self.conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = get_activation(activation)

        # Optional residual projection if in/out channels differ
        if use_residual:
            if in_channels != out_channels:
                self.res_proj = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                )
            else:
                self.res_proj = nn.Identity()
        else:
            self.res_proj = None

        # Optional attention (SE / CBAM / none)
        self.attention = build_attention(
            channels=out_channels,
            attention_type=attention_type,
            reduction=16,
            spatial_kernel_size=7,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = (
            nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.attention is not None:
            x = self.attention(x)

        if self.use_residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            x = x + identity

        x = self.pool(x)
        x = self.dropout(x)
        return x


class MainCNN(nn.Module):
    """
    Main configurable CNN architecture for 7-class FER on RAF-DB.

    Default: 4 blocks with 32 → 64 → 128 → 256 channels, GAP, then small FC head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        base_channels: int = 32,
        num_blocks: int = 4,
        use_depthwise: bool = True,
        use_residual: bool = True,
        attention_type: str = "se",
        activation: str = "elu",
        dropout_block: float = 0.25,
        hidden_dim: int = 256,
        dropout_head: float = 0.4,
        use_se: Optional[bool] = None,  # NEW: for compatibility with train_model(...)
    ):
        super().__init__()

        assert num_blocks >= 1, "num_blocks must be at least 1"

        self.in_channels = in_channels
        self.num_classes = num_classes

        # If use_se is explicitly given, override attention_type
        if use_se is not None:
            attention_type = "se" if use_se else "none"

        blocks = []
        channels_in = in_channels

        for i in range(num_blocks):
            channels_out = base_channels * (2 ** i)
            block = ConvBlock(
                in_channels=channels_in,
                out_channels=channels_out,
                use_depthwise=use_depthwise,
                activation=activation,
                dropout=dropout_block,
                use_residual=use_residual,
                # e.g. attention only from block 2+:
                attention_type=attention_type if i >= 1 else "none",
            )
            blocks.append(block)
            channels_in = channels_out

        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels_in, hidden_dim),
            get_activation(activation),
            nn.Dropout(p=dropout_head),
            nn.Linear(hidden_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Kaiming initialization for convs, and reasonable defaults for the rest.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DepthwiseSeparableConv2d)):
                if isinstance(m, DepthwiseSeparableConv2d):
                    nn.init.kaiming_normal_(
                        m.depthwise.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.depthwise.bias is not None:
                        nn.init.zeros_(m.depthwise.bias)
                    nn.init.kaiming_normal_(
                        m.pointwise.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.pointwise.bias is not None:
                        nn.init.zeros_(m.pointwise.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """
    Factory that reads the config dict and returns a MainCNN instance.

    Expected in cfg:
      cfg["dataset"]["channels"]
      cfg["model"]["num_classes"], ["base_channels"], ["num_blocks"], etc.
    """
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    in_channels = int(dataset_cfg.get("channels", model_cfg.get("in_channels", 1)))
    num_classes = int(model_cfg.get("num_classes", 7))
    base_channels = int(model_cfg.get("base_channels", 32))
    num_blocks = int(model_cfg.get("num_blocks", 4))
    use_depthwise = bool(model_cfg.get("use_depthwise", True))
    use_residual = bool(model_cfg.get("use_residual", True))
    attention_type = model_cfg.get("attention_type", "se")
    activation = model_cfg.get("activation", "elu")
    dropout_block = float(model_cfg.get("dropout_block", 0.25))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    dropout_head = float(model_cfg.get("dropout_head", 0.4))
    use_se = model_cfg.get("use_se", None)  # optional in config

    model = MainCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        num_blocks=num_blocks,
        use_depthwise=use_depthwise,
        use_residual=use_residual,
        attention_type=attention_type,
        activation=activation,
        dropout_block=dropout_block,
        hidden_dim=hidden_dim,
        dropout_head=dropout_head,
        use_se=use_se,
    )
    return model
