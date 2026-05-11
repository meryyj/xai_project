import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def get_activation(name: str = "elu"):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation {name}")


# ------------------------------------------------------
# Depthwise separable Conv block (optionnel)
# ------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """
    Conv 3x3 depthwise + Conv 1x1 pointwise
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ------------------------------------------------------
# SE Block (attention canal) - optionnel
# ------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ------------------------------------------------------
# ConvBlock générique
# ------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation: str = "elu",
        depthwise_separable: bool = False,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()

        if depthwise_separable:
            conv = DepthwiseSeparableConv(in_channels, out_channels)
        else:
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )

        self.conv = conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
        self.use_se = use_se
        self.se = SEBlock(out_channels, se_reduction) if use_se else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_se:
            x = self.se(x)
        return x


# ------------------------------------------------------
# Réseau principal EmotionCNN
# ------------------------------------------------------
class EmotionCNN(nn.Module):
    """
    Architecture type :

    Block1: [ConvBlock(3->32) x2] -> MaxPool -> Dropout(0.2)
    Block2: [ConvBlock(32->64) x2] -> MaxPool -> Dropout(0.25)
    Block3: [ConvBlock(64->128) x2] -> MaxPool -> Dropout(0.3)
    Block4: [ConvBlock(128->256) x2] -> MaxPool -> Dropout(0.3)
    GAP -> FC(256) -> Dropout(0.4) -> FC(num_classes)
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        base_channels: int = 32,
        activation: str = "elu",
        depthwise_separable: bool = False,
        use_se: bool = False,
        dropout_blocks = (0.2, 0.25, 0.3, 0.3),
        dropout_head: float = 0.4,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # Block 1
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, c1, activation, depthwise_separable, use_se),
            ConvBlock(c1, c1, activation, depthwise_separable, use_se),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(dropout_blocks[0])

        # Block 2
        self.block2 = nn.Sequential(
            ConvBlock(c1, c2, activation, depthwise_separable, use_se),
            ConvBlock(c2, c2, activation, depthwise_separable, use_se),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(dropout_blocks[1])

        # Block 3
        self.block3 = nn.Sequential(
            ConvBlock(c2, c3, activation, depthwise_separable, use_se),
            ConvBlock(c3, c3, activation, depthwise_separable, use_se),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(dropout_blocks[2])

        # Block 4
        self.block4 = nn.Sequential(
            ConvBlock(c3, c4, activation, depthwise_separable, use_se),
            ConvBlock(c4, c4, activation, depthwise_separable, use_se),
        )
        self.pool4 = nn.MaxPool2d(2)
        self.drop4 = nn.Dropout(dropout_blocks[3])

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Head
        self.fc1 = nn.Linear(c4, 256)
        self.act_head = get_activation(activation)
        self.drop_head = nn.Dropout(dropout_head)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (B, 3, 128, 128)

        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.block3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.block4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # GAP
        x = self.gap(x)          # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        x = self.fc1(x)
        x = self.act_head(x)
        x = self.drop_head(x)
        x = self.fc_out(x)       # (B, num_classes)

        return x