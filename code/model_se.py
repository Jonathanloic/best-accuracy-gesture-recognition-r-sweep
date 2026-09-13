from __future__ import annotations

import torch
from torch import nn


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation (SE) adapted to 1D feature maps [B, C, T].

    Squeeze: global average pooling over the temporal dimension.
    Excitation: C -> C/r -> C with ReLU then sigmoid gating.
    Scale: multiply the original feature map by the learned channel weights.

    The reduction ratio r=16 follows Hu et al., Squeeze-and-Excitation Networks.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")

        hidden_channels = max(1, channels // reduction_ratio)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            # The SE paper formulates excitation using W1 and W2 weight matrices.
            # bias=False keeps the implementation aligned with that formulation.
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.shape

        # [B, C, T] -> [B, C, 1] -> [B, C]
        descriptor = self.squeeze(x).view(batch_size, channels)

        # [B, C] -> [B, C] -> [B, C, 1]
        weights = self.excitation(descriptor).view(batch_size, channels, 1)

        # Channel-wise recalibration; shape is unchanged.
        return x * weights


class SEDepthwiseSeparableConv(nn.Module):
    """Best-Accuracy block with SE after ReLU and before Dropout.

    Depthwise -> Pointwise -> BatchNorm -> MaxPool -> ReLU
    -> Squeeze-and-Excitation -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        p_drop: float = 0.2,
        reduction_ratio: int = 16,
    ) -> None:
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.act = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation1D(
            channels=out_channels,
            reduction_ratio=reduction_ratio,
        )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.se(x)
        x = self.drop(x)
        return x


class SEDepthwiseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, ...],
        kernel_size: int,
        p_drop: float,
        reduction_ratio: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for width in widths:
            layers.append(
                SEDepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    reduction_ratio=reduction_ratio,
                )
            )
            prev_channels = width

        self.features = nn.Sequential(*layers)

        # Classifier stays identical to the Best Accuracy baseline.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev_channels, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def build_best_accuracy_se_model(
    in_channels: int = 4,
    num_classes: int = 6,
    reduction_ratio: int = 16,
) -> SEDepthwiseCNN:
    """Best Accuracy baseline + one SE block after every convolutional block."""
    return SEDepthwiseCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        widths=(32, 32, 32, 32, 32, 32),
        kernel_size=15,
        p_drop=0.2,
        reduction_ratio=reduction_ratio,
    )
