from __future__ import annotations

import torch
from torch import nn


class ChannelAttention1D(nn.Module):
    """DanHAR-style channel attention adapted to 1D feature maps [B, C, T]."""

    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")

        hidden_channels = max(1, channels // reduction_ratio)

        # Shared two-layer projection for both average-pooled and max-pooled
        # channel descriptors: C -> C/r -> C.
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = torch.mean(x, dim=2, keepdim=True)
        max_descriptor = torch.amax(x, dim=2, keepdim=True)

        attention = self.shared_mlp(avg_descriptor) + self.shared_mlp(max_descriptor)
        attention = self.sigmoid(attention)
        return x * attention


class TemporalAttention1D(nn.Module):
    """DanHAR-style temporal attention adapted to 1D feature maps [B, C, T]."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("temporal attention kernel_size must be a positive odd number")

        # Average and max channel descriptors are concatenated -> [B, 2, T].
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_descriptor = torch.mean(x, dim=1, keepdim=True)
        max_descriptor = torch.amax(x, dim=1, keepdim=True)

        descriptor = torch.cat([avg_descriptor, max_descriptor], dim=1)
        attention = self.sigmoid(self.conv(descriptor))
        return x * attention


class ChannelTemporalAttention1D(nn.Module):
    """Sequential channel -> temporal attention, as used in DanHAR."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        temporal_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention1D(
            channels=channels,
            reduction_ratio=reduction_ratio,
        )
        self.temporal = TemporalAttention1D(kernel_size=temporal_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.temporal(x)
        return x


class AttentionDepthwiseSeparableConv(nn.Module):
    """
    Best-Accuracy depthwise-separable block with dual attention inserted
    after ReLU and before Dropout.

    Depthwise -> Pointwise -> BN -> MaxPool -> ReLU
    -> Channel Attention -> Temporal Attention -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        stride: int = 1,
        p_drop: float = 0.2,
        reduction_ratio: int = 16,
        temporal_kernel_size: int = 7,
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
        self.attention = ChannelTemporalAttention1D(
            channels=out_channels,
            reduction_ratio=reduction_ratio,
            temporal_kernel_size=temporal_kernel_size,
        )
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.act(x)
        x = self.attention(x)
        x = self.drop(x)
        return x


class AttentionDepthwiseCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, ...],
        kernel_size: int,
        p_drop: float,
        reduction_ratio: int,
        temporal_kernel_size: int,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_channels = in_channels

        for width in widths:
            layers.append(
                AttentionDepthwiseSeparableConv(
                    in_channels=prev_channels,
                    out_channels=width,
                    kernel_size=kernel_size,
                    p_drop=p_drop,
                    reduction_ratio=reduction_ratio,
                    temporal_kernel_size=temporal_kernel_size,
                )
            )
            prev_channels = width

        self.features = nn.Sequential(*layers)

        # Classifier is intentionally unchanged from the Best Accuracy baseline.
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


def build_best_accuracy_channel_temporal_model(
    in_channels: int = 4,
    num_classes: int = 6,
    reduction_ratio: int = 16,
) -> AttentionDepthwiseCNN:
    """Best Accuracy baseline + channel-temporal attention after every block."""
    return AttentionDepthwiseCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        widths=(32, 32, 32, 32, 32, 32),
        kernel_size=15,
        p_drop=0.2,
        reduction_ratio=reduction_ratio,
        temporal_kernel_size=7,
    )
