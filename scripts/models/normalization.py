"""Normalization for convolutional feature maps in NCHW layout."""

import torch
from torch import nn


class ChannelLayerNorm(nn.LayerNorm):
    """Normalize channels independently at each spatial position.

    Affine parameters have shape (C,), independent of image resolution.
    """

    def __init__(self, channels: int) -> None:
        super().__init__(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features.movedim(1, -1)).movedim(-1, 1)


def build_feature_normalization(name: str, channels: int) -> nn.Module:
    """Build affine normalization without changing the feature-map shape."""
    if name == "batch":
        return nn.BatchNorm2d(channels, momentum=0.01)
    if name == "layer":
        return ChannelLayerNorm(channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    raise ValueError(f"Unsupported feature normalization: {name}")
