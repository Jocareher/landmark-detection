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


class SourceAdaptiveInstanceNorm2d(nn.Module):
    """Match each instance to a fixed source-domain channel-statistics prototype.

    The prototype is learned from training images only, stored as checkpoint
    buffers, and never changed by validation, inference, or TTA.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.01) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("source_mean", torch.zeros(channels))
        self.register_buffer("source_std", torch.ones(channels))
        self.register_buffer("source_count", torch.zeros((), dtype=torch.long))
        self.collect_source = False
        self._calibration = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("AdaIN expects NCHW feature maps.")
        mean = features.float().mean(dim=(2, 3), keepdim=True)
        std = features.float().var(dim=(2, 3), keepdim=True, unbiased=False).add(self.eps).sqrt()
        if self._calibration is not None:
            with torch.no_grad():
                self._calibration[0] += mean.detach().sum(dim=0).flatten()
                self._calibration[1] += std.detach().sum(dim=0).flatten()
                self._calibration[2] += features.shape[0]
        elif self.collect_source and self.training:
            with torch.no_grad():
                batch_mean = mean.detach().mean(dim=0).flatten()
                batch_std = std.detach().mean(dim=0).flatten()
                if self.source_count.item() == 0:
                    self.source_mean.copy_(batch_mean)
                    self.source_std.copy_(batch_std)
                else:
                    self.source_mean.lerp_(batch_mean, self.momentum)
                    self.source_std.lerp_(batch_std, self.momentum)
                self.source_count.add_(features.shape[0])
        normalized = (features.float() - mean) / std
        output = normalized * self.source_std.view(1, -1, 1, 1)
        output = output + self.source_mean.view(1, -1, 1, 1)
        output = output * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return output.to(features.dtype)

    def begin_calibration(self) -> None:
        self._calibration = [torch.zeros_like(self.source_mean),
                             torch.zeros_like(self.source_std), 0]

    def finish_calibration(self) -> int:
        if self._calibration is None or self._calibration[2] == 0:
            raise RuntimeError("AdaIN calibration received no source images.")
        mean_sum, std_sum, count = self._calibration
        self.source_mean.copy_(mean_sum / count)
        self.source_std.copy_(std_sum / count)
        self.source_count.fill_(count)
        self._calibration = None
        return count


def build_feature_normalization(name: str, channels: int) -> nn.Module:
    """Build affine normalization without changing the feature-map shape."""
    if name == "batch":
        return nn.BatchNorm2d(channels, momentum=0.01)
    if name == "layer":
        return ChannelLayerNorm(channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    if name == "adain":
        return SourceAdaptiveInstanceNorm2d(channels)
    raise ValueError(f"Unsupported feature normalization: {name}")
