from __future__ import annotations

import torch
from torch import nn


def _build_activation(name: str) -> nn.Module:
    """Build one supported activation layer."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported normalizer activation: {name}")


def _build_normalization(name: str, channels: int) -> nn.Module | None:
    """Build an optional internal normalization layer."""
    if name == "none":
        return None
    if name == "group":
        groups = min(4, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported normalizer normalization: {name}")


class ResidualImageNormalizer(nn.Module):
    """Apply a shallow, geometry-preserving residual correction to RGB tensors.

    The input is expected to be in the same channel-normalized tensor space used
    by the existing landmarker. Consequently, output clamping is disabled by
    default: clamping such tensors to ``[0, 1]`` would alter the established
    preprocessing contract.
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        num_layers: int = 3,
        kernel_size: int = 3,
        activation: str = "relu",
        normalization: str = "none",
        residual_scale: float = 0.05,
        initialize_identity: bool = True,
        clamp_output: bool = False,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        """Initialize the residual image normalizer."""
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError("Normalizer channel counts must be positive.")
        if num_layers <= 0:
            raise ValueError("Normalizer num_layers must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Normalizer kernel_size must be a positive odd integer.")
        if residual_scale < 0:
            raise ValueError("Normalizer residual_scale must be non-negative.")
        if clamp_output and clamp_min >= clamp_max:
            raise ValueError("clamp_min must be smaller than clamp_max.")

        padding = kernel_size // 2
        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(
                nn.Conv2d(
                    input_channels,
                    input_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
        else:
            in_channels = input_channels
            for _ in range(num_layers - 1):
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                )
                normalization_layer = _build_normalization(
                    normalization, hidden_channels
                )
                if normalization_layer is not None:
                    layers.append(normalization_layer)
                layers.append(_build_activation(activation))
                in_channels = hidden_channels
            layers.append(
                nn.Conv2d(
                    hidden_channels,
                    input_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

        self.delta_network = nn.Sequential(*layers)
        self.residual_scale = float(residual_scale)
        self.clamp_output = bool(clamp_output)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        self.activation_name = activation
        self.normalization_name = normalization
        self.initialize_identity = bool(initialize_identity)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolutions and optionally make the module exactly identity."""
        convolution_layers = [
            module
            for module in self.delta_network.modules()
            if isinstance(module, nn.Conv2d)
        ]
        for module in convolution_layers:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.initialize_identity and convolution_layers:
            nn.init.zeros_(convolution_layers[-1].weight)
            if convolution_layers[-1].bias is not None:
                nn.init.zeros_(convolution_layers[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return the input plus a bounded residual correction."""
        if image.ndim != 4 or image.shape[1] != self.input_channels:
            raise ValueError(
                "Expected image tensor with shape "
                f"(B, {self.input_channels}, H, W), got {tuple(image.shape)}."
            )
        residual = self.residual_scale * torch.tanh(self.delta_network(image))
        normalized = image + residual
        if self.clamp_output:
            normalized = normalized.clamp(self.clamp_min, self.clamp_max)
        return normalized

    def architecture_config(self) -> dict[str, object]:
        """Return the serializable architecture configuration."""
        return {
            "type": "residual",
            "input_channels": self.input_channels,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "activation": self.activation_name,
            "normalization": self.normalization_name,
            "residual_scale": self.residual_scale,
            "initialize_identity": self.initialize_identity,
            "clamp_output": self.clamp_output,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
        }
