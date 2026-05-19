from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import normalize_heatmaps_to_probabilities, spatial_softmax_2d


class PerLandmarkHeatmapLoss(Protocol):
    """Callable protocol for heatmap losses that return one loss per landmark."""

    def __call__(
        self,
        predicted_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        """Return a tensor with shape ``(B, K)``."""


class MeanSquaredHeatmapLoss(nn.Module):
    """Per-landmark mean squared heatmap loss preserving the legacy MSE regime."""

    def forward(
        self,
        predicted_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean squared error averaged over each heatmap channel."""
        loss = F.mse_loss(predicted_heatmaps, target_heatmaps, reduction="none")
        return loss.flatten(start_dim=2).mean(dim=-1)


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss for heatmap regression.

    This implements the formulation from Wang et al., "Adaptive Wing Loss for
    Robust Face Alignment via Heatmap Regression" using the common piecewise
    definition:

    ``omega * log(1 + |e / epsilon| ** (alpha - y))`` for small errors, and a
    target-dependent linear continuation for larger errors.

    The module returns one averaged loss value per batch item and landmark
    channel, so visibility masks can be applied outside the loss.
    """

    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ) -> None:
        """Store Adaptive Wing hyperparameters."""
        super().__init__()
        self.omega = float(omega)
        self.theta = float(theta)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)

    def forward(
        self,
        predicted_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-landmark Adaptive Wing heatmap loss."""
        target = target_heatmaps.float()
        prediction = predicted_heatmaps.float()
        error = (target - prediction).abs()
        exponent = self.alpha - target
        theta_over_epsilon = self.theta / self.epsilon

        small_error_loss = self.omega * torch.log1p(
            torch.pow(error / self.epsilon, exponent)
        )
        a = (
            self.omega
            * exponent
            * torch.pow(theta_over_epsilon, exponent - 1.0)
            / (self.epsilon * (1.0 + torch.pow(theta_over_epsilon, exponent)))
        )
        c = self.theta * a - self.omega * torch.log1p(
            torch.pow(theta_over_epsilon, exponent)
        )
        large_error_loss = a * error - c
        loss = torch.where(error < self.theta, small_error_loss, large_error_loss)
        return loss.flatten(start_dim=2).mean(dim=-1)


class WassersteinHeatmapLoss(nn.Module):
    """Efficient separable 2D Wasserstein-style heatmap loss.

    The original 2D Wasserstein objective treats heatmaps as spatial
    distributions and penalizes transport-aware spatial discrepancy. To keep the
    current 72-landmark, 64x64 setup practical, this implementation compares the
    cumulative distributions of the x and y marginals rather than constructing a
    dense 4096x4096 transport matrix per landmark. This is a separable
    Wasserstein-style approximation: it is spatially aware, differentiable, and
    pairs naturally with barycenter decoding.
    """

    def __init__(
        self,
        softmax_temperature: float = 1.0,
        epsilon: float = 1e-8,
        validate_normalization: bool = False,
    ) -> None:
        """Store normalization and softmax hyperparameters."""
        super().__init__()
        self.softmax_temperature = float(softmax_temperature)
        self.epsilon = float(epsilon)
        self.validate_normalization = bool(validate_normalization)

    def forward(
        self,
        predicted_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-landmark separable Wasserstein-style heatmap loss."""
        predicted_probabilities = spatial_softmax_2d(
            predicted_heatmaps,
            temperature=self.softmax_temperature,
        )
        target_probabilities = normalize_heatmaps_to_probabilities(
            target_heatmaps,
            eps=self.epsilon,
        )
        if self.validate_normalization:
            self._assert_normalized(predicted_probabilities, "predicted")
            self._assert_normalized(target_probabilities, "target")

        pred_x = predicted_probabilities.sum(dim=2)
        target_x = target_probabilities.sum(dim=2)
        pred_y = predicted_probabilities.sum(dim=3)
        target_y = target_probabilities.sum(dim=3)
        cdf_x_loss = (
            (pred_x.cumsum(dim=-1) - target_x.cumsum(dim=-1)).square().mean(dim=-1)
        )
        cdf_y_loss = (
            (pred_y.cumsum(dim=-1) - target_y.cumsum(dim=-1)).square().mean(dim=-1)
        )
        return cdf_x_loss + cdf_y_loss

    def _assert_normalized(self, probabilities: torch.Tensor, name: str) -> None:
        """Raise if a probability map does not sum to one per heatmap channel."""
        sums = probabilities.flatten(start_dim=2).sum(dim=-1)
        if not torch.allclose(
            sums,
            torch.ones_like(sums),
            rtol=1e-3,
            atol=1e-4,
        ):
            raise ValueError(f"{name} heatmap probabilities are not normalized.")


def compute_masked_heatmap_loss(
    loss_fn: PerLandmarkHeatmapLoss,
    predicted_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average a per-landmark heatmap loss with an optional visibility mask."""
    per_landmark_loss = loss_fn(predicted_heatmaps, target_heatmaps)
    if target_visibility is None:
        return per_landmark_loss.mean()
    mask = target_visibility.to(
        device=per_landmark_loss.device,
        dtype=per_landmark_loss.dtype,
    )
    return (per_landmark_loss * mask).sum() / mask.sum().clamp_min(1.0)


def build_landmark_heatmap_loss(config) -> nn.Module:
    """Build the configured heatmap loss module."""
    landmark_loss = str(config.landmark_loss)
    if landmark_loss == "mse":
        return MeanSquaredHeatmapLoss()
    if landmark_loss == "adaptive_wing":
        return AdaptiveWingLoss(
            omega=config.adaptive_wing_omega,
            theta=config.adaptive_wing_theta,
            epsilon=config.adaptive_wing_epsilon,
            alpha=config.adaptive_wing_alpha,
        )
    if landmark_loss == "wasserstein":
        return WassersteinHeatmapLoss(
            softmax_temperature=config.wasserstein_softmax_temperature,
            epsilon=config.wasserstein_epsilon,
            validate_normalization=config.wasserstein_validate_normalization,
        )
    raise ValueError(f"Unsupported landmark loss regime: {landmark_loss}")
