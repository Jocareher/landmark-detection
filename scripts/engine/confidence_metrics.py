from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metrics import spatial_softmax_2d


@dataclass
class HeatmapConfidenceMetrics:
    """Container for per-landmark heatmap confidence signals."""

    heatmap_max: torch.Tensor
    heatmap_entropy: torch.Tensor
    heatmap_variance: torch.Tensor
    peak_sharpness: torch.Tensor


def compute_heatmap_confidence_metrics(
    heatmaps: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> HeatmapConfidenceMetrics:
    """Compute confidence metrics for each landmark heatmap in a batch.

    Parameters
    ----------
    heatmaps : torch.Tensor
        Predicted heatmaps with shape ``(B, K, H, W)``.
    temperature : float, optional
        Spatial softmax temperature used when converting heatmaps to
        probabilities for entropy and spatial variance.
    eps : float, optional
        Numerical stability value for logarithms and divisions.

    Returns
    -------
    HeatmapConfidenceMetrics
        Per-landmark metrics, each with shape ``(B, K)``.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps with shape (B, K, H, W), got {tuple(heatmaps.shape)}."
        )

    batch_size, num_landmarks, heatmap_height, heatmap_width = heatmaps.shape
    flattened = heatmaps.float().reshape(batch_size, num_landmarks, -1)
    heatmap_max = flattened.max(dim=-1).values
    probabilities = spatial_softmax_2d(heatmaps, temperature=temperature)
    flat_probabilities = probabilities.reshape(batch_size, num_landmarks, -1)
    entropy = -(flat_probabilities * flat_probabilities.clamp_min(eps).log()).sum(
        dim=-1
    )

    x_coords = torch.arange(
        heatmap_width, device=heatmaps.device, dtype=probabilities.dtype
    )
    y_coords = torch.arange(
        heatmap_height, device=heatmaps.device, dtype=probabilities.dtype
    )
    expected_x = (probabilities.sum(dim=2) * x_coords).sum(dim=-1)
    expected_y = (probabilities.sum(dim=3) * y_coords).sum(dim=-1)
    variance_x = (
        probabilities.sum(dim=2) * (x_coords[None, None, :] - expected_x[..., None]).square()
    ).sum(dim=-1)
    variance_y = (
        probabilities.sum(dim=3) * (y_coords[None, None, :] - expected_y[..., None]).square()
    ).sum(dim=-1)
    heatmap_variance = variance_x + variance_y

    peak_sharpness = compute_peak_sharpness(heatmaps)
    return HeatmapConfidenceMetrics(
        heatmap_max=heatmap_max,
        heatmap_entropy=entropy,
        heatmap_variance=heatmap_variance,
        peak_sharpness=peak_sharpness,
    )


def compute_peak_sharpness(heatmaps: torch.Tensor) -> torch.Tensor:
    """Estimate peak sharpness as the gap between the top two local maxima.

    Local maxima are detected with a 3x3 max-pooling pass. If a heatmap has fewer
    than two detected local maxima, the function falls back to the gap between
    the two largest flattened heatmap responses.
    """
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps with shape (B, K, H, W), got {tuple(heatmaps.shape)}."
        )

    pooled = F.max_pool2d(heatmaps.float(), kernel_size=3, stride=1, padding=1)
    local_peak_mask = heatmaps.float() == pooled
    flattened = heatmaps.float().reshape(*heatmaps.shape[:2], -1)
    local_values = flattened.masked_fill(
        ~local_peak_mask.reshape(*heatmaps.shape[:2], -1),
        float("-inf"),
    )
    local_top2 = local_values.topk(k=min(2, local_values.shape[-1]), dim=-1).values
    flat_top2 = flattened.topk(k=min(2, flattened.shape[-1]), dim=-1).values

    if local_top2.shape[-1] < 2:
        return torch.zeros_like(flattened[..., 0])

    local_gap = local_top2[..., 0] - local_top2[..., 1]
    flat_gap = flat_top2[..., 0] - flat_top2[..., 1]
    valid_local_gap = torch.isfinite(local_top2[..., 1])
    return torch.where(valid_local_gap, local_gap, flat_gap)
