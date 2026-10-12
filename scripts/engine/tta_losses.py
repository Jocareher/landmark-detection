"""Differentiable, label-free objectives, reduced per image before batch averaging."""
from __future__ import annotations

import math
import torch
from .metrics import heatmap_distribution_statistics
from .pca_shape_prior import compute_pca_projection_loss


def validate_tta_weights(pca_weight: float, entropy_weight: float, spread_weight: float) -> None:
    weights = (pca_weight, entropy_weight, spread_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("TTA loss weights must be finite and nonnegative.")
    if not any(w > 0 for w in weights):
        raise ValueError("At least one TTA loss weight must be positive.")


def compute_tta_losses(heatmaps, landmarks, pca_prior, *, pca_weight=1.0,
                       entropy_weight=0.0, spread_weight=0.0, pca_loss_space="image"):
    """Return [B] raw signals, weighted contributions and their sum.

    p = spatial softmax(logits), with temperature fixed at 1.
    H = -sum(p log(p))/log(HW); S = 2 trace(Cov_p(r)), r in [0,1]^2.
    Each heatmap signal averages ALL landmarks; no visibility or GT is used.
    A zero PCA weight bypasses the prior and projection entirely.
    """
    validate_tta_weights(pca_weight, entropy_weight, spread_weight)
    statistics = heatmap_distribution_statistics(heatmaps)
    raw = {"heatmap_entropy_loss": statistics["entropy"].mean(-1),
           "heatmap_spread_loss": statistics["spread"].mean(-1)}
    zero = heatmaps.new_zeros(heatmaps.shape[0])
    if pca_weight > 0:
        if pca_prior is None:
            raise ValueError("A PCA prior is required when tta_pca_weight > 0.")
        raw["pca_reconstruction_loss"] = compute_pca_projection_loss(
            landmarks, pca_prior, loss_space=pca_loss_space, reduction="none")
    else:
        raw["pca_reconstruction_loss"] = torch.full_like(zero, math.nan)
    result = dict(raw)
    for signal, weight, key in (("pca", pca_weight, "pca_reconstruction_loss"),
                               ("entropy", entropy_weight, "heatmap_entropy_loss"),
                               ("spread", spread_weight, "heatmap_spread_loss")):
        result[f"weighted_{signal}_loss"] = raw[key] * weight if weight > 0 else zero
    result["sample_tta_loss"] = sum(result[f"weighted_{signal}_loss"]
                                     for signal in ("pca", "entropy", "spread"))
    return result
