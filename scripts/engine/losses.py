from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_visibility_weighted_heatmap_loss(
    predicted_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_visibility: torch.Tensor,
    invisible_landmark_weight: float = 1.0,
) -> torch.Tensor:
    """Compute a GT-visibility-weighted heatmap loss."""
    if invisible_landmark_weight < 0.0:
        raise ValueError(
            "invisible_landmark_weight must be non-negative, "
            f"got {invisible_landmark_weight}."
        )

    per_pixel_loss = F.mse_loss(
        predicted_heatmaps,
        target_heatmaps,
        reduction="none",
    )
    per_landmark_loss = per_pixel_loss.mean(dim=(-1, -2))
    landmark_weights = torch.where(
        target_visibility > 0,
        torch.ones_like(per_landmark_loss),
        torch.full_like(per_landmark_loss, float(invisible_landmark_weight)),
    )
    weighted_loss = per_landmark_loss * landmark_weights
    normalization = landmark_weights.sum().clamp_min(1.0)
    return weighted_loss.sum() / normalization


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    lambda_heatmap: float = 1.0,
    lambda_visibility: float = 1.0,
    invisible_landmark_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the weighted loss for the heatmap and visibility prediction heads."""
    predicted_heatmaps = outputs["heatmaps"]
    predicted_visibility_logits = outputs["visibility_logits"]
    target_heatmaps = batch["heatmaps"]
    target_visibility = batch["visibility"]

    if invisible_landmark_weight == 1.0:
        heatmap_loss = heatmap_loss_fn(predicted_heatmaps, target_heatmaps)
    else:
        heatmap_loss = compute_visibility_weighted_heatmap_loss(
            predicted_heatmaps=predicted_heatmaps,
            target_heatmaps=target_heatmaps,
            target_visibility=target_visibility,
            invisible_landmark_weight=invisible_landmark_weight,
        )
    visibility_loss = visibility_loss_fn(predicted_visibility_logits, target_visibility)
    total_loss = lambda_heatmap * heatmap_loss + lambda_visibility * visibility_loss
    return {
        "total_loss": total_loss,
        "heatmap_loss": heatmap_loss,
        "visibility_loss": visibility_loss,
    }
