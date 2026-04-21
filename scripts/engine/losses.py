from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .pca_shape_prior import (
    compute_pca_projection_loss,
    softargmax_heatmaps_to_image_coords,
)


def compute_visible_landmark_heatmap_loss(
    predicted_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    target_visibility: torch.Tensor,
) -> torch.Tensor:
    """Compute heatmap MSE only for landmark channels marked visible in GT."""
    channel_mask = (
        target_visibility.to(
            device=predicted_heatmaps.device,
            dtype=predicted_heatmaps.dtype,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    squared_error = F.mse_loss(
        predicted_heatmaps,
        target_heatmaps,
        reduction="none",
    )
    masked_error = squared_error * channel_mask
    normalizer = (
        channel_mask.sum() * predicted_heatmaps.shape[-1] * predicted_heatmaps.shape[-2]
    )
    return masked_error.sum() / normalizer.clamp_min(1.0)


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    lambda_vis: float = 1.0,
    lambda_lmk_vis: float = 1.0,
    lambda_lmk_full: float = 1.0,
    lambda_pca_projection: float = 0.0,
    pca_shape_prior: dict[str, Any] | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the experiment loss for visibility, visible landmarks, and full landmarks."""
    predicted_full_heatmaps = outputs["heatmaps"]
    predicted_visible_heatmaps = outputs["visible_heatmaps"]
    predicted_visibility_logits = outputs["visibility_logits"]
    target_heatmaps = batch["heatmaps"]
    target_visibility = batch["visibility"]

    full_landmark_loss = heatmap_loss_fn(predicted_full_heatmaps, target_heatmaps)
    visible_landmark_loss = compute_visible_landmark_heatmap_loss(
        predicted_heatmaps=predicted_visible_heatmaps,
        target_heatmaps=target_heatmaps,
        target_visibility=target_visibility,
    )
    visibility_loss = visibility_loss_fn(predicted_visibility_logits, target_visibility)
    pca_projection_loss = predicted_full_heatmaps.new_zeros(())
    if pca_shape_prior is not None and lambda_pca_projection > 0.0:
        if image_height is None or image_width is None:
            raise ValueError(
                "image_height and image_width are required for PCA projection loss."
            )
        predicted_landmarks = softargmax_heatmaps_to_image_coords(
            heatmaps=predicted_full_heatmaps,
            image_height=image_height,
            image_width=image_width,
        )
        pca_projection_loss = compute_pca_projection_loss(
            predicted_landmarks=predicted_landmarks,
            pca_prior=pca_shape_prior,
        ).to(dtype=predicted_full_heatmaps.dtype)
    total_loss = (
        lambda_vis * visibility_loss
        + lambda_lmk_vis * visible_landmark_loss
        + lambda_lmk_full * full_landmark_loss
        + lambda_pca_projection * pca_projection_loss
    )
    return {
        "total_loss": total_loss,
        "full_landmark_loss": full_landmark_loss,
        "visible_landmark_loss": visible_landmark_loss,
        "visibility_loss": visibility_loss,
        "pca_projection_loss": pca_projection_loss,
    }
