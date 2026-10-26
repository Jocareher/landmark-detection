from __future__ import annotations

from typing import Any

import torch

from .landmark_losses import PerLandmarkHeatmapLoss, compute_masked_heatmap_loss
from .pca_shape_prior import (
    compute_pca_projection_terms,
    softargmax_heatmaps_to_image_coords,
)


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    heatmap_loss_fn: PerLandmarkHeatmapLoss,
    visibility_loss_fn: torch.nn.Module,
    lambda_vis: float = 1.0,
    lambda_lmk_vis: float = 1.0,
    lambda_lmk_full: float = 1.0,
    lambda_pca_projection: float = 0.0,
    pca_shape_prior: dict[str, Any] | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    pca_loss_space: str = "image",
    pca_mahalanobis_enabled: bool = False,
    pca_mahalanobis_weight: float = 1.0,
    pca_mahalanobis_c: float = 10.0,
    pca_mahalanobis_percentile: float | None = None,
    pca_mahalanobis_eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute the experiment loss for visibility, visible landmarks, and full landmarks."""
    predicted_full_heatmaps = outputs["heatmaps"]
    predicted_visible_heatmaps = outputs["visible_heatmaps"]
    predicted_visibility_logits = outputs["visibility_logits"]
    target_heatmaps = batch["heatmaps"]
    target_visibility = batch["visibility"]

    full_landmark_loss = compute_masked_heatmap_loss(
        loss_fn=heatmap_loss_fn,
        predicted_heatmaps=predicted_full_heatmaps,
        target_heatmaps=target_heatmaps,
        target_visibility=None,
    )
    visible_landmark_loss = compute_masked_heatmap_loss(
        loss_fn=heatmap_loss_fn,
        predicted_heatmaps=predicted_visible_heatmaps,
        target_heatmaps=target_heatmaps,
        target_visibility=target_visibility,
    )
    visibility_loss = visibility_loss_fn(predicted_visibility_logits, target_visibility)
    pca_projection_loss = predicted_full_heatmaps.new_zeros(())
    pca_terms = {key: predicted_full_heatmaps.new_zeros(()) for key in (
        "reconstruction", "mahalanobis_squared", "mahalanobis_penalty", "weighted_mahalanobis")}
    if pca_shape_prior is not None and lambda_pca_projection > 0.0:
        if image_height is None or image_width is None:
            raise ValueError(
                "image_height and image_width are required for PCA projection loss."
            )
        pca_device_type = predicted_full_heatmaps.device.type
        with torch.autocast(device_type=pca_device_type, enabled=False):
            predicted_landmarks = softargmax_heatmaps_to_image_coords(
                heatmaps=predicted_full_heatmaps.float(),
                image_height=image_height,
                image_width=image_width,
            )
            pca_terms = compute_pca_projection_terms(
                predicted_landmarks=predicted_landmarks,
                pca_prior=pca_shape_prior,
                loss_space=pca_loss_space,
                mahalanobis_enabled=pca_mahalanobis_enabled,
                mahalanobis_weight=pca_mahalanobis_weight,
                mahalanobis_c=pca_mahalanobis_c,
                mahalanobis_percentile=pca_mahalanobis_percentile,
                mahalanobis_eps=pca_mahalanobis_eps,
            )
        # Keep the potentially large coefficient penalty in FP32 under AMP.
        pca_projection_loss = pca_terms["loss"]
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
        "pca_loss": pca_projection_loss,
        "pca_reconstruction_loss": pca_terms["reconstruction"],
        "pca_mahalanobis_squared": pca_terms["mahalanobis_squared"],
        "pca_mahalanobis_penalty": pca_terms["mahalanobis_penalty"],
        "weighted_pca_mahalanobis_loss": pca_terms["weighted_mahalanobis"],
    }
