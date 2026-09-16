from __future__ import annotations

import math
from typing import Any

import torch

from .landmark_losses import PerLandmarkHeatmapLoss, compute_masked_heatmap_loss
from .pca_shape_prior import (
    compute_pca_regularization_terms,
    decode_landmarks_for_shape_loss,
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
    pca_mahalanobis_limit: float = 2.0,
    pca_variance_floor: float = 1e-4,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
    lambda_pca_mahalanobis: float = 0.0,
    pca_regularization: str = "bounded_reconstruction",
    pca_coefficient_alpha: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Compute the experiment loss for visibility, visible landmarks, and full landmarks."""
    for name, weight in (("lambda_pca_projection", lambda_pca_projection),
                         ("lambda_pca_mahalanobis", lambda_pca_mahalanobis)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if pca_regularization not in {"bounded_reconstruction", "mahalanobis_reconstruction", "mahalanobis"}:
        raise ValueError(f"Unknown PCA regularization: {pca_regularization}")
    if pca_regularization != "mahalanobis" and lambda_pca_mahalanobis != 0:
        raise ValueError("Reconstruction modes require lambda_pca_mahalanobis=0; use pca_regularization='mahalanobis' for the previous loss.")
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
    pca_terms = {
        key: predicted_full_heatmaps.new_zeros((), dtype=torch.float32)
        for key in (
            "pca_loss", "pca_subspace_loss", "pca_mahalanobis_loss",
            "pca_mahalanobis_sq_per_component", "pca_outside_fraction",
            "pca_restricted_mahalanobis_sq_per_component",
            "pca_projection_mse", "pca_bounded_loss", "pca_coefficient_loss", "pca_clipped_fraction",
        )
    }
    if lambda_pca_projection > 0.0 or lambda_pca_mahalanobis > 0.0:
        if pca_shape_prior is None:
            raise ValueError("A positive PCA projection or Mahalanobis weight requires a PCA prior.")
        if image_height is None or image_width is None:
            raise ValueError(
                "image_height and image_width are required for PCA regularization."
            )
        pca_device_type = predicted_full_heatmaps.device.type
        with torch.autocast(device_type=pca_device_type, enabled=False):
            predicted_landmarks = decode_landmarks_for_shape_loss(
                heatmaps=predicted_full_heatmaps.float(),
                image_height=image_height,
                image_width=image_width,
                decoder=coordinate_decoder,
                temperature=wasserstein_softmax_temperature,
            )
            pca_terms.update(compute_pca_regularization_terms(
                predicted_landmarks=predicted_landmarks,
                pca_prior=pca_shape_prior,
                mahalanobis_limit=pca_mahalanobis_limit,
                variance_floor=pca_variance_floor,
                coefficient_alpha=pca_coefficient_alpha if pca_regularization == "bounded_reconstruction" else None,
                mahalanobis_reconstruction=pca_regularization == "mahalanobis_reconstruction",
            ))
    # The logged PCA loss is the actual weighted contribution to the objective.
    pca_terms["pca_loss"] = (
        lambda_pca_projection * pca_terms["pca_bounded_loss" if pca_regularization != "mahalanobis" else "pca_subspace_loss"]
        + lambda_pca_mahalanobis * pca_terms["pca_mahalanobis_loss"]
    )
    total_loss = (
        lambda_vis * visibility_loss
        + lambda_lmk_vis * visible_landmark_loss
        + lambda_lmk_full * full_landmark_loss
        + pca_terms["pca_loss"]
    )
    return {
        "total_loss": total_loss,
        "full_landmark_loss": full_landmark_loss,
        "visible_landmark_loss": visible_landmark_loss,
        "visibility_loss": visibility_loss,
        **pca_terms,
    }
