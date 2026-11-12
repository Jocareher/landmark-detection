from __future__ import annotations

from typing import Any
import math

import torch

from .landmark_losses import PerLandmarkHeatmapLoss, compute_masked_heatmap_loss
from .pca_shape_prior import (
    compute_pca_projection_terms,
    compute_grouped_training_pca_terms,
    softargmax_heatmaps_to_image_coords,
)


def resolve_pca_training_weights(legacy_lambda, legacy_beta, reconstruction=None, mahalanobis=None):
    """Explicit weights override the legacy product; require a complete explicit pair."""
    if (reconstruction is None) != (mahalanobis is None):
        raise ValueError("Set BOTH lambda_pca_reconstruction and lambda_pca_mahalanobis.")
    weights = ((legacy_lambda, legacy_lambda * legacy_beta) if reconstruction is None
               else (reconstruction, mahalanobis))
    if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
        raise ValueError("PCA training weights must be finite and nonnegative.")
    return tuple(float(w) for w in weights)


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
    pca_training_prior_mode: str = "global",
    lambda_pca_reconstruction: float | None = None,
    lambda_pca_mahalanobis: float | None = None,
    compute_pca_diagnostics: bool = False,
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
    wr, wh = resolve_pca_training_weights(lambda_pca_projection, pca_mahalanobis_weight,
                                        lambda_pca_reconstruction, lambda_pca_mahalanobis)
    explicit = lambda_pca_reconstruction is not None
    if explicit and wh > 0 and not pca_mahalanobis_enabled:
        raise ValueError("Positive lambda_pca_mahalanobis requires pca_mahalanobis_enabled.")
    needed = wr > 0 or wh > 0 or compute_pca_diagnostics
    pca_projection_loss = predicted_full_heatmaps.new_zeros(())
    pca_terms = {key: predicted_full_heatmaps.new_zeros(()) for key in (
        "reconstruction", "mahalanobis_squared", "mahalanobis_penalty", "weighted_mahalanobis", "mahalanobis_active_fraction")}
    if pca_training_prior_mode not in {"global", "gt_grouped"}:
        raise ValueError(f"Unsupported training PCA prior mode: {pca_training_prior_mode}")
    if needed and pca_shape_prior is None:
        raise ValueError("PCA training weights or diagnostics require pca_shape_prior.")
    if needed:
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
            projection = (compute_grouped_training_pca_terms
                          if pca_training_prior_mode == "gt_grouped" else compute_pca_projection_terms)
            routing = {"yaw_angles": batch.get("yaw_angle")} if pca_training_prior_mode == "gt_grouped" else {"reduction": "none"}
            pca_terms = projection(
                predicted_landmarks=predicted_landmarks,
                pca_prior=pca_shape_prior,
                loss_space=pca_loss_space,
                mahalanobis_enabled=pca_mahalanobis_enabled,
                mahalanobis_weight=1.0 if explicit else pca_mahalanobis_weight,
                mahalanobis_c=pca_mahalanobis_c,
                mahalanobis_percentile=pca_mahalanobis_percentile,
                mahalanobis_eps=pca_mahalanobis_eps,
                **routing,
            )
            if "mahalanobis_active_fraction" not in pca_terms:
                pca_terms["mahalanobis_active_fraction"] = (pca_terms["mahalanobis_penalty"] > 0).float().mean()
                pca_terms = {key: value.mean() for key, value in pca_terms.items()}
        # Keep the potentially large coefficient penalty in FP32 under AMP.
        pca_projection_loss = pca_terms["loss"]
    weighted_reconstruction = wr * pca_terms["reconstruction"]
    weighted_mahalanobis = wh * pca_terms["mahalanobis_penalty"]
    weighted_pca = weighted_reconstruction + weighted_mahalanobis
    total_loss = (
        lambda_vis * visibility_loss
        + lambda_lmk_vis * visible_landmark_loss
        + lambda_lmk_full * full_landmark_loss
        + weighted_pca
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
        "weighted_pca_reconstruction_loss": weighted_reconstruction,
        "weighted_pca_total_mahalanobis_loss": weighted_mahalanobis,
        "weighted_pca_loss": weighted_pca,
        "pca_mahalanobis_active_fraction": pca_terms["mahalanobis_active_fraction"],
    }
