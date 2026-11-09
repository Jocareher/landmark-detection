"""Differentiable TTA objectives, reduced per image before batch averaging."""
from __future__ import annotations

import math
import torch
from .metrics import heatmap_distribution_statistics
from .pca_shape_prior import compute_pca_projection_terms, validate_pca_mahalanobis


# BabyLand orientation class IDs: left, quarter_left, frontal, quarter_right, right.
POSE5_YAW_BY_CLASS = {0: "-75", 1: "-45", 2: "0", 3: "45", 4: "75"}
PCA_PRIOR_MODES = ("global", "min_pose", "gt_pose")


def resolve_gt_pose_labels(class_ids, batch_size):
    if class_ids is None or len(class_ids) != batch_size:
        raise ValueError("gt_pose requires one annotated orientation class per image.")
    labels = []
    for value in class_ids:
        value = value.item() if isinstance(value, torch.Tensor) else value
        if value not in POSE5_YAW_BY_CLASS:
            raise ValueError(f"gt_pose requires an orientation class in [0, 4], got {value!r}.")
        labels.append(POSE5_YAW_BY_CLASS[value])
    return labels


def validate_tta_weights(pca_weight: float, entropy_weight: float, spread_weight: float) -> None:
    weights = (pca_weight, entropy_weight, spread_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("TTA loss weights must be finite and nonnegative.")
    if not any(w > 0 for w in weights):
        raise ValueError("At least one TTA loss weight must be positive.")


def compute_tta_losses(heatmaps, landmarks, pca_prior, *, pca_weight=1.0,
                       entropy_weight=0.0, spread_weight=0.0, pca_loss_space="image", pca_prior_mode="global",
                       pca_mahalanobis_enabled=False, pca_mahalanobis_weight=1.0,
                       pca_mahalanobis_c=10.0, pca_mahalanobis_percentile=None,
                       pca_mahalanobis_eps=1e-8, gt_pose_labels=None):
    """Return [B] raw signals, weighted contributions and their sum.

    p = spatial softmax(logits), with temperature fixed at 1.
    H = -sum(p log(p))/log(HW); S = 2 trace(Cov_p(r)), r in [0,1]^2.
    Each heatmap signal averages ALL landmarks; no visibility or GT is used.
    A zero PCA weight bypasses the prior and projection entirely.
    """
    validate_tta_weights(pca_weight, entropy_weight, spread_weight)
    validate_pca_mahalanobis(pca_mahalanobis_enabled, pca_mahalanobis_weight,
                            pca_mahalanobis_c, pca_mahalanobis_eps,
                            pca_mahalanobis_percentile)
    options = dict(mahalanobis_enabled=pca_mahalanobis_enabled,
                   mahalanobis_weight=pca_mahalanobis_weight,
                   mahalanobis_c=pca_mahalanobis_c,
                   mahalanobis_percentile=pca_mahalanobis_percentile,
                   mahalanobis_eps=pca_mahalanobis_eps)
    if pca_prior_mode not in PCA_PRIOR_MODES:
        raise ValueError(f"Unsupported PCA prior mode: {pca_prior_mode}")
    statistics = heatmap_distribution_statistics(heatmaps)
    raw = {"heatmap_entropy_loss": statistics["entropy"].mean(-1),
           "heatmap_spread_loss": statistics["spread"].mean(-1)}
    zero = heatmaps.new_zeros(heatmaps.shape[0])
    selection = {}
    if pca_weight > 0:
        if pca_prior is None:
            raise ValueError("A PCA prior is required when tta_pca_weight > 0.")
        if pca_prior_mode in {"min_pose", "gt_pose"}:
            priors = pca_prior.get("pose_priors", {})
            if not priors:
                raise ValueError(f"{pca_prior_mode} requires a prior containing pose_priors.")
            labels = list(priors)
            if pca_prior_mode == "gt_pose":
                if set(labels) != set(POSE5_YAW_BY_CLASS.values()):
                    raise ValueError("gt_pose requires a five-pose PCA prior bank.")
                if (gt_pose_labels is None or len(gt_pose_labels) != landmarks.shape[0]
                        or any(label not in priors for label in gt_pose_labels)):
                    raise ValueError("gt_pose requires one valid prior label per image.")
                indices = torch.tensor([labels.index(label) for label in gt_pose_labels],
                                       device=landmarks.device, dtype=torch.long)
            candidates_terms = [compute_pca_projection_terms(
                landmarks, {"alignment": pca_prior["alignment"], "global_prior": priors[label]},
                loss_space=pca_loss_space, reduction="none", **options) for label in labels]
            candidates = torch.stack([term["loss"] for term in candidates_terms], dim=1)
            if not bool(torch.isfinite(candidates).all()):
                raise FloatingPointError("Non-finite candidate pose PCA loss.")
            if pca_prior_mode == "min_pose":
                indices = candidates.argmin(dim=1)
            terms = {key: torch.stack([term[key] for term in candidates_terms], dim=1)
                     .gather(1, indices[:, None]).squeeze(1) for key in candidates_terms[0]}
            selection["pca_selected_yaw"] = torch.tensor([float(label) for label in labels], device=indices.device)[indices]
            for index, label in enumerate(labels):
                selection[f"pca_candidate_loss_yaw_{label}"] = candidates[:, index]
        else:
            terms = compute_pca_projection_terms(
                landmarks, pca_prior, loss_space=pca_loss_space, reduction="none", **options)
        for key, source in (("pca_loss", "loss"), ("pca_reconstruction_loss", "reconstruction"),
                            ("pca_mahalanobis_squared", "mahalanobis_squared"),
                            ("pca_mahalanobis_penalty", "mahalanobis_penalty"),
                            ("weighted_pca_mahalanobis_loss", "weighted_mahalanobis")):
            raw[key] = terms[source]
    else:
        for key in ("pca_loss", "pca_reconstruction_loss", "pca_mahalanobis_squared",
                    "pca_mahalanobis_penalty", "weighted_pca_mahalanobis_loss"):
            raw[key] = torch.full_like(zero, math.nan)
    result = {**raw, **selection}
    for signal, weight, key in (("pca", pca_weight, "pca_loss"),
                               ("entropy", entropy_weight, "heatmap_entropy_loss"),
                               ("spread", spread_weight, "heatmap_spread_loss")):
        result[f"weighted_{signal}_loss"] = raw[key] * weight if weight > 0 else zero
    result["sample_tta_loss"] = sum(result[f"weighted_{signal}_loss"]
                                     for signal in ("pca", "entropy", "spread"))
    return result
