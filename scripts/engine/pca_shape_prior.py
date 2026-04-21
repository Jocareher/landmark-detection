from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PCA_NORMALIZATION_TYPE = "centroid_bbox_sqrt_area"
PCA_NORMALIZATION_EPS = 1e-6


def normalize_landmark_shapes(
    landmarks: torch.Tensor,
    eps: float = PCA_NORMALIZATION_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Center landmark shapes by centroid and scale them by landmark bbox area."""
    if landmarks.ndim == 2:
        landmarks = landmarks.unsqueeze(0)
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            "Expected landmarks with shape (B, N, 2) or (N, 2), "
            f"got {tuple(landmarks.shape)}."
        )

    landmarks = landmarks.float()
    centroid = landmarks.mean(dim=1, keepdim=True)
    centered = landmarks - centroid
    min_xy = landmarks.amin(dim=1, keepdim=True)
    max_xy = landmarks.amax(dim=1, keepdim=True)
    bbox_size = (max_xy - min_xy).clamp_min(eps)
    scale = torch.sqrt(bbox_size[..., 0] * bbox_size[..., 1]).unsqueeze(-1)
    normalized = centered / scale.clamp_min(eps)
    return normalized, centroid, scale


def flatten_landmark_shapes(landmarks: torch.Tensor) -> torch.Tensor:
    """Flatten `(B, N, 2)` landmark tensors to `(B, 2N)` shape vectors."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (B, N, 2), got {tuple(landmarks.shape)}."
        )
    return landmarks.reshape(landmarks.shape[0], -1)


def fit_pca_shape_prior(
    landmarks: torch.Tensor,
    num_components: int,
    eps: float = PCA_NORMALIZATION_EPS,
) -> dict[str, Any]:
    """Fit a PCA prior from normalized training landmark shapes."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (M, N, 2), got {tuple(landmarks.shape)}."
        )
    if landmarks.shape[0] < 2:
        raise ValueError("At least two landmark shapes are required to fit PCA.")

    normalized_shapes, _, scales = normalize_landmark_shapes(landmarks, eps=eps)
    shape_vectors = flatten_landmark_shapes(normalized_shapes)
    mean_shape = shape_vectors.mean(dim=0)
    centered = shape_vectors - mean_shape

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    max_components = min(num_components, vh.shape[0], shape_vectors.shape[0] - 1)
    if max_components < 1:
        raise ValueError("The requested PCA prior would contain zero components.")

    explained_variance_all = singular_values.square() / (shape_vectors.shape[0] - 1)
    total_variance = explained_variance_all.sum().clamp_min(eps)
    explained_variance_ratio_all = explained_variance_all / total_variance

    return {
        "mean_shape": mean_shape.cpu(),
        "components": vh[:max_components].cpu(),
        "explained_variance": explained_variance_all[:max_components].cpu(),
        "explained_variance_ratio": explained_variance_ratio_all[:max_components].cpu(),
        "all_explained_variance": explained_variance_all.cpu(),
        "all_explained_variance_ratio": explained_variance_ratio_all.cpu(),
        "num_components": int(max_components),
        "requested_num_components": int(num_components),
        "num_train_shapes": int(shape_vectors.shape[0]),
        "num_landmarks": int(landmarks.shape[1]),
        "shape_vector_size": int(shape_vectors.shape[1]),
        "normalization": {
            "type": PCA_NORMALIZATION_TYPE,
            "center": "landmark_centroid",
            "scale": "sqrt_landmark_bbox_area",
            "eps": float(eps),
        },
        "train_scale_mean": float(scales.mean().item()),
        "train_scale_std": float(scales.std(unbiased=False).item()),
    }


def load_pca_shape_prior(
    prior_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a saved PCA shape prior and move tensor fields to the target device."""
    payload = torch.load(prior_path, map_location=device, weights_only=False)
    required_keys = {"mean_shape", "components", "normalization", "num_landmarks"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(
            f"Invalid PCA prior '{prior_path}'. Missing keys: {sorted(missing_keys)}"
        )
    if payload["normalization"].get("type") != PCA_NORMALIZATION_TYPE:
        raise ValueError(
            f"Unsupported PCA normalization '{payload['normalization'].get('type')}'. "
            f"Expected '{PCA_NORMALIZATION_TYPE}'."
        )
    for key in (
        "mean_shape",
        "components",
        "explained_variance",
        "explained_variance_ratio",
        "all_explained_variance",
        "all_explained_variance_ratio",
    ):
        if key in payload and isinstance(payload[key], torch.Tensor):
            payload[key] = payload[key].to(device=device, dtype=torch.float32)
    return payload


def softargmax_heatmaps_to_image_coords(
    heatmaps: torch.Tensor,
    image_height: int,
    image_width: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Decode heatmaps into differentiable image-space landmark coordinates."""
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps with shape (B, N, H, W), got {tuple(heatmaps.shape)}."
        )

    batch_size, num_landmarks, heatmap_height, heatmap_width = heatmaps.shape
    heatmaps_float = heatmaps.float() / max(float(temperature), PCA_NORMALIZATION_EPS)
    probabilities = F.softmax(
        heatmaps_float.reshape(batch_size, num_landmarks, -1),
        dim=-1,
    )
    x_coords = torch.linspace(
        0,
        heatmap_width - 1,
        heatmap_width,
        device=heatmaps.device,
        dtype=probabilities.dtype,
    )
    y_coords = torch.linspace(
        0,
        heatmap_height - 1,
        heatmap_height,
        device=heatmaps.device,
        dtype=probabilities.dtype,
    )
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    pred_x = (probabilities * grid_x.reshape(1, 1, -1)).sum(dim=-1)
    pred_y = (probabilities * grid_y.reshape(1, 1, -1)).sum(dim=-1)
    landmarks = torch.stack([pred_x, pred_y], dim=-1)
    landmarks[..., 0] *= image_width / float(heatmap_width)
    landmarks[..., 1] *= image_height / float(heatmap_height)
    return landmarks


def compute_pca_projection_loss(
    predicted_landmarks: torch.Tensor,
    pca_prior: dict[str, Any],
) -> torch.Tensor:
    """Penalize the residual after projecting predicted shapes onto the PCA subspace."""
    expected_landmarks = int(pca_prior["num_landmarks"])
    if predicted_landmarks.shape[1] != expected_landmarks:
        raise ValueError(
            f"PCA prior expects {expected_landmarks} landmarks, "
            f"got {predicted_landmarks.shape[1]}."
        )

    eps = float(pca_prior["normalization"].get("eps", PCA_NORMALIZATION_EPS))
    normalized_shapes, _, _ = normalize_landmark_shapes(predicted_landmarks, eps=eps)
    shape_vectors = flatten_landmark_shapes(normalized_shapes)
    mean_shape = pca_prior["mean_shape"].to(
        device=shape_vectors.device,
        dtype=shape_vectors.dtype,
    )
    components = pca_prior["components"].to(
        device=shape_vectors.device,
        dtype=shape_vectors.dtype,
    )
    centered = shape_vectors - mean_shape
    coefficients = centered @ components.T
    reconstructed = mean_shape + coefficients @ components
    return F.mse_loss(shape_vectors, reconstructed)
