from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..utils.synthetic_labels import SYNTHETIC_CLASS_ID_TO_NAME


PCA_ALIGNMENT_METHOD = "procrustes"
PCA_ALIGNMENT_ALLOW_REFLECTION = False
PCA_ALIGNMENT_EPS = 1e-6


def flatten_landmark_shapes(landmarks: torch.Tensor) -> torch.Tensor:
    """Flatten `(B, N, 2)` landmark tensors to `(B, 2N)` shape vectors."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (B, N, 2), got {tuple(landmarks.shape)}."
        )
    return landmarks.reshape(landmarks.shape[0], -1)


def normalize_shape_for_procrustes(
    points: torch.Tensor,
    eps: float = PCA_ALIGNMENT_EPS,
) -> torch.Tensor:
    """Center a shape and normalize it to unit Frobenius norm."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(
            f"Expected points with shape (N, 2), got {tuple(points.shape)}."
        )
    points = points.to(dtype=torch.float32)
    centered = points - points.mean(dim=0, keepdim=True)
    scale = torch.linalg.norm(centered)
    if float(scale.item()) <= eps:
        raise ValueError("Shape has near-zero scale and cannot be normalized.")
    return centered / scale.clamp_min(eps)


def estimate_similarity_transform_torch(
    source: torch.Tensor,
    target: torch.Tensor,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    eps: float = PCA_ALIGNMENT_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate a similarity transform mapping source points onto target points."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 2:
        raise ValueError(
            f"Expected matching (N, 2) source/target shapes, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}."
        )

    source = source.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    source_centered = source - source_center
    target_centered = target - target_center

    source_scale = torch.linalg.norm(source_centered)
    target_scale = torch.linalg.norm(target_centered)
    if float(source_scale.item()) <= eps or float(target_scale.item()) <= eps:
        raise ValueError("Degenerate shape encountered during Procrustes alignment.")

    source_normalized = source_centered / source_scale.clamp_min(eps)
    target_normalized = target_centered / target_scale.clamp_min(eps)
    covariance = source_normalized.T @ target_normalized
    u_matrix, _, vh_matrix = torch.linalg.svd(covariance, full_matrices=False)
    rotation = u_matrix @ vh_matrix

    if not allow_reflection and float(torch.det(rotation).item()) < 0.0:
        correction = torch.eye(2, device=rotation.device, dtype=rotation.dtype)
        correction[-1, -1] = -1.0
        rotation = u_matrix @ correction @ vh_matrix

    scale = target_scale / source_scale.clamp_min(eps)
    translation = target_center - scale * (source_center @ rotation)
    return rotation, scale, translation


def apply_similarity_transform_torch(
    coords: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply a row-vector similarity transform to one `(N, 2)` landmark shape."""
    coords = coords.to(dtype=torch.float32)
    return (scale * (coords @ rotation)) + translation


def align_shape_to_reference_torch(
    coords: torch.Tensor,
    reference_shape: torch.Tensor,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    eps: float = PCA_ALIGNMENT_EPS,
) -> torch.Tensor:
    """Align one shape onto a reference shape with no-reflection Procrustes."""
    rotation, scale, translation = estimate_similarity_transform_torch(
        source=coords,
        target=reference_shape,
        allow_reflection=allow_reflection,
        eps=eps,
    )
    return apply_similarity_transform_torch(coords, rotation, scale, translation)


def generalized_procrustes_analysis_torch(
    shapes: torch.Tensor,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
    eps: float = PCA_ALIGNMENT_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run generalized Procrustes analysis on one class-specific shape set."""
    if shapes.ndim != 3 or shapes.shape[-1] != 2:
        raise ValueError(
            f"Expected shapes with shape (M, N, 2), got {tuple(shapes.shape)}."
        )
    if shapes.shape[0] < 2:
        raise ValueError("At least two shapes are required for Procrustes PCA.")

    finite_mask = torch.isfinite(shapes).all(dim=2).all(dim=1)
    valid_shapes = shapes[finite_mask]
    if valid_shapes.shape[0] < 2:
        raise ValueError("At least two finite shapes are required for Procrustes PCA.")

    reference_shape = normalize_shape_for_procrustes(valid_shapes[0], eps=eps)
    aligned_shapes = torch.empty_like(valid_shapes)

    for _ in range(max_iterations):
        for sample_index in range(valid_shapes.shape[0]):
            aligned_shapes[sample_index] = align_shape_to_reference_torch(
                coords=valid_shapes[sample_index],
                reference_shape=reference_shape,
                allow_reflection=allow_reflection,
                eps=eps,
            )
        new_reference = normalize_shape_for_procrustes(
            aligned_shapes.mean(dim=0),
            eps=eps,
        )
        difference = torch.mean((new_reference - reference_shape).square())
        reference_shape = new_reference
        if float(difference.item()) < tolerance:
            break

    return aligned_shapes, reference_shape


def _select_num_components(
    explained_variance_ratio_all: torch.Tensor,
    max_components: int,
    requested_num_components: int | None,
    explained_variance_threshold: float | None,
) -> int:
    if max_components < 1:
        raise ValueError("The requested PCA prior would contain zero components.")
    if explained_variance_threshold is not None:
        if not 0.0 < float(explained_variance_threshold) <= 1.0:
            raise ValueError(
                "explained_variance_threshold must be in the interval (0, 1]."
            )
        cumulative = explained_variance_ratio_all.cumsum(dim=0)
        threshold_tensor = torch.tensor(
            float(explained_variance_threshold),
            device=cumulative.device,
            dtype=cumulative.dtype,
        )
        selected = int(torch.searchsorted(cumulative, threshold_tensor).item()) + 1
        return min(max(selected, 1), max_components)
    requested = (
        32 if requested_num_components is None else int(requested_num_components)
    )
    if requested < 1:
        raise ValueError("num_components must be at least 1.")
    return min(requested, max_components)


def fit_single_class_pca_prior(
    landmarks: torch.Tensor,
    class_idx: int,
    num_components: int | None = 32,
    explained_variance_threshold: float | None = None,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    eps: float = PCA_ALIGNMENT_EPS,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Fit one PCA prior for a single orientation class after GPA alignment."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (M, N, 2), got {tuple(landmarks.shape)}."
        )

    finite_mask = torch.isfinite(landmarks).all(dim=2).all(dim=1)
    valid_landmarks = landmarks[finite_mask].float()
    if valid_landmarks.shape[0] < 2:
        raise ValueError(
            f"Class {class_idx} requires at least two finite shapes to fit PCA, "
            f"got {valid_landmarks.shape[0]}."
        )

    aligned_shapes, reference_shape = generalized_procrustes_analysis_torch(
        valid_landmarks,
        allow_reflection=allow_reflection,
        eps=eps,
    )
    shape_vectors = flatten_landmark_shapes(aligned_shapes)
    mean_shape = shape_vectors.mean(dim=0)
    centered = shape_vectors - mean_shape

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    max_components = min(
        vh.shape[0], shape_vectors.shape[0] - 1, shape_vectors.shape[1]
    )
    if max_components < 1:
        raise ValueError(
            f"Class {class_idx} does not have enough samples to fit at least one PCA component."
        )

    explained_variance_all = singular_values.square() / (shape_vectors.shape[0] - 1)
    total_variance = explained_variance_all.sum().clamp_min(eps)
    explained_variance_ratio_all = explained_variance_all / total_variance
    selected_components = _select_num_components(
        explained_variance_ratio_all=explained_variance_ratio_all,
        max_components=max_components,
        requested_num_components=num_components,
        explained_variance_threshold=explained_variance_threshold,
    )

    prior = {
        "class_idx": int(class_idx),
        "class_name": SYNTHETIC_CLASS_ID_TO_NAME[int(class_idx)],
        "mean_shape": mean_shape.cpu(),
        "components": vh[:selected_components].cpu(),
        "explained_variance": explained_variance_all[:selected_components].cpu(),
        "explained_variance_ratio": explained_variance_ratio_all[
            :selected_components
        ].cpu(),
        "all_explained_variance": explained_variance_all.cpu(),
        "all_explained_variance_ratio": explained_variance_ratio_all.cpu(),
        "num_components": int(selected_components),
        "requested_num_components": (
            None if num_components is None else int(num_components)
        ),
        "explained_variance_threshold": (
            None
            if explained_variance_threshold is None
            else float(explained_variance_threshold)
        ),
        "num_samples": int(valid_landmarks.shape[0]),
        "num_landmarks": int(landmarks.shape[1]),
        "shape_vector_size": int(shape_vectors.shape[1]),
        "reference_shape": reference_shape.cpu(),
    }
    return prior, aligned_shapes.cpu()


def fit_class_conditioned_pca_shape_prior(
    landmarks: torch.Tensor,
    class_indices: torch.Tensor,
    num_components: int | None = 32,
    explained_variance_threshold: float | None = None,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    eps: float = PCA_ALIGNMENT_EPS,
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    """Fit one Procrustes-aligned PCA prior per class_idx."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (M, N, 2), got {tuple(landmarks.shape)}."
        )
    class_indices = class_indices.reshape(-1).to(dtype=torch.int64)
    if landmarks.shape[0] != class_indices.shape[0]:
        raise ValueError(
            f"Landmark/class batch mismatch: {landmarks.shape[0]} vs {class_indices.shape[0]}."
        )

    priors: dict[int, dict[str, Any]] = {}
    aligned_shapes_by_class: dict[int, torch.Tensor] = {}
    for class_idx, class_name in SYNTHETIC_CLASS_ID_TO_NAME.items():
        class_mask = class_indices == int(class_idx)
        class_landmarks = landmarks[class_mask]
        if class_landmarks.shape[0] < 2:
            raise ValueError(
                f"Class {class_idx} ({class_name}) has only {class_landmarks.shape[0]} "
                "train samples. At least two are required to fit PCA."
            )
        prior, aligned_shapes = fit_single_class_pca_prior(
            landmarks=class_landmarks,
            class_idx=int(class_idx),
            num_components=num_components,
            explained_variance_threshold=explained_variance_threshold,
            allow_reflection=allow_reflection,
            eps=eps,
        )
        priors[int(class_idx)] = prior
        aligned_shapes_by_class[int(class_idx)] = aligned_shapes

    payload = {
        "priors": priors,
        "alignment": {
            "method": PCA_ALIGNMENT_METHOD,
            "num_landmarks": int(landmarks.shape[1]),
            "allow_reflection": bool(allow_reflection),
            "eps": float(eps),
        },
        "class_mapping": {
            int(class_idx): class_name
            for class_idx, class_name in SYNTHETIC_CLASS_ID_TO_NAME.items()
        },
    }
    return payload, aligned_shapes_by_class


def load_pca_shape_prior(
    prior_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a saved class-conditioned PCA prior and move tensor fields to device."""
    payload = torch.load(prior_path, map_location=device, weights_only=False)
    required_keys = {"priors", "alignment", "class_mapping"}
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(
            f"Invalid PCA prior '{prior_path}'. Missing keys: {sorted(missing_keys)}"
        )
    if payload["alignment"].get("method") != PCA_ALIGNMENT_METHOD:
        raise ValueError(
            f"Unsupported PCA alignment method '{payload['alignment'].get('method')}'. "
            f"Expected '{PCA_ALIGNMENT_METHOD}'."
        )

    normalized_priors: dict[int, dict[str, Any]] = {}
    for raw_class_idx, prior in payload["priors"].items():
        class_idx = int(raw_class_idx)
        for key in (
            "mean_shape",
            "components",
            "explained_variance",
            "explained_variance_ratio",
            "all_explained_variance",
            "all_explained_variance_ratio",
            "reference_shape",
        ):
            if key in prior and isinstance(prior[key], torch.Tensor):
                prior[key] = prior[key].to(device=device, dtype=torch.float32)
        normalized_priors[class_idx] = prior
    payload["priors"] = normalized_priors
    payload["class_mapping"] = {
        int(raw_class_idx): class_name
        for raw_class_idx, class_name in payload["class_mapping"].items()
    }
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
    heatmaps_float = heatmaps.float() / max(float(temperature), PCA_ALIGNMENT_EPS)
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
    class_indices: torch.Tensor,
    pca_prior: dict[str, Any],
) -> torch.Tensor:
    """Penalize projection residuals using the class-conditioned PCA subspace."""
    if predicted_landmarks.ndim != 3 or predicted_landmarks.shape[-1] != 2:
        raise ValueError(
            "Expected predicted_landmarks with shape (B, N, 2), "
            f"got {tuple(predicted_landmarks.shape)}."
        )

    predicted_landmarks = predicted_landmarks.to(dtype=torch.float32)
    class_indices = class_indices.reshape(-1).to(dtype=torch.int64)
    if predicted_landmarks.shape[0] != class_indices.shape[0]:
        raise ValueError(
            f"Predicted landmark/class batch mismatch: "
            f"{predicted_landmarks.shape[0]} vs {class_indices.shape[0]}."
        )

    alignment_config = pca_prior["alignment"]
    allow_reflection = bool(alignment_config.get("allow_reflection", False))
    eps = float(alignment_config.get("eps", PCA_ALIGNMENT_EPS))

    sample_losses: list[torch.Tensor] = []
    for sample_index in range(predicted_landmarks.shape[0]):
        class_idx = int(class_indices[sample_index].item())
        if class_idx not in pca_prior["priors"]:
            raise ValueError(
                f"Missing PCA prior for class_idx={class_idx}. "
                f"Available classes: {sorted(pca_prior['priors'])}."
            )
        current_prior = pca_prior["priors"][class_idx]
        expected_landmarks = int(current_prior["num_landmarks"])
        current_shape = predicted_landmarks[sample_index]
        if current_shape.shape[0] != expected_landmarks:
            raise ValueError(
                f"PCA prior for class_idx={class_idx} expects {expected_landmarks} landmarks, "
                f"got {current_shape.shape[0]}."
            )

        reference_shape = current_prior["reference_shape"].to(
            device=current_shape.device,
            dtype=current_shape.dtype,
        )
        aligned_shape = align_shape_to_reference_torch(
            coords=current_shape,
            reference_shape=reference_shape,
            allow_reflection=allow_reflection,
            eps=eps,
        )
        shape_vector = aligned_shape.reshape(1, -1)
        mean_shape = current_prior["mean_shape"].to(
            device=shape_vector.device,
            dtype=shape_vector.dtype,
        )
        components = current_prior["components"].to(
            device=shape_vector.device,
            dtype=shape_vector.dtype,
        )
        centered = shape_vector - mean_shape
        coefficients = centered @ components.T
        reconstructed = mean_shape + coefficients @ components
        sample_losses.append(F.mse_loss(shape_vector, reconstructed))

    return torch.stack(sample_losses).mean()
