from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .metrics import decode_heatmaps_to_image_coords

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
    """Run generalized Procrustes analysis on a set of landmark shapes."""
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


def fit_global_pca_shape_prior(
    landmarks: torch.Tensor,
    num_components: int | None = 32,
    explained_variance_threshold: float | None = None,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    eps: float = PCA_ALIGNMENT_EPS,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Fit one global PCA prior after pooling all finite training shapes."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (M, N, 2), got {tuple(landmarks.shape)}."
        )

    finite_mask = torch.isfinite(landmarks).all(dim=2).all(dim=1)
    valid_landmarks = landmarks[finite_mask].float()
    if valid_landmarks.shape[0] < 2:
        raise ValueError(
            f"Global PCA prior requires at least two finite shapes, got {valid_landmarks.shape[0]}."
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
            "The global training shapes do not have enough samples to fit at least one PCA component."
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
        "scope": "global",
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


def build_global_pca_shape_prior_payload(
    landmarks: torch.Tensor,
    num_components: int | None = 32,
    explained_variance_threshold: float | None = None,
    allow_reflection: bool = PCA_ALIGNMENT_ALLOW_REFLECTION,
    eps: float = PCA_ALIGNMENT_EPS,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Fit a single Procrustes-aligned PCA prior from all synthetic training samples."""
    if landmarks.ndim != 3 or landmarks.shape[-1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (M, N, 2), got {tuple(landmarks.shape)}."
        )
    global_prior, aligned_shapes = fit_global_pca_shape_prior(
        landmarks=landmarks,
        num_components=num_components,
        explained_variance_threshold=explained_variance_threshold,
        allow_reflection=allow_reflection,
        eps=eps,
    )
    payload = {
        "global_prior": global_prior,
        "alignment": {
            "method": PCA_ALIGNMENT_METHOD,
            "num_landmarks": int(landmarks.shape[1]),
            "allow_reflection": bool(allow_reflection),
            "eps": float(eps),
        },
    }
    return payload, aligned_shapes


def load_pca_shape_prior(
    prior_path: str | Path,
    device: torch.device | str,
) -> dict[str, Any]:
    """Load a saved global PCA prior and move tensor fields to device."""
    payload = torch.load(prior_path, map_location=device, weights_only=False)
    required_keys = {"global_prior", "alignment"}
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

    global_prior = payload["global_prior"]
    for key in (
        "mean_shape",
        "components",
        "explained_variance",
        "explained_variance_ratio",
        "all_explained_variance",
        "all_explained_variance_ratio",
        "reference_shape",
    ):
        if key in global_prior and isinstance(global_prior[key], torch.Tensor):
            global_prior[key] = global_prior[key].to(device=device, dtype=torch.float32)
    payload["global_prior"] = global_prior
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


def decode_landmarks_for_shape_loss(
    heatmaps: torch.Tensor,
    image_height: int,
    image_width: int,
    decoder: str = "argmax_subpixel",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Use inference coordinates, with a soft-argmax surrogate for argmax gradients.

    The straight-through gradient is an approximation, not the derivative of
    argmax. Barycenter decoding uses its actual differentiable computation.
    """
    decoded = decode_heatmaps_to_image_coords(
        heatmaps.float(), image_height, image_width,
        use_subpixel=True, decoder=decoder, softmax_temperature=temperature,
    )
    if decoder == "barycenter":
        return decoded
    surrogate = softargmax_heatmaps_to_image_coords(
        heatmaps, image_height, image_width, temperature=temperature,
    )
    return decoded.detach() + (surrogate - surrogate.detach())


def align_shape_for_regularization(
    coords: torch.Tensor,
    reference_shape: torch.Tensor,
    allow_reflection: bool = False,
    eps: float = PCA_ALIGNMENT_EPS,
) -> torch.Tensor:
    """Align in 2D without SVD gradients, including initially collapsed predictions.

    For nondegenerate shapes this uses the same norm ratio and optimal rotation
    as the prior's Procrustes alignment. At undefined rotations use identity;
    at zero scale clamp the denominator so heatmap supervision can recover.
    """
    source = coords - coords.mean(dim=0)
    target_center = reference_shape.mean(dim=0)
    target = reference_shape - target_center
    source = source / torch.linalg.vector_norm(source).clamp_min(eps)
    target_scale = torch.linalg.vector_norm(target).clamp_min(eps)
    target_unit = target / target_scale

    def rotate(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        covariance = points.T @ target_unit
        a = covariance[0, 0] + covariance[1, 1]
        b = covariance[0, 1] - covariance[1, 0]
        magnitude = torch.linalg.vector_norm(torch.stack([a, b]))
        rotation = torch.stack([a, b, -b, a]).reshape(2, 2) / magnitude.clamp_min(eps)
        rotation = torch.where(
            magnitude > eps, rotation,
            torch.eye(2, device=coords.device, dtype=coords.dtype),
        )
        return points @ rotation, magnitude

    aligned, score = rotate(source)
    if allow_reflection:
        reflected, reflected_score = rotate(source * source.new_tensor([-1., 1.]))
        aligned = torch.where(reflected_score > score, reflected, aligned)
    return aligned * target_scale + target_center


def restrict_coefficients_mahalanobis(
    coefficients: torch.Tensor,
    variances: torch.Tensor,
    limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Radially restrict b to D_M^2/K <= limit in whitened PCA coordinates.

    This is nearest-point projection in the whitened metric, not Euclidean
    projection onto an anisotropic ellipsoid. Inside coefficients are unchanged.
    The caller supplies positive, stabilized variances. No gradient is detached.
    """
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("limit must be finite and positive.")
    radius = math.sqrt(coefficients.shape[-1] * limit)
    distance = torch.linalg.vector_norm(coefficients / variances.sqrt(), dim=-1, keepdim=True)
    factor = radius / distance.clamp_min(radius)
    restricted = coefficients * factor
    before = distance.squeeze(-1).square() / coefficients.shape[-1]
    after = (restricted.square() / variances).mean(dim=-1)
    return restricted, before, after


def compute_pca_regularization_terms(
    predicted_landmarks: torch.Tensor,
    pca_prior: dict[str, Any],
    mahalanobis_limit: float = 2.0,
    variance_floor: float = 1e-4,
    coefficient_alpha: float | None = None,
    mahalanobis_reconstruction: bool = False,
) -> dict[str, torch.Tensor]:
    """Penalize off-subspace residuals and only excessive Mahalanobis distance.

    q = D_M^2 / K uses retained PCA variances with a relative eigenvalue floor.
    The Mahalanobis loss is relu(q / limit - 1)^2, so its gradient is zero
    throughout the accepted region. The subspace term is squared residual
    length divided by 2N coordinates (144 for 72 landmarks), matching the
    original projection MSE; it does not shrink coefficients.
    With coefficient_alpha set, instead return the MSE to the PCA reconstruction
    clipped to +/- alpha times each original standard deviation. Gradients flow
    through alignment, projection and clipping; no moving target is detached.
    With mahalanobis_reconstruction=True, radially restrict coefficients to
    D_M^2/K <= limit and use only MSE to that reconstruction, without a hinge.
    The limit is a tunable tolerance, not a calibrated probability percentile.
    """
    if mahalanobis_reconstruction and coefficient_alpha is not None:
        raise ValueError("Choose coefficient clipping or Mahalanobis reconstruction, not both.")
    if coefficient_alpha is not None and (not math.isfinite(coefficient_alpha) or coefficient_alpha <= 0):
        raise ValueError("coefficient_alpha must be finite and positive.")
    if not math.isfinite(mahalanobis_limit) or mahalanobis_limit <= 0:
        raise ValueError("mahalanobis_limit must be finite and positive.")
    if not math.isfinite(variance_floor) or not 0 < variance_floor <= 1:
        raise ValueError("variance_floor must be in (0, 1].")
    if predicted_landmarks.ndim != 3 or predicted_landmarks.shape[-1] != 2:
        raise ValueError(
            "Expected predicted_landmarks with shape (B, N, 2), "
            f"got {tuple(predicted_landmarks.shape)}."
        )

    predicted_landmarks = predicted_landmarks.to(dtype=torch.float32)

    alignment_config = pca_prior["alignment"]
    allow_reflection = bool(alignment_config.get("allow_reflection", False))
    eps = float(alignment_config.get("eps", PCA_ALIGNMENT_EPS))
    global_prior = pca_prior["global_prior"]
    expected_landmarks = int(global_prior["num_landmarks"])

    variances = global_prior["explained_variance"].detach().to(predicted_landmarks)
    if (
        variances.ndim != 1 or variances.numel() == 0
        or variances.numel() != global_prior["components"].shape[0]
        or not torch.isfinite(variances).all()
        or (variances < 0).any() or variances.max() <= 0
    ):
        raise ValueError("PCA prior must contain finite, nonnegative component variances with positive total variance.")
    safe_variances = variances.clamp_min(
        (variances.max() * variance_floor).clamp_min(torch.finfo(torch.float32).tiny)
    )
    residuals: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    bounded_residuals = []
    coefficient_residuals = []
    clipped_fractions = []
    clipped_shapes = []
    restricted_distances = []
    for sample_index in range(predicted_landmarks.shape[0]):
        current_shape = predicted_landmarks[sample_index]
        if current_shape.shape[0] != expected_landmarks:
            raise ValueError(
                f"Global PCA prior expects {expected_landmarks} landmarks, "
                f"got {current_shape.shape[0]}."
            )

        reference_shape = global_prior["reference_shape"].to(
            device=current_shape.device,
            dtype=current_shape.dtype,
        )
        aligned_shape = align_shape_for_regularization(
            coords=current_shape,
            reference_shape=reference_shape,
            allow_reflection=allow_reflection,
            eps=eps,
        )
        shape_vector = aligned_shape.reshape(1, -1)
        mean_shape = global_prior["mean_shape"].to(
            device=shape_vector.device,
            dtype=shape_vector.dtype,
        )
        components = global_prior["components"].to(
            device=shape_vector.device,
            dtype=shape_vector.dtype,
        )
        centered = shape_vector - mean_shape
        coefficients = centered @ components.T
        reconstructed = mean_shape + coefficients @ components
        if coefficient_alpha is not None or mahalanobis_reconstruction:
            if mahalanobis_reconstruction:
                bounded_coefficients, _, after = restrict_coefficients_mahalanobis(
                    coefficients, safe_variances, mahalanobis_limit,
                )
                restricted_distances.append(after.mean())
                clipped = coefficients != bounded_coefficients
            else:
                # Box bounds use original variances, including zero-width modes.
                bounds = coefficient_alpha * variances.sqrt()
                bounded_coefficients = torch.clamp(coefficients, min=-bounds, max=bounds)
                clipped = coefficients.abs() > bounds
            bounded_shape = mean_shape + bounded_coefficients @ components
            bounded_residuals.append((shape_vector - bounded_shape).square().mean())
            coefficient_residuals.append((coefficients - bounded_coefficients).square().sum() / (2 * expected_landmarks))
            clipped_fractions.append(clipped.float().mean())
            clipped_shapes.append(clipped.any().float())
        residuals.append((shape_vector - reconstructed).square().sum())
        distances.append((coefficients.square() / safe_variances).mean())

    residuals_tensor = torch.stack(residuals)
    distances_tensor = torch.stack(distances)
    subspace_loss = residuals_tensor.mean() / (2 * expected_landmarks)
    if coefficient_alpha is not None or mahalanobis_reconstruction:
        bounded_loss = torch.stack(bounded_residuals).mean()
        zero = bounded_loss.new_zeros(())
        return {
            "pca_loss": bounded_loss,
            "pca_bounded_loss": bounded_loss,
            "pca_coefficient_loss": torch.stack(coefficient_residuals).mean(),
            "pca_clipped_fraction": torch.stack(clipped_fractions).mean(),
            "pca_outside_fraction": torch.stack(clipped_shapes).mean(),
            "pca_subspace_loss": subspace_loss,
            "pca_projection_mse": subspace_loss,
            "pca_mahalanobis_loss": zero,
            "pca_mahalanobis_sq_per_component": distances_tensor.mean() if mahalanobis_reconstruction else zero,
            "pca_restricted_mahalanobis_sq_per_component": torch.stack(restricted_distances).mean() if mahalanobis_reconstruction else zero,
        }
    mahalanobis_loss = F.relu(distances_tensor / mahalanobis_limit - 1).square().mean()
    return {
        "pca_loss": subspace_loss + mahalanobis_loss,
        "pca_subspace_loss": subspace_loss,
        "pca_mahalanobis_loss": mahalanobis_loss,
        "pca_mahalanobis_sq_per_component": distances_tensor.mean(),
        "pca_outside_fraction": (distances_tensor > mahalanobis_limit).float().mean(),
        "pca_projection_mse": subspace_loss,
    }


def compute_pca_projection_loss(
    predicted_landmarks: torch.Tensor,
    pca_prior: dict[str, Any],
) -> torch.Tensor:
    """Legacy unnormalized projection MSE, retained for comparison only."""
    return compute_pca_regularization_terms(predicted_landmarks, pca_prior)["pca_projection_mse"]
