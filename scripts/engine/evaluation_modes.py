from __future__ import annotations

import numpy as np

from .geometry_metrics import compute_per_landmark_point_to_line_distances


def compute_masked_natural_per_landmark_nme(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    target_visibility: np.ndarray,
    predicted_visibility: np.ndarray,
    normalization_fn,
    inclusion_mode: str = "visible_intersection",
    eps: float = 1e-6,
) -> tuple[dict[int, float], dict[int, float], float | None, float | None]:
    """Compute natural-image NME under one landmark inclusion mode."""
    finite_target_mask = np.isfinite(target_landmarks[:, 0]) & np.isfinite(
        target_landmarks[:, 1]
    )
    finite_prediction_mask = np.isfinite(predicted_landmarks[:, 0]) & np.isfinite(
        predicted_landmarks[:, 1]
    )
    if inclusion_mode == "visible_intersection":
        normalization_mask = (target_visibility == 1) & finite_target_mask
        valid_mask = (
            normalization_mask & (predicted_visibility == 1) & finite_prediction_mask
        )
    elif inclusion_mode == "gt_valid":
        normalization_mask = finite_target_mask
        valid_mask = finite_target_mask & finite_prediction_mask
    else:
        raise ValueError(f"Unsupported natural evaluation mode: {inclusion_mode}")

    if normalization_mask.sum() == 0:
        return {}, {}, None, None

    safe_predicted_landmarks = np.nan_to_num(
        predicted_landmarks.astype(np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_target_landmarks = np.nan_to_num(
        target_landmarks.astype(np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    normalization = normalization_fn(
        target_landmarks=safe_target_landmarks[normalization_mask],
        eps=eps,
    )
    point_errors = np.linalg.norm(
        safe_predicted_landmarks - safe_target_landmarks, axis=1
    )
    normalized_errors = point_errors / normalization
    point_to_line_errors = (
        compute_per_landmark_point_to_line_distances(
            predicted_landmarks=safe_predicted_landmarks,
            target_landmarks=safe_target_landmarks,
        )
        / normalization
    )
    per_landmark_errors = {
        int(landmark_index): float(normalized_errors[landmark_index])
        for landmark_index in np.flatnonzero(valid_mask)
    }
    per_landmark_point_to_line_errors = {
        int(landmark_index): float(point_to_line_errors[landmark_index])
        for landmark_index in np.flatnonzero(valid_mask)
    }
    if not per_landmark_errors:
        return {}, {}, None, None
    return (
        per_landmark_errors,
        per_landmark_point_to_line_errors,
        float(np.mean(list(per_landmark_errors.values()))),
        float(np.mean(list(per_landmark_point_to_line_errors.values()))),
    )
