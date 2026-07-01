from __future__ import annotations

import math

import pytest
import torch

from scripts.engine.inference import apply_optional_pca_inference_correction
from scripts.engine.pca_shape_prior import (
    build_global_pca_shape_prior_payload,
    project_landmarks_with_pca,
)


def _make_training_shapes() -> torch.Tensor:
    theta = torch.linspace(0.0, 2.0 * math.pi, 72, dtype=torch.float32)[:-1]
    theta = torch.cat([theta, torch.tensor([2.0 * math.pi], dtype=torch.float32)])
    base = torch.stack([torch.cos(theta), 0.7 * torch.sin(theta)], dim=1)
    mode = torch.stack([0.08 * torch.sin(2.0 * theta), 0.05 * torch.cos(3.0 * theta)], dim=1)
    coeffs = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)
    return torch.stack([base + coeff * mode for coeff in coeffs], dim=0)


def _make_prior() -> tuple[dict[str, object], torch.Tensor]:
    prior_payload, aligned_shapes = build_global_pca_shape_prior_payload(
        landmarks=_make_training_shapes(),
        num_components=2,
    )
    return prior_payload, aligned_shapes


def test_pca_projection_preserves_shape_device_dtype_and_size() -> None:
    prior_payload, aligned_shapes = _make_prior()
    predicted = (aligned_shapes[:2] * 120.0 + 80.0).to(dtype=torch.float64)

    corrected = project_landmarks_with_pca(
        predicted_landmarks=predicted,
        pca_prior=prior_payload,
        num_components=2,
        alpha=1.0,
    )

    assert corrected.shape == predicted.shape
    assert corrected.device == predicted.device
    assert corrected.dtype == predicted.dtype
    assert torch.isfinite(corrected).all()


def test_pca_projection_alpha_zero_returns_original_landmarks() -> None:
    prior_payload, aligned_shapes = _make_prior()
    predicted = aligned_shapes[:1] * 100.0 + 50.0

    corrected = project_landmarks_with_pca(
        predicted_landmarks=predicted,
        pca_prior=prior_payload,
        num_components=1,
        alpha=0.0,
    )

    assert torch.equal(corrected, predicted)


def test_pca_projection_alpha_one_is_full_reconstruction() -> None:
    prior_payload, aligned_shapes = _make_prior()
    predicted = aligned_shapes[:1] * 100.0 + 50.0

    hard_corrected = project_landmarks_with_pca(
        predicted_landmarks=predicted,
        pca_prior=prior_payload,
        num_components=1,
        alpha=1.0,
    )
    soft_corrected = project_landmarks_with_pca(
        predicted_landmarks=predicted,
        pca_prior=prior_payload,
        num_components=1,
        alpha=0.5,
    )

    assert torch.allclose(soft_corrected, 0.5 * predicted + 0.5 * hard_corrected)


def test_pca_projection_validates_alpha_and_component_count() -> None:
    prior_payload, aligned_shapes = _make_prior()
    predicted = aligned_shapes[:1]

    with pytest.raises(ValueError, match="alpha"):
        project_landmarks_with_pca(predicted, prior_payload, alpha=-0.1)
    with pytest.raises(ValueError, match="at least 1"):
        project_landmarks_with_pca(predicted, prior_payload, num_components=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        project_landmarks_with_pca(predicted, prior_payload, num_components=99)


def test_disabled_pca_inference_adapter_leaves_predictions_unchanged() -> None:
    predicted = torch.rand(2, 72, 2)

    corrected, stats = apply_optional_pca_inference_correction(
        predicted_landmarks=predicted,
        apply_pca_inference=False,
    )

    assert corrected is predicted
    assert stats == {"mean_displacement": 0.0, "max_displacement": 0.0}


def test_invalid_prediction_sample_is_preserved() -> None:
    prior_payload, aligned_shapes = _make_prior()
    predicted = aligned_shapes[:2].clone()
    predicted[1, 0, 0] = float("nan")

    with pytest.warns(RuntimeWarning, match="skipped sample 1"):
        corrected = project_landmarks_with_pca(
            predicted_landmarks=predicted,
            pca_prior=prior_payload,
            num_components=1,
            alpha=1.0,
        )

    assert torch.allclose(corrected[1], predicted[1], equal_nan=True)


def test_projection_inverse_alignment_preserves_in_subspace_shape() -> None:
    prior_payload, aligned_shapes = _make_prior()
    predicted = aligned_shapes[1:2]

    corrected = project_landmarks_with_pca(
        predicted_landmarks=predicted,
        pca_prior=prior_payload,
        num_components=2,
        alpha=1.0,
    )

    assert torch.allclose(corrected, predicted, atol=5e-4, rtol=5e-4)
