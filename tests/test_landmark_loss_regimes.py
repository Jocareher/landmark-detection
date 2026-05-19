from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.engine.landmark_losses import (
    AdaptiveWingLoss,
    WassersteinHeatmapLoss,
    build_landmark_heatmap_loss,
    compute_masked_heatmap_loss,
)
from scripts.engine.losses import compute_multitask_loss
from scripts.engine.metrics import (
    decode_heatmaps_to_image_coords,
    decoder_from_landmark_loss,
    normalize_heatmaps_to_probabilities,
    spatial_softmax_2d,
)


def _config(landmark_loss: str) -> SimpleNamespace:
    return SimpleNamespace(
        landmark_loss=landmark_loss,
        adaptive_wing_omega=14.0,
        adaptive_wing_theta=0.5,
        adaptive_wing_epsilon=1.0,
        adaptive_wing_alpha=2.1,
        wasserstein_softmax_temperature=1.0,
        wasserstein_epsilon=1e-8,
        wasserstein_validate_normalization=True,
    )


def test_decoder_selection() -> None:
    assert decoder_from_landmark_loss("mse") == "argmax_subpixel"
    assert decoder_from_landmark_loss("adaptive_wing") == "argmax_subpixel"
    assert decoder_from_landmark_loss("wasserstein") == "barycenter"


def test_adaptive_wing_loss_is_finite_for_full_and_visible_branches() -> None:
    predicted = torch.randn(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)
    visibility = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    loss_fn = AdaptiveWingLoss()

    full_loss = compute_masked_heatmap_loss(loss_fn, predicted, target)
    visible_loss = compute_masked_heatmap_loss(loss_fn, predicted, target, visibility)

    assert torch.isfinite(full_loss)
    assert torch.isfinite(visible_loss)


def test_wasserstein_loss_normalization_and_barycenter_decoder() -> None:
    predicted = torch.randn(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)
    visibility = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    loss_fn = WassersteinHeatmapLoss(validate_normalization=True)

    predicted_probabilities = spatial_softmax_2d(predicted)
    target_probabilities = normalize_heatmaps_to_probabilities(target)
    assert torch.allclose(
        predicted_probabilities.flatten(start_dim=2).sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-5,
    )
    assert torch.allclose(
        target_probabilities.flatten(start_dim=2).sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-5,
    )

    full_loss = compute_masked_heatmap_loss(loss_fn, predicted, target)
    visible_loss = compute_masked_heatmap_loss(loss_fn, predicted, target, visibility)
    decoded = decode_heatmaps_to_image_coords(
        predicted,
        image_height=64,
        image_width=64,
        decoder="barycenter",
    )

    assert torch.isfinite(full_loss)
    assert torch.isfinite(visible_loss)
    assert decoded.shape == (2, 3, 2)
    assert torch.isfinite(decoded).all()


def test_loss_factory_builds_all_regimes() -> None:
    for landmark_loss in ("mse", "adaptive_wing", "wasserstein"):
        loss_fn = build_landmark_heatmap_loss(_config(landmark_loss))
        predicted = torch.randn(1, 2, 8, 8)
        target = torch.rand(1, 2, 8, 8)
        loss = compute_masked_heatmap_loss(loss_fn, predicted, target)
        assert torch.isfinite(loss)


def test_multitask_loss_accepts_all_landmark_loss_regimes() -> None:
    outputs = {
        "heatmaps": torch.randn(2, 3, 8, 8),
        "visible_heatmaps": torch.randn(2, 3, 8, 8),
        "visibility_logits": torch.randn(2, 3),
    }
    batch = {
        "heatmaps": torch.rand(2, 3, 8, 8),
        "visibility": torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32),
    }
    visibility_loss = torch.nn.BCEWithLogitsLoss()

    for landmark_loss in ("mse", "adaptive_wing", "wasserstein"):
        heatmap_loss = build_landmark_heatmap_loss(_config(landmark_loss))
        loss_dict = compute_multitask_loss(
            outputs=outputs,
            batch=batch,
            heatmap_loss_fn=heatmap_loss,
            visibility_loss_fn=visibility_loss,
        )
        assert torch.isfinite(loss_dict["total_loss"])
        assert torch.isfinite(loss_dict["full_landmark_loss"])
        assert torch.isfinite(loss_dict["visible_landmark_loss"])
