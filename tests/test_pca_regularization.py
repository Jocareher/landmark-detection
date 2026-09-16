from __future__ import annotations

from copy import deepcopy
import csv

import pytest
import torch

from scripts.engine.landmark_losses import MeanSquaredHeatmapLoss
from scripts.engine.losses import compute_multitask_loss
from scripts.engine.metrics import decode_heatmaps_to_image_coords
from scripts.engine.pca_shape_prior import (
    align_shape_for_regularization,
    align_shape_to_reference_torch,
    build_global_pca_shape_prior_payload,
    compute_pca_projection_loss,
    compute_pca_regularization_terms,
    decode_landmarks_for_shape_loss,
    load_pca_shape_prior,
    restrict_coefficients_mahalanobis,
)


@pytest.fixture
def shapes_and_prior():
    generator = torch.Generator().manual_seed(42)
    base = torch.tensor([[-2., -1.], [0., -1.5], [2., -1.], [-1., 1.], [1., 1.], [0., 2.]])
    shapes = base + 0.15 * torch.randn(30, 6, 2, generator=generator)
    prior, _ = build_global_pca_shape_prior_payload(shapes, num_components=4)
    return shapes, prior


def test_accepted_nonmean_shape_has_no_mahalanobis_force(shapes_and_prior):
    shapes, prior = shapes_and_prior
    prediction = shapes[:1].clone().requires_grad_()
    distance = compute_pca_regularization_terms(prediction, prior)["pca_mahalanobis_sq_per_component"]
    assert distance > 0  # This is not the mean shape.
    terms = compute_pca_regularization_terms(prediction, prior, mahalanobis_limit=2 * distance.item())
    terms["pca_mahalanobis_loss"].backward()
    assert terms["pca_mahalanobis_loss"] == 0
    assert terms["pca_outside_fraction"] == 0
    assert torch.count_nonzero(prediction.grad) == 0


def test_outlier_receives_gradient_that_reduces_mahalanobis_distance(shapes_and_prior):
    shapes, prior = shapes_and_prior
    prediction = shapes[:1].clone().requires_grad_()
    initial = compute_pca_regularization_terms(prediction, prior)
    limit = initial["pca_mahalanobis_sq_per_component"].item() / 2
    terms = compute_pca_regularization_terms(prediction, prior, mahalanobis_limit=limit)
    terms["pca_mahalanobis_loss"].backward()
    assert terms["pca_outside_fraction"] == 1
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.norm() > 0
    corrected = prediction.detach() - 0.001 * prediction.grad / prediction.grad.norm()
    after = compute_pca_regularization_terms(corrected, prior, mahalanobis_limit=limit)
    assert after["pca_mahalanobis_sq_per_component"] < terms["pca_mahalanobis_sq_per_component"]


def test_subspace_residual_still_detects_errors_inside_mahalanobis_region(shapes_and_prior):
    shapes, prior = shapes_and_prior
    terms = compute_pca_regularization_terms(shapes[:1], prior, mahalanobis_limit=1e6)
    assert terms["pca_mahalanobis_loss"] == 0
    assert terms["pca_subspace_loss"] > 0
    torch.testing.assert_close(terms["pca_loss"], terms["pca_subspace_loss"])
    torch.testing.assert_close(terms["pca_projection_mse"], compute_pca_projection_loss(shapes[:1], prior))


def test_variance_weights_and_floor(shapes_and_prior):
    shapes, prior = shapes_and_prior
    original = compute_pca_regularization_terms(shapes[:1], prior)
    broader = deepcopy(prior)
    broader["global_prior"]["explained_variance"] *= 4
    relaxed = compute_pca_regularization_terms(shapes[:1], broader)
    torch.testing.assert_close(
        relaxed["pca_mahalanobis_sq_per_component"],
        original["pca_mahalanobis_sq_per_component"] / 4,
    )
    torch.testing.assert_close(relaxed["pca_subspace_loss"], original["pca_subspace_loss"])
    prior["global_prior"]["explained_variance"][-1] = 0
    prediction = shapes[:1].clone().requires_grad_()
    terms = compute_pca_regularization_terms(prediction, prior)
    terms["pca_loss"].backward()
    assert all(torch.isfinite(value) for value in terms.values())
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("allow_reflection", [False, True])
def test_training_alignment_matches_prior_alignment(shapes_and_prior, allow_reflection):
    shapes, prior = shapes_and_prior
    reference = prior["global_prior"]["reference_shape"]
    for shape in (shapes[0], shapes[0] * torch.tensor([-1., 1.])):
        expected = align_shape_to_reference_torch(shape, reference, allow_reflection=allow_reflection)
        actual = align_shape_for_regularization(shape, reference, allow_reflection=allow_reflection)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_loss_is_invariant_to_translation_scale_and_inplane_rotation(shapes_and_prior):
    shapes, prior = shapes_and_prior
    rotation = torch.tensor([[0.6, 0.8], [-0.8, 0.6]])
    transformed = 7 * (shapes @ rotation) + torch.tensor([31., -9.])
    original = compute_pca_regularization_terms(shapes, prior)
    moved = compute_pca_regularization_terms(transformed, prior)
    for key in original:
        torch.testing.assert_close(original[key], moved[key], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("decoder", ["argmax_subpixel", "barycenter"])
def test_shape_loss_uses_final_coordinates_with_nonzero_heatmap_gradients(decoder):
    heatmaps = torch.randn(2, 6, 8, 8, generator=torch.Generator().manual_seed(2), requires_grad=True)
    coords = decode_landmarks_for_shape_loss(heatmaps, 64, 80, decoder=decoder, temperature=0.7)
    expected = decode_heatmaps_to_image_coords(heatmaps, 64, 80, decoder=decoder, softmax_temperature=0.7)
    torch.testing.assert_close(coords, expected, rtol=0, atol=0)
    coords.square().mean().backward()
    assert torch.isfinite(heatmaps.grad).all()
    assert heatmaps.grad.norm() > 0


@pytest.mark.parametrize("decoder", ["argmax_subpixel", "barycenter"])
def test_regularizer_alone_backpropagates_through_heatmaps(shapes_and_prior, decoder):
    _, prior = shapes_and_prior
    heatmaps = torch.randn(2, 6, 8, 8, generator=torch.Generator().manual_seed(12), requires_grad=True)
    result = compute_multitask_loss(
        pca_regularization="mahalanobis",
        outputs={"heatmaps": heatmaps, "visible_heatmaps": heatmaps, "visibility_logits": torch.zeros(2, 6)},
        batch={"heatmaps": torch.zeros_like(heatmaps), "visibility": torch.ones(2, 6)},
        heatmap_loss_fn=MeanSquaredHeatmapLoss(), visibility_loss_fn=torch.nn.BCEWithLogitsLoss(),
        lambda_vis=0, lambda_lmk_vis=0, lambda_lmk_full=0,
        lambda_pca_projection=0.1, pca_shape_prior=prior,
        image_height=64, image_width=64, coordinate_decoder=decoder,
        wasserstein_softmax_temperature=0.7,
    )
    result["total_loss"].backward()
    assert result["pca_loss"] > 0
    assert torch.isfinite(heatmaps.grad).all()
    assert heatmaps.grad.norm() > 0


@pytest.mark.parametrize("decoder", ["argmax_subpixel", "barycenter"])
def test_initially_flat_heatmaps_have_finite_loss_and_gradients(shapes_and_prior, decoder):
    _, prior = shapes_and_prior
    heatmaps = torch.zeros(1, 6, 8, 8, requires_grad=True)
    coords = decode_landmarks_for_shape_loss(heatmaps, 64, 64, decoder=decoder)
    terms = compute_pca_regularization_terms(coords, prior)
    terms["pca_loss"].backward()
    assert torch.isfinite(terms["pca_loss"])
    assert torch.isfinite(heatmaps.grad).all()


def test_existing_prior_format_loads_without_rebuilding(shapes_and_prior, tmp_path):
    shapes, prior = shapes_and_prior
    path = tmp_path / "prior.pt"
    torch.save(prior, path)
    loaded = load_pca_shape_prior(path, device="cpu")
    assert torch.isfinite(compute_pca_regularization_terms(shapes, loaded)["pca_loss"])


def test_72_landmark_residual_is_original_mse_divided_by_144():
    generator = torch.Generator().manual_seed(17)
    shapes = torch.randn(8, 72, 2, generator=generator)
    prior, _ = build_global_pca_shape_prior_payload(shapes, num_components=3)
    predictions = torch.randn(2, 72, 2, generator=generator, requires_grad=True)
    global_prior = prior["global_prior"]
    aligned = torch.stack([
        align_shape_to_reference_torch(shape, global_prior["reference_shape"])
        for shape in predictions
    ]).flatten(1)
    centered = aligned - global_prior["mean_shape"]
    reconstruction = global_prior["mean_shape"] + (centered @ global_prior["components"].T) @ global_prior["components"]
    expected = (aligned - reconstruction).square().sum(dim=1).mean() / 144
    actual = compute_pca_regularization_terms(predictions, prior)["pca_subspace_loss"]
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, predictions)[0]
    actual_grad = torch.autograd.grad(actual, predictions)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("alpha,beta", [(1., 0.), (0., 0.1), (0.3, 0.002), (0., 0.)])
def test_independent_weights_and_logged_contribution(shapes_and_prior, alpha, beta):
    _, prior = shapes_and_prior
    heatmaps = torch.randn(1, 6, 8, 8, generator=torch.Generator().manual_seed(8), requires_grad=True)
    coords = decode_landmarks_for_shape_loss(heatmaps, 64, 64, decoder="barycenter")
    terms = compute_pca_regularization_terms(coords, prior)
    result = compute_multitask_loss(
        pca_regularization="mahalanobis",
        outputs={"heatmaps": heatmaps, "visible_heatmaps": heatmaps, "visibility_logits": torch.zeros(1, 6)},
        batch={"heatmaps": torch.zeros_like(heatmaps), "visibility": torch.ones(1, 6)},
        heatmap_loss_fn=MeanSquaredHeatmapLoss(), visibility_loss_fn=torch.nn.BCEWithLogitsLoss(),
        lambda_vis=0, lambda_lmk_vis=0, lambda_lmk_full=0,
        lambda_pca_projection=alpha, lambda_pca_mahalanobis=beta,
        pca_shape_prior=prior if alpha or beta else None,
        image_height=64, image_width=64, coordinate_decoder="barycenter",
    )
    expected = alpha * terms["pca_subspace_loss"] + beta * terms["pca_mahalanobis_loss"]
    torch.testing.assert_close(result["total_loss"], expected)
    torch.testing.assert_close(result["pca_loss"], expected)
    expected_grad = torch.autograd.grad(expected, heatmaps, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(result["total_loss"], heatmaps)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_mahalanobis_only_requires_prior():
    with pytest.raises(ValueError, match="requires a PCA prior"):
        compute_multitask_loss(
        pca_regularization="mahalanobis",
            outputs={"heatmaps": torch.zeros(1, 6, 8, 8), "visible_heatmaps": torch.zeros(1, 6, 8, 8), "visibility_logits": torch.zeros(1, 6)},
            batch={"heatmaps": torch.zeros(1, 6, 8, 8), "visibility": torch.ones(1, 6)},
            heatmap_loss_fn=MeanSquaredHeatmapLoss(), visibility_loss_fn=torch.nn.BCEWithLogitsLoss(),
            lambda_pca_mahalanobis=0.1,
        )


@pytest.mark.parametrize("options", [
    {"mahalanobis_limit": 0}, {"mahalanobis_limit": float("nan")},
    {"variance_floor": 0}, {"variance_floor": 2},
])
def test_invalid_tolerances_are_rejected(shapes_and_prior, options):
    shapes, prior = shapes_and_prior
    with pytest.raises(ValueError):
        compute_pca_regularization_terms(shapes, prior, **options)


@pytest.mark.parametrize("mode", ["mahalanobis", "bounded_reconstruction", "mahalanobis_reconstruction"])
def test_training_wires_tolerances_and_logs_metrics_without_pca_at_inference(shapes_and_prior, tmp_path, mode):
    from scripts.engine.inference import run_inference
    from scripts.engine.train import train_model

    _, prior = shapes_and_prior
    prior_path = tmp_path / "global_prior.pt"
    torch.save(prior, prior_path)

    class HeatmapModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.heatmaps = torch.nn.Parameter(torch.randn(1, 6, 8, 8, generator=torch.Generator().manual_seed(18)))

        def forward(self, images):
            return {
                "heatmaps": self.heatmaps.expand(images.shape[0], -1, -1, -1),
                "visible_heatmaps": self.heatmaps.expand(images.shape[0], -1, -1, -1),
                "visibility_logits": self.heatmaps.mean(dim=(-1, -2)).expand(images.shape[0], -1),
            }

    model = HeatmapModel()
    batch = {"image": torch.zeros(1, 3, 64, 64), "heatmaps": torch.zeros(1, 6, 8, 8), "visibility": torch.ones(1, 6)}
    coords = decode_landmarks_for_shape_loss(model.heatmaps, 64, 64, decoder="barycenter", temperature=0.7)
    expected = compute_pca_regularization_terms(coords, prior, mahalanobis_limit=0.3, variance_floor=0.01, coefficient_alpha=0.2 if mode == "bounded_reconstruction" else None, mahalanobis_reconstruction=mode == "mahalanobis_reconstruction")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    summary = train_model(
        pca_regularization=mode, pca_coefficient_alpha=0.2,
        model=model, train_loader=[batch], val_loader=[batch], optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        heatmap_loss_fn=MeanSquaredHeatmapLoss(), visibility_loss_fn=torch.nn.BCEWithLogitsLoss(),
        device=torch.device("cpu"), num_epochs=1, output_dir=tmp_path / "run",
        lambda_pca_projection=0.01, lambda_pca_mahalanobis=0.003 if mode == "mahalanobis" else 0, pca_prior_path=prior_path,
        pca_mahalanobis_limit=0.3, pca_variance_floor=0.01,
        coordinate_decoder="barycenter", wasserstein_softmax_temperature=0.7,
        use_wandb=False, use_amp=False, visualize_every_n_epochs=0,
    )
    assert summary["history"]["train"][0]["pca_loss"] == pytest.approx(0.01 * (expected["pca_bounded_loss"] if mode != "mahalanobis" else expected["pca_subspace_loss"]).item() + 0.003 * expected["pca_mahalanobis_loss"].item())
    with open(summary["results_csv"], newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert float(rows[0]["pca_mahalanobis_sq_per_component"]) == pytest.approx(expected["pca_mahalanobis_sq_per_component"].item())
    prior_path.unlink()  # Inference must be independent of the training prior.
    predictions = run_inference(model, [batch], torch.device("cpu"), compute_nme=False, coordinate_decoder="barycenter", wasserstein_softmax_temperature=0.7)
    torch.testing.assert_close(
        predictions["predictions"],
        decode_heatmaps_to_image_coords(model.heatmaps.detach(), 64, 64, decoder="barycenter", softmax_temperature=0.7),
    )


def test_bounded_loss_decomposes_and_does_not_shrink_accepted_coefficients(shapes_and_prior):
    shapes, prior = shapes_and_prior
    prediction = shapes[:1].clone().requires_grad_()
    inside = compute_pca_regularization_terms(prediction, prior, coefficient_alpha=1000)
    assert inside['pca_coefficient_loss'] == 0
    assert inside['pca_clipped_fraction'] == 0
    torch.testing.assert_close(inside['pca_bounded_loss'], inside['pca_projection_mse'])
    grad = torch.autograd.grad(inside['pca_coefficient_loss'], prediction)[0]
    assert torch.count_nonzero(grad) == 0
    outside = compute_pca_regularization_terms(prediction, prior, coefficient_alpha=0.01)
    assert outside['pca_coefficient_loss'] > 0
    assert outside['pca_outside_fraction'] == 1
    torch.testing.assert_close(outside['pca_bounded_loss'], outside['pca_subspace_loss'] + outside['pca_coefficient_loss'])
    assert outside['pca_mahalanobis_loss'] == 0
    outside['pca_loss'].backward()
    assert torch.isfinite(prediction.grad).all()


def test_bounded_loss_matches_explicit_clipped_reconstruction(shapes_and_prior):
    shapes, prior = shapes_and_prior
    prediction = shapes[:2].clone().requires_grad_()
    g = prior['global_prior']
    s = torch.stack([align_shape_for_regularization(x, g['reference_shape']) for x in prediction]).flatten(1)
    b = (s - g['mean_shape']) @ g['components'].T
    bounds = 0.2 * g['explained_variance'].sqrt()
    clipped = b.maximum(-bounds).minimum(bounds)
    target = g['mean_shape'] + clipped @ g['components']
    expected = (s - target).square().mean()
    actual = compute_pca_regularization_terms(prediction, prior, coefficient_alpha=0.2)['pca_bounded_loss']
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, prediction)[0], torch.autograd.grad(expected, prediction)[0])


@pytest.mark.parametrize('alpha', [0, -1, float('nan'), float('inf')])
def test_bounded_invalid_alpha(shapes_and_prior, alpha):
    shapes, prior = shapes_and_prior
    with pytest.raises(ValueError, match='coefficient_alpha'):
        compute_pca_regularization_terms(shapes, prior, coefficient_alpha=alpha)


def test_bounded_flat_heatmaps_and_zero_variance_have_finite_gradients(shapes_and_prior):
    _, prior = shapes_and_prior
    prior['global_prior']['explained_variance'][-1] = 0
    heatmaps = torch.zeros(1, 6, 8, 8, requires_grad=True)
    c = decode_landmarks_for_shape_loss(heatmaps, 64, 64, decoder='barycenter')
    loss = compute_pca_regularization_terms(c, prior, coefficient_alpha=3)['pca_bounded_loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(heatmaps.grad).all()


def test_radial_mahalanobis_restriction_preserves_inside_and_bounds_outside():
    b = torch.tensor([[0., 0.], [0.2, -0.3], [10., -12.]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([1., 4.], dtype=torch.float64)
    restricted, before, after = restrict_coefficients_mahalanobis(b, v, limit=2.)
    torch.testing.assert_close(restricted[:2], b[:2], rtol=0, atol=0)
    assert before[-1] > 2
    torch.testing.assert_close(after[-1], torch.tensor(2., dtype=torch.float64))
    assert (after <= 2. + 1e-12).all()
    ratios = restricted[-1] / b[-1]
    torch.testing.assert_close(ratios[0], ratios[1])
    grad = torch.autograd.grad((b[:2] - restricted[:2]).square().sum(), b)[0]
    assert torch.count_nonzero(grad) == 0


def test_radial_reconstruction_gradient_matches_finite_differences():
    b = torch.tensor([[4., -3.]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([1., 0.1], dtype=torch.float64)
    def loss(coefficients):
        restricted, _, _ = restrict_coefficients_mahalanobis(coefficients, v, 2.)
        return (coefficients - restricted).square().mean()
    assert torch.autograd.gradcheck(loss, (b,))


def test_mahalanobis_reconstruction_matches_reference_and_decomposes(shapes_and_prior):
    shapes, prior = shapes_and_prior
    prediction = shapes[:2].clone().requires_grad_()
    g = prior['global_prior']
    aligned = torch.stack([align_shape_for_regularization(x, g['reference_shape']) for x in prediction]).flatten(1)
    b = (aligned - g['mean_shape']) @ g['components'].T
    v = g['explained_variance'].clamp_min(g['explained_variance'].max() * 1e-4)
    q = (b.square() / v).mean(-1, keepdim=True)
    restricted = b * torch.sqrt(0.01 / q.clamp_min(0.01))
    expected = (aligned - (g['mean_shape'] + restricted @ g['components'])).square().mean()
    terms = compute_pca_regularization_terms(prediction, prior, mahalanobis_limit=0.01, mahalanobis_reconstruction=True)
    torch.testing.assert_close(terms['pca_bounded_loss'], expected)
    torch.testing.assert_close(terms['pca_bounded_loss'], terms['pca_subspace_loss'] + terms['pca_coefficient_loss'])
    assert terms['pca_mahalanobis_loss'] == 0
    assert terms['pca_restricted_mahalanobis_sq_per_component'] <= 0.010001
    terms['pca_loss'].backward()
    assert torch.isfinite(prediction.grad).all()


def test_mahalanobis_reconstruction_flat_heatmaps_and_zero_variance(shapes_and_prior):
    _, prior = shapes_and_prior
    prior['global_prior']['explained_variance'][-1] = 0
    heatmaps = torch.zeros(1, 6, 8, 8, requires_grad=True)
    coords = decode_landmarks_for_shape_loss(heatmaps, 64, 64, decoder='barycenter')
    terms = compute_pca_regularization_terms(coords, prior, mahalanobis_reconstruction=True)
    terms['pca_loss'].backward()
    assert all(torch.isfinite(x) for x in terms.values())
    assert torch.isfinite(heatmaps.grad).all()


def test_cannot_mix_box_and_ellipsoid_restrictions(shapes_and_prior):
    shapes, prior = shapes_and_prior
    with pytest.raises(ValueError, match='not both'):
        compute_pca_regularization_terms(shapes, prior, coefficient_alpha=3, mahalanobis_reconstruction=True)
