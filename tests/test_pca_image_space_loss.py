"""Image-space reconstruction and gradients through the complete similarity map."""

import torch
import pytest
import torch.nn.functional as F

from scripts.engine.pca_shape_prior import (
    apply_similarity_transform_torch,
    compute_pca_projection_loss,
    estimate_similarity_transform_torch,
    invert_similarity_transform_torch,
    align_shape_to_reference_torch,
)


def _example():
    reference = torch.tensor([[-2., -1.], [1., -1.], [2., 0.5], [0., 2.], [-1., 0.]])
    reference -= reference.mean(0)
    reference /= torch.linalg.vector_norm(reference)
    component = torch.tensor([[1., 0., -1., 0., 0., 1., 0., -1., 0., 0.]])
    component /= torch.linalg.vector_norm(component)
    prior = {
        'alignment': {'method': 'procrustes', 'allow_reflection': False, 'eps': 1e-6},
        'global_prior': {'num_landmarks': 5, 'reference_shape': reference,
                         'mean_shape': reference.reshape(1, -1), 'components': component},
    }
    predicted = reference * 30 + torch.tensor([60., 80.])
    predicted[2] += torch.tensor([6., -3.])
    return predicted.unsqueeze(0), prior


def test_inverse_and_image_space_loss_identity():
    predicted, prior = _example()
    x = predicted[0]
    pca = prior['global_prior']
    rotation, scale, translation = estimate_similarity_transform_torch(x, pca['reference_shape'])
    z = apply_similarity_transform_torch(x, rotation, scale, translation)
    torch.testing.assert_close(invert_similarity_transform_torch(z, rotation, scale, translation), x)
    coefficients = (z.reshape(1, -1) - pca['mean_shape']) @ pca['components'].T
    reconstruction = pca['mean_shape'] + coefficients @ pca['components']
    aligned_loss = F.mse_loss(z.reshape(1, -1), reconstruction)
    loss = compute_pca_projection_loss(predicted, prior)
    torch.testing.assert_close(loss, aligned_loss / scale.square(), rtol=1e-5, atol=1e-5)
    assert loss > aligned_loss


def test_loss_scales_quadratically_and_averages_samples():
    predicted, prior = _example()
    loss = compute_pca_projection_loss(predicted, prior)
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]])
    transformed = 2 * (predicted @ rotation) + torch.tensor([9., -20.])
    transformed_loss = compute_pca_projection_loss(transformed, prior)
    torch.testing.assert_close(transformed_loss, 4 * loss, rtol=1e-4, atol=1e-5)
    batch_loss = compute_pca_projection_loss(torch.cat([predicted, transformed]), prior)
    torch.testing.assert_close(batch_loss, (loss + transformed_loss) / 2)


def test_complete_gradient_includes_prediction_scale():
    predicted, prior = _example()
    multiplier = torch.tensor(1., requires_grad=True)
    loss = compute_pca_projection_loss(multiplier * predicted, prior)
    gradient, = torch.autograd.grad(loss, multiplier)
    # L(a X) = a^2 L(X); detaching inverse scale would miss this derivative.
    torch.testing.assert_close(gradient, 2 * loss.detach(), rtol=2e-4, atol=2e-4)
    predicted = predicted.requires_grad_()
    loss = compute_pca_projection_loss(predicted, prior)
    gradient, = torch.autograd.grad(loss, predicted)
    step = 0.01
    offset = torch.zeros_like(predicted)
    offset[0, 2, 0] = step
    numerical = (compute_pca_projection_loss(predicted.detach() + offset, prior)
                 - compute_pca_projection_loss(predicted.detach() - offset, prior)) / (2 * step)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[0, 2, 0], numerical, rtol=3e-3, atol=1e-3)


def test_exactly_representable_shape_has_zero_loss():
    _, prior = _example()
    predicted = prior['global_prior']['reference_shape'].unsqueeze(0) * 30 + 50
    loss = compute_pca_projection_loss(predicted, prior)
    assert loss < 1e-9


def test_original_loss_matches_historical_formula_and_is_similarity_invariant():
    predicted, prior = _example()
    pca = prior['global_prior']
    aligned = align_shape_to_reference_torch(predicted[0], pca['reference_shape']).reshape(1, -1)
    reconstruction = pca['mean_shape'] + ((aligned - pca['mean_shape']) @ pca['components'].T) @ pca['components']
    original_loss = compute_pca_projection_loss(predicted, prior, loss_space='aligned')
    torch.testing.assert_close(original_loss, F.mse_loss(aligned, reconstruction), rtol=0, atol=0)
    rotation = torch.tensor([[0.8, -0.6], [0.6, 0.8]])
    transformed = 3 * (predicted @ rotation) + torch.tensor([11., -6.])
    torch.testing.assert_close(
        compute_pca_projection_loss(transformed, prior, loss_space='aligned'),
        original_loss, rtol=1e-4, atol=1e-7,
    )
    batch_loss = compute_pca_projection_loss(torch.cat([predicted, transformed]), prior, loss_space='aligned')
    torch.testing.assert_close(batch_loss, original_loss, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize('space', ['aligned', 'image'])
def test_both_loss_modes_have_finite_nonzero_gradients(space):
    predicted, prior = _example()
    predicted.requires_grad_()
    loss = compute_pca_projection_loss(predicted, prior, loss_space=space)
    gradient, = torch.autograd.grad(loss, predicted)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_invalid_pca_loss_space_is_rejected():
    predicted, prior = _example()
    with pytest.raises(ValueError, match='PCA loss space'):
        compute_pca_projection_loss(predicted, prior, loss_space='unknown')


@pytest.mark.parametrize('space', ['aligned', 'image'])
def test_multitask_loss_uses_selected_pca_mode(space, monkeypatch):
    from scripts.engine import losses
    from scripts.engine.landmark_losses import MeanSquaredHeatmapLoss

    predicted, prior = _example()
    monkeypatch.setattr(losses, 'softargmax_heatmaps_to_image_coords', lambda **kwargs: predicted)
    heatmaps = torch.randn(1, 5, 8, 8)
    outputs = {'heatmaps': heatmaps, 'visible_heatmaps': heatmaps,
               'visibility_logits': torch.zeros(1, 5)}
    batch = {'heatmaps': torch.zeros_like(heatmaps), 'visibility': torch.ones(1, 5)}
    result = losses.compute_multitask_loss(
        outputs, batch, MeanSquaredHeatmapLoss(), torch.nn.BCEWithLogitsLoss(),
        lambda_pca_projection=0.7, pca_shape_prior=prior,
        image_height=32, image_width=32, pca_loss_space=space,
    )
    expected = compute_pca_projection_loss(predicted, prior, loss_space=space)
    torch.testing.assert_close(result['pca_loss'], expected)
    torch.testing.assert_close(result['total_loss'], result['full_landmark_loss']
                               + result['visible_landmark_loss'] + result['visibility_loss'] + 0.7 * expected)


@pytest.mark.parametrize('entrypoint', ['main', 'evaluate'])
def test_yaml_pca_mode_and_cli_override(entrypoint, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module(f'scripts.{entrypoint}')
    config = tmp_path / 'config.yaml'
    config.write_text('arguments:\n  pca_loss_space: aligned\n  checkpoint: /tmp/model.pth\n')
    monkeypatch.setattr('sys.argv', ['experiment', '--config', str(config)])
    assert module.build_config_from_args(module.parse_args()).pca_loss_space == 'aligned'
    monkeypatch.setattr('sys.argv', ['experiment', '--config', str(config), '--pca-loss-space', 'image'])
    assert module.build_config_from_args(module.parse_args()).pca_loss_space == 'image'
