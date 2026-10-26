"""Coefficient regularization: thresholds, gradients, pose selection and opt-out."""
from copy import deepcopy

import pytest
import torch

from scripts.engine.pca_shape_prior import (
    chi_square_critical_value, compute_pca_projection_loss, compute_pca_projection_terms,
    pca_mahalanobis_threshold, validate_pca_mahalanobis,
    validate_pca_mahalanobis_prior,
)
from scripts.engine.tta_losses import compute_tta_losses
from test_pca_image_space_loss import _example


def example():
    x, prior = _example()
    # Full orthonormal basis makes reconstruction exact, even far from the mean.
    prior['global_prior']['components'] = torch.eye(10)
    prior['global_prior']['explained_variance'] = torch.full((10,), 1e-4)
    return x, prior


def test_theoretical_critical_value_and_percentile_override():
    assert chi_square_critical_value(32, 95.0) == pytest.approx(46.19425952, rel=1e-8)
    assert chi_square_critical_value(2, 95.0) == pytest.approx(5.99146455, rel=1e-8)
    assert chi_square_critical_value(1, 95.0) == pytest.approx(3.84145882, rel=1e-8)
    x, prior = example()
    terms = compute_pca_projection_terms(
        x, prior, 'aligned', mahalanobis_enabled=True,
        mahalanobis_c=1e9, mahalanobis_percentile=95.0)
    critical = chi_square_critical_value(10, 95.0)
    torch.testing.assert_close(
        terms['mahalanobis_penalty'],
        (terms['mahalanobis_squared'] - critical).clamp_min(0))
    assert terms['mahalanobis_penalty'] > 0  # The manual c=1e9 was overridden.


def test_pose_candidates_use_their_own_component_counts():
    x, prior = example()
    narrow = deepcopy(prior['global_prior'])
    wide = deepcopy(narrow)
    narrow['components'] = narrow['components'][:2]
    narrow['explained_variance'] = narrow['explained_variance'][:2]
    prior['pose_priors'] = {'-75': narrow, '75': wide}
    assert pca_mahalanobis_threshold(narrow, 10, 95.0) == pytest.approx(5.99146455)
    assert pca_mahalanobis_threshold(wide, 10, 95.0) == pytest.approx(18.30703805)
    results = compute_tta_losses(
        torch.randn(1, 5, 8, 8), x, prior, pca_prior_mode='min_pose',
        pca_mahalanobis_enabled=True, pca_mahalanobis_percentile=95.0)
    for label, candidate in prior['pose_priors'].items():
        expected = compute_pca_projection_loss(
            x, {'alignment': prior['alignment'], 'global_prior': candidate},
            mahalanobis_enabled=True, mahalanobis_percentile=95.0,
            reduction='none')
        torch.testing.assert_close(results[f'pca_candidate_loss_yaw_{label}'], expected)


@pytest.mark.parametrize('space', ['aligned', 'image'])
def test_new_term_detects_exactly_reconstructible_outliers_and_backpropagates(space):
    x, prior = example()
    x.requires_grad_()
    baseline = compute_pca_projection_loss(x, prior, space)
    assert baseline.item() < 1e-8
    terms = compute_pca_projection_terms(x, prior, space, mahalanobis_enabled=True,
                                         mahalanobis_weight=.002, mahalanobis_c=10)
    assert terms['mahalanobis_squared'] > 10
    torch.testing.assert_close(terms['loss'], baseline + .002 * (terms['mahalanobis_squared'] - 10))
    gradient, = torch.autograd.grad(terms['loss'], x)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_threshold_is_subtracted_directly_and_inactive_region_has_zero_gradient():
    x, prior = example()
    x.requires_grad_()
    a = compute_pca_projection_terms(x, prior, 'aligned', mahalanobis_enabled=True, mahalanobis_c=5)
    b = compute_pca_projection_terms(x, prior, 'aligned', mahalanobis_enabled=True, mahalanobis_c=10)
    torch.testing.assert_close(a['mahalanobis_penalty'] - b['mahalanobis_penalty'], torch.tensor(5.))
    inactive = compute_pca_projection_terms(x, prior, 'aligned', mahalanobis_enabled=True,
                                          mahalanobis_c=a['mahalanobis_squared'].item() + 1)
    assert inactive['mahalanobis_penalty'] == 0
    grad, = torch.autograd.grad(inactive['mahalanobis_penalty'], x)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_disabled_preserves_legacy_prior_and_zero_weight_preserves_loss():
    x, legacy = _example()  # No eigenvalues required when disabled.
    original = compute_pca_projection_loss(x, legacy, 'aligned')
    torch.testing.assert_close(original, compute_pca_projection_loss(
        x, legacy, 'aligned', mahalanobis_enabled=False, mahalanobis_c=0))
    x, prior = example()
    torch.testing.assert_close(compute_pca_projection_loss(x, prior, 'aligned'),
        compute_pca_projection_loss(x, prior, 'aligned', mahalanobis_enabled=True, mahalanobis_weight=0))
    with pytest.raises(ValueError, match='explained_variance'):
        compute_pca_projection_loss(x, legacy, mahalanobis_enabled=True)


def test_variance_floor_and_per_image_reduction():
    x, prior = example()
    prior['global_prior']['explained_variance'][0] = 0
    terms = compute_pca_projection_terms(x.repeat(2,1,1), prior, 'aligned', reduction='none',
                                        mahalanobis_enabled=True)
    reduced = compute_pca_projection_terms(x.repeat(2,1,1), prior, 'aligned', mahalanobis_enabled=True)
    for key, values in terms.items():
        assert values.shape == (2,) and torch.isfinite(values).all()
        torch.testing.assert_close(values.mean(), reduced[key])


def test_min_pose_selects_complete_objective_and_correct_winner_gradient():
    x, prior = example()
    narrow = deepcopy(prior['global_prior'])
    wide = deepcopy(narrow)
    wide['explained_variance'] *= 1000
    prior['pose_priors'] = {'-75': narrow, '75': wide}
    x.requires_grad_()
    results = compute_tta_losses(torch.randn(1,5,8,8), x, prior, pca_loss_space='aligned',
                                pca_prior_mode='min_pose', pca_mahalanobis_enabled=True,
                                pca_mahalanobis_weight=.01, pca_mahalanobis_c=0)
    assert results['pca_selected_yaw'].item() == 75
    selected = compute_pca_projection_loss(x, {'alignment': prior['alignment'], 'global_prior': wide},
        'aligned', mahalanobis_enabled=True, mahalanobis_weight=.01, mahalanobis_c=0)
    torch.testing.assert_close(results['sample_tta_loss'].mean(), selected)
    g1, = torch.autograd.grad(results['sample_tta_loss'].mean(), x, retain_graph=True)
    g2, = torch.autograd.grad(selected, x)
    torch.testing.assert_close(g1, g2)


@pytest.mark.parametrize('kwargs', [{'c': -1}, {'c': float('nan')}, {'weight': -1},
                                   {'weight': float('inf')}, {'eps': 0}, {'enabled': 'true'},
                                   {'percentile': 0}, {'percentile': 100},
                                   {'percentile': float('nan')}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        validate_pca_mahalanobis(**kwargs)


def test_invalid_prior_fails_preflight():
    _, prior = example()
    prior['global_prior']['explained_variance'][0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        validate_pca_mahalanobis_prior(prior)


def test_yaml_options_and_hpc_preservation(tmp_path, monkeypatch):
    from pathlib import Path
    import importlib.util
    import yaml
    from scripts.config import load_yaml_config
    config = load_yaml_config('configs/normalizer_experiments.yaml')
    assert config.pca_mahalanobis_enabled is False
    assert config.pca_mahalanobis_c == 10
    assert config.pca_mahalanobis_percentile == 95.0
    spec = importlib.util.spec_from_file_location('run_job', 'slurm/run_job.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = yaml.safe_load(Path('configs/pca_tta_evaluation.yaml').read_text())
    original['arguments'].update(pca_mahalanobis_enabled=True, pca_mahalanobis_c=5,
                                 pca_mahalanobis_percentile=90.0,
                                 pca_mahalanobis_weight=.0001)
    plan = dict(mode='tta', run_dir=str(tmp_path/'run'), scope='normalizer',
                checkpoint='/weights/model.pth', normalizer_checkpoint=None,
                dataset='babyland', resources=dict(batch_size=2, workers=0, steps=1, learning_rate=1e-4),
                paths=dict(pca_prior='/weights/pca.pt', babyland_crops='/data/crops',
                           babyland_labels='/data/labels', babyland_source_root='/data/images'))
    snapshot = tmp_path/'tta.yaml'
    snapshot.write_text(yaml.safe_dump({'arguments': module.resolved_arguments(plan, original['arguments'])}))
    from scripts.evaluate import parse_args, build_config_from_args
    monkeypatch.setattr('sys.argv', ['evaluate', '--config', str(snapshot)])
    loaded = build_config_from_args(parse_args())
    assert loaded.pca_mahalanobis_enabled is True
    assert loaded.pca_mahalanobis_c == 5
    assert loaded.pca_mahalanobis_percentile == 90.0
    assert loaded.pca_mahalanobis_weight == .0001
    monkeypatch.setattr('sys.argv', ['evaluate', '--config', str(snapshot),
                                   '--no-pca-mahalanobis-enabled', '--pca-mahalanobis-c', '10',
                                   '--pca-mahalanobis-percentile', '99'])
    loaded = build_config_from_args(parse_args())
    assert loaded.pca_mahalanobis_enabled is False and loaded.pca_mahalanobis_c == 10
    assert loaded.pca_mahalanobis_percentile == 99.0


def test_enabled_tta_records_components_and_restores_source(tmp_path):
    from scripts.engine.pca_tta import PCAGuidedTTA, PCATTAConfig
    from scripts.models import NormalizedLandmarker, ResidualImageNormalizer
    from test_pca_tta import TinyLandmarker
    torch.manual_seed(123)
    _, prior = example()
    model = NormalizedLandmarker(TinyLandmarker(num_landmarks=5),
        ResidualImageNormalizer(hidden_channels=4, num_layers=2))
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    adapter = PCAGuidedTTA(model, prior, torch.device('cpu'), tmp_path,
        PCATTAConfig(steps=1, probe_count=0, pca_loss_space='aligned',
            pca_mahalanobis_enabled=True, pca_mahalanobis_c=0,
            pca_mahalanobis_weight=.0001))
    adapter.adapt_batch(torch.randn(2,3,8,8), ['a','b'])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    assert adapter.failed_samples == 0
    for row in adapter.trajectory_rows:
        assert row['pca_mahalanobis_squared'] > 0
        assert row['pca_loss'] == pytest.approx(row['pca_reconstruction_loss'] +
                                                row['weighted_pca_mahalanobis_loss'])
    summary = adapter.finalize()
    assert summary['pca_mahalanobis_enabled'] is True
    assert summary['pca_mahalanobis_c'] == 0
    assert (tmp_path/'curves'/'pca_loss_mean.png').exists()


def test_supervised_training_routes_coefficient_loss_to_heatmap_gradients():
    from scripts.engine.losses import compute_multitask_loss
    from scripts.engine.landmark_losses import MeanSquaredHeatmapLoss
    torch.manual_seed(7)
    _, prior = example()
    heatmaps = torch.randn(1,5,8,8, requires_grad=True)
    visibility = torch.ones(1,5)
    losses = compute_multitask_loss(
        {'heatmaps': heatmaps, 'visible_heatmaps': heatmaps,
         'visibility_logits': torch.zeros(1,5)},
        {'heatmaps': heatmaps.detach(), 'visibility': visibility},
        MeanSquaredHeatmapLoss(), torch.nn.BCEWithLogitsLoss(),
        lambda_vis=0, lambda_lmk_vis=0, lambda_lmk_full=0,
        lambda_pca_projection=.3, pca_shape_prior=prior,
        image_height=32, image_width=32, pca_loss_space='aligned',
        pca_mahalanobis_enabled=True, pca_mahalanobis_c=0,
        pca_mahalanobis_weight=.001)
    assert losses['pca_mahalanobis_penalty'] > 0
    torch.testing.assert_close(losses['total_loss'], .3 * (
        losses['pca_reconstruction_loss'] + .001 * losses['pca_mahalanobis_penalty']))
    losses['total_loss'].backward()
    assert torch.isfinite(heatmaps.grad).all() and heatmaps.grad.abs().max() > 0
