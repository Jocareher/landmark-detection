import math
from copy import deepcopy

import pytest
import torch

from scripts.engine.metrics import heatmap_distribution_statistics, compute_heatmap_uncertainty
from scripts.engine.tta_losses import compute_tta_losses
from scripts.engine.pca_tta import PCATTAConfig, PCAGuidedTTA, _aggregate_trajectories
from scripts.models import NormalizedLandmarker, ResidualImageNormalizer
from test_pca_tta import TinyLandmarker, _pca_prior


class BatchLandmarker(TinyLandmarker):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(3)

    def forward(self, images):
        return super().forward(self.norm(images))


def test_normalized_signals_limits_and_gradients():
    uniform = torch.zeros(1, 2, 2, 2, dtype=torch.double)
    stats = heatmap_distribution_statistics(uniform)
    torch.testing.assert_close(stats['entropy'], torch.ones(1, 2, dtype=torch.double))
    torch.testing.assert_close(stats['spread'], torch.ones(1, 2, dtype=torch.double))
    peak = uniform.clone()
    peak[..., 0, 0] = 100
    for value in heatmap_distribution_statistics(peak).values():
        assert value.max() < 1e-10
    logits = torch.randn(2, 3, 4, 5, dtype=torch.double, requires_grad=True)
    for signal in ('entropy', 'spread'):
        assert torch.autograd.gradcheck(lambda x: heatmap_distribution_statistics(x)[signal], (logits,))
        grad, = torch.autograd.grad(heatmap_distribution_statistics(logits)[signal].mean(), logits)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert not compute_heatmap_uncertainty(logits)['entropy'].requires_grad
    shifted = heatmap_distribution_statistics(logits + 10)
    for key, value in heatmap_distribution_statistics(logits).items():
        torch.testing.assert_close(shifted[key], value)


@pytest.mark.parametrize('weights', [(0,1,0),(0,0,1),(0,1,1),(1,1,0),(1,0,1),(1,1,1),(1,0,0)])
def test_weighted_objective_and_batch_reduction(weights):
    heatmaps = torch.randn(2, 4, 6, 6, requires_grad=True)
    landmarks = torch.tensor([[[-1.,-1.],[1.,-1.],[1.2,1.],[0.,1.]]]).repeat(2,1,1).requires_grad_()
    result = compute_tta_losses(heatmaps, landmarks, _pca_prior() if weights[0] else None,
                               pca_weight=weights[0], entropy_weight=weights[1], spread_weight=weights[2], pca_loss_space='aligned')
    torch.testing.assert_close(result['sample_tta_loss'], sum(result[f'weighted_{k}_loss'] for k in ('pca','entropy','spread')))
    result['sample_tta_loss'].mean().backward()
    if weights[1] or weights[2]:
        assert heatmaps.grad.abs().sum() > 0
    if not weights[0]:
        assert torch.isnan(result['pca_reconstruction_loss']).all()
        assert not result['weighted_pca_loss'].any()
    single = compute_tta_losses(heatmaps[:1], landmarks[:1], _pca_prior() if weights[0] else None,
                               pca_weight=weights[0], entropy_weight=weights[1], spread_weight=weights[2], pca_loss_space='aligned')
    torch.testing.assert_close(single['sample_tta_loss'][0], result['sample_tta_loss'][0])


@pytest.mark.parametrize('weights', [(0,0,0),(-1,1,0),(math.nan,1,0),(1,math.inf,0)])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        PCATTAConfig(pca_weight=weights[0], entropy_weight=weights[1], spread_weight=weights[2]).validate()


@pytest.mark.parametrize('scope', ['normalizer', 'normalizer_all_norms'])
@pytest.mark.parametrize('weights', [(0,1,0),(0,0,1),(1,1,1)])
def test_heatmap_and_mixed_tta_restore_source_and_report(tmp_path, scope, weights):
    torch.manual_seed(5)
    model = NormalizedLandmarker(BatchLandmarker(), ResidualImageNormalizer(
        hidden_channels=4, num_layers=2, normalization='batch', initialize_identity=False))
    source = deepcopy(model.state_dict())
    adapter = PCAGuidedTTA(model, _pca_prior() if weights[0] else None, torch.device('cpu'), tmp_path,
                          PCATTAConfig(steps=1, probe_count=0, adaptation_scope=scope,
                                       pca_weight=weights[0], entropy_weight=weights[1], spread_weight=weights[2]))
    images = torch.randn(2,3,8,8)
    first = adapter.adapt_batch(images, ['a','b'])
    second = adapter.adapt_batch(images, ['c','d'])
    torch.testing.assert_close(first['heatmaps'], second['heatmaps'])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, source[key])
    assert not adapter.failed_samples
    assert all(math.isfinite(row['sample_tta_loss']) for row in adapter.trajectory_rows)
    adapter.record_evaluation_metrics(sample_id='a', orientation='frontal',
        initial_nme_box_gt_valid=0.1, final_nme_box_gt_valid=0.09,
        initial_hausdorff_box_gt_valid=0.2, final_hausdorff_box_gt_valid=0.18,
        number_of_valid_landmarks=4)
    summary = adapter.finalize()
    assert summary['loss_weights']['entropy'] == weights[1]
    assert (tmp_path/'figures/tta_loss_components.png').exists()
    assert (tmp_path/'figures/tta_objective_curve.png').exists()
    assert _aggregate_trajectories(adapter.trajectory_rows)
    assert (tmp_path/'figures/pca_loss_reduction_vs_nme_change.png').exists() == bool(weights[0])
    if not weights[0]:
        assert summary['pca_loss_mode'] is None


def test_missing_prior_rejected_before_model_changes(tmp_path):
    model = NormalizedLandmarker(TinyLandmarker(), ResidualImageNormalizer(hidden_channels=4, num_layers=2))
    before = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError, match='prior'):
        PCAGuidedTTA(model, None, torch.device('cpu'), tmp_path, PCATTAConfig())
    assert before == [p.requires_grad for p in model.parameters()]
