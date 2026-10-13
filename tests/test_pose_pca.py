from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from scripts.engine.pca_shape_prior import (
    PCA_POSE_ANGLES, build_pose_pca_shape_prior_payload, compute_min_pose_pca_loss,
    compute_pca_projection_loss, load_pca_shape_prior,
)
from scripts.engine.pca_tta import PCAGuidedTTA, PCATTAConfig
from scripts.engine.tta_losses import compute_tta_losses
from scripts.models import NormalizedLandmarker, ResidualImageNormalizer
from test_pca_tta import TinyLandmarker, _pca_prior


def examples():
    torch.manual_seed(22)
    yaw = torch.tensor(PCA_POSE_ANGLES['pose13']).repeat_interleave(5)
    base = torch.tensor([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
    shapes = base + torch.randn(len(yaw),4,2)*0.1
    shapes[:,0,0] += yaw.float()/300
    return shapes, yaw


@pytest.mark.parametrize('mode', ['pose5','pose13'])
def test_fit_exact_pose_priors_and_loading(tmp_path, mode):
    shapes, yaw = examples()
    payload, aligned, by_pose = build_pose_pca_shape_prior_payload(shapes, yaw, mode, num_components=2)
    assert list(payload['pose_priors']) == [str(x) for x in PCA_POSE_ANGLES[mode]]
    assert aligned.shape[0] == 5 * len(PCA_POSE_ANGLES[mode])
    assert payload['num_excluded_samples'] == (40 if mode == 'pose5' else 0)
    for label, prior in payload['pose_priors'].items():
        assert prior['num_samples'] == 5 and prior['num_components'] == 2
        torch.testing.assert_close(prior['reference_shape'], payload['global_prior']['reference_shape'])
        # Reference means come from exactly the selected group in shared coordinates.
        torch.testing.assert_close(prior['mean_shape'], by_pose[label].flatten(1).mean(0))
    path = tmp_path/'prior.pt'
    torch.save(payload, path)
    loaded = load_pca_shape_prior(path, 'cpu')
    assert loaded['prior_mode'] == mode
    compute_min_pose_pca_loss(shapes[:2], loaded, 'aligned')


@pytest.mark.parametrize('space', ['aligned','image'])
def test_per_image_minimum_and_winning_gradient(space):
    shapes, yaw = examples()
    payload, _, _ = build_pose_pca_shape_prior_payload(shapes, yaw, 'pose5', num_components=1)
    predictions = (shapes[::7] + torch.randn_like(shapes[::7])*0.2).requires_grad_()
    minimum, indices, candidates, labels = compute_min_pose_pca_loss(predictions, payload, space)
    torch.testing.assert_close(minimum, candidates.min(1).values)
    assert indices.unique().numel() > 1
    expected = torch.stack([compute_pca_projection_loss(predictions[i:i+1],
        {'alignment': payload['alignment'], 'global_prior': payload['pose_priors'][labels[int(index)]]},
        loss_space=space) for i,index in enumerate(indices)])
    grad, = torch.autograd.grad(minimum.mean(), predictions, retain_graph=True)
    wanted, = torch.autograd.grad(expected.mean(), predictions)
    torch.testing.assert_close(grad, wanted)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_pose_labels_and_missing_models_rejected():
    shapes, yaw = examples()
    with pytest.raises(ValueError, match='finite yaw'):
        build_pose_pca_shape_prior_payload(shapes, yaw.float()*float('nan'))
    with pytest.raises(ValueError, match='at least two'):
        build_pose_pca_shape_prior_payload(shapes[yaw != 0], yaw[yaw != 0])
    with pytest.raises(ValueError, match='pose_priors'):
        compute_min_pose_pca_loss(shapes, _pca_prior())


def test_pose_batch_tta_restores_weights_and_reports_candidates(tmp_path):
    shapes, yaw = examples()
    prior, _, _ = build_pose_pca_shape_prior_payload(shapes, yaw, num_components=1)
    model = NormalizedLandmarker(TinyLandmarker(), ResidualImageNormalizer(
        normalization='batch', hidden_channels=4, num_layers=2, initialize_identity=False))
    source = deepcopy(model.state_dict())
    adapter = PCAGuidedTTA(model, prior, torch.device('cpu'), tmp_path,
        PCATTAConfig(pca_prior_mode='min_pose', pca_loss_space='aligned', episode_unit='batch',
                     steps=1, probe_count=0, entropy_weight=0.1))
    images = torch.randn(3,3,8,8)
    first = adapter.adapt_batch(images, ['a','b','c'])
    second = adapter.adapt_batch(images, ['d','e','f'])
    assert adapter.episode_batch_sizes == [3,3]
    torch.testing.assert_close(first['heatmaps'], second['heatmaps'])
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value, source[key])
    assert adapter.failed_samples == 0
    for row in adapter.trajectory_rows:
        assert row['pca_reconstruction_loss'] == min(value for key,value in row.items() if key.startswith('pca_candidate_loss_'))
        assert row['pca_selected_yaw'] in PCA_POSE_ANGLES['pose5']
    result = adapter.finalize()
    assert result['adaptation_unit'] == 'batch' and result['pca_prior_mode'] == 'min_pose'
    assert 'final_pca_selected_yaw' in adapter.summary_rows[0]


def test_heatmap_only_min_pose_does_not_need_prior():
    losses = compute_tta_losses(torch.randn(2,4,5,5), None, None, pca_prior_mode='min_pose',
                               pca_weight=0, entropy_weight=1)
    assert torch.isfinite(losses['sample_tta_loss']).all()
    assert 'pca_selected_yaw' not in losses


@pytest.mark.parametrize('mode', ['global', 'pose5', 'pose13'])
def test_builder_main_saves_bank_and_pose_previews(tmp_path, monkeypatch, mode):
    from scripts import build_pca_shape_prior as builder
    shapes, yaw = examples()
    args = SimpleNamespace(prior_mode=mode, pca_num_components=2, pca_explained_variance=None,
                           output_path=tmp_path/'prior.pt', dataset_root=tmp_path/'dataset',
                           image_height=32, image_width=32, previews_dir=tmp_path/'previews')
    monkeypatch.setattr(builder, 'parse_args', lambda: args)
    monkeypatch.setattr(builder, 'collect_train_landmarks',
                        lambda args, include_yaw=False: (shapes,yaw) if include_yaw else shapes)
    calls = []
    def preview(prior_payload, aligned_shapes, previews_dir):
        previews_dir.mkdir(parents=True, exist_ok=True)
        calls.append(previews_dir)
    monkeypatch.setattr(builder, 'save_pca_analysis_plots', preview)
    builder.main()
    payload = load_pca_shape_prior(args.output_path, 'cpu')
    assert payload['source_split'] == 'train' and payload['prior_mode'] == mode
    assert len(calls) == 1 + (len(PCA_POSE_ANGLES[mode]) if mode != 'global' else 0)
    assert (args.previews_dir/'pose_pca_summary.json').exists() == (mode != 'global')
