import csv
import json
from copy import deepcopy

import numpy as np
import torch

from scripts.engine.runtime_reporting import PredictionTiming, project_prediction_batch, save_training_curves
from scripts.engine.inference import export_inference_outputs
from scripts.engine.pca_tta import PCAGuidedTTA, PCATTAConfig
from scripts.models import NormalizedLandmarker, ResidualImageNormalizer
from test_pca_tta import TinyLandmarker


def batch_sample(name='a'):
    return {'image': torch.zeros(3,8,8), 'metadata': {
        'sample_id': name, 'image_path': '/tmp/a.png', 'original_size': (16,16),
        'transformed_size': (8,8), 'crop_size': (16,16), 'transform_crop_to_orig':
        torch.tensor([[1.,0.,10.],[0.,1.,20.],[0.,0.,1.]]),
        'source_image_path': '/tmp/original.png', 'crop_image_path': '/tmp/a.png'}}


def test_prediction_timings_cover_phases_and_amortized_batch_cost(tmp_path):
    timer = PredictionTiming('cpu')
    timer.begin(['a','b'])
    for name in ('transfer','inference','decode','reprojection'):
        with timer.stage(name):
            pass
    timer.finish()
    summary = timer.save(tmp_path)
    row = timer.rows[0]
    assert row['end_to_end_seconds'] >= sum(row[f'{name}_seconds'] for name in ('transfer','inference','decode','reprojection'))
    assert summary['end_to_end']['amortized_mean_seconds_per_image'] == row['end_to_end_seconds']/2
    assert summary['num_images'] == 2
    assert (tmp_path/'timing/prediction_batches.csv').exists()


def test_reprojection_includes_inverse_crop_transform():
    batch = next(iter(torch.utils.data.DataLoader([batch_sample()], batch_size=1)))
    landmarks = torch.tensor([[[1.,2.],[3.,4.]]])
    crop, original = project_prediction_batch(landmarks, batch['metadata'], affine=True)[0]
    torch.testing.assert_close(crop, torch.tensor([[2.,4.],[6.,8.]]))
    np.testing.assert_allclose(original, [[12.,24.],[16.,28.]])


def test_inference_exports_timing_without_extra_forward_and_preserves_predictions(tmp_path):
    class Model(TinyLandmarker):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, images):
            self.calls += 1
            return super().forward(images)
    model = Model()
    loader = torch.utils.data.DataLoader([batch_sample('a'),batch_sample('b')], batch_size=2)
    result = export_inference_outputs(model, loader, torch.device('cpu'), tmp_path,
                                     save_overlays=False, project_to_original=True)
    assert model.calls == 1
    assert result['timing']['num_images'] == 2
    assert result['timing']['inference']['count'] == 1
    assert result['timing']['reprojection']['total_seconds'] >= 0
    assert (tmp_path/'predictions/labels/a.txt').exists()


def test_all_training_loss_components_are_saved_and_plotted(tmp_path):
    history = {'train': [{'total_loss':2., 'pca_loss':.1, 'image_tv_loss':.3}],
               'val': [{'total_loss':3., 'pca_loss':.2, 'image_tv_loss':.4}]}
    save_training_curves(history, tmp_path)
    with (tmp_path/'curves/train_val_losses.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row['split'] for row in rows} == {'train','val'}
    assert rows[0]['epoch'] == '1' and rows[1]['pca_loss'] == '0.2'
    for name in ('total_loss','pca_loss','image_tv_loss'):
        assert (tmp_path/f'curves/{name}.png').exists()


def test_tta_timings_and_curves_preserve_episodic_restoration(tmp_path):
    model = NormalizedLandmarker(TinyLandmarker(), ResidualImageNormalizer(
        hidden_channels=4, num_layers=2, initialize_identity=False))
    source = deepcopy(model.state_dict())
    adapter = PCAGuidedTTA(model, None, torch.device('cpu'), tmp_path,
        PCATTAConfig(pca_weight=0, entropy_weight=1, episode_unit='batch', steps=1, probe_count=0))
    adapter.adapt_batch(torch.randn(2,3,8,8), ['a','b'])
    summary = adapter.finalize()
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value, source[key])
    assert len(adapter.forward_timings) == 2
    assert len(adapter.adaptation_timings) == 1
    assert summary['timing']['inference']['count'] == 2
    for path in ('timing/model_forwards.csv', 'timing/adaptation_batches.csv',
                 'curves/tta_losses_per_image.csv','curves/tta_losses_aggregate.csv',
                 'curves/heatmap_entropy_loss_mean.png'):
        assert (tmp_path/path).exists()


def test_training_records_total_validation_inference_and_all_curves(tmp_path):
    from scripts.engine.train import train_model
    model = TinyLandmarker()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    sample = {'image': torch.randn(3,8,8), 'heatmaps': torch.rand(4,8,8),
              'visibility': torch.ones(4), 'landmarks': torch.tensor([[1.,1.],[6.,1.],[6.,6.],[1.,6.]])}
    loader = torch.utils.data.DataLoader([sample], batch_size=1)
    result = train_model(model, loader, loader, optimizer, scheduler,
        torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss(), torch.device('cpu'), 1, tmp_path,
        use_wandb=False, use_amp=False, visualize_every_n_epochs=0)
    timing = json.loads((tmp_path/'timing/training_summary.json').read_text())
    assert timing['completed_epochs'] == 1
    assert timing['training_total_seconds'] >= timing['train_epoch_seconds'] + timing['val_epoch_seconds']
    assert result['history']['val'][0]['validation_inference_seconds'] > 0
    assert result['history']['train'][0]['validation_inference_seconds'] is None
    with (tmp_path/'curves/train_val_losses.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and {'total_loss','full_landmark_loss','visible_landmark_loss','visibility_loss','pca_loss','image_l1_loss','image_tv_loss'} <= set(rows[0])


def test_accelerator_timing_synchronizes_stage_boundaries(monkeypatch):
    from scripts.engine import runtime_reporting
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(str(device)))
    timer = PredictionTiming(torch.device('cuda:0'))
    timer.begin(['a'])
    with timer.stage('inference'):
        pass
    timer.finish()
    assert calls == ['cuda:0'] * 5
    calls.clear()
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: calls.append('mps'))
    runtime_reporting.clock('mps')
    assert calls == ['mps']
