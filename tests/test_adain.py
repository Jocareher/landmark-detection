"""AdaIN source reference and checkpoint behavior."""
import torch

from scripts.models.normalization import SourceAdaptiveInstanceNorm2d
from scripts.models.image_normalizer import ResidualImageNormalizer
from scripts.models.hrn import HRNetLandmarkVisibility
from scripts.models.normalized_landmarker import NormalizedLandmarker, build_model_from_checkpoints


def test_source_reference_calibration_and_freeze():
    layer = SourceAdaptiveInstanceNorm2d(2)
    source = torch.tensor([[[[1., 3.], [5., 7.]], [[2., 4.], [6., 8.]]]])
    layer.train()
    layer.collect_source = True
    layer(source)
    assert layer.source_count.item() == 1
    layer.eval()
    layer.begin_calibration()
    layer(source)
    assert layer.finish_calibration() == 1
    before = (layer.source_mean.clone(), layer.source_std.clone(), layer.source_count.clone())
    target = torch.randn(1, 2, 2, 2, requires_grad=True)
    output = layer(target)
    assert torch.allclose(output.mean((2, 3)).flatten(), layer.source_mean, atol=1e-4)
    output.sum().backward()
    assert target.grad is not None
    assert all(torch.equal(current, saved) for current, saved in zip(
        (layer.source_mean, layer.source_std, layer.source_count), before))
    layer.train()
    layer.collect_source = False  # TTA may train affine weights without changing reference.
    layer(target.detach())
    assert torch.equal(layer.source_count, before[2])


def test_adain_full_checkpoint_reconstruction():
    from types import SimpleNamespace
    from scripts.engine.pca_tta import PCAGuidedTTA
    model = NormalizedLandmarker(
        HRNetLandmarkVisibility(head_normalization='adain'),
        ResidualImageNormalizer(normalization='adain'),
    )
    payload = {'model_state_dict': model.state_dict(),
               'landmarker_architecture': model.landmarker.architecture_config(),
               'normalizer_architecture': model.normalizer.architecture_config()}
    restored = build_model_from_checkpoints(payload)
    assert isinstance(restored.normalizer.delta_network[1], SourceAdaptiveInstanceNorm2d)
    assert all(torch.equal(value, restored.state_dict()[key])
               for key, value in model.state_dict().items())
    adapter = PCAGuidedTTA.__new__(PCAGuidedTTA)
    adapter.model = restored
    adapter.config = SimpleNamespace(adaptation_scope='normalizer_head_norms')
    assert len(adapter._select_head_modules()) == 3


def test_final_calibration_uses_unaugmented_train_images(monkeypatch):
    from types import SimpleNamespace
    from torch.utils.data import Dataset, DataLoader
    from scripts import main as training_main

    class SourceDataset(Dataset):
        def __init__(self):
            self.transform = lambda image: image + 100  # training augmentation
            self.images = [torch.arange(16., dtype=torch.float32).view(1, 4, 4) + value
                           for value in (2., 6.)]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return {'image': self.transform(self.images[index])}

    model = SourceAdaptiveInstanceNorm2d(1)
    monkeypatch.setattr(training_main, 'build_transforms',
                        lambda config: (lambda image: image + 100, lambda image: image))
    loader = DataLoader(SourceDataset(), batch_size=1)
    config = SimpleNamespace(eval_batch_size=1, batch_size=1, num_workers=0)
    assert training_main.calibrate_adain_source(model, loader, config, torch.device('cpu')) == 2
    assert torch.allclose(model.source_mean, torch.tensor([11.5]))
    assert model.source_count.item() == 2
    assert model._calibration is None


def test_local_adain_yaml_builds_training_configuration(monkeypatch):
    from pathlib import Path
    from scripts.main import parse_args, build_config_from_args

    config_path = Path(__file__).resolve().parents[1] / 'configs/adain_normalizer_experiments.yaml'
    monkeypatch.setattr('sys.argv', ['train', '--config', str(config_path)])
    config = build_config_from_args(parse_args())
    assert config.normalizer_internal_normalization == 'adain'
    assert config.head_normalization == 'adain'
    assert config.num_unfrozen_stages == 1
