from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.engine.train import save_checkpoint
from scripts.engine.normalizer_experiments import save_modular_checkpoints
from scripts.models import HRNetLandmarkVisibility, NormalizedLandmarker, ResidualImageNormalizer, build_model_from_checkpoints
from scripts.models.normalization import ChannelLayerNorm


@pytest.mark.parametrize('kind', ['layer', 'instance'])
def test_normalization_statistics_identity_and_gradients(kind):
    normalizer = ResidualImageNormalizer(normalization=kind, hidden_channels=5)
    image = torch.randn(2, 3, 9, 7)
    torch.testing.assert_close(normalizer(image), image, rtol=0, atol=0)
    norm = normalizer.delta_network[1]
    features = torch.randn(2, 5, 9, 7, requires_grad=True)
    result = norm(features)
    dims = 1 if kind == 'layer' else (2, 3)
    torch.testing.assert_close(result.mean(dim=dims), torch.zeros_like(result.mean(dim=dims)), atol=1e-6, rtol=0)
    result.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert norm.weight.grad is not None
    norm.eval()
    torch.testing.assert_close(norm(features), result)
    # Statistics do not depend on other images in the batch.
    torch.testing.assert_close(norm(features[:1]), result[:1])


@pytest.mark.parametrize('kind', ['layer', 'instance'])
def test_finetune_partition_and_checkpoint_roundtrip(kind, tmp_path: Path):
    torch.set_num_threads(1)
    source = NormalizedLandmarker(
        HRNetLandmarkVisibility(num_landmarks=4, head_normalization=kind),
        ResidualImageNormalizer(normalization=kind, hidden_channels=5),
    )
    source.configure_joint_finetune(num_unfrozen_stages=1, unfreeze_stem=False)
    source.train()
    for name, param in source.landmarker.backbone.named_parameters():
        assert param.requires_grad == name.startswith(('transition3.', 'stage4.'))
    assert not source.landmarker.backbone.stage3.training
    assert source.landmarker.backbone.stage4.training
    expected_type = ChannelLayerNorm if kind == 'layer' else nn.InstanceNorm2d
    for head in (source.landmarker.visibility_feature_head, source.landmarker.visible_landmark_feature_head, source.landmarker.full_landmark_fusion_head):
        assert isinstance(head[1], expected_type)
        assert all(p.requires_grad for p in head.parameters())
    assert isinstance(source.landmarker.backbone.bn1, nn.BatchNorm2d)
    source.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = source(image)
    save_checkpoint(tmp_path / 'best_model.pth', 0, source, None, {})
    paths = save_modular_checkpoints(source, tmp_path, None, 'normalizer_joint_finetune', tmp_path / 'resolved.yaml', True, True, 'barycenter', 'wasserstein', 'test')
    for path, normalizer_path in ((paths['full_model_best.pth'], None), (paths['landmarker_best.pth'], paths['normalizer_best.pth'])):
        restored = build_model_from_checkpoints(
            torch.load(path, weights_only=False),
            normalizer_checkpoint=torch.load(normalizer_path, weights_only=False) if normalizer_path else None,
        ).eval()
        with torch.no_grad():
            actual = restored(image)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    source.configure_normalizer_only()
    source.train()
    assert not source.landmarker.training
    assert all(not p.requires_grad for p in source.landmarker.parameters())
    assert all(p.requires_grad for p in source.normalizer.parameters())
    source(image)['heatmaps'].square().mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in source.normalizer.parameters())
    assert all(p.grad is None for p in source.landmarker.parameters())


@pytest.mark.parametrize("kind", ["layer", "instance"])
def test_cli_selects_matched_normalization(kind, monkeypatch):
    from scripts.main import parse_args, build_config_from_args, build_model

    monkeypatch.setattr("sys.argv", [
        "train", "--experiment-mode", "normalizer_joint_finetune",
        "--normalizer-normalization", kind, "--head-normalization", kind,
        "--transfer-mode", "fine_tuning", "--num-unfrozen-stages", "1",
    ])
    config = build_config_from_args(parse_args())
    config.pretrained_weights = None  # No external weights needed for this check.
    model = build_model(config)
    assert model.normalizer.normalization_name == kind
    assert model.landmarker.head_normalization == kind
    assert model.landmarker.backbone.stage4.training
    assert not model.landmarker.backbone.stage3.training
