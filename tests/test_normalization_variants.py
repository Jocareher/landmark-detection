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


@pytest.mark.parametrize('normalizer_kind,head_kind', [
    ('none', 'batch'), ('layer', 'layer'), ('instance', 'instance'),
])
def test_finetune_partition_and_checkpoint_roundtrip(normalizer_kind, head_kind, tmp_path: Path):
    torch.set_num_threads(1)
    source = NormalizedLandmarker(
        HRNetLandmarkVisibility(num_landmarks=4, head_normalization=head_kind),
        ResidualImageNormalizer(normalization=normalizer_kind, hidden_channels=5),
    )
    source.configure_joint_finetune(num_unfrozen_stages=1, unfreeze_stem=False)
    source.train()
    for name, param in source.landmarker.backbone.named_parameters():
        assert param.requires_grad == name.startswith(('transition3.', 'stage4.'))
    assert not source.landmarker.backbone.stage3.training
    assert source.landmarker.backbone.stage4.training
    expected_type = {'batch': nn.BatchNorm2d, 'layer': ChannelLayerNorm,
                     'instance': nn.InstanceNorm2d}[head_kind]
    for head in (source.landmarker.visibility_feature_head, source.landmarker.visible_landmark_feature_head, source.landmarker.full_landmark_fusion_head):
        assert isinstance(head[1], expected_type)
        assert all(p.requires_grad for p in head.parameters())
    assert isinstance(source.landmarker.backbone.bn1, nn.BatchNorm2d)
    source.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = source(image)
    save_checkpoint(tmp_path / 'best_model.pth', 0, source, None, {}, pca_loss_space='aligned')
    paths = save_modular_checkpoints(source, tmp_path, None, 'normalizer_joint_finetune', tmp_path / 'resolved.yaml', True, True, 'barycenter', 'wasserstein', 'test')
    for path, normalizer_path in ((paths['full_model_best.pth'], None), (paths['landmarker_best.pth'], paths['normalizer_best.pth'])):
        assert torch.load(path, weights_only=False)['pca_loss_space'] == 'aligned'
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


def test_early_instance_norm_finetune_and_checkpoint(tmp_path: Path):
    torch.set_num_threads(1)
    model = NormalizedLandmarker(
        HRNetLandmarkVisibility(num_landmarks=4, head_normalization="batch",
                                layer1_output_instance_norm=True),
        ResidualImageNormalizer(normalization="none", final_instance_norm=True),
    )
    model.configure_joint_finetune(num_unfrozen_stages=1, unfreeze_stem=False)
    model.train()
    assert isinstance(model.normalizer.delta_network[-3], nn.InstanceNorm2d)
    assert isinstance(model.normalizer.delta_network[-2], nn.ReLU)
    assert isinstance(model.landmarker.backbone.layer1_output_norm, nn.InstanceNorm2d)
    assert all(not p.requires_grad for p in model.landmarker.backbone.layer1.parameters())
    assert all(not p.requires_grad for p in model.landmarker.backbone.bn1.parameters())
    assert all(p.requires_grad for p in model.landmarker.backbone.layer1_output_norm.parameters())
    assert all(p.requires_grad for p in model.landmarker.backbone.stage4.parameters())
    assert all(isinstance(head[1], nn.BatchNorm2d) for head in (
        model.landmarker.visibility_feature_head,
        model.landmarker.visible_landmark_feature_head,
        model.landmarker.full_landmark_fusion_head,
    ))
    assert not model.landmarker.backbone.bn1.training
    assert not model.landmarker.backbone.layer1.training
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model.eval()(image)
    payload = {
        "model_state_dict": model.state_dict(),
        "landmarker_architecture": model.landmarker.architecture_config(),
        "normalizer_architecture": model.normalizer.architecture_config(),
    }
    restored = build_model_from_checkpoints(payload).eval()
    with torch.no_grad():
        actual = restored(image)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_legacy_final_instance_norm_checkpoint_preserves_post_activation_position():
    model = NormalizedLandmarker(
        HRNetLandmarkVisibility(num_landmarks=4, layer1_output_instance_norm=True),
        ResidualImageNormalizer(final_instance_norm=True,
                                final_instance_norm_position="after_activation",
                                initialize_identity=False),
    )
    architecture = model.normalizer.architecture_config()
    architecture.pop("final_instance_norm_position")
    restored = build_model_from_checkpoints({
        "model_state_dict": model.state_dict(),
        "landmarker_architecture": model.landmarker.architecture_config(),
        "normalizer_architecture": architecture,
    })
    assert restored.normalizer.final_instance_norm_position == "after_activation"
    image = torch.randn(1, 3, 16, 16)
    torch.testing.assert_close(restored.normalizer(image), model.normalizer(image), rtol=0, atol=0)


def test_final_only_and_hidden_instance_norm_order_and_identity():
    for normalization, final, expected_count in (("none", True, 1), ("instance", False, 2)):
        model = ResidualImageNormalizer(normalization=normalization, final_instance_norm=final)
        layers = list(model.delta_network)
        assert sum(isinstance(layer, nn.InstanceNorm2d) for layer in layers) == expected_count
        for index, layer in enumerate(layers):
            if isinstance(layer, nn.InstanceNorm2d):
                assert isinstance(layers[index - 1], nn.Conv2d)
                assert isinstance(layers[index + 1], nn.ReLU)
        assert isinstance(layers[-1], nn.Conv2d)
        image = torch.randn(1, 3, 16, 16)
        torch.testing.assert_close(model(image), image, rtol=0, atol=0)
    with pytest.raises(ValueError, match="every hidden block"):
        ResidualImageNormalizer(normalization="instance", final_instance_norm=True)


def test_batch_normalizer_and_full_instance_backbone_finetune(tmp_path: Path):
    torch.set_num_threads(1)
    batch_normalizer = ResidualImageNormalizer(normalization="batch")
    assert sum(isinstance(module, nn.BatchNorm2d) for module in batch_normalizer.modules()) == 2

    official = HRNetLandmarkVisibility(num_landmarks=4)
    source = {key: value.clone() for key, value in official.backbone.state_dict().items()
              if key in ("conv1.weight", "bn1.weight", "bn1.bias", "bn1.running_mean")}
    pretrained_path = tmp_path / "pretrained.pth"
    torch.save(source, pretrained_path)
    model = NormalizedLandmarker(
        HRNetLandmarkVisibility(num_landmarks=4, head_normalization="instance",
                                backbone_normalization="instance"),
        ResidualImageNormalizer(normalization="instance"),
    )
    audit = model.landmarker.load_official_hrnet_pretrained(str(pretrained_path), verbose=False)
    assert "backbone.conv1.weight" in audit["loaded_keys"]
    assert "backbone.bn1.weight" in audit["loaded_keys"]
    assert "bn1.running_mean" in audit["skipped_keys"]
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.landmarker.backbone.modules())
    torch.testing.assert_close(model.landmarker.backbone.conv1.weight, source["conv1.weight"])
    model.configure_joint_finetune(num_unfrozen_stages=1, unfreeze_stem=False)
    model.train()
    assert all(parameter.requires_grad for module in model.landmarker.backbone.modules()
               if isinstance(module, nn.InstanceNorm2d) for parameter in module.parameters())
    assert all(not parameter.requires_grad for parameter in model.landmarker.backbone.conv1.parameters())
    assert all(not parameter.requires_grad for parameter in model.landmarker.backbone.layer1[0].conv1.parameters())
    assert all(parameter.requires_grad for parameter in model.landmarker.backbone.stage4.parameters())
    assert all(isinstance(head[1], nn.InstanceNorm2d) for head in (
        model.landmarker.visibility_feature_head,
        model.landmarker.visible_landmark_feature_head,
        model.landmarker.full_landmark_fusion_head,
    ))
    output = model(torch.randn(1, 3, 64, 64))
    output["heatmaps"].square().mean().backward()
    assert model.landmarker.backbone.bn1.weight.grad is not None
    assert model.landmarker.backbone.stage2[0].branches[0][0].bn1.weight.grad is not None
    assert model.landmarker.backbone.conv1.weight.grad is None
    payload = {
        "model_state_dict": model.state_dict(),
        "landmarker_architecture": model.landmarker.architecture_config(),
        "normalizer_architecture": model.normalizer.architecture_config(),
    }
    restored = build_model_from_checkpoints(payload)
    assert restored.landmarker.backbone_normalization == "instance"
    assert not any(isinstance(module, nn.BatchNorm2d) for module in restored.landmarker.backbone.modules())
