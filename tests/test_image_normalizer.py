from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from scripts.config import load_yaml_config
from scripts.engine.normalizer_experiments import run_normalizer_diagnostics
from scripts.models import (
    NormalizedLandmarker,
    ResidualImageNormalizer,
    load_normalized_checkpoint,
)


class DummyLandmarker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {
            "heatmaps": features,
            "visible_heatmaps": features,
            "visibility_logits": features.mean(dim=(-2, -1)),
        }


def test_residual_normalizer_preserves_shape_and_identity() -> None:
    normalizer = ResidualImageNormalizer(initialize_identity=True)
    images = torch.randn(2, 3, 32, 32)
    normalized = normalizer(images)
    assert normalized.shape == images.shape
    torch.testing.assert_close(normalized, images, atol=0.0, rtol=0.0)


def test_wrapper_preserves_output_dictionary_and_freezing() -> None:
    wrapper = NormalizedLandmarker(
        landmarker=DummyLandmarker(),
        normalizer=ResidualImageNormalizer(initialize_identity=True),
    )
    wrapper.configure_normalizer_only()
    outputs = wrapper(torch.randn(2, 3, 16, 16))
    assert set(outputs) == {"heatmaps", "visible_heatmaps", "visibility_logits"}
    assert all(
        not parameter.requires_grad for parameter in wrapper.landmarker.parameters()
    )
    assert all(parameter.requires_grad for parameter in wrapper.normalizer.parameters())


def test_yaml_nested_values_override_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
experiment:
  name: normalizer_sanity
normalizer:
  hidden_channels: 7
training:
  landmark_loss: wasserstein
evaluation:
  datasets:
    synbaby: false
""",
        encoding="utf-8",
    )
    config = load_yaml_config(config_path)
    assert config.experiment_mode == "normalizer_sanity"
    assert config.normalizer_hidden_channels == 7
    assert config.landmark_loss == "wasserstein"
    assert config.evaluate_synbaby is False


def test_diagnostics_save_visual_example_with_tiny_batch(tmp_path: Path) -> None:
    wrapper = NormalizedLandmarker(
        landmarker=DummyLandmarker(),
        normalizer=ResidualImageNormalizer(initialize_identity=True),
    )
    batch = {
        "image": torch.zeros(1, 3, 8, 8),
        "metadata": {"sample_id": ["tiny"]},
    }
    summary = run_normalizer_diagnostics(
        model=wrapper,
        dataloader=[batch],
        device=torch.device("cpu"),
        output_dir=tmp_path,
        dataset_name="tiny",
        coordinate_decoder="barycenter",
        softmax_temperature=1.0,
        visibility_threshold=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_visual_examples=1,
    )
    assert summary["max_absolute_difference"] == 0.0
    assert (tmp_path / "visualizations/tiny/side_by_side/tiny.png").exists()
    assert (tmp_path / "metrics/tiny_normalizer_diagnostics.json").exists()


def test_legacy_checkpoint_loads_into_wrapped_landmarker() -> None:
    source = DummyLandmarker()
    wrapper = NormalizedLandmarker(
        landmarker=DummyLandmarker(),
        normalizer=ResidualImageNormalizer(initialize_identity=True),
    )
    load_normalized_checkpoint(
        wrapper, {"model_state_dict": source.state_dict()}, strict=True
    )
    for expected, actual in zip(source.parameters(), wrapper.landmarker.parameters()):
        torch.testing.assert_close(expected, actual)
