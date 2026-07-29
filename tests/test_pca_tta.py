from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from scripts.engine.pca_tta import PCAGuidedTTA, PCATTAConfig
from scripts.models import NormalizedLandmarker, ResidualImageNormalizer


class TinyLandmarker(nn.Module):
    """Small differentiable heatmap model used to exercise TTA mechanics."""

    def __init__(self, num_landmarks: int = 4) -> None:
        super().__init__()
        self.heatmap_head = nn.Conv2d(3, num_landmarks, kernel_size=1)
        self.visibility_head = nn.Linear(3, num_landmarks)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        heatmaps = self.heatmap_head(images)
        pooled = images.mean(dim=(-2, -1))
        return {
            "heatmaps": heatmaps,
            "visible_heatmaps": heatmaps,
            "visibility_logits": self.visibility_head(pooled),
        }


def _pca_prior() -> dict[str, object]:
    reference = torch.tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=torch.float32,
    )
    mean = reference.reshape(1, -1)
    component = torch.tensor(
        [[1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    component = component / torch.linalg.norm(component, dim=1, keepdim=True)
    return {
        "alignment": {
            "method": "procrustes",
            "allow_reflection": False,
            "eps": 1e-6,
        },
        "global_prior": {
            "num_landmarks": 4,
            "reference_shape": reference,
            "mean_shape": mean,
            "components": component,
        },
    }


def test_pca_tta_is_episodic_and_restores_source_normalizer(
    tmp_path: Path,
    capsys,
) -> None:
    torch.manual_seed(7)
    model = NormalizedLandmarker(
        landmarker=TinyLandmarker(),
        normalizer=ResidualImageNormalizer(
            hidden_channels=4,
            num_layers=2,
            residual_scale=0.1,
            initialize_identity=False,
        ),
    )
    source_normalizer = deepcopy(model.normalizer.state_dict())
    source_landmarker = deepcopy(model.landmarker.state_dict())
    adapter = PCAGuidedTTA(
        model=model,
        pca_prior=_pca_prior(),
        device=torch.device("cpu"),
        output_dir=tmp_path / "tta",
        config=PCATTAConfig(
            steps=2,
            learning_rate=1e-3,
            monitor_steps=(0, 1, 2),
            probe_count=0,
        ),
    )
    assert all(
        not parameter.requires_grad for parameter in model.landmarker.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.normalizer.parameters())
    image = torch.randn(1, 3, 12, 12)

    first = adapter.adapt_batch(image, ["first"])
    second = adapter.adapt_batch(image, ["second"])

    assert first["heatmaps"].shape == (1, 4, 12, 12)
    torch.testing.assert_close(first["heatmaps"], second["heatmaps"])
    assert len(adapter.trajectory_rows) == 6
    first_losses = [
        row["pca_reconstruction_loss"]
        for row in adapter.trajectory_rows
        if row["sample_id"] == "first"
    ]
    second_losses = [
        row["pca_reconstruction_loss"]
        for row in adapter.trajectory_rows
        if row["sample_id"] == "second"
    ]
    assert first_losses == second_losses
    for key, expected in source_normalizer.items():
        torch.testing.assert_close(model.normalizer.state_dict()[key], expected)
    for key, expected in source_landmarker.items():
        torch.testing.assert_close(model.landmarker.state_dict()[key], expected)

    summary = adapter.finalize()
    assert summary["processed_samples"] == 2
    assert summary["failed_samples"] == 0
    assert (tmp_path / "tta/trajectories.csv").exists()
    assert (tmp_path / "tta/image_summary.csv").exists()
    assert (tmp_path / "tta/aggregate_curves.csv").exists()
    assert (tmp_path / "tta/figures/pca_reconstruction_loss_curve.png").exists()
    terminal_output = capsys.readouterr().out
    assert "landmarker_trainable=0" in terminal_output
    assert "sample=first" in terminal_output
    assert "sample=second" in terminal_output
    assert terminal_output.count("source_normalizer_restored=yes") == 2
    assert terminal_output.count("optimizer=fresh") == 2
    assert terminal_output.count("source_normalizer_restored_after=yes") == 2


def test_pca_tta_uses_reconstruction_loss_without_regularization(
    tmp_path: Path,
) -> None:
    model = NormalizedLandmarker(
        landmarker=TinyLandmarker(),
        normalizer=ResidualImageNormalizer(hidden_channels=4, num_layers=2),
    )
    adapter = PCAGuidedTTA(
        model=model,
        pca_prior=_pca_prior(),
        device=torch.device("cpu"),
        output_dir=tmp_path,
        config=PCATTAConfig(steps=1, monitor_steps=(0, 1), probe_count=1),
    )

    adapter.adapt_batch(torch.randn(1, 3, 10, 10), ["sample"])

    assert adapter.trajectory_rows
    for row in adapter.trajectory_rows:
        assert row["total_tta_loss"] == row["pca_reconstruction_loss"]
    assert (tmp_path / "probes/sample/step_0000.png").exists()
    assert (tmp_path / "probes/sample/step_0001.png").exists()
    assert (tmp_path / "probes/sample/adaptation_grid.png").exists()
