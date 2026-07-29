from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import evaluate as standalone_evaluate
from scripts.engine import full_evaluation


def test_infanface_cli_resolves_dedicated_protocol(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--eval-mode",
            "infanface",
            "--checkpoint",
            "model.pth",
            "--dataset-root",
            "crops",
            "--natural-gt-root",
            "labels",
            "--natural-source-root",
            "sources",
        ],
    )

    args = standalone_evaluate.parse_args()
    config = standalone_evaluate.build_config_from_args(args)

    assert config.dataset_protocol == "infanface"
    assert config.eval_mode == "natural"
    assert config.infanface_crop_root == Path("crops")
    assert config.infanface_gt_root == Path("labels")
    assert config.infanface_source_root == Path("sources")


def test_legacy_natural_mode_remains_a_babyland_alias() -> None:
    assert standalone_evaluate.resolve_dataset_protocol("natural") == "babyland"
    assert standalone_evaluate.resolve_dataset_protocol("babyland") == "babyland"
    assert standalone_evaluate.resolve_dataset_protocol("infanface") == "infanface"


def test_dataset_name_is_derived_instead_of_exposed_as_cli_argument(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--checkpoint", "model.pth"],
    )

    args = standalone_evaluate.parse_args()

    assert not hasattr(args, "dataset_name")


def test_pca_tta_cli_options_are_resolved(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--eval-mode",
            "babyland",
            "--checkpoint",
            "full_model.pth",
            "--pca-tta",
            "--pca-prior-path",
            "prior.pt",
            "--pca-tta-steps",
            "7",
            "--pca-tta-learning-rate",
            "0.00002",
            "--pca-tta-monitor-steps",
            "0",
            "1",
            "7",
        ],
    )

    args = standalone_evaluate.parse_args()
    config = standalone_evaluate.build_config_from_args(args)

    assert config.pca_tta_enabled is True
    assert config.pca_prior_path == Path("prior.pt")
    assert config.pca_tta_steps == 7
    assert config.pca_tta_learning_rate == 0.00002
    assert config.pca_tta_monitor_steps == (0, 1, 7)


def test_infanface_evaluator_reuses_provided_dataloader(
    tmp_path: Path, monkeypatch
) -> None:
    class Loader:
        dataset = [object(), object()]

    loader = Loader()
    config = SimpleNamespace(
        infanface_crop_root=tmp_path / "crops",
        infanface_gt_root=tmp_path / "labels",
        infanface_source_root=None,
        visibility_threshold=0.5,
        save_inference_overlays=True,
        show_landmark_indices=False,
        overlay_point_radius=2,
        overlay_line_width=2,
        overlay_connection_color="#ffffff",
        save_natural_crop_overlays=False,
        landmark_loss="wasserstein",
        coordinate_decoder="barycenter",
        wasserstein_softmax_temperature=1.0,
        use_landmark_names_in_boxplot=True,
    )
    monkeypatch.setattr(
        full_evaluation,
        "build_inference_dataloader",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provided dataloader should be reused")
        ),
    )
    monkeypatch.setattr(
        full_evaluation,
        "export_inference_outputs",
        lambda **_kwargs: {
            "num_samples": 2,
            "predictions": torch.empty(0),
            "prediction_labels_dir": str(tmp_path / "predictions" / "labels"),
            "prediction_overlays_dir": str(tmp_path / "predictions" / "images"),
            "prediction_crop_overlays_dir": None,
        },
    )
    monkeypatch.setattr(
        full_evaluation,
        "benchmark_infantface_prediction_directory",
        lambda **_kwargs: {
            "total_images": 2,
            "images_with_prediction": 2,
            "mean_nme_box": 0.1,
            "mean_hausdorff_box": 0.2,
        },
    )

    result = full_evaluation.evaluate_infanface(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        config=config,
        output_dir=tmp_path,
        dataloader=loader,
    )

    assert result["inference"]["num_samples"] == 2
    assert result["metrics"]["mean_hausdorff_box"] == 0.2
    assert result["output_dir"] == str(tmp_path)
