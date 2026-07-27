from __future__ import annotations

import ast
import csv
from argparse import Namespace
from pathlib import Path

import torch
import yaml
from torch import nn

from scripts.config import apply_argparse_arguments, build_config, load_yaml_config
from scripts.engine.evaluation_reporting import (
    collect_official_metric_rows,
    write_evaluation_report,
    write_official_metric_exports,
)
from scripts.engine.evaluate import save_metrics_summary_csv
from scripts.engine.normalizer_experiments import (
    _save_visual_example,
    run_normalizer_diagnostics,
    save_modular_checkpoints,
)
from scripts.engine.normalizer_monitoring import NormalizerProbeMonitor
from scripts.main import build_optimizer, validate_experiment_config
from scripts.models import (
    HRNetLandmarkVisibility,
    NormalizedLandmarker,
    ResidualImageNormalizer,
    build_model_from_checkpoints,
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


def test_official_metric_exports_unwrap_metrics_and_keep_hausdorff(
    tmp_path: Path,
) -> None:
    summaries = {
        "babyland": {
            "mean_nme_box_gt_valid": 0.1041,
            "mean_hausdorff_box_gt_valid": 0.201,
        },
        "infanface": {
            "inference": {"total_images": 20},
            "metrics": {
                "mean_nme_box": 0.09,
                "mean_hausdorff_box": 0.18,
            },
        },
    }

    rows = collect_official_metric_rows(summaries)
    write_official_metric_exports(tmp_path, rows)

    row_keys = {(row["dataset"], row["metric"]) for row in rows}
    assert ("babyland", "mean_hausdorff_box_gt_valid") in row_keys
    assert ("infanface", "mean_hausdorff_box") in row_keys
    with (tmp_path / "official_metrics_wide.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        wide_rows = list(csv.DictReader(handle))
    assert wide_rows[0]["mean_nme_box_gt_valid"] == "0.1041"
    assert wide_rows[1]["mean_hausdorff_box"] == "0.18"
    assert (tmp_path / "official_metrics_long.csv").exists()
    assert (tmp_path / "official_metrics_copy_paste.tsv").exists()


def test_official_metric_exports_include_orientation_nme_and_hausdorff(
    tmp_path: Path,
) -> None:
    summaries = {
        "babyland": {
            "mean_nme_box_gt_valid": 0.1041,
            "orientation_metrics": {
                "frontal": {
                    "mean_nme_box_gt_valid": 0.08,
                    "mean_hausdorff_box_gt_valid": 0.15,
                }
            },
        }
    }

    rows = collect_official_metric_rows(summaries)
    paths = write_evaluation_report(tmp_path, summaries)

    orientation_rows = [row for row in rows if row["category"] == "Orientation"]
    assert {row["metric"] for row in orientation_rows} == {
        "orientation_frontal_mean_nme_box_gt_valid",
        "orientation_frontal_mean_hausdorff_box_gt_valid",
    }
    report = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "### Orientation" in report
    assert "orientation_frontal_mean_hausdorff_box_gt_valid" in report


def test_dataset_metrics_summary_includes_all_orientation_hausdorff_statistics(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "metrics_summary.csv"
    save_metrics_summary_csv(
        output_path,
        {
            "orientation_sample_counts": {"frontal": 4},
            "orientation_metrics": {
                "frontal": {
                    "mean_nme_box_gt_valid": 0.08,
                    "mean_hausdorff_box_gt_valid": 0.15,
                    "median_hausdorff_box_gt_valid": 0.14,
                    "p90_hausdorff_box_gt_valid": 0.20,
                    "p95_hausdorff_box_gt_valid": 0.22,
                    "p99_hausdorff_box_gt_valid": 0.25,
                }
            },
        },
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        metrics = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert metrics["mean_hausdorff_box_gt_valid_frontal"] == "0.1500"
    assert metrics["p99_hausdorff_box_gt_valid_frontal"] == "0.2500"


def test_repository_wide_evaluation_report_is_experiment_agnostic(
    tmp_path: Path,
) -> None:
    paths = write_evaluation_report(
        reports_dir=tmp_path,
        evaluation_summaries={
            "synbaby": {
                "mean_nme_box": 0.05,
                "mean_hausdorff_box": 0.12,
            }
        },
    )

    report = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "independently of the experiment mode" in report
    assert "mean_nme_box" in report
    assert "mean_hausdorff_box" in report
    assert Path(paths["wide_csv"]).exists()


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


def test_joint_finetune_trains_normalizer_stage4_transition3_and_heads_only() -> None:
    wrapper = NormalizedLandmarker(
        landmarker=HRNetLandmarkVisibility(num_landmarks=72),
        normalizer=ResidualImageNormalizer(initialize_identity=True),
    )
    wrapper.configure_joint_finetune(num_unfrozen_stages=1, unfreeze_stem=False)
    trainable_names = {
        name
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }

    assert any(name.startswith("normalizer.") for name in trainable_names)
    assert any(
        name.startswith("landmarker.backbone.transition3") for name in trainable_names
    )
    assert any(
        name.startswith("landmarker.backbone.stage4") for name in trainable_names
    )
    assert any(
        name.startswith("landmarker.visibility_feature_head")
        for name in trainable_names
    )
    assert not any(
        name.startswith("landmarker.backbone.stage3") for name in trainable_names
    )
    assert not any(
        name.startswith("landmarker.backbone.conv1") for name in trainable_names
    )

    wrapper.train()
    assert wrapper.landmarker.backbone.stage3.training is False
    assert wrapper.landmarker.backbone.stage4.training is True


def test_stem_finetune_trains_stem_stage4_and_heads_only() -> None:
    model = HRNetLandmarkVisibility(num_landmarks=72)
    model.set_stem_stage4_finetune(unfreeze_stage1=False)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert any(name.startswith("backbone.conv1") for name in trainable_names)
    assert any(name.startswith("backbone.bn1") for name in trainable_names)
    assert any(name.startswith("backbone.conv2") for name in trainable_names)
    assert any(name.startswith("backbone.bn2") for name in trainable_names)
    assert any(name.startswith("backbone.transition3") for name in trainable_names)
    assert any(name.startswith("backbone.stage4") for name in trainable_names)
    assert any(name.startswith("visibility_feature_head") for name in trainable_names)
    assert not any(name.startswith("backbone.layer1") for name in trainable_names)
    assert not any(name.startswith("backbone.transition1") for name in trainable_names)
    assert not any(name.startswith("backbone.stage2") for name in trainable_names)
    assert not any(name.startswith("backbone.transition2") for name in trainable_names)
    assert not any(name.startswith("backbone.stage3") for name in trainable_names)

    model.train()
    assert model.backbone.bn1.training is True
    assert model.backbone.layer1.training is False
    assert model.backbone.stage2.training is False
    assert model.backbone.stage3.training is False
    assert model.backbone.stage4.training is True


def test_stem_finetune_can_optionally_train_stage1() -> None:
    model = HRNetLandmarkVisibility(num_landmarks=72)
    model.set_stem_stage4_finetune(unfreeze_stage1=True)

    assert all(
        parameter.requires_grad for parameter in model.backbone.layer1.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in model.backbone.stage2.parameters()
    )


def test_stem_finetune_uses_common_learning_rate_by_default() -> None:
    model = HRNetLandmarkVisibility(num_landmarks=72)
    model.set_stem_stage4_finetune(unfreeze_stage1=False)
    config = build_config()
    config.experiment_mode = "hrnet_stem_finetune"
    config.learning_rate = 1.0e-4
    config.use_differential_learning_rates = False

    optimizer = build_optimizer(model, config)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1.0e-4


def test_stem_finetune_supports_differential_component_learning_rates() -> None:
    model = HRNetLandmarkVisibility(num_landmarks=72)
    model.set_stem_stage4_finetune(unfreeze_stage1=True)
    config = build_config()
    config.experiment_mode = "hrnet_stem_finetune"
    config.use_differential_learning_rates = True
    config.stem_learning_rate = 1.0e-5
    config.stage1_learning_rate = 2.0e-5
    config.stage4_learning_rate = 3.0e-5
    config.heads_learning_rate = 1.0e-4

    optimizer = build_optimizer(model, config)
    learning_rates = {
        str(group["name"]): float(group["lr"]) for group in optimizer.param_groups
    }

    assert learning_rates == {
        "stem": 1.0e-5,
        "stage1": 2.0e-5,
        "stage4": 3.0e-5,
        "heads": 1.0e-4,
    }


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
    assert (tmp_path / "image_comparisons/tiny/side_by_side/tiny.png").exists()
    assert (tmp_path / "image_comparisons/tiny/residual_abs_fixed/tiny.png").exists()
    assert (tmp_path / "image_comparisons/tiny/residual_signed/tiny.png").exists()
    assert (
        tmp_path / "image_comparisons/tiny/normalized_change_amplified/tiny.png"
    ).exists()
    assert (tmp_path / "image_comparisons/README.md").exists()
    assert (tmp_path / "diagnostic_tables/tiny_normalizer_diagnostics.json").exists()


def test_residual_visualizations_use_fixed_scale_and_preserve_sign(
    tmp_path: Path,
) -> None:
    input_image = torch.zeros(3, 4, 4)
    normalized_image = input_image.clone()
    normalized_image[0] += 0.01
    normalized_image[2] -= 0.01

    _save_visual_example(
        input_image=input_image,
        normalized_image=normalized_image,
        output_root=tmp_path,
        sample_id="signed",
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        residual_display_scale=0.02,
        residual_amplification=10.0,
    )

    from PIL import Image

    signed_image = Image.open(tmp_path / "residual_signed/signed.png")
    fixed_absolute_image = Image.open(tmp_path / "residual_abs_fixed/signed.png")
    signed = torch.tensor(bytearray(signed_image.tobytes())).reshape(4, 4, 3)
    fixed_absolute = torch.tensor(bytearray(fixed_absolute_image.tobytes())).reshape(
        4, 4, 3
    )
    panel = Image.open(tmp_path / "side_by_side/signed.png")

    assert int(signed[0, 0, 0]) > 128
    assert abs(int(signed[0, 0, 1]) - 128) <= 1
    assert int(signed[0, 0, 2]) < 128
    assert 120 <= int(fixed_absolute[0, 0, 0]) <= 135
    assert int(fixed_absolute[0, 0, 1]) == 0
    assert 120 <= int(fixed_absolute[0, 0, 2]) <= 135
    assert panel.width == 5 * input_image.shape[-1]
    assert panel.height > input_image.shape[-2]


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


def test_full_and_split_normalizer_checkpoints_reconstruct_same_model() -> None:
    source = NormalizedLandmarker(
        landmarker=HRNetLandmarkVisibility(num_landmarks=72),
        normalizer=ResidualImageNormalizer(
            hidden_channels=5,
            num_layers=2,
            initialize_identity=True,
        ),
    )
    architecture = source.normalizer.architecture_config()
    full_model = build_model_from_checkpoints(
        {
            "model_state_dict": source.state_dict(),
            "normalizer_architecture": architecture,
        },
        num_landmarks=72,
    )
    split_model = build_model_from_checkpoints(
        {"model_state_dict": source.landmarker.state_dict()},
        num_landmarks=72,
        normalizer_checkpoint={
            "normalizer_state_dict": source.normalizer.state_dict(),
            "architecture": architecture,
        },
    )

    assert isinstance(full_model, NormalizedLandmarker)
    assert isinstance(split_model, NormalizedLandmarker)
    for expected, actual in zip(source.parameters(), full_model.parameters()):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(source.parameters(), split_model.parameters()):
        torch.testing.assert_close(expected, actual)


def test_modular_export_always_includes_landmarker_without_normalizer(
    tmp_path: Path,
) -> None:
    model = NormalizedLandmarker(
        landmarker=HRNetLandmarkVisibility(num_landmarks=72),
        normalizer=ResidualImageNormalizer(initialize_identity=True),
    )
    saved = save_modular_checkpoints(
        model=model,
        output_dir=tmp_path,
        base_checkpoint_path=None,
        experiment_mode="normalizer_train_frozen_landmarker",
        resolved_config_path=tmp_path / "resolved.yaml",
        landmarker_updated=False,
        normalizer_updated=True,
        decoder_name="barycenter",
        loss_pipeline_name="wasserstein",
        evaluation_protocol="test",
    )

    landmarker_payload = torch.load(
        saved["landmarker_best.pth"], map_location="cpu", weights_only=False
    )
    assert "model_state_dict" in landmarker_payload
    assert all(
        not key.startswith("normalizer.")
        for key in landmarker_payload["model_state_dict"]
    )
    assert Path(saved["full_model_best.pth"]).exists()
    assert Path(saved["normalizer_best.pth"]).exists()


def test_shared_yaml_supports_argparser_names_and_inverse_flags() -> None:
    config = load_yaml_config("configs/normalizer_experiments.yaml")
    assert config.runs_dir == Path("/home/jocareher/Downloads/landmark-detection/runs")
    assert config.checkpoint_path is None
    assert config.batch_size == 32
    assert config.num_epochs == 60
    assert config.learning_rate == 1.0e-4
    assert config.landmark_loss == "wasserstein"
    assert config.pca_prior_path == Path(
        "/home/jocareher/Downloads/landmark-detection/weights/pca_prior_13y_k32.pt"
    )
    assert config.lambda_pca_projection == 1.0
    assert config.transfer_mode == "fine_tuning"
    assert config.num_unfrozen_stages == 1
    assert config.use_wandb is True
    assert config.use_amp is True
    assert config.use_cache is True
    assert config.save_config is False
    assert config.enable_photometric_augmentations is True


def test_joint_finetune_requires_pretrained_hrnet_but_not_landmarker_checkpoint(
    tmp_path: Path,
) -> None:
    pretrained_weights = tmp_path / "hrnet.pth"
    pretrained_weights.touch()
    config = build_config()
    config.experiment_mode = "normalizer_joint_finetune"
    config.checkpoint_path = None
    config.pretrained_weights = pretrained_weights
    config.landmark_loss = "wasserstein"
    config.coordinate_decoder = "barycenter"
    config.transfer_mode = "fine_tuning"
    config.num_unfrozen_stages = 1

    validate_experiment_config(config)

    assert config.train_normalizer is True
    assert config.freeze_landmarker is False
    assert config.finetune_last_backbone_stage is True
    assert config.train_heads is True


def test_stem_finetune_requires_pretrained_hrnet_and_no_checkpoint(
    tmp_path: Path,
) -> None:
    pretrained_weights = tmp_path / "hrnet.pth"
    pretrained_weights.touch()
    config = build_config()
    config.experiment_mode = "hrnet_stem_finetune"
    config.checkpoint_path = None
    config.pretrained_weights = pretrained_weights
    config.transfer_mode = "fine_tuning"

    validate_experiment_config(config)

    assert config.train_normalizer is False
    assert config.freeze_landmarker is False
    assert config.finetune_last_backbone_stage is True
    assert config.train_heads is True


def test_shared_yaml_lists_every_argparser_destination() -> None:
    syntax_tree = ast.parse(Path("scripts/main.py").read_text(encoding="utf-8"))
    parser_destinations: set[str] = set()
    for node in ast.walk(syntax_tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        option = node.args[0].value
        if not isinstance(option, str) or not option.startswith("--"):
            continue
        destination = option.removeprefix("--").replace("-", "_")
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                destination = str(keyword.value.value)
        if destination != "config":
            parser_destinations.add(destination)

    yaml_payload = yaml.safe_load(
        Path("configs/normalizer_experiments.yaml").read_text(encoding="utf-8")
    )
    assert set(yaml_payload["arguments"]) == parser_destinations


def test_every_yaml_argument_reaches_resolved_config() -> None:
    yaml_payload = yaml.safe_load(
        Path("configs/normalizer_experiments.yaml").read_text(encoding="utf-8")
    )["arguments"]
    config = apply_argparse_arguments(build_config(), Namespace(**yaml_payload))

    assert config.transfer_mode == "fine_tuning"
    assert config.num_unfrozen_stages == 1
    assert config.unfreeze_stem is False
    assert config.use_amp is True
    assert config.use_cache is True
    assert config.normalizer_monitoring_enabled is True
    assert config.normalizer_monitor_steps == [0, 1, 5, 10, 20]


def test_fixed_probe_monitor_saves_panels_metrics_and_tta_losses(
    tmp_path: Path,
) -> None:
    wrapper = NormalizedLandmarker(
        landmarker=DummyLandmarker(),
        normalizer=ResidualImageNormalizer(initialize_identity=True),
    )
    probe_batch = {
        "image": torch.zeros(1, 3, 8, 8),
        "landmarks": torch.tensor([[[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]]),
        "visibility": torch.ones(1, 3),
        "metadata": {"sample_id": ["fixed-probe"]},
    }
    monitor = NormalizerProbeMonitor(
        model=wrapper,
        probe_batch=probe_batch,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        coordinate_decoder="barycenter",
        softmax_temperature=1.0,
        max_images=1,
    )
    summary = monitor.capture(
        stage="tta",
        step=0,
        adaptation_loss=1.0,
        structural_prior_loss=0.5,
    )
    monitor.capture(
        stage="tta",
        step=1,
        adaptation_loss=0.8,
        structural_prior_loss=0.4,
        is_final=True,
    )

    assert summary["mean_mean_absolute_pixel_difference"] == 0.0
    assert summary["geometry_warning_count"] == 0.0
    assert (tmp_path / "panels/tta_step_000000/fixed-probe.png").exists()
    assert (tmp_path / "checkpoint_grids/fixed-probe.png").exists()
    assert (tmp_path / "animations/fixed-probe.gif").exists()
    assert (tmp_path / "probe_metrics.csv").exists()
    assert (tmp_path / "probe_metrics_by_checkpoint.csv").exists()
    assert (tmp_path / "probe_metrics_final_summary.csv").exists()
    assert (tmp_path / "plots/probe_metric_trajectories.png").exists()
    assert (tmp_path / "plots/final_probe_profile.png").exists()
    assert (tmp_path / "adaptation_losses.csv").exists()
    assert (tmp_path / "adaptation_losses.png").exists()
