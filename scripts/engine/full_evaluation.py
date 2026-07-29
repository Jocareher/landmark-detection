from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..dataset import build_natural_evaluation_dataloader
from ..inference import build_inference_dataloader
from .benchmark import benchmark_infantface_prediction_directory
from .evaluate import evaluate_checkpoint
from .evaluate_natural import evaluate_natural_checkpoint
from .evaluation_reporting import write_evaluation_report
from .inference import export_inference_outputs


def _optional_path(value: str | Path | None) -> Path | None:
    """Convert an optional config path value to Path."""
    if value is None:
        return None
    return Path(value)


def _require_path(config: ExperimentConfig, field_name: str, dataset_name: str) -> Path:
    """Fetch a required dataset-specific path from config."""
    value = _optional_path(getattr(config, field_name, None))
    if value is None:
        raise ValueError(
            f"{dataset_name} evaluation is enabled but config.{field_name} is not set."
        )
    return value


def _format_metric(value: Any) -> str:
    """Format optional numeric metrics for terminal summaries."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_evaluation_readme(evaluation_root: Path) -> Path:
    """Document the canonical evaluation artifact layout and metric scope."""
    path = evaluation_root / "README.md"
    path.write_text(
        "\n".join(
            [
                "# Evaluation outputs",
                "",
                "Each evaluated dataset uses the same top-level structure:",
                "",
                "```text",
                "<dataset>/",
                "  figures/",
                "  predictions/",
                "    images/",
                "    labels/",
                "  metrics_summary.csv",
                "  summary.json",
                "  per_image_nme.csv",
                "  per_image_per_landmark_nme.csv",
                "```",
                "",
                "`figures/` contains evaluation plots. `predictions/images/` contains overlays and `predictions/labels/` contains exported landmark files. Dataset-level tables stay directly in the dataset folder.",
                "",
                "`reports/` is the only consolidated reporting directory. It contains the Markdown report and long, wide, and tab-separated tables intended for spreadsheet use.",
                "",
                "Orientation metrics are identified with the `orientation_<orientation>_` prefix in consolidated exports and include NME and Hausdorff whenever the dataset protocol supports them.",
                "",
                "Normalizer-only image-change diagnostics are intentionally stored outside this tree under `normalizer_diagnostics/`; they are not substitutes for model evaluation metrics.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _print_visibility_summary(summary: dict[str, Any], dataset_name: str) -> None:
    """Print visibility precision/recall/F1 metrics when available."""
    metrics = summary.get("visibility_metrics")
    if not metrics:
        return
    for label, display_name in (
        ("global", "Global"),
        ("visible", "Visible class"),
        ("invisible", "Invisible class"),
    ):
        current = metrics.get(label, {})
        print(
            f"[INFO] {dataset_name} visibility {display_name}: "
            f"precision={current.get('precision', 0.0):.4f} "
            f"recall={current.get('recall', 0.0):.4f} "
            f"f1={current.get('f1', 0.0):.4f}"
        )


def print_evaluation_summary(
    dataset_name: str,
    summary: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Print a concise dataset-specific evaluation summary."""
    output_dir = Path(output_dir)
    print(f"[INFO] {dataset_name} evaluation summary:")
    for key, label in (
        ("num_samples", "samples"),
        ("total_images", "total images"),
        ("images_with_prediction", "images with prediction"),
        ("images_without_prediction", "images without prediction"),
        ("num_samples_with_geometric_metrics", "samples with geometric metrics"),
        ("valid_landmarks_used", "valid landmarks used"),
    ):
        if key in summary:
            print(f"[INFO]   {label}: {summary[key]}")
    if "detection_rate" in summary and summary["detection_rate"] is not None:
        print(f"[INFO]   detection rate: {summary['detection_rate'] * 100.0:.2f}%")
    metric_labels = (
        (
            "mean_nme_box_visible_intersection",
            "mean NME box visible-intersection",
        ),
        (
            "median_nme_box_visible_intersection",
            "median NME box visible-intersection",
        ),
        (
            "mean_nme_box_point_to_line_visible_intersection",
            "mean NME box point-to-line visible-intersection",
        ),
        (
            "median_nme_box_point_to_line_visible_intersection",
            "median NME box point-to-line visible-intersection",
        ),
        (
            "mean_hausdorff_box_visible_intersection",
            "mean Hausdorff box visible-intersection",
        ),
        (
            "median_hausdorff_box_visible_intersection",
            "median Hausdorff box visible-intersection",
        ),
        ("mean_nme_box", "mean NME box"),
        ("median_nme_box", "median NME box"),
        ("mean_nme_box_point_to_line", "mean NME box point-to-line"),
        ("median_nme_box_point_to_line", "median NME box point-to-line"),
        ("mean_hausdorff_box", "mean Hausdorff box"),
        ("median_hausdorff_box", "median Hausdorff box"),
        ("mean_nme_box_gt_valid", "mean NME box GT-valid"),
        ("median_nme_box_gt_valid", "median NME box GT-valid"),
        (
            "mean_nme_box_point_to_line_gt_valid",
            "mean NME box point-to-line GT-valid",
        ),
        (
            "median_nme_box_point_to_line_gt_valid",
            "median NME box point-to-line GT-valid",
        ),
        ("mean_hausdorff_box_gt_valid", "mean Hausdorff box GT-valid"),
        ("median_hausdorff_box_gt_valid", "median Hausdorff box GT-valid"),
        ("mean_nme_box_non_contour", "mean NME box without contour"),
        ("median_nme_box_non_contour", "median NME box without contour"),
        (
            "mean_nme_box_point_to_line_non_contour",
            "mean NME box point-to-line without contour",
        ),
        (
            "median_nme_box_point_to_line_non_contour",
            "median NME box point-to-line without contour",
        ),
        (
            "mean_hausdorff_box_non_contour",
            "mean Hausdorff box without contour",
        ),
        (
            "median_hausdorff_box_non_contour",
            "median Hausdorff box without contour",
        ),
        ("mean_nme_interocular", "mean NME interocular"),
    )
    has_explicit_visible_intersection = (
        "mean_nme_box_visible_intersection" in summary
        or "mean_nme_box_point_to_line_visible_intersection" in summary
    )
    legacy_visible_keys = {
        "mean_nme_box",
        "median_nme_box",
        "mean_nme_box_point_to_line",
        "median_nme_box_point_to_line",
    }
    for key, label in metric_labels:
        if has_explicit_visible_intersection and key in legacy_visible_keys:
            continue
        if key in summary:
            print(f"[INFO]   {label}: {_format_metric(summary[key])}")
    _print_visibility_summary(summary, dataset_name)
    if summary.get("prediction_labels_dir") is not None:
        print(f"[INFO]   prediction labels: {summary['prediction_labels_dir']}")
    if summary.get("prediction_overlays_dir") is not None:
        print(f"[INFO]   prediction overlays: {summary['prediction_overlays_dir']}")
    if summary.get("prediction_crop_overlays_dir") is not None:
        print(f"[INFO]   crop overlays: {summary['prediction_crop_overlays_dir']}")
    print(f"[INFO]   output dir: {output_dir}")


def evaluate_synbaby(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the synthetic SynBaby evaluation."""
    output_dir = Path(output_dir)
    print("[INFO] SynBaby evaluation started...")
    print(f"[INFO] SynBaby output dir: {output_dir}")
    print(f"[INFO] SynBaby samples queued: {len(dataloader.dataset)}")
    print(
        "[INFO] SynBaby decoder: "
        f"{config.coordinate_decoder} (landmark_loss={config.landmark_loss})"
    )
    print(
        "[INFO] SynBaby metrics, reports, predictions, and plots are being generated..."
    )
    summary = evaluate_checkpoint(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        visibility_threshold=config.visibility_threshold,
        save_predictions=config.save_test_predictions_after_training,
        save_overlays=config.save_test_overlays_after_training,
        show_indices=config.show_landmark_indices,
        use_landmark_names_in_boxplot=config.use_landmark_names_in_boxplot,
        point_radius=config.overlay_point_radius,
        line_width=config.overlay_line_width,
        line_color=config.overlay_connection_color,
        landmark_loss=config.landmark_loss,
        coordinate_decoder=config.coordinate_decoder,
        wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
    )
    print("[INFO] SynBaby evaluation finished.")
    print_evaluation_summary("SynBaby", summary, output_dir)
    return summary


def evaluate_babyland(
    model: torch.nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run BabyLand crop inference, original-image reprojection, and evaluation."""
    crop_root = _require_path(config, "babyland_crop_root", "BabyLand")
    gt_root = _require_path(config, "babyland_gt_root", "BabyLand")
    source_root = _optional_path(getattr(config, "babyland_source_root", None))
    output_dir = Path(output_dir)

    print("[INFO] BabyLand evaluation started...")
    print(f"[INFO] BabyLand crop root: {crop_root}")
    print(f"[INFO] BabyLand GT root: {gt_root}")
    if source_root is not None:
        print(f"[INFO] BabyLand source root: {source_root}")
    print(f"[INFO] BabyLand output dir: {output_dir}")
    print("[INFO] BabyLand dataloader construction started...")
    dataloader = build_natural_evaluation_dataloader(
        export_root=crop_root,
        gt_root=gt_root,
        source_root=source_root,
        config=config,
    )
    print(f"[INFO] BabyLand samples queued: {len(dataloader.dataset)}")
    print(
        "[INFO] BabyLand decoder: "
        f"{config.coordinate_decoder} (landmark_loss={config.landmark_loss})"
    )
    print("[INFO] BabyLand inference started on crop images...")
    print(
        "[INFO] BabyLand reprojection started from crop to original-image coordinates..."
    )
    print("[INFO] BabyLand metrics computation started...")
    summary = evaluate_natural_checkpoint(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        visibility_threshold=config.visibility_threshold,
        save_predictions=config.save_evaluation_predictions,
        save_overlays=config.save_evaluation_overlays,
        show_indices=config.show_landmark_indices,
        use_landmark_names_in_boxplot=config.use_landmark_names_in_boxplot,
        point_radius=config.overlay_point_radius,
        line_width=config.overlay_line_width,
        line_color=config.overlay_connection_color,
        save_crop_overlays=config.save_natural_crop_overlays,
        landmark_loss=config.landmark_loss,
        coordinate_decoder=config.coordinate_decoder,
        wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
    )
    print("[INFO] BabyLand metrics computed.")
    print("[INFO] BabyLand evaluation finished.")
    print_evaluation_summary("BabyLand", summary, output_dir)
    return summary


def evaluate_infanface(
    model: torch.nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: str | Path,
    dataloader: torch.utils.data.DataLoader | None = None,
    print_summary: bool = True,
    tta_adapter: Any | None = None,
) -> dict[str, Any]:
    """Run InfAnFace crop inference, reprojection, and benchmark evaluation."""
    crop_root = _require_path(config, "infanface_crop_root", "InfAnFace")
    gt_root = _require_path(config, "infanface_gt_root", "InfAnFace")
    source_root = _optional_path(getattr(config, "infanface_source_root", None))
    output_dir = Path(output_dir)

    inference_config = type(config)(**vars(config).copy())
    inference_config.project_to_original = True
    inference_config.source_root = source_root

    print("[INFO] InfAnFace evaluation started...")
    print(f"[INFO] InfAnFace crop root: {crop_root}")
    print(f"[INFO] InfAnFace GT root: {gt_root}")
    if source_root is not None:
        print(f"[INFO] InfAnFace source root: {source_root}")
    print(f"[INFO] InfAnFace output dir: {output_dir}")
    print(f"[INFO] InfAnFace predictions output dir: {output_dir / 'predictions'}")
    print(f"[INFO] InfAnFace figures output dir: {output_dir / 'figures'}")
    print("[INFO] InfAnFace dataloader construction started...")
    if dataloader is None:
        dataloader = build_inference_dataloader(crop_root, inference_config)
    print(f"[INFO] InfAnFace samples queued: {len(dataloader.dataset)}")
    print(
        "[INFO] InfAnFace decoder: "
        f"{config.coordinate_decoder} (landmark_loss={config.landmark_loss})"
    )
    print("[INFO] InfAnFace inference started on crop images...")
    print(
        "[INFO] InfAnFace reprojection started from crop to original-image coordinates..."
    )
    inference_summary = export_inference_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        visibility_threshold=config.visibility_threshold,
        save_overlays=config.save_inference_overlays,
        show_indices=config.show_landmark_indices,
        point_radius=config.overlay_point_radius,
        line_width=config.overlay_line_width,
        line_color=config.overlay_connection_color,
        project_to_original=True,
        save_crop_overlays=config.save_natural_crop_overlays,
        landmark_loss=config.landmark_loss,
        coordinate_decoder=config.coordinate_decoder,
        wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
        tta_adapter=tta_adapter,
    )
    inference_summary_for_return = {
        key: value for key, value in inference_summary.items() if key != "predictions"
    }
    print(
        "[INFO] InfAnFace reprojected prediction labels saved: "
        f"{inference_summary_for_return['prediction_labels_dir']}"
    )

    print("[INFO] InfAnFace final evaluation started...")
    benchmark_summary = benchmark_infantface_prediction_directory(
        gt_root=gt_root,
        prediction_root=inference_summary_for_return["prediction_labels_dir"],
        output_dir=output_dir,
        use_landmark_names_in_boxplot=config.use_landmark_names_in_boxplot,
        fixed_log_y_limits=(1e-3, 1.0),
    )
    print("[INFO] InfAnFace final evaluation finished.")
    if print_summary:
        print_evaluation_summary("InfAnFace", benchmark_summary, output_dir)
    return {
        "inference": inference_summary_for_return,
        "metrics": benchmark_summary,
        "output_dir": str(output_dir),
    }


def run_full_evaluation(
    model: torch.nn.Module,
    synbaby_dataloader: torch.utils.data.DataLoader | None,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run all enabled experiment evaluations in the configured sequence."""
    evaluation_root = Path(config.output_dir) / config.evaluation_dirname
    evaluation_root.mkdir(parents=True, exist_ok=True)
    _write_evaluation_readme(evaluation_root)

    print("[INFO] Full evaluation started...")
    print(
        "[INFO] Full evaluation loss/decoder: "
        f"landmark_loss={config.landmark_loss}, "
        f"coordinate_decoder={config.coordinate_decoder}"
    )
    summaries: dict[str, Any] = {}
    output_dirs = {
        "synbaby": evaluation_root / "synbaby",
        "babyland": evaluation_root / "babyland",
        "infanface": evaluation_root / "infanface",
    }

    if config.evaluate_synbaby:
        if synbaby_dataloader is None:
            raise ValueError(
                "SynBaby evaluation is enabled but no dataloader was provided."
            )
        summaries["synbaby"] = evaluate_synbaby(
            model=model,
            dataloader=synbaby_dataloader,
            device=device,
            config=config,
            output_dir=output_dirs["synbaby"],
        )
    else:
        print("[INFO] SynBaby evaluation disabled by config.")

    if config.evaluate_babyland:
        summaries["babyland"] = evaluate_babyland(
            model=model,
            device=device,
            config=config,
            output_dir=output_dirs["babyland"],
        )
    else:
        print("[INFO] BabyLand evaluation disabled by config.")

    if config.evaluate_infanface:
        summaries["infanface"] = evaluate_infanface(
            model=model,
            device=device,
            config=config,
            output_dir=output_dirs["infanface"],
        )
    else:
        print("[INFO] InfAnFace evaluation disabled by config.")

    report_paths = write_evaluation_report(
        reports_dir=evaluation_root / "reports",
        evaluation_summaries=summaries,
    )
    print("[INFO] Full evaluation completed.")
    print("[INFO] Evaluation outputs:")
    for dataset_name, dataset_output_dir in output_dirs.items():
        if getattr(config, f"evaluate_{dataset_name}"):
            print(f"[INFO]   {dataset_name}: {dataset_output_dir}")
    print(f"[INFO]   consolidated report: {report_paths['markdown']}")
    print(f"[INFO]   Excel-ready CSV: {report_paths['wide_csv']}")
    print(f"[INFO]   copy/paste TSV: {report_paths['copy_paste_tsv']}")

    return {
        "evaluation_root": str(evaluation_root),
        "output_dirs": {key: str(value) for key, value in output_dirs.items()},
        "report_paths": report_paths,
        "summaries": summaries,
    }
