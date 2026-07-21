from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import (
    ExperimentConfig,
    build_config,
    config_to_serializable_dict,
    resolve_evaluation_output_dir,
)
from scripts.dataset import build_dataloaders, build_natural_evaluation_dataloader
from scripts.engine.evaluate import evaluate_checkpoint
from scripts.engine.evaluate_natural import evaluate_natural_checkpoint
from scripts.engine.metrics import decoder_from_landmark_loss
from scripts.models import build_model_from_checkpoints
from scripts.utils import get_default_device, set_seed


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for standalone checkpoint evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained landmark detection checkpoint on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-mode",
        choices=["synthetic", "natural"],
        default="synthetic",
        help="Evaluation mode. Synthetic expects the repository split layout, while natural expects a detector-export root with images/metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to evaluate on the test split.",
    )
    parser.add_argument(
        "--normalizer-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional normalizer-only checkpoint to combine with a landmarker-only "
            "--checkpoint. Do not use it with a full-model checkpoint."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=defaults.dataset_root,
        help="Synthetic dataset root in synthetic mode, or detector-export root in natural mode.",
    )
    parser.add_argument(
        "--natural-gt-root",
        type=Path,
        default=None,
        help="Directory that contains GT txt files for the original natural images. Required in natural mode.",
    )
    parser.add_argument(
        "--natural-source-root",
        type=Path,
        default=None,
        help="Optional root used to resolve relative source_image_path values from natural detector metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where evaluation metrics and figures will be written. If omitted, it is derived from the checkpoint name.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.cache_dir,
        help="Directory used to store cached dataset files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.eval_batch_size,
        help="Mini-batch size for the test dataloader. If omitted, the training batch size from config is reused.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.num_workers,
        help="Number of worker processes for the dataloader.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=defaults.device,
        help="Device used to run evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed used for deterministic evaluation.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=defaults.visibility_threshold,
        help="Threshold applied to visibility logits to obtain binary predictions.",
    )
    parser.add_argument(
        "--landmark-loss",
        choices=["mse", "adaptive_wing", "wasserstein"],
        default=defaults.landmark_loss,
        help="Loss regime used by the checkpoint; controls coordinate decoding.",
    )
    parser.add_argument(
        "--wasserstein-softmax-temperature",
        type=float,
        default=defaults.wasserstein_softmax_temperature,
        help="Spatial softmax temperature used by barycenter decoding.",
    )
    parser.add_argument(
        "--use-landmark-names",
        action="store_true",
        default=defaults.use_landmark_names_in_boxplot,
        help="Use landmark names instead of indices on evaluation boxplots.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save the resolved configuration JSON next to the evaluation outputs.",
    )
    parser.add_argument(
        "--save-crop-overlays",
        action="store_true",
        default=defaults.save_natural_crop_overlays,
        help="In natural mode, additionally save qualitative overlays on the detector crops under predictions/crops/.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """
    Build an evaluation config from CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    ExperimentConfig
        Resolved config object.
    """
    config = build_config()
    config.dataset_root = args.dataset_root
    config.eval_mode = args.eval_mode
    config.natural_gt_root = args.natural_gt_root
    config.natural_source_root = args.natural_source_root
    config.cache_dir = args.cache_dir
    config.eval_batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device
    config.target_mode = "regression"
    config.seed = args.seed
    config.visibility_threshold = args.visibility_threshold
    config.landmark_loss = args.landmark_loss
    config.coordinate_decoder = decoder_from_landmark_loss(args.landmark_loss)
    config.wasserstein_softmax_temperature = args.wasserstein_softmax_temperature
    config.use_landmark_names_in_boxplot = args.use_landmark_names
    config.save_natural_crop_overlays = args.save_crop_overlays

    if args.output_dir is None:
        resolve_evaluation_output_dir(config, args.checkpoint)
    else:
        config.output_dir = args.output_dir

    return config


def maybe_save_config(
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    """
    Save the resolved config as JSON.

    Parameters
    ----------
    config : ExperimentConfig
        Resolved config.
    output_dir : Path
        Evaluation output directory.
    """
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_serializable_dict(config), indent=2),
        encoding="utf-8",
    )


def _normalizer_architecture_from_config(config: ExperimentConfig) -> dict[str, object]:
    """Return fallback normalizer metadata for legacy full checkpoints."""
    return {
        "type": "residual",
        "input_channels": config.normalizer_input_channels,
        "hidden_channels": config.normalizer_hidden_channels,
        "num_layers": config.normalizer_num_layers,
        "kernel_size": config.normalizer_kernel_size,
        "activation": config.normalizer_activation,
        "normalization": config.normalizer_internal_normalization,
        "residual_scale": config.normalizer_residual_scale,
        "initialize_identity": config.normalizer_initialize_identity,
        "clamp_output": config.normalizer_clamp_output,
        "clamp_min": config.normalizer_clamp_min,
        "clamp_max": config.normalizer_clamp_max,
    }


def print_visibility_metrics(summary: dict) -> None:
    """Print visibility precision, recall, and F1 metrics."""
    metrics = summary.get("visibility_metrics", {})
    for label, display_name in (
        ("global", "Global"),
        ("visible", "Visible class"),
        ("invisible", "Invisible class"),
    ):
        current = metrics.get(label, {})
        print(
            f"[INFO] Visibility {display_name}: "
            f"precision={current.get('precision', 0.0):.4f} "
            f"recall={current.get('recall', 0.0):.4f} "
            f"f1={current.get('f1', 0.0):.4f}"
        )


def main() -> None:
    """
    Run standalone evaluation.
    """
    args = parse_args()
    config = build_config_from_args(args)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = get_default_device(config.device)

    if args.eval_mode == "natural" and args.natural_gt_root is None:
        raise ValueError(
            "--natural-gt-root is required when --eval-mode natural is used."
        )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Eval mode: {args.eval_mode}")
    print(f"[INFO] Dataset root: {config.dataset_root}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Output dir: {output_dir}")
    print(
        "[INFO] Loss/decoder: "
        f"landmark_loss={config.landmark_loss}, "
        f"coordinate_decoder={config.coordinate_decoder}"
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer_checkpoint = (
        torch.load(args.normalizer_checkpoint, map_location="cpu", weights_only=False)
        if args.normalizer_checkpoint is not None
        else None
    )
    model = build_model_from_checkpoints(
        checkpoint,
        num_landmarks=config.num_landmarks,
        normalizer_checkpoint=normalizer_checkpoint,
        fallback_normalizer_architecture=_normalizer_architecture_from_config(config),
    )
    model.to(device)

    if args.save_config:
        maybe_save_config(config, output_dir)

    if args.eval_mode == "synthetic":
        dataloaders = build_dataloaders(config)
        summary = evaluate_checkpoint(
            model=model,
            dataloader=dataloaders["test"],
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
            landmark_loss=config.landmark_loss,
            coordinate_decoder=config.coordinate_decoder,
            wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
        )
    else:
        dataloader = build_natural_evaluation_dataloader(
            export_root=config.dataset_root,
            gt_root=args.natural_gt_root,
            source_root=args.natural_source_root,
            config=config,
        )
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

    print("[INFO] Evaluation finished.")
    has_explicit_visible_intersection = (
        "mean_nme_box_visible_intersection" in summary
        or "mean_nme_box_point_to_line_visible_intersection" in summary
    )
    if has_explicit_visible_intersection:
        for key, label in (
            (
                "mean_nme_box_visible_intersection",
                "Mean NME box visible-intersection",
            ),
            (
                "median_nme_box_visible_intersection",
                "Median NME box visible-intersection",
            ),
            (
                "mean_nme_box_point_to_line_visible_intersection",
                "Mean NME box point-to-line visible-intersection",
            ),
            (
                "median_nme_box_point_to_line_visible_intersection",
                "Median NME box point-to-line visible-intersection",
            ),
            ("mean_nme_box_gt_valid", "Mean NME box GT-valid"),
            ("median_nme_box_gt_valid", "Median NME box GT-valid"),
            (
                "mean_nme_box_point_to_line_gt_valid",
                "Mean NME box point-to-line GT-valid",
            ),
            (
                "median_nme_box_point_to_line_gt_valid",
                "Median NME box point-to-line GT-valid",
            ),
        ):
            metric_value = summary.get(key)
            if metric_value is not None:
                print(f"[INFO] {label}: {metric_value:.4f}")
            else:
                print(f"[INFO] {label}: n/a")
    else:
        if summary["mean_nme_box"] is not None:
            print(f"[INFO] Mean NME box: {summary['mean_nme_box']:.4f}")
        else:
            print("[INFO] Mean NME box: n/a")
        if summary["median_nme_box"] is not None:
            print(f"[INFO] Median NME box: {summary['median_nme_box']:.4f}")
        else:
            print("[INFO] Median NME box: n/a")
        if summary.get("mean_nme_box_point_to_line") is not None:
            print(
                "[INFO] Mean NME box point-to-line: "
                f"{summary['mean_nme_box_point_to_line']:.4f}"
            )
        else:
            print("[INFO] Mean NME box point-to-line: n/a")
        if summary.get("median_nme_box_point_to_line") is not None:
            print(
                "[INFO] Median NME box point-to-line: "
                f"{summary['median_nme_box_point_to_line']:.4f}"
            )
        else:
            print("[INFO] Median NME box point-to-line: n/a")
    if summary["mean_nme_interocular"] is not None:
        print(f"[INFO] Mean NME interocular: {summary['mean_nme_interocular']:.4f}")
    print_visibility_metrics(summary)
    print(f"[INFO] Labels dir: {summary['prediction_labels_dir']}")
    if summary["prediction_overlays_dir"] is not None:
        print(f"[INFO] Overlays dir: {summary['prediction_overlays_dir']}")
    if summary.get("prediction_crop_overlays_dir") is not None:
        print(f"[INFO] Crop overlays dir: {summary['prediction_crop_overlays_dir']}")


if __name__ == "__main__":
    main()
