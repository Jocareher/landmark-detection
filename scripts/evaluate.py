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
from scripts.dataset import build_dataloaders
from scripts.engine.evaluate import evaluate_checkpoint
from scripts.models import HRNetLandmarkVisibility
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
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint to evaluate on the test split.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=defaults.dataset_root,
        help="Root directory that contains the dataset splits.",
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
    config.cache_dir = args.cache_dir
    config.eval_batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device
    config.seed = args.seed
    config.visibility_threshold = args.visibility_threshold
    config.use_landmark_names_in_boxplot = args.use_landmark_names

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


def build_model(config: ExperimentConfig) -> HRNetLandmarkVisibility:
    """
    Instantiate the model.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    HRNetLandmarkVisibility
        Instantiated model.
    """
    model = HRNetLandmarkVisibility(num_landmarks=config.num_landmarks)
    return model


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

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Dataset root: {config.dataset_root}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Output dir: {output_dir}")

    dataloaders = build_dataloaders(config)

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if args.save_config:
        maybe_save_config(config, output_dir)

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
    )

    print("[INFO] Evaluation finished.")
    print(f"[INFO] Mean NME: {summary['mean_nme']:.6f}")
    print(f"[INFO] Median NME: {summary['median_nme']:.6f}")
    print(f"[INFO] Visibility accuracy: {summary['visibility_accuracy']:.6f}")
    print(f"[INFO] Labels dir: {summary['prediction_labels_dir']}")
    if summary["prediction_overlays_dir"] is not None:
        print(f"[INFO] Overlays dir: {summary['prediction_overlays_dir']}")


if __name__ == "__main__":
    main()
