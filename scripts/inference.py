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
    resolve_inference_output_dir,
)
from scripts.dataset import build_dataloaders
from scripts.engine.inference import export_inference_outputs
from scripts.models import HRNetLandmarkVisibility
from scripts.utils import get_default_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone inference."""
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Run standalone inference from a trained landmark detection checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint used to generate predictions on the test split.",
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
        help="Directory where predicted labels and overlays will be written. If omitted, it is derived from the checkpoint name.",
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
        help="Mini-batch size for inference. If omitted, the training batch size from config is reused.",
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
        help="Device used to run inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed used for deterministic inference.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=defaults.visibility_threshold,
        help="Threshold applied to visibility logits to obtain binary predictions.",
    )
    parser.add_argument(
        "--disable-overlays",
        action="store_true",
        default=not defaults.save_inference_overlays,
        help="Disable writing image overlays with predicted landmarks.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        default=defaults.show_landmark_indices,
        help="Draw landmark indices next to each predicted point.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save the resolved configuration JSON next to the inference outputs.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build an inference config from CLI overrides."""
    config = build_config()
    config.dataset_root = args.dataset_root
    config.cache_dir = args.cache_dir
    config.eval_batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device
    config.seed = args.seed
    config.visibility_threshold = args.visibility_threshold
    config.save_inference_overlays = not args.disable_overlays
    config.show_landmark_indices = args.show_indices

    if args.output_dir is None:
        resolve_inference_output_dir(config, args.checkpoint)
    else:
        config.output_dir = args.output_dir

    return config


def maybe_save_config(config: ExperimentConfig, output_dir: Path) -> None:
    """Save the resolved config as JSON."""
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_serializable_dict(config), indent=2),
        encoding="utf-8",
    )


def build_model(config: ExperimentConfig) -> HRNetLandmarkVisibility:
    """Instantiate the model used for inference."""
    return HRNetLandmarkVisibility(num_landmarks=config.num_landmarks)


def main() -> None:
    """Run standalone inference."""
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

    summary = export_inference_outputs(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        output_dir=output_dir,
        visibility_threshold=config.visibility_threshold,
        save_overlays=config.save_inference_overlays,
        show_indices=config.show_landmark_indices,
    )

    print("[INFO] Inference finished.")
    print(f"[INFO] Samples processed: {summary['num_samples']}")
    print(f"[INFO] Labels dir: {summary['prediction_labels_dir']}")
    if summary["prediction_overlays_dir"] is not None:
        print(f"[INFO] Overlays dir: {summary['prediction_overlays_dir']}")


if __name__ == "__main__":
    main()
