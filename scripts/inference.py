from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import ExperimentConfig
from scripts.dataset import build_dataloaders
from scripts.engine.inference import export_inference_outputs
from scripts.models import HRNetLandmarkVisibility
from scripts.utils import get_default_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone inference."""
    parser = argparse.ArgumentParser(
        description="Run standalone inference from a trained landmark detection checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visibility-threshold", type=float, default=None)
    parser.add_argument("--disable-overlays", action="store_true")
    parser.add_argument("--show-indices", action="store_true")
    parser.add_argument("--save-config", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build an inference config from CLI overrides."""
    config = ExperimentConfig()

    if args.dataset_root is not None:
        config.dataset_root = args.dataset_root
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir
    if args.batch_size is not None:
        config.eval_batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.visibility_threshold is not None:
        config.visibility_threshold = args.visibility_threshold

    if args.output_dir is None:
        config.output_dir = args.checkpoint.parent / f"{args.checkpoint.stem}_inference"

    return config


def maybe_save_config(config: ExperimentConfig, output_dir: Path) -> None:
    """Save the resolved config as JSON."""
    serialized = {
        key: str(value)
        if isinstance(value, Path)
        else list(value)
        if isinstance(value, tuple)
        else value
        for key, value in config.to_dict().items()
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(serialized, indent=2),
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
        save_overlays=not args.disable_overlays,
        show_indices=args.show_indices,
    )

    print("[INFO] Inference finished.")
    print(f"[INFO] Samples processed: {summary['num_samples']}")
    print(f"[INFO] Labels dir: {summary['prediction_labels_dir']}")
    if summary["prediction_overlays_dir"] is not None:
        print(f"[INFO] Overlays dir: {summary['prediction_overlays_dir']}")


if __name__ == "__main__":
    main()
