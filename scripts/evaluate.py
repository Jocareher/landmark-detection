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
    parser = argparse.ArgumentParser(
        description="Evaluate a trained landmark detection checkpoint on the test split.",
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
    parser.add_argument("--use-landmark-names", action="store_true")
    parser.add_argument("--save-config", action="store_true")
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
    if args.use_landmark_names:
        config.use_landmark_names_in_boxplot = True

    if args.output_dir is None:
        checkpoint_stem = args.checkpoint.stem
        config.output_dir = (
            args.checkpoint.parent / f"{checkpoint_stem}_{config.evaluation_dirname}"
        )

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
        save_predictions=False,
        save_overlays=False,
        show_indices=False,
        use_landmark_names_in_boxplot=config.use_landmark_names_in_boxplot,
    )

    print("[INFO] Evaluation finished.")
    print(f"[INFO] Mean NME: {summary['mean_nme']:.6f}")
    print(f"[INFO] Median NME: {summary['median_nme']:.6f}")
    print(f"[INFO] Visibility accuracy: {summary['visibility_accuracy']:.6f}")


if __name__ == "__main__":
    main()
