from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import (
    ExperimentConfig,
    build_config,
    config_to_serializable_dict,
)
from scripts.dataset import build_dataloaders, build_natural_evaluation_dataloader
from scripts.engine.confidence_error_analysis import run_confidence_error_analysis
from scripts.engine.metrics import decoder_from_landmark_loss
from scripts.engine.pca_shape_prior import load_pca_shape_prior
from scripts.models import HRNetLandmarkVisibility
from scripts.utils import get_default_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for confidence-error analysis."""
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Analyze confidence-error relationships for BabyLand-72 landmark predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON or YAML config with keys matching ExperimentConfig fields.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["synthetic", "natural"],
        default=None,
        help="Analyze a synthetic split or detector-aligned natural crops.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint path. Can also be provided as checkpoint_path in --config.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Synthetic dataset root, or detector-export root in natural mode.",
    )
    parser.add_argument(
        "--natural-gt-root",
        type=Path,
        default=None,
        help="Directory with natural GT txt files. Required in natural mode.",
    )
    parser.add_argument(
        "--natural-source-root",
        type=Path,
        default=None,
        help="Optional root for resolving natural source image paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where analysis artifacts will be written.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--landmark-loss",
        choices=["mse", "adaptive_wing", "wasserstein"],
        default=None,
        help="Loss regime used by the checkpoint; controls coordinate decoding.",
    )
    parser.add_argument(
        "--wasserstein-softmax-temperature",
        type=float,
        default=None,
        help="Spatial softmax temperature for barycenter decoding and confidence probabilities.",
    )
    parser.add_argument(
        "--tta-samples",
        type=int,
        default=None,
        help="Number of mild test-time augmentations used for consistency. Use 0 to skip.",
    )
    parser.add_argument(
        "--compute-pca-shape-plausibility",
        action="store_true",
        help="Compute optional PCA shape reconstruction error.",
    )
    parser.add_argument(
        "--pca-prior-path",
        type=Path,
        default=None,
        help="PCA prior .pt file used when PCA shape plausibility is enabled.",
    )
    parser.add_argument(
        "--failure-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Normalized-error thresholds used to define localization failures.",
    )
    parser.add_argument(
        "--retention-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Fractions retained by confidence-ranked pseudo-label diagnostics.",
    )
    parser.add_argument(
        "--max-visual-examples",
        type=int,
        default=None,
        help="Maximum qualitative examples saved across confidence/error buckets.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit for quick validation.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=None,
        help="Threshold for predicted visibility columns and example overlays.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        default=None,
        help="Disable synthetic dataset cache loading/writing.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save the resolved analysis config JSON in the output directory.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build an analysis config from defaults, optional config file, and CLI args."""
    config = build_config()
    config_file_keys: set[str] = set()
    if args.config is not None:
        config_file_keys = _apply_config_file(config, args.config)

    config.eval_mode = _arg_or_config(args, "eval_mode", config, "eval_mode", "natural")
    config.checkpoint_path = _resolve_checkpoint_path(args, config)
    config.dataset_root = _arg_or_config(args, "dataset_root", config, "dataset_root")
    config.natural_gt_root = _arg_or_config(
        args, "natural_gt_root", config, "natural_gt_root"
    )
    config.natural_source_root = _arg_or_config(
        args,
        "natural_source_root",
        config,
        "natural_source_root",
    )
    config.cache_dir = _arg_or_config(args, "cache_dir", config, "cache_dir")
    config.eval_batch_size = _arg_or_config(
        args, "batch_size", config, "eval_batch_size"
    )
    config.num_workers = _arg_or_config(args, "num_workers", config, "num_workers")
    config.device = _arg_or_config(args, "device", config, "device")
    config.seed = _arg_or_config(args, "seed", config, "seed")
    config.target_mode = "regression"
    config.use_cache = (
        config.use_cache if args.disable_cache is None else not args.disable_cache
    )
    config.landmark_loss = _arg_or_config(
        args, "landmark_loss", config, "landmark_loss"
    )
    config.coordinate_decoder = decoder_from_landmark_loss(config.landmark_loss)
    config.wasserstein_softmax_temperature = _arg_or_config(
        args,
        "wasserstein_softmax_temperature",
        config,
        "wasserstein_softmax_temperature",
    )
    config.visibility_threshold = _arg_or_config(
        args,
        "visibility_threshold",
        config,
        "visibility_threshold",
    )
    config.pca_prior_path = _arg_or_config(
        args, "pca_prior_path", config, "pca_prior_path"
    )
    config.tta_samples = _arg_or_config(args, "tta_samples", config, "tta_samples", 0)
    config.compute_pca_shape_plausibility = bool(
        args.compute_pca_shape_plausibility
        or getattr(config, "compute_pca_shape_plausibility", False)
    )
    config.failure_thresholds = _arg_or_config(
        args,
        "failure_thresholds",
        config,
        "failure_thresholds",
        [0.05, 0.08, 0.10],
    )
    config.retention_fractions = _arg_or_config(
        args,
        "retention_fractions",
        config,
        "retention_fractions",
        [0.10, 0.25, 0.50, 1.0],
    )
    config.max_visual_examples = _arg_or_config(
        args,
        "max_visual_examples",
        config,
        "max_visual_examples",
        12,
    )
    config.max_batches = _arg_or_config(
        args, "max_batches", config, "max_batches", None
    )
    config.output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(config.output_dir)
        if "output_dir" in config_file_keys
        else config.checkpoint_path.parent
        / f"{config.checkpoint_path.stem}_confidence_error"
    )
    return config


def main() -> None:
    """Run the confidence-error analysis CLI."""
    args = parse_args()
    config = build_config_from_args(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.checkpoint_path is None:
        raise ValueError("--checkpoint or checkpoint_path in --config is required.")
    if config.eval_mode == "natural" and config.natural_gt_root is None:
        raise ValueError(
            "--natural-gt-root is required when --eval-mode natural is used."
        )
    if config.compute_pca_shape_plausibility and config.pca_prior_path is None:
        raise ValueError(
            "--pca-prior-path is required when PCA shape plausibility is enabled."
        )

    set_seed(config.seed)
    device = get_default_device(config.device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Eval mode: {config.eval_mode}")
    print(f"[INFO] Dataset root: {config.dataset_root}")
    print(f"[INFO] Checkpoint: {config.checkpoint_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] TTA samples: {config.tta_samples}")

    model = build_model(config)
    checkpoint = torch.load(
        config.checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if config.eval_mode == "natural":
        dataloader = build_natural_evaluation_dataloader(
            export_root=config.dataset_root,
            gt_root=config.natural_gt_root,
            source_root=config.natural_source_root,
            config=config,
        )
        dataset_description = (
            f"natural detector crops={config.dataset_root}; gt={config.natural_gt_root}"
        )
    else:
        dataloaders = build_dataloaders(config)
        dataloader = dataloaders["test"]
        dataset_description = f"synthetic test split={config.dataset_root}/test"

    pca_prior = None
    if config.compute_pca_shape_plausibility:
        pca_prior = load_pca_shape_prior(config.pca_prior_path, device=device)

    if args.save_config or bool(getattr(config, "save_config", False)):
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config_to_serializable_dict(config), indent=2),
            encoding="utf-8",
        )

    summary = run_confidence_error_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        checkpoint_path=config.checkpoint_path,
        dataset_description=dataset_description,
        eval_mode=config.eval_mode,
        coordinate_decoder=config.coordinate_decoder,
        wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
        tta_samples=config.tta_samples,
        pca_prior=pca_prior,
        failure_thresholds=config.failure_thresholds,
        retention_fractions=config.retention_fractions,
        max_visual_examples=config.max_visual_examples,
        max_batches=config.max_batches,
        visibility_threshold=config.visibility_threshold,
    )

    print("[INFO] Confidence-error analysis finished.")
    print(f"[INFO] Images: {summary['num_images']}")
    print(f"[INFO] Landmark rows: {summary['num_landmark_rows']}")
    print(f"[INFO] Evaluable landmark rows: {summary['num_evaluable_landmark_rows']}")
    print(f"[INFO] Mean image NME fraction: {summary['mean_nme_fraction']:.4f}")
    print(f"[INFO] Mean image NME percent: {summary['mean_nme_percent']:.2f}")
    print(f"[INFO] Report: {summary['outputs']['report']}")


def _apply_config_file(config: ExperimentConfig, config_path: Path) -> set[str]:
    raw = _load_config_file(config_path)
    for key, value in raw.items():
        if key.endswith("_root") or key.endswith("_dir") or key.endswith("_path"):
            value = Path(value) if value is not None else None
        setattr(config, key, value)
    return set(raw.keys())


def _load_config_file(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as error:
            raise RuntimeError(
                "PyYAML is required to read YAML config files."
            ) from error
        loaded = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    return loaded


def _arg_or_config(
    args: argparse.Namespace,
    arg_name: str,
    config: ExperimentConfig,
    config_name: str,
    default: Any | None = None,
) -> Any:
    """Return a CLI override, then config-file value, then an explicit fallback."""
    arg_value = getattr(args, arg_name)
    if arg_value is not None:
        return arg_value
    if hasattr(config, config_name):
        return getattr(config, config_name)
    return default


def _resolve_checkpoint_path(
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> Path | None:
    """Resolve checkpoint from CLI or config-file aliases."""
    if args.checkpoint is not None:
        return args.checkpoint
    for field_name in ("checkpoint_path", "checkpoint"):
        value = getattr(config, field_name, None)
        if value is not None:
            return Path(value)
    return None


def build_model(config: ExperimentConfig) -> HRNetLandmarkVisibility:
    """Instantiate the BabyLand-72 landmark model used by evaluation scripts."""
    return HRNetLandmarkVisibility(num_landmarks=config.num_landmarks)


if __name__ == "__main__":
    main()
