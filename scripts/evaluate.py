from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

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
from scripts.engine.full_evaluation import evaluate_infanface, print_evaluation_summary
from scripts.engine.metrics import decoder_from_landmark_loss
from scripts.engine.normalizer_experiments import (
    run_normalizer_diagnostics,
    write_combined_normalizer_diagnostics,
)
from scripts.engine.pca_shape_prior import load_pca_shape_prior
from scripts.engine.pca_tta import PCAGuidedTTA, PCATTAConfig
from scripts.models import NormalizedLandmarker, build_model_from_checkpoints
from scripts.inference import build_inference_dataloader
from scripts.utils import get_default_device, set_seed


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for standalone checkpoint evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args()
    yaml_defaults = _load_yaml_argument_defaults(config_args.config)
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained landmark detection checkpoint on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="Optional evaluation YAML. Explicit CLI values take precedence.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["synthetic", "natural", "babyland", "infanface"],
        default="synthetic",
        help=(
            "Dataset evaluation protocol. 'natural' is retained as a backward-"
            "compatible alias for 'babyland'. InfantFace uses its dedicated "
            "72-to-68 benchmark protocol."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        required=yaml_defaults.get("checkpoint") is None,
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
        help=(
            "Synthetic dataset root in synthetic mode, or detector-export crop "
            "root containing images/ and metadata/ in BabyLand/InfantFace mode."
        ),
    )
    parser.add_argument(
        "--natural-gt-root",
        type=Path,
        default=None,
        help=(
            "Directory containing original-image GT txt files. Required for "
            "BabyLand and InfantFace."
        ),
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
        action=argparse.BooleanOptionalAction,
        default=defaults.use_landmark_names_in_boxplot,
        help="Use landmark names instead of indices on evaluation boxplots.",
    )
    parser.add_argument(
        "--save-config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save the resolved configuration JSON next to the evaluation outputs.",
    )
    parser.add_argument(
        "--save-crop-overlays",
        action=argparse.BooleanOptionalAction,
        default=defaults.save_natural_crop_overlays,
        help="In natural mode, additionally save qualitative overlays on the detector crops under predictions/crops/.",
    )
    parser.add_argument(
        "--disable-normalizer-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip image-change and prediction-drift diagnostics for normalized models.",
    )
    parser.add_argument(
        "--normalizer-visual-examples",
        type=int,
        default=32,
        help="Maximum number of normalizer comparison panels to save.",
    )
    parser.add_argument(
        "--normalizer-residual-display-scale",
        type=float,
        default=0.02,
        help=(
            "Fixed RGB residual magnitude mapped to full intensity in signed and "
            "absolute diagnostic images. The same scale is used for every sample."
        ),
    )
    parser.add_argument(
        "--normalizer-residual-amplification",
        type=float,
        default=25.0,
        help="Amplification used by the input-plus-residual diagnostic preview.",
    )
    parser.add_argument(
        "--pca-tta",
        action=argparse.BooleanOptionalAction,
        default=defaults.pca_tta_enabled,
        help=(
            "Enable episodic per-image TTA. Only the external normalizer is "
            "updated and PCA reconstruction loss is the sole objective."
        ),
    )
    parser.add_argument(
        "--pca-prior-path",
        type=Path,
        default=defaults.pca_prior_path,
        help="Source-trained PCA shape-prior checkpoint required by --pca-tta.",
    )
    parser.add_argument(
        "--pca-tta-steps",
        type=int,
        default=defaults.pca_tta_steps,
        help="Number of independent normalizer updates performed per target image.",
    )
    parser.add_argument(
        "--pca-tta-learning-rate",
        type=float,
        default=defaults.pca_tta_learning_rate,
        help="Adam learning rate used for the normalizer during each TTA episode.",
    )
    parser.add_argument(
        "--pca-tta-monitor-steps",
        type=int,
        nargs="+",
        default=list(defaults.pca_tta_monitor_steps),
        help="Adaptation steps saved for fixed target-image probe panels.",
    )
    parser.add_argument(
        "--pca-tta-probe-count",
        type=int,
        default=defaults.pca_tta_probe_count,
        help="Number of target images receiving detailed visual trajectories.",
    )
    parser.add_argument(
        "--pca-tta-difference-display-max",
        type=float,
        default=defaults.pca_tta_difference_display_max,
        help="Fixed absolute RGB difference mapped to full probe-map intensity.",
    )
    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_wandb,
        help="Log aggregate PCA-TTA curves, image summaries, and probes to W&B.",
    )
    parser.add_argument(
        "--wandb-project",
        default=defaults.wandb_project,
        help="Weights & Biases project used by optional TTA logging.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name for this standalone TTA evaluation.",
    )
    _apply_yaml_argument_defaults(parser, yaml_defaults)
    return parser.parse_args()


def _load_yaml_argument_defaults(config_path: Path | None) -> dict[str, Any]:
    """Read standalone evaluation parser defaults from one optional YAML file."""
    if config_path is None:
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("The evaluation YAML root must be a mapping.")
    arguments = payload.get("arguments", payload)
    if not isinstance(arguments, dict):
        raise ValueError("The evaluation YAML 'arguments' entry must be a mapping.")
    return dict(arguments)


def _apply_yaml_argument_defaults(
    parser: argparse.ArgumentParser,
    yaml_defaults: dict[str, Any],
) -> None:
    """Validate and type-convert YAML values against every parser destination."""
    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"}
    }
    unknown = sorted(set(yaml_defaults).difference(actions))
    if unknown:
        raise ValueError(f"Unsupported evaluation YAML arguments: {unknown}")

    converted: dict[str, Any] = {}
    for destination, value in yaml_defaults.items():
        action = actions[destination]
        if value is None:
            converted[destination] = None
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(
                    f"YAML argument '{destination}' must be true or false."
                )
            converted[destination] = value
            continue
        if action.nargs in {"+", "*"}:
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"YAML argument '{destination}' must be a list because the "
                    "corresponding CLI option accepts multiple values."
                )
            typed_value = [
                action.type(item) if action.type is not None else item for item in value
            ]
        else:
            typed_value = action.type(value) if action.type is not None else value
        if action.choices is not None:
            values = typed_value if isinstance(typed_value, list) else [typed_value]
            invalid = [item for item in values if item not in action.choices]
            if invalid:
                raise ValueError(
                    f"Invalid YAML value for '{destination}': {invalid}. "
                    f"Expected one of {list(action.choices)}."
                )
        converted[destination] = typed_value
    parser.set_defaults(**converted)


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
    dataset_protocol = resolve_dataset_protocol(args.eval_mode)
    config.eval_mode = "synthetic" if dataset_protocol == "synthetic" else "natural"
    config.dataset_protocol = dataset_protocol
    config.natural_gt_root = args.natural_gt_root
    config.natural_source_root = args.natural_source_root
    if dataset_protocol == "infanface":
        config.infanface_crop_root = args.dataset_root
        config.infanface_gt_root = args.natural_gt_root
        config.infanface_source_root = args.natural_source_root
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
    config.pca_tta_enabled = args.pca_tta
    config.pca_prior_path = args.pca_prior_path
    config.pca_tta_steps = args.pca_tta_steps
    config.pca_tta_learning_rate = args.pca_tta_learning_rate
    config.pca_tta_monitor_steps = tuple(args.pca_tta_monitor_steps)
    config.pca_tta_probe_count = args.pca_tta_probe_count
    config.pca_tta_difference_display_max = args.pca_tta_difference_display_max
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    config.config_path = args.config

    if args.output_dir is None:
        resolve_evaluation_output_dir(config, args.checkpoint)
    else:
        config.output_dir = args.output_dir

    return config


def resolve_dataset_protocol(eval_mode: str) -> str:
    """Resolve legacy evaluation modes to an explicit dataset protocol."""
    return "babyland" if eval_mode == "natural" else eval_mode


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


def main() -> None:
    """
    Run standalone evaluation.
    """
    args = parse_args()
    if args.normalizer_visual_examples < 0:
        raise ValueError("--normalizer-visual-examples cannot be negative.")
    if args.normalizer_residual_display_scale <= 0:
        raise ValueError("--normalizer-residual-display-scale must be positive.")
    if args.normalizer_residual_amplification <= 0:
        raise ValueError("--normalizer-residual-amplification must be positive.")
    config = build_config_from_args(args)
    dataset_protocol = config.dataset_protocol

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = get_default_device(config.device)

    if dataset_protocol in {"babyland", "infanface"} and args.natural_gt_root is None:
        raise ValueError(
            "--natural-gt-root is required for BabyLand and InfantFace evaluation."
        )

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Dataset protocol: {dataset_protocol}")
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
    if isinstance(model, NormalizedLandmarker):
        loading_mode = (
            "landmarker + separate normalizer checkpoints"
            if args.normalizer_checkpoint is not None
            else "full normalized-model checkpoint"
        )
        print(f"[INFO] Loaded {loading_mode}; normalizer diagnostics are available.")
        print(
            "[INFO] Checkpoint load verified | strict=True missing_keys=0 "
            "unexpected_keys=0 landmarker=complete normalizer=complete"
        )
    else:
        print("[INFO] Loaded landmarker-only checkpoint; no normalizer is active.")
        print(
            "[INFO] Checkpoint load verified | strict=True missing_keys=0 "
            "unexpected_keys=0 landmarker=complete"
        )

    tta_adapter = None
    wandb_module = None
    if config.pca_tta_enabled:
        if dataset_protocol == "synthetic":
            raise ValueError("PCA-guided TTA is supported only for natural datasets.")
        if not isinstance(model, NormalizedLandmarker):
            raise ValueError(
                "--pca-tta requires a full normalized-model checkpoint or a "
                "landmarker checkpoint combined with --normalizer-checkpoint."
            )
        if config.pca_prior_path is None:
            raise ValueError("--pca-prior-path is required when --pca-tta is enabled.")
        if config.use_wandb:
            import wandb as wandb_module

            wandb_module.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config_to_serializable_dict(config),
            )
        pca_prior = load_pca_shape_prior(config.pca_prior_path, device=device)
        tta_adapter = PCAGuidedTTA(
            model=model,
            pca_prior=pca_prior,
            device=device,
            output_dir=output_dir / "tta",
            config=PCATTAConfig(
                steps=config.pca_tta_steps,
                learning_rate=config.pca_tta_learning_rate,
                monitor_steps=tuple(config.pca_tta_monitor_steps),
                probe_count=config.pca_tta_probe_count,
                difference_display_max=config.pca_tta_difference_display_max,
                normalization_mean=tuple(config.normalization_mean),
                normalization_std=tuple(config.normalization_std),
            ),
            wandb_module=wandb_module,
        )
        print(
            "[INFO] PCA-guided episodic TTA enabled | "
            f"steps={config.pca_tta_steps} "
            f"lr={config.pca_tta_learning_rate:g} "
            f"prior={config.pca_prior_path}"
        )

    if args.save_config:
        maybe_save_config(config, output_dir)

    artifact_summary: dict = {}
    if dataset_protocol == "synthetic":
        dataloaders = build_dataloaders(config)
        evaluation_dataloader = dataloaders["test"]
        summary = evaluate_checkpoint(
            model=model,
            dataloader=evaluation_dataloader,
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
        artifact_summary = summary
    elif dataset_protocol == "babyland":
        evaluation_dataloader = build_natural_evaluation_dataloader(
            export_root=config.dataset_root,
            gt_root=args.natural_gt_root,
            source_root=args.natural_source_root,
            config=config,
        )
        summary = evaluate_natural_checkpoint(
            model=model,
            dataloader=evaluation_dataloader,
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
            tta_adapter=tta_adapter,
        )
        artifact_summary = summary
    else:
        inference_config = type(config)(**vars(config).copy())
        inference_config.project_to_original = True
        inference_config.source_root = args.natural_source_root
        evaluation_dataloader = build_inference_dataloader(
            config.dataset_root,
            inference_config,
        )
        infanface_result = evaluate_infanface(
            model=model,
            device=device,
            config=config,
            output_dir=output_dir,
            dataloader=evaluation_dataloader,
            print_summary=False,
            tta_adapter=tta_adapter,
        )
        summary = infanface_result["metrics"]
        artifact_summary = infanface_result["inference"]

    if tta_adapter is not None:
        tta_summary = tta_adapter.finalize()
        artifact_summary["tta_dir"] = str(output_dir / "tta")
        artifact_summary["tta_failed_samples"] = tta_summary["failed_samples"]
        print(
            "[INFO] PCA TTA reports saved | "
            f"dir={output_dir / 'tta'} "
            f"failed={tta_summary['failed_samples']}/"
            f"{tta_summary['processed_samples']}"
        )

    if isinstance(model, NormalizedLandmarker):
        if args.disable_normalizer_diagnostics:
            print("[INFO] Normalizer diagnostics disabled by CLI option.")
        else:
            dataset_name = dataset_protocol
            diagnostics_dir = output_dir / "normalizer_diagnostics"
            print(
                "[INFO] Generating normalizer diagnostics | "
                f"dataset={dataset_name} examples={args.normalizer_visual_examples} "
                f"fixed_scale={args.normalizer_residual_display_scale:g} "
                f"amplification={args.normalizer_residual_amplification:g}x"
            )
            diagnostic_summary = run_normalizer_diagnostics(
                model=model,
                dataloader=evaluation_dataloader,
                device=device,
                output_dir=diagnostics_dir,
                dataset_name=dataset_name,
                coordinate_decoder=config.coordinate_decoder,
                softmax_temperature=config.wasserstein_softmax_temperature,
                visibility_threshold=config.visibility_threshold,
                mean=config.normalization_mean,
                std=config.normalization_std,
                num_visual_examples=args.normalizer_visual_examples,
                save_visual_examples=args.normalizer_visual_examples > 0,
                residual_display_scale=args.normalizer_residual_display_scale,
                residual_amplification=args.normalizer_residual_amplification,
            )
            write_combined_normalizer_diagnostics(
                diagnostics_dir, {dataset_name: diagnostic_summary}
            )
            print(f"[INFO] Normalizer diagnostics dir: {diagnostics_dir}")

    print("[INFO] Evaluation finished.")
    terminal_summary = dict(summary)
    for key in (
        "prediction_labels_dir",
        "prediction_overlays_dir",
        "prediction_crop_overlays_dir",
    ):
        if artifact_summary.get(key) is not None:
            terminal_summary[key] = artifact_summary[key]
    display_name = {
        "synthetic": "SynBaby",
        "babyland": "BabyLand",
        "infanface": "InfAnFace",
    }[dataset_protocol]
    print_evaluation_summary(display_name, terminal_summary, output_dir)
    if wandb_module is not None:
        wandb_module.finish()


if __name__ == "__main__":
    main()
