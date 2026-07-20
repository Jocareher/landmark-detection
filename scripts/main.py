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
    load_yaml_config,
    resolve_output_dir,
    save_resolved_config_files,
)
from scripts.dataset import build_dataloaders, build_natural_evaluation_dataloader
from scripts.inference import build_inference_dataloader
from scripts.engine.normalizer_experiments import (
    run_normalizer_diagnostics,
    save_modular_checkpoints,
    write_combined_normalizer_diagnostics,
    write_experiment_report,
)
from scripts.engine.landmark_losses import build_landmark_heatmap_loss
from scripts.engine.metrics import decoder_from_landmark_loss
from scripts.models import (
    HRNetLandmarkVisibility,
    NormalizedLandmarker,
    ResidualImageNormalizer,
    load_normalized_checkpoint,
)
from scripts.utils import (
    get_default_device,
    save_model_summary,
    save_reproducibility_metadata,
    set_seed,
    tee_terminal_output,
)
from scripts.utils.visualization import save_dataset_preview_grid


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the end-to-end training pipeline."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args()
    defaults = (
        load_yaml_config(config_args.config)
        if config_args.config is not None
        else build_config()
    )
    parser = argparse.ArgumentParser(
        description="Train the model on train/val and then evaluate the best checkpoint on test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="Optional YAML configuration. CLI values take precedence.",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=[
            "none",
            "normalizer_sanity",
            "normalizer_train_frozen_landmarker",
            "normalizer_joint_finetune",
        ],
        default=defaults.experiment_mode,
        help="Residual image normalizer experiment mode.",
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
        default=defaults.runs_dir,
        help="Base directory where training runs will be created.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.cache_dir,
        help="Directory used to store cached dataset files.",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=defaults.pretrained_weights,
        help="Path to the pretrained HRNet weights loaded before training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults.checkpoint_path,
        help="Optional checkpoint to load before continuing training.",
    )
    parser.add_argument(
        "--checkpoint-strict",
        action=argparse.BooleanOptionalAction,
        default=defaults.checkpoint_strict,
        help="Require an exact checkpoint key match.",
    )
    parser.add_argument(
        "--normalizer-hidden-channels",
        type=int,
        default=defaults.normalizer_hidden_channels,
    )
    parser.add_argument(
        "--normalizer-num-layers", type=int, default=defaults.normalizer_num_layers
    )
    parser.add_argument(
        "--normalizer-kernel-size", type=int, default=defaults.normalizer_kernel_size
    )
    parser.add_argument(
        "--normalizer-activation",
        choices=["relu", "gelu"],
        default=defaults.normalizer_activation,
    )
    parser.add_argument(
        "--normalizer-normalization",
        choices=["none", "group", "instance"],
        default=defaults.normalizer_internal_normalization,
    )
    parser.add_argument(
        "--normalizer-residual-scale",
        type=float,
        default=defaults.normalizer_residual_scale,
    )
    parser.add_argument(
        "--normalizer-initialize-identity",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalizer_initialize_identity,
    )
    parser.add_argument(
        "--normalizer-clamp-output",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalizer_clamp_output,
    )
    parser.add_argument(
        "--normalizer-image-regularization",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalizer_image_regularization_enabled,
    )
    parser.add_argument(
        "--normalizer-lambda-l1", type=float, default=defaults.normalizer_lambda_l1
    )
    parser.add_argument(
        "--normalizer-lambda-tv", type=float, default=defaults.normalizer_lambda_tv
    )
    parser.add_argument(
        "--normalizer-visual-examples",
        type=int,
        default=defaults.normalizer_visual_examples,
    )
    parser.add_argument(
        "--normalizer-monitoring",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalizer_monitoring_enabled,
        help="Monitor a fixed validation probe set while training the normalizer.",
    )
    parser.add_argument(
        "--normalizer-monitor-probes",
        type=int,
        default=defaults.normalizer_monitor_probes,
        help="Number of unchanged validation images used for normalizer monitoring.",
    )
    parser.add_argument(
        "--normalizer-monitor-steps",
        type=int,
        nargs="+",
        default=defaults.normalizer_monitor_steps,
        help="One-based source-training epochs to capture, plus 0 for initialization.",
    )
    parser.add_argument(
        "--normalizer-tta-monitor-steps",
        type=int,
        nargs="+",
        default=defaults.normalizer_tta_monitor_steps,
        help="Future TTA steps to capture; the final step is always captured.",
    )
    parser.add_argument(
        "--normalizer-monitor-difference-max",
        type=float,
        default=defaults.normalizer_monitor_difference_max,
        help="Fixed RGB difference represented by white in every difference panel.",
    )
    parser.add_argument(
        "--normalizer-monitor-registration-warning-px",
        type=float,
        default=defaults.normalizer_monitor_registration_warning_px,
        help="Absolute phase-correlation shift that triggers a geometry warning.",
    )
    parser.add_argument(
        "--normalizer-monitor-edge-correlation-warning",
        type=float,
        default=defaults.normalizer_monitor_edge_correlation_warning,
        help="Edge correlation below which a geometry warning is emitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Mini-batch size for the training split.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults.eval_batch_size,
        help="Mini-batch size for validation and test. If omitted, training batch size is used.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.num_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=defaults.learning_rate,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--landmark-loss",
        choices=["mse", "adaptive_wing", "wasserstein"],
        default=defaults.landmark_loss,
        help="Heatmap landmark loss regime.",
    )
    parser.add_argument(
        "--adaptive-wing-omega",
        type=float,
        default=defaults.adaptive_wing_omega,
        help="Adaptive Wing Loss omega parameter.",
    )
    parser.add_argument(
        "--adaptive-wing-theta",
        type=float,
        default=defaults.adaptive_wing_theta,
        help="Adaptive Wing Loss theta threshold.",
    )
    parser.add_argument(
        "--adaptive-wing-epsilon",
        type=float,
        default=defaults.adaptive_wing_epsilon,
        help="Adaptive Wing Loss epsilon parameter.",
    )
    parser.add_argument(
        "--adaptive-wing-alpha",
        type=float,
        default=defaults.adaptive_wing_alpha,
        help="Adaptive Wing Loss alpha parameter.",
    )
    parser.add_argument(
        "--wasserstein-softmax-temperature",
        type=float,
        default=defaults.wasserstein_softmax_temperature,
        help="Spatial softmax temperature used by Wasserstein loss and barycenter decoding.",
    )
    parser.add_argument(
        "--wasserstein-epsilon",
        type=float,
        default=defaults.wasserstein_epsilon,
        help="Numerical epsilon used by Wasserstein heatmap normalization.",
    )
    parser.add_argument(
        "--pca-prior-path",
        type=Path,
        default=defaults.pca_prior_path,
        help="Path to the precomputed train-set PCA shape prior .pt file.",
    )
    parser.add_argument(
        "--lambda-pca-projection",
        type=float,
        default=defaults.lambda_pca_projection,
        help="Weight assigned to the PCA projection loss on final landmarks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=defaults.device,
        help="Device where the model will run.",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["feature_extractor", "fine_tuning"],
        default=defaults.transfer_mode,
        help="Transfer learning strategy applied to the HRNet backbone.",
    )
    parser.add_argument(
        "--num-unfrozen-stages",
        type=int,
        default=defaults.num_unfrozen_stages,
        help="Number of backbone stages to unfreeze in fine-tuning mode.",
    )
    parser.add_argument(
        "--unfreeze-stem",
        action="store_true",
        default=defaults.unfreeze_stem,
        help="Unfreeze the HRNet stem layers as part of transfer learning.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=defaults.use_wandb,
        help="Enable Weights & Biases experiment tracking.",
    )
    parser.add_argument(
        "--wandb-project",
        default=defaults.wandb_project,
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=defaults.wandb_run_name,
        help="Explicit run name. If omitted, one is generated automatically.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        default=not defaults.use_amp,
        help="Disable automatic mixed precision training.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        default=not defaults.use_cache,
        help="Disable dataset cache loading and writing.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=defaults.run_smoke_test,
        help="Run a single optimization step before full training.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=defaults.save_config,
        help="Save resolved config JSON into output dir.",
    )
    parser.add_argument(
        "--enable-photometric-augmentations",
        action="store_true",
        default=defaults.enable_photometric_augmentations,
        help="Enable photometric training augmentations.",
    )
    parser.add_argument(
        "--enable-geometric-augmentations",
        action="store_true",
        default=defaults.enable_geometric_augmentations,
        help="Enable geometric training augmentations that also transform landmarks.",
    )
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=defaults.color_jitter_brightness,
        help="Maximum relative brightness jitter strength.",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=defaults.color_jitter_contrast,
        help="Maximum relative contrast jitter strength.",
    )
    parser.add_argument(
        "--saturation-jitter",
        type=float,
        default=defaults.color_jitter_saturation,
        help="Maximum relative saturation jitter strength.",
    )
    parser.add_argument(
        "--color-jitter-probability",
        type=float,
        default=defaults.color_jitter_probability,
        help="Application probability for color jitter.",
    )
    parser.add_argument(
        "--blur-probability",
        type=float,
        default=defaults.blur_probability,
        help="Application probability for Gaussian blur.",
    )
    parser.add_argument(
        "--blur-radius-min",
        type=float,
        default=defaults.blur_radius_min,
        help="Minimum Gaussian blur radius.",
    )
    parser.add_argument(
        "--blur-radius-max",
        type=float,
        default=defaults.blur_radius_max,
        help="Maximum Gaussian blur radius.",
    )
    parser.add_argument(
        "--noise-probability",
        type=float,
        default=defaults.noise_probability,
        help="Application probability for Gaussian noise.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=defaults.noise_std,
        help="Gaussian noise standard deviation in normalized [0, 1] image space.",
    )
    parser.add_argument(
        "--jpeg-probability",
        type=float,
        default=defaults.jpeg_probability,
        help="Application probability for JPEG compression simulation.",
    )
    parser.add_argument(
        "--jpeg-quality-min",
        type=int,
        default=defaults.jpeg_quality_min,
        help="Minimum JPEG quality used during compression simulation.",
    )
    parser.add_argument(
        "--jpeg-quality-max",
        type=int,
        default=defaults.jpeg_quality_max,
        help="Maximum JPEG quality used during compression simulation.",
    )
    parser.add_argument(
        "--rgb-shift-probability",
        type=float,
        default=defaults.rgb_shift_probability,
        help="Application probability for additive RGB channel perturbations.",
    )
    parser.add_argument(
        "--rgb-shift-limit",
        type=float,
        default=defaults.rgb_shift_limit,
        help="Maximum additive RGB channel shift in normalized [0, 1] image space.",
    )
    parser.add_argument(
        "--geometric-probability",
        type=float,
        default=defaults.geometric_probability,
        help="Application probability for geometric training augmentation.",
    )
    parser.add_argument(
        "--geometric-max-translation",
        type=float,
        default=defaults.geometric_max_translation,
        help="Maximum translation as a fraction of image width and height.",
    )
    parser.add_argument(
        "--geometric-scale-min",
        type=float,
        default=defaults.geometric_scale_min,
        help="Minimum isotropic scale factor for geometric augmentation.",
    )
    parser.add_argument(
        "--geometric-scale-max",
        type=float,
        default=defaults.geometric_scale_max,
        help="Maximum isotropic scale factor for geometric augmentation.",
    )
    parser.add_argument(
        "--geometric-max-rotation-deg",
        type=float,
        default=defaults.geometric_max_rotation_deg,
        help="Maximum absolute rotation in degrees for geometric augmentation.",
    )
    parser.add_argument(
        "--evaluate-synbaby",
        action=argparse.BooleanOptionalAction,
        default=defaults.evaluate_synbaby,
        help="Enable or disable SynBaby evaluation in the full evaluation pipeline.",
    )
    parser.add_argument(
        "--evaluate-babyland",
        action=argparse.BooleanOptionalAction,
        default=defaults.evaluate_babyland,
        help="Enable or disable BabyLand evaluation in the full evaluation pipeline.",
    )
    parser.add_argument(
        "--evaluate-infanface",
        action=argparse.BooleanOptionalAction,
        default=defaults.evaluate_infanface,
        help="Enable or disable InfAnFace evaluation in the full evaluation pipeline.",
    )
    parser.add_argument(
        "--babyland-crop-root",
        type=Path,
        default=defaults.babyland_crop_root,
        help="BabyLand detector-export crop root containing images/ and metadata/.",
    )
    parser.add_argument(
        "--babyland-gt-root",
        type=Path,
        default=defaults.babyland_gt_root,
        help="BabyLand original-image GT label root.",
    )
    parser.add_argument(
        "--babyland-source-root",
        type=Path,
        default=defaults.babyland_source_root,
        help="Optional root used to resolve BabyLand original source image paths.",
    )
    parser.add_argument(
        "--infanface-crop-root",
        type=Path,
        default=defaults.infanface_crop_root,
        help="InfAnFace detector-export crop root containing images/ and metadata/.",
    )
    parser.add_argument(
        "--infanface-gt-root",
        type=Path,
        default=defaults.infanface_gt_root,
        help="InfAnFace original-image GT label root.",
    )
    parser.add_argument(
        "--infanface-source-root",
        type=Path,
        default=defaults.infanface_source_root,
        help="Optional root used to resolve InfAnFace original source image paths.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Merge CLI overrides into the default experiment configuration."""
    config = (
        load_yaml_config(args.config) if args.config is not None else build_config()
    )
    config.experiment_mode = args.experiment_mode
    config.checkpoint_path = args.checkpoint
    config.checkpoint_strict = args.checkpoint_strict
    config.normalizer_enabled = args.experiment_mode != "none"
    config.normalizer_hidden_channels = args.normalizer_hidden_channels
    config.normalizer_num_layers = args.normalizer_num_layers
    config.normalizer_kernel_size = args.normalizer_kernel_size
    config.normalizer_activation = args.normalizer_activation
    config.normalizer_internal_normalization = args.normalizer_normalization
    config.normalizer_residual_scale = args.normalizer_residual_scale
    config.normalizer_initialize_identity = args.normalizer_initialize_identity
    config.normalizer_clamp_output = args.normalizer_clamp_output
    config.normalizer_image_regularization_enabled = (
        args.normalizer_image_regularization
    )
    config.normalizer_lambda_l1 = args.normalizer_lambda_l1
    config.normalizer_lambda_tv = args.normalizer_lambda_tv
    config.normalizer_visual_examples = args.normalizer_visual_examples
    config.normalizer_monitoring_enabled = args.normalizer_monitoring
    config.normalizer_monitor_probes = args.normalizer_monitor_probes
    config.normalizer_monitor_steps = tuple(args.normalizer_monitor_steps)
    config.normalizer_tta_monitor_steps = tuple(args.normalizer_tta_monitor_steps)
    config.normalizer_monitor_difference_max = args.normalizer_monitor_difference_max
    config.normalizer_monitor_registration_warning_px = (
        args.normalizer_monitor_registration_warning_px
    )
    config.normalizer_monitor_edge_correlation_warning = (
        args.normalizer_monitor_edge_correlation_warning
    )
    config.dataset_root = args.dataset_root
    config.runs_dir = args.output_dir
    config.cache_dir = args.cache_dir
    config.pretrained_weights = args.pretrained_weights
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.landmark_loss = args.landmark_loss
    config.coordinate_decoder = decoder_from_landmark_loss(args.landmark_loss)
    config.adaptive_wing_omega = args.adaptive_wing_omega
    config.adaptive_wing_theta = args.adaptive_wing_theta
    config.adaptive_wing_epsilon = args.adaptive_wing_epsilon
    config.adaptive_wing_alpha = args.adaptive_wing_alpha
    config.wasserstein_softmax_temperature = args.wasserstein_softmax_temperature
    config.wasserstein_epsilon = args.wasserstein_epsilon
    config.pca_prior_path = args.pca_prior_path
    config.lambda_pca_projection = args.lambda_pca_projection
    config.seed = args.seed
    config.device = args.device
    config.transfer_mode = args.transfer_mode
    config.num_unfrozen_stages = args.num_unfrozen_stages
    config.unfreeze_stem = args.unfreeze_stem
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    config.use_amp = not args.disable_amp
    config.use_cache = not args.disable_cache
    config.run_smoke_test = args.smoke_test
    config.save_config = args.save_config
    config.enable_photometric_augmentations = args.enable_photometric_augmentations
    config.enable_geometric_augmentations = args.enable_geometric_augmentations
    config.color_jitter_brightness = args.brightness_jitter
    config.color_jitter_contrast = args.contrast_jitter
    config.color_jitter_saturation = args.saturation_jitter
    config.color_jitter_probability = args.color_jitter_probability
    config.blur_probability = args.blur_probability
    config.blur_radius_min = args.blur_radius_min
    config.blur_radius_max = args.blur_radius_max
    config.noise_probability = args.noise_probability
    config.noise_std = args.noise_std
    config.jpeg_probability = args.jpeg_probability
    config.jpeg_quality_min = args.jpeg_quality_min
    config.jpeg_quality_max = args.jpeg_quality_max
    config.rgb_shift_probability = args.rgb_shift_probability
    config.rgb_shift_limit = args.rgb_shift_limit
    config.geometric_probability = args.geometric_probability
    config.geometric_max_translation = args.geometric_max_translation
    config.geometric_scale_min = args.geometric_scale_min
    config.geometric_scale_max = args.geometric_scale_max
    config.geometric_max_rotation_deg = args.geometric_max_rotation_deg
    config.evaluate_synbaby = args.evaluate_synbaby
    config.evaluate_babyland = args.evaluate_babyland
    config.evaluate_infanface = args.evaluate_infanface
    config.babyland_crop_root = args.babyland_crop_root
    config.babyland_gt_root = args.babyland_gt_root
    config.babyland_source_root = args.babyland_source_root
    config.infanface_crop_root = args.infanface_crop_root
    config.infanface_gt_root = args.infanface_gt_root
    config.infanface_source_root = args.infanface_source_root
    return config


def maybe_save_config(config: ExperimentConfig) -> None:
    """Persist the resolved configuration inside the active run directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serialized = config_to_serializable_dict(config)
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )


def validate_full_evaluation_paths(config: ExperimentConfig) -> None:
    """Fail early when an enabled evaluation dataset is missing required paths."""
    required_fields = []
    if config.evaluate_babyland:
        required_fields.extend(
            [
                (
                    "babyland_crop_root",
                    "BabyLand detector-export crop root",
                    "BabyLand",
                ),
                ("babyland_gt_root", "BabyLand GT label root", "BabyLand"),
            ]
        )
    if config.evaluate_infanface:
        required_fields.extend(
            [
                (
                    "infanface_crop_root",
                    "InfAnFace detector-export crop root",
                    "InfAnFace",
                ),
                ("infanface_gt_root", "InfAnFace GT label root", "InfAnFace"),
            ]
        )

    for field_name, description, dataset_name in required_fields:
        value = getattr(config, field_name, None)
        if value is None:
            raise ValueError(
                f"{description} is required because {dataset_name} evaluation is enabled."
            )
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    for field_name, description in (
        ("babyland_source_root", "BabyLand source image root"),
        ("infanface_source_root", "InfAnFace source image root"),
    ):
        value = getattr(config, field_name, None)
        if value is not None and not Path(value).exists():
            raise FileNotFoundError(f"{description} not found: {value}")


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate and resolve invariants of the normalizer experiment modes."""
    if config.experiment_mode == "none":
        return
    if config.checkpoint_path is None:
        raise ValueError(
            f"{config.experiment_mode} requires checkpoint.path/--checkpoint."
        )
    if not Path(config.checkpoint_path).exists():
        raise FileNotFoundError(
            f"Landmarker checkpoint not found: {config.checkpoint_path}"
        )
    if config.landmark_loss != "wasserstein":
        raise ValueError(
            "Normalizer experiments must use the existing Wasserstein landmark loss "
            "and barycenter decoder; set landmark_loss: wasserstein."
        )
    if config.coordinate_decoder != "barycenter":
        raise ValueError(
            "Wasserstein normalizer experiments require barycenter decoding."
        )
    if config.normalizer_monitor_probes <= 0:
        raise ValueError("normalizer_monitor_probes must be positive.")
    if config.normalizer_monitor_difference_max <= 0:
        raise ValueError("normalizer_monitor_difference_max must be positive.")
    if any(step < 0 for step in config.normalizer_monitor_steps):
        raise ValueError("normalizer_monitor_steps cannot contain negative values.")
    if any(step < 0 for step in config.normalizer_tta_monitor_steps):
        raise ValueError("normalizer_tta_monitor_steps cannot contain negative values.")
    if config.experiment_mode == "normalizer_sanity":
        config.num_epochs = 0
        config.train_normalizer = False
        config.freeze_landmarker = True
        config.finetune_last_backbone_stage = False
        config.train_heads = False
        if "--evaluate-synbaby" not in sys.argv:
            config.evaluate_synbaby = False
    elif config.experiment_mode == "normalizer_train_frozen_landmarker":
        config.train_normalizer = True
        config.freeze_landmarker = True
        config.finetune_last_backbone_stage = False
        config.train_heads = False
    elif config.experiment_mode == "normalizer_joint_finetune":
        config.train_normalizer = True
        config.freeze_landmarker = False
        config.finetune_last_backbone_stage = True
        config.train_heads = True
    if not config.normalizer_image_regularization_enabled:
        config.normalizer_lambda_l1 = 0.0
        config.normalizer_lambda_tv = 0.0


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    """Instantiate the model, load pretrained weights, and configure trainable layers."""
    landmarker = HRNetLandmarkVisibility(num_landmarks=config.num_landmarks)
    if (
        config.pretrained_weights is not None
        and Path(config.pretrained_weights).exists()
    ):
        landmarker.load_official_hrnet_pretrained(
            str(config.pretrained_weights), verbose=True
        )
    if not config.normalizer_enabled:
        landmarker.set_transfer_learning_mode(
            mode=config.transfer_mode,
            num_unfrozen_stages=config.num_unfrozen_stages,
            unfreeze_stem=config.unfreeze_stem,
        )
        return landmarker

    normalizer = ResidualImageNormalizer(
        input_channels=config.normalizer_input_channels,
        hidden_channels=config.normalizer_hidden_channels,
        num_layers=config.normalizer_num_layers,
        kernel_size=config.normalizer_kernel_size,
        activation=config.normalizer_activation,
        normalization=config.normalizer_internal_normalization,
        residual_scale=config.normalizer_residual_scale,
        initialize_identity=config.normalizer_initialize_identity,
        clamp_output=config.normalizer_clamp_output,
        clamp_min=config.normalizer_clamp_min,
        clamp_max=config.normalizer_clamp_max,
    )
    model = NormalizedLandmarker(landmarker=landmarker, normalizer=normalizer)
    if config.experiment_mode == "normalizer_sanity":
        model.freeze_landmarker()
        model.freeze_normalizer()
    elif config.experiment_mode == "normalizer_train_frozen_landmarker":
        model.configure_normalizer_only()
    elif config.experiment_mode == "normalizer_joint_finetune":
        model.configure_joint_finetune(num_unfrozen_stages=1, unfreeze_stem=False)
    else:
        raise ValueError(f"Unsupported experiment mode: {config.experiment_mode}")
    return model


def build_normalizer_diagnostic_loaders(
    config: ExperimentConfig,
    synbaby_dataloader: torch.utils.data.DataLoader | None,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build diagnostic loaders without changing any official evaluation path."""
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    if config.evaluate_synbaby and synbaby_dataloader is not None:
        loaders["synbaby"] = synbaby_dataloader
    if config.evaluate_babyland:
        loaders["babyland"] = build_natural_evaluation_dataloader(
            export_root=config.babyland_crop_root,
            gt_root=config.babyland_gt_root,
            source_root=config.babyland_source_root,
            config=config,
        )
    if config.evaluate_infanface:
        inference_config = type(config)(**vars(config).copy())
        inference_config.project_to_original = True
        inference_config.source_root = config.infanface_source_root
        loaders["infanface"] = build_inference_dataloader(
            config.infanface_crop_root, inference_config
        )
    return loaders


def finalize_normalizer_experiment(
    model: NormalizedLandmarker,
    config: ExperimentConfig,
    device: torch.device,
    synbaby_dataloader: torch.utils.data.DataLoader | None,
    evaluation_summary: dict[str, object],
) -> None:
    """Generate diagnostics, modular checkpoints, and the experiment report."""
    diagnostic_loaders = build_normalizer_diagnostic_loaders(config, synbaby_dataloader)
    diagnostics = {
        dataset_name: run_normalizer_diagnostics(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=config.output_dir,
            dataset_name=dataset_name,
            coordinate_decoder=config.coordinate_decoder,
            softmax_temperature=config.wasserstein_softmax_temperature,
            visibility_threshold=config.visibility_threshold,
            mean=config.normalization_mean,
            std=config.normalization_std,
            changed_pixel_thresholds=config.normalizer_changed_pixel_thresholds,
            num_visual_examples=config.normalizer_visual_examples,
            save_visual_examples=config.save_normalizer_visual_examples,
        )
        for dataset_name, dataloader in diagnostic_loaders.items()
    }
    write_combined_normalizer_diagnostics(config.output_dir, diagnostics)
    checkpoint_paths = save_modular_checkpoints(
        model=model,
        output_dir=config.output_dir,
        base_checkpoint_path=config.checkpoint_path,
        experiment_mode=config.experiment_mode,
        resolved_config_path=config.output_dir / "configs" / "resolved_config.yaml",
        landmarker_updated=config.experiment_mode == "normalizer_joint_finetune",
        normalizer_updated=config.experiment_mode != "normalizer_sanity",
        decoder_name=config.coordinate_decoder,
        loss_pipeline_name=config.landmark_loss,
        evaluation_protocol="Existing SynBaby/BabyLand/InfAnFace evaluation pipeline",
    )
    warnings: list[str] = []
    if config.experiment_mode == "normalizer_sanity":
        for dataset_name, summary in diagnostics.items():
            if summary["max_absolute_difference"] > config.normalizer_sanity_atol:
                warnings.append(
                    f"{dataset_name}: identity image difference exceeded the "
                    f"absolute tolerance ({config.normalizer_sanity_atol:g})."
                )
            if summary["mean_landmark_displacement_px"] > config.normalizer_sanity_atol:
                warnings.append(
                    f"{dataset_name}: decoded landmark drift exceeded the identity "
                    f"tolerance ({config.normalizer_sanity_atol:g} px)."
                )
    report_path = write_experiment_report(
        output_dir=config.output_dir,
        experiment_mode=config.experiment_mode,
        objective="Evaluate a shallow residual appearance adapter before BabyLand-72.",
        checkpoint_path=config.checkpoint_path,
        parameter_counts=model.parameter_counts(),
        evaluation_summaries=evaluation_summary.get("summaries", {}),
        diagnostics=diagnostics,
        checkpoint_paths=checkpoint_paths,
        normalizer_architecture=model.normalizer.architecture_config()
        if model.normalizer is not None
        else None,
        training_protocol=(
            "Inference only; normalizer and landmarker frozen."
            if config.experiment_mode == "normalizer_sanity"
            else (
                "SynBaby-supervised normalizer training; landmarker frozen."
                if config.experiment_mode == "normalizer_train_frozen_landmarker"
                else "SynBaby-supervised normalizer plus HRNet stage 4 and head fine-tuning."
            )
        ),
        warnings=warnings,
    )
    commit_report_source = (
        Path(__file__).resolve().parent.parent / "reports" / "commit_report.md"
    )
    if commit_report_source.exists():
        run_commit_report = config.output_dir / "reports" / "commit_report.md"
        run_commit_report.write_text(
            commit_report_source.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if config.use_wandb:
        import wandb

        wandb_payload: dict[str, float] = {}
        for dataset_name, summary in diagnostics.items():
            for metric_name, value in summary.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    wandb_payload[f"normalizer/{dataset_name}/{metric_name}"] = float(
                        value
                    )
        _add_numeric_wandb_metrics(
            wandb_payload,
            evaluation_summary.get("summaries", {}),
            prefix="evaluation",
        )
        if wandb_payload:
            wandb.log(wandb_payload)
        wandb.finish()
    print(f"[INFO] Normalizer experiment report: {report_path}")


def _add_numeric_wandb_metrics(
    output: dict[str, float], value: object, prefix: str
) -> None:
    """Flatten numeric evaluation values into W&B metric names."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _add_numeric_wandb_metrics(output, nested_value, prefix=f"{prefix}/{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def main() -> None:
    """Execute the end-to-end experiment pipeline from the command line."""
    args = parse_args()
    from scripts.engine import run_full_evaluation, smoke_test_single_batch, train_model

    config = build_config_from_args(args)
    validate_experiment_config(config)
    resolve_output_dir(config)
    validate_full_evaluation_paths(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tee_terminal_output(config.output_dir / "train.log") as train_log_path:
        print("[INFO] Parsing CLI arguments...")
        print(f"[INFO] Terminal log file: {train_log_path}")
        print(f"[INFO] Run directory: {config.output_dir}")
        print(f"[INFO] Dataset root: {config.dataset_root}")
        print(f"[INFO] Run name: {config.wandb_run_name}")
        print(f"[INFO] Seed: {config.seed}")
        print("[INFO] Setting global seed and deterministic runtime options...")
        set_seed(config.seed)
        device = get_default_device(config.device)
        print(f"[INFO] Using device: {device}")
        print("[INFO] Building datasets and dataloaders...")
        dataloaders = build_dataloaders(config)
        print(
            "[INFO] Dataloaders ready | "
            f"train={len(dataloaders['train'].dataset)} "
            f"val={len(dataloaders['val'].dataset)} "
            f"test={len(dataloaders['test'].dataset)}"
        )
        if config.save_preview_batches:
            preview_dir = config.output_dir / "previews"
            print(f"[INFO] Saving deterministic dataset previews into {preview_dir}...")
            save_dataset_preview_grid(
                dataset=dataloaders["train"].dataset,
                output_path=preview_dir / "train_preview.png",
                title="Train Preview",
                num_samples=dataloaders["train"].batch_size or config.batch_size,
                seed=config.preview_seed,
                show_indices=config.show_landmark_indices,
                point_radius=max(3, config.overlay_point_radius // 2),
                line_width=max(2, config.overlay_line_width // 4),
                line_color=config.overlay_connection_color,
                mean=config.normalization_mean,
                std=config.normalization_std,
            )
            save_dataset_preview_grid(
                dataset=dataloaders["val"].dataset,
                output_path=preview_dir / "val_preview.png",
                title="Validation Preview",
                num_samples=dataloaders["val"].batch_size
                or config.eval_batch_size
                or config.batch_size,
                seed=config.preview_seed,
                show_indices=config.show_landmark_indices,
                point_radius=max(3, config.overlay_point_radius // 2),
                line_width=max(2, config.overlay_line_width // 4),
                line_color=config.overlay_connection_color,
                mean=config.normalization_mean,
                std=config.normalization_std,
            )
        print("[INFO] Building model...")
        model = build_model(config)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in trainable_parameters
        )
        print(
            "[INFO] Model ready | "
            f"total_params={total_parameters} "
            f"trainable_params={trainable_parameter_count}"
        )

        if config.checkpoint_path is not None:
            print(f"[INFO] Loading checkpoint from {config.checkpoint_path}...")
            checkpoint = torch.load(
                config.checkpoint_path, map_location="cpu", weights_only=False
            )
            if isinstance(model, NormalizedLandmarker):
                load_normalized_checkpoint(
                    model,
                    checkpoint,
                    strict=config.checkpoint_strict,
                    load_normalizer=config.experiment_mode != "normalizer_sanity",
                )
            else:
                model.load_state_dict(
                    checkpoint["model_state_dict"], strict=config.checkpoint_strict
                )
            print(f"Loaded checkpoint: {config.checkpoint_path}")

        if config.save_config or config.experiment_mode != "none":
            print("[INFO] Saving resolved config files...")
            maybe_save_config(config)
            save_resolved_config_files(config, config.output_dir)

        print("[INFO] Writing reproducibility metadata...")
        save_reproducibility_metadata(
            output_dir=config.output_dir,
            parsed_args=vars(args),
            resolved_config=config_to_serializable_dict(config),
            include_git_diff=config.include_git_diff,
            include_pip_freeze=config.include_pip_freeze,
        )
        print("[INFO] Writing model summary...")
        save_model_summary(
            model=model,
            output_dir=config.output_dir,
            input_size=(1, 3, config.image_size[0], config.image_size[1]),
        )

        if config.experiment_mode == "normalizer_sanity":
            print(
                "[INFO] Running identity-normalizer sanity evaluation; no optimization."
            )
            if config.use_wandb:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config_to_serializable_dict(config),
                    reinit=True,
                )
            model.to(device)
            full_evaluation_summary = run_full_evaluation(
                model=model,
                synbaby_dataloader=dataloaders["test"]
                if config.evaluate_synbaby
                else None,
                device=device,
                config=config,
            )
            finalize_normalizer_experiment(
                model=model,
                config=config,
                device=device,
                synbaby_dataloader=dataloaders["test"]
                if config.evaluate_synbaby
                else None,
                evaluation_summary=full_evaluation_summary,
            )
            return

        print("[INFO] Building optimizer, scheduler, and losses...")
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.lr_milestones),
            gamma=config.lr_gamma,
        )
        heatmap_loss_fn = build_landmark_heatmap_loss(config)
        visibility_loss_fn = torch.nn.BCEWithLogitsLoss()
        print(
            "[INFO] Training setup ready | "
            f"epochs={config.num_epochs} "
            f"batch_size={config.batch_size} "
            f"lr={config.learning_rate} "
            f"lambda_vis={config.lambda_vis} "
            f"lambda_lmk_vis={config.lambda_lmk_vis} "
            f"lambda_lmk_full={config.lambda_lmk_full} "
            f"landmark_loss={config.landmark_loss} "
            f"coordinate_decoder={config.coordinate_decoder} "
            f"lambda_pca_projection={config.lambda_pca_projection} "
            f"pca_prior_path={config.pca_prior_path} "
        )
        if config.landmark_loss == "adaptive_wing":
            print(
                "[INFO] Adaptive Wing hyperparameters | "
                f"omega={config.adaptive_wing_omega} "
                f"theta={config.adaptive_wing_theta} "
                f"epsilon={config.adaptive_wing_epsilon} "
                f"alpha={config.adaptive_wing_alpha}"
            )
        if config.landmark_loss == "wasserstein":
            print(
                "[INFO] Wasserstein hyperparameters | "
                f"softmax_temperature={config.wasserstein_softmax_temperature} "
                f"epsilon={config.wasserstein_epsilon}"
            )

        if config.run_smoke_test:
            print("[INFO] Running smoke test on one training batch...")
            model.to(device)
            smoke_test_single_batch(
                model=model,
                dataloader=dataloaders["train"],
                device=device,
                heatmap_loss_fn=heatmap_loss_fn,
                visibility_loss_fn=visibility_loss_fn,
                optimizer=optimizer,
                lambda_vis=config.lambda_vis,
                lambda_lmk_vis=config.lambda_lmk_vis,
                lambda_lmk_full=config.lambda_lmk_full,
                lambda_pca_projection=config.lambda_pca_projection,
                pca_prior_path=config.pca_prior_path,
                coordinate_decoder=config.coordinate_decoder,
                wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
            )

        print("[INFO] Starting training loop...")
        summary = train_model(
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            optimizer=optimizer,
            scheduler=scheduler,
            heatmap_loss_fn=heatmap_loss_fn,
            visibility_loss_fn=visibility_loss_fn,
            device=device,
            num_epochs=config.num_epochs,
            output_dir=config.output_dir,
            lambda_vis=config.lambda_vis,
            lambda_lmk_vis=config.lambda_lmk_vis,
            lambda_lmk_full=config.lambda_lmk_full,
            lambda_pca_projection=config.lambda_pca_projection,
            lambda_image_l1=config.normalizer_lambda_l1,
            lambda_image_tv=config.normalizer_lambda_tv,
            pca_prior_path=config.pca_prior_path,
            patience=config.patience,
            project_name=config.wandb_project,
            run_name=config.wandb_run_name,
            use_wandb=config.use_wandb,
            wandb_config=config_to_serializable_dict(config),
            finish_wandb=config.experiment_mode == "none",
            use_amp=config.use_amp,
            landmark_loss=config.landmark_loss,
            coordinate_decoder=config.coordinate_decoder,
            wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
            visualize_every_n_epochs=config.visualize_every_n_epochs,
            num_visualization_images=config.num_visualization_images,
            normalizer_monitoring_enabled=config.normalizer_monitoring_enabled,
            normalizer_monitor_probes=config.normalizer_monitor_probes,
            normalizer_monitor_steps=tuple(config.normalizer_monitor_steps),
            normalizer_monitor_difference_max=config.normalizer_monitor_difference_max,
            normalizer_monitor_registration_warning_px=(
                config.normalizer_monitor_registration_warning_px
            ),
            normalizer_monitor_edge_correlation_warning=(
                config.normalizer_monitor_edge_correlation_warning
            ),
            normalization_mean=tuple(config.normalization_mean),
            normalization_std=tuple(config.normalization_std),
        )
        print("[INFO] Training finished.")
        print(f"[INFO] Best epoch: {summary['best_epoch']}")
        print(f"[INFO] Best val loss: {summary['best_val_loss']:.4f}")
        print(f"[INFO] Results CSV: {summary['results_csv']}")

        best_checkpoint_path = config.output_dir / "best_model.pth"
        print(
            f"[INFO] Loading best checkpoint for final test evaluation: {best_checkpoint_path}"
        )
        best_checkpoint = torch.load(
            best_checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])
        model.to(device)

        full_evaluation_summary = run_full_evaluation(
            model=model,
            synbaby_dataloader=dataloaders["test"],
            device=device,
            config=config,
        )
        print(
            "[INFO] Full evaluation summary available for datasets: "
            f"{', '.join(full_evaluation_summary['summaries'])}"
        )
        if isinstance(model, NormalizedLandmarker):
            finalize_normalizer_experiment(
                model=model,
                config=config,
                device=device,
                synbaby_dataloader=dataloaders["test"]
                if config.evaluate_synbaby
                else None,
                evaluation_summary=full_evaluation_summary,
            )


if __name__ == "__main__":
    main()
