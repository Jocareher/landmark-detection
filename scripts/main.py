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
    resolve_output_dir,
)
from scripts.dataset import build_dataloaders
from scripts.models import HRNetLandmarkVisibility
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
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Train the model on train/val and then evaluate the best checkpoint on test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=None,
        help="Optional checkpoint to load before continuing training.",
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
        default=False,
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
    config = build_config()
    config.dataset_root = args.dataset_root
    config.runs_dir = args.output_dir
    config.cache_dir = args.cache_dir
    config.pretrained_weights = args.pretrained_weights
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
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


def build_model(config: ExperimentConfig) -> HRNetLandmarkVisibility:
    """Instantiate the model, load pretrained weights, and configure trainable layers."""
    model = HRNetLandmarkVisibility(num_landmarks=config.num_landmarks)
    if (
        config.pretrained_weights is not None
        and Path(config.pretrained_weights).exists()
    ):
        model.load_official_hrnet_pretrained(
            str(config.pretrained_weights), verbose=True
        )
    model.set_transfer_learning_mode(
        mode=config.transfer_mode,
        num_unfrozen_stages=config.num_unfrozen_stages,
        unfreeze_stem=config.unfreeze_stem,
    )
    return model


def main() -> None:
    """Execute the end-to-end experiment pipeline from the command line."""
    args = parse_args()
    from scripts.engine import run_full_evaluation, smoke_test_single_batch, train_model

    config = build_config_from_args(args)
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

        if args.checkpoint is not None:
            print(f"[INFO] Loading checkpoint from {args.checkpoint}...")
            checkpoint = torch.load(
                args.checkpoint, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint: {args.checkpoint}")

        if args.save_config:
            print("[INFO] Saving resolved config JSON...")
            maybe_save_config(config)

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
        heatmap_loss_fn = torch.nn.MSELoss()
        visibility_loss_fn = torch.nn.BCEWithLogitsLoss()
        print(
            "[INFO] Training setup ready | "
            f"epochs={config.num_epochs} "
            f"batch_size={config.batch_size} "
            f"lr={config.learning_rate} "
            f"lambda_vis={config.lambda_vis} "
            f"lambda_lmk_vis={config.lambda_lmk_vis} "
            f"lambda_lmk_full={config.lambda_lmk_full} "
            f"lambda_pca_projection={config.lambda_pca_projection} "
            f"pca_prior_path={config.pca_prior_path} "
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
            pca_prior_path=config.pca_prior_path,
            patience=config.patience,
            project_name=config.wandb_project,
            run_name=config.wandb_run_name,
            use_wandb=config.use_wandb,
            use_amp=config.use_amp,
            visualize_every_n_epochs=config.visualize_every_n_epochs,
            num_visualization_images=config.num_visualization_images,
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


if __name__ == "__main__":
    main()
