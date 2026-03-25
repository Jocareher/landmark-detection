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
from scripts.models import HRNetLandmarkVisibility
from scripts.utils import (
    get_default_device,
    save_model_summary,
    save_reproducibility_metadata,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the end-to-end training pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full landmark detection pipeline: train, validate, and test."
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, help="Checkpoint to load before continuing training."
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda-heatmap", type=float)
    parser.add_argument("--lambda-visibility", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--transfer-mode", choices=["feature_extractor", "fine_tuning"])
    parser.add_argument("--num-unfrozen-stages", type=int)
    parser.add_argument("--unfreeze-stem", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save resolved config JSON into output dir.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Merge CLI overrides into the default experiment configuration."""
    config = ExperimentConfig()
    if args.dataset_root is not None:
        config.dataset_root = args.dataset_root
    if args.output_dir is not None:
        config.runs_dir = args.output_dir
    if args.cache_dir is not None:
        config.cache_dir = args.cache_dir
    if args.pretrained_weights is not None:
        config.pretrained_weights = args.pretrained_weights
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.lambda_heatmap is not None:
        config.lambda_heatmap = args.lambda_heatmap
    if args.lambda_visibility is not None:
        config.lambda_visibility = args.lambda_visibility
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.transfer_mode is not None:
        config.transfer_mode = args.transfer_mode
    if args.num_unfrozen_stages is not None:
        config.num_unfrozen_stages = args.num_unfrozen_stages
    if args.unfreeze_stem:
        config.unfreeze_stem = True
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.disable_amp:
        config.use_amp = False
    if args.disable_cache:
        config.use_cache = False
    if args.smoke_test:
        config.run_smoke_test = True
    return config


def maybe_save_config(config: ExperimentConfig) -> None:
    """Persist the resolved configuration inside the active run directory."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    serialized = {
        key: str(value)
        if isinstance(value, Path)
        else list(value)
        if isinstance(value, tuple)
        else value
        for key, value in config.to_dict().items()
    }
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )


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
    from scripts.engine import evaluate_checkpoint, smoke_test_single_batch, train_model

    print("[INFO] Parsing CLI arguments...")
    config = build_config_from_args(args)
    config.resolve_output_dir()
    config.output_dir.mkdir(parents=True, exist_ok=True)
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
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.save_config:
        print("[INFO] Saving resolved config JSON...")
        maybe_save_config(config)

    print("[INFO] Writing reproducibility metadata...")
    save_reproducibility_metadata(
        output_dir=config.output_dir,
        parsed_args=vars(args),
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
        trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
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
        f"lambda_heatmap={config.lambda_heatmap} "
        f"lambda_visibility={config.lambda_visibility}"
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
            lambda_heatmap=config.lambda_heatmap,
            lambda_visibility=config.lambda_visibility,
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
        lambda_heatmap=config.lambda_heatmap,
        lambda_visibility=config.lambda_visibility,
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
    print(f"[INFO] Best val loss: {summary['best_val_loss']:.6f}")
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

    test_output_dir = config.output_dir / config.evaluation_dirname
    print("[INFO] Evaluating best model on test split...")
    test_summary = evaluate_checkpoint(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        output_dir=test_output_dir,
        visibility_threshold=config.visibility_threshold,
        save_predictions=False,
        save_overlays=False,
        show_indices=False,
        use_landmark_names_in_boxplot=config.use_landmark_names_in_boxplot,
    )
    print("[INFO] Test evaluation finished.")
    print(f"[INFO] Test mean NME: {test_summary['mean_nme']:.6f}")
    print(f"[INFO] Test median NME: {test_summary['median_nme']:.6f}")
    print(f"[INFO] Test visibility accuracy: {test_summary['visibility_accuracy']:.6f}")
    print(f"[INFO] Test evaluation dir: {test_output_dir}")


if __name__ == "__main__":
    main()
