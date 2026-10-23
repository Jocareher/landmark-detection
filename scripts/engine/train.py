from __future__ import annotations

import csv
import json
from .runtime_reporting import clock, save_training_curves
import math
import time
from pathlib import Path
from typing import Any

import torch

from .losses import compute_multitask_loss
from .metrics import (
    AverageMeter,
    compute_box_normalized_nme,
    decode_heatmaps_to_image_coords,
    compute_heatmap_uncertainty,
)
from .normalizer_experiments import compute_image_regularization
from .normalizer_monitoring import (
    NormalizerProbeMonitor,
    should_capture_source_step,
)
from .pca_shape_prior import load_pca_shape_prior
from ..models import NormalizedLandmarker
from ..utils.training_progress import TrainingProgressReporter


def _forward_with_normalized_images(
    model: torch.nn.Module, images: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run either a plain landmarker or a normalizer-wrapped landmarker."""
    if hasattr(model, "normalize_images") and hasattr(model, "forward_normalized"):
        normalized_images = model.normalize_images(images)
        return model.forward_normalized(normalized_images), normalized_images
    return model(images), images


def _add_image_regularization(
    loss_dict: dict[str, torch.Tensor],
    images: torch.Tensor,
    normalized_images: torch.Tensor,
    lambda_image_l1: float,
    lambda_image_tv: float,
) -> None:
    """Add residual-image regularization without changing the landmark losses."""
    image_losses = compute_image_regularization(
        images,
        normalized_images,
        lambda_l1=lambda_image_l1,
        lambda_tv=lambda_image_tv,
    )
    loss_dict.update(image_losses)
    loss_dict["total_loss"] = (
        loss_dict["total_loss"] + image_losses["image_regularization_loss"]
    )


def _compute_visible_box_normalized_nme(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    visibility: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-image box-normalized NME over visible landmarks only.

    The normalization box is computed from the complete target shape, matching
    ``compute_box_normalized_nme``. Only the final point-error average is masked
    by target visibility. Images without a valid visible landmark return NaN
    and are excluded from the running diagnostic average.
    """
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Predictions and targets must have shape (B, K, 2).")
    if visibility.shape != targets.shape[:2]:
        raise ValueError("Visibility must have shape (B, K).")
    finite_targets = torch.isfinite(targets).all(dim=-1)
    finite_predictions = torch.isfinite(predictions).all(dim=-1)
    valid = (visibility > 0) & finite_targets & finite_predictions
    minimum = targets.amin(dim=1)
    maximum = targets.amax(dim=1)
    box_size = maximum - minimum
    normalization = torch.sqrt((box_size[:, 0] * box_size[:, 1]).clamp_min(float(eps)))
    point_errors = torch.linalg.vector_norm(predictions - targets, dim=-1)
    counts = valid.sum(dim=1)
    visible_error = (point_errors * valid).sum(dim=1) / counts.clamp_min(1)
    values = visible_error / normalization
    return torch.where(
        counts > 0,
        values,
        torch.full_like(values, float("nan")),
    )


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    lambda_vis: float = 1.0,
    lambda_lmk_vis: float = 1.0,
    lambda_lmk_full: float = 1.0,
    lambda_pca_projection: float = 0.0,
    lambda_image_l1: float = 0.0,
    lambda_image_tv: float = 0.0,
    pca_shape_prior: dict[str, Any] | None = None,
    training: bool = True,
    use_subpixel_decode: bool = False,
    use_amp: bool = True,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
    progress_desc: str | None = None,
    progress_reporter: TrainingProgressReporter | None = None,
    learning_rate: float = 0.0,
    pca_loss_space: str = "image",
    pca_mahalanobis_enabled: bool = False,
    pca_mahalanobis_weight: float = 1.0,
    pca_mahalanobis_c: float = 10.0,
    pca_mahalanobis_eps: float = 1e-8,
) -> dict[str, float]:
    """Run one full training or validation epoch and aggregate split metrics."""
    if training and optimizer is None:
        raise ValueError("An optimizer is required when training=True.")

    model.train(training)
    from ..models.normalization import SourceAdaptiveInstanceNorm2d
    for module in model.modules():
        if isinstance(module, SourceAdaptiveInstanceNorm2d):
            module.collect_source = training
    total_loss_meter = AverageMeter()
    full_landmark_loss_meter = AverageMeter()
    visible_landmark_loss_meter = AverageMeter()
    visibility_loss_meter = AverageMeter()
    pca_projection_loss_meter = AverageMeter()
    pca_term_meters = {key: AverageMeter() for key in ('pca_reconstruction_loss', 'pca_mahalanobis_squared', 'pca_mahalanobis_penalty', 'weighted_pca_mahalanobis_loss')}
    image_l1_loss_meter = AverageMeter()
    image_tv_loss_meter = AverageMeter()
    nme_meter = AverageMeter()
    entropy_meter = AverageMeter()
    spread_meter = AverageMeter()
    visible_nme_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    autocast_device = device.type
    epoch_start_time = clock(device)
    validation_inference_seconds = 0.0
    validation_inference_images = 0

    for batch in dataloader:
        batch_start_time = time.time()
        images = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        visibility = batch["visibility"].to(device, non_blocking=True)
        batch_on_device: dict[str, torch.Tensor] = {
            "heatmaps": heatmaps,
            "visibility": visibility,
        }
        if "landmarks" in batch:
            batch_on_device["landmarks"] = batch["landmarks"].to(
                device, non_blocking=True
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs, normalized_images = _forward_with_normalized_images(
                    model, images
                )
                loss_dict = compute_multitask_loss(
                    outputs=outputs,
                    batch=batch_on_device,
                    heatmap_loss_fn=heatmap_loss_fn,
                    visibility_loss_fn=visibility_loss_fn,
                    lambda_vis=lambda_vis,
                    lambda_lmk_vis=lambda_lmk_vis,
                    lambda_lmk_full=lambda_lmk_full,
                    lambda_pca_projection=lambda_pca_projection,
                    pca_loss_space=pca_loss_space,
                    pca_mahalanobis_enabled=pca_mahalanobis_enabled,
                    pca_mahalanobis_weight=pca_mahalanobis_weight,
                    pca_mahalanobis_c=pca_mahalanobis_c,
                    pca_mahalanobis_eps=pca_mahalanobis_eps,
                    pca_shape_prior=pca_shape_prior,
                    image_height=images.shape[2],
                    image_width=images.shape[3],
                )
                _add_image_regularization(
                    loss_dict,
                    images,
                    normalized_images,
                    lambda_image_l1,
                    lambda_image_tv,
                )
            total_loss = loss_dict["total_loss"]
            if use_amp:
                if scaler is None:
                    raise ValueError(
                        "A GradScaler is required when use_amp=True and training=True."
                    )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
        else:
            with torch.inference_mode():
                with torch.autocast(device_type=autocast_device, enabled=use_amp):
                    forward_started = clock(device)
                    outputs, normalized_images = _forward_with_normalized_images(
                        model, images
                    )
                    validation_inference_seconds += clock(device)-forward_started
                    validation_inference_images += images.shape[0]
                    loss_dict = compute_multitask_loss(
                        outputs=outputs,
                        batch=batch_on_device,
                        heatmap_loss_fn=heatmap_loss_fn,
                        visibility_loss_fn=visibility_loss_fn,
                        lambda_vis=lambda_vis,
                        lambda_lmk_vis=lambda_lmk_vis,
                        lambda_lmk_full=lambda_lmk_full,
                        lambda_pca_projection=lambda_pca_projection,
                        pca_loss_space=pca_loss_space,
                        pca_mahalanobis_enabled=pca_mahalanobis_enabled,
                        pca_mahalanobis_weight=pca_mahalanobis_weight,
                        pca_mahalanobis_c=pca_mahalanobis_c,
                        pca_mahalanobis_eps=pca_mahalanobis_eps,
                        pca_shape_prior=pca_shape_prior,
                        image_height=images.shape[2],
                        image_width=images.shape[3],
                    )
                    _add_image_regularization(
                        loss_dict,
                        images,
                        normalized_images,
                        lambda_image_l1,
                        lambda_image_tv,
                    )
            total_loss = loss_dict["total_loss"]

        if not bool(torch.isfinite(total_loss)) and progress_reporter is not None:
            phase = progress_desc or ("TRAIN" if training else "VALIDATION")
            progress_reporter.error(f"Non-finite total loss detected during {phase}.")

        batch_size = images.size(0)
        uncertainty = compute_heatmap_uncertainty(outputs["heatmaps"])
        entropy_meter.update(float(uncertainty["entropy"].mean()), batch_size)
        spread_meter.update(float(uncertainty["spread"].mean()), batch_size)
        total_loss_meter.update(total_loss.item(), batch_size)
        full_landmark_loss_meter.update(
            loss_dict["full_landmark_loss"].item(), batch_size
        )
        visible_landmark_loss_meter.update(
            loss_dict["visible_landmark_loss"].item(), batch_size
        )
        visibility_loss_meter.update(loss_dict["visibility_loss"].item(), batch_size)
        pca_projection_loss_meter.update(loss_dict["pca_loss"].item(), batch_size)
        for key, meter in pca_term_meters.items():
            meter.update(loss_dict[key].item(), batch_size)
        image_l1_loss_meter.update(loss_dict["image_l1_loss"].item(), batch_size)
        image_tv_loss_meter.update(loss_dict["image_tv_loss"].item(), batch_size)

        if "landmarks" in batch_on_device:
            pred_landmarks = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"].detach(),
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=use_subpixel_decode,
                decoder=coordinate_decoder,
                softmax_temperature=wasserstein_softmax_temperature,
            )
            nme_values = compute_box_normalized_nme(
                preds=pred_landmarks, targets=batch_on_device["landmarks"]
            )
            nme_meter.update(float(nme_values.mean()), batch_size)
            visible_nme_values = _compute_visible_box_normalized_nme(
                predictions=pred_landmarks,
                targets=batch_on_device["landmarks"],
                visibility=visibility,
            )
            finite_visible_nme = visible_nme_values[torch.isfinite(visible_nme_values)]
            if finite_visible_nme.numel() > 0:
                visible_nme_meter.update(
                    float(finite_visible_nme.mean()),
                    int(finite_visible_nme.numel()),
                )

        batch_time_meter.update(time.time() - batch_start_time)
        if progress_reporter is not None:
            update = (
                progress_reporter.update_train_batch
                if training
                else progress_reporter.update_validation_batch
            )
            update(
                total_loss=total_loss_meter.avg,
                nme=nme_meter.avg if nme_meter.count else None,
                visible_nme=(
                    visible_nme_meter.avg if visible_nme_meter.count else None
                ),
                learning_rate=learning_rate,
            )

    return {
        "mean_heatmap_entropy": entropy_meter.avg,
        "mean_heatmap_spread": spread_meter.avg,
        "total_loss": total_loss_meter.avg,
        "full_landmark_loss": full_landmark_loss_meter.avg,
        "visible_landmark_loss": visible_landmark_loss_meter.avg,
        "visibility_loss": visibility_loss_meter.avg,
        "pca_loss": pca_projection_loss_meter.avg,
        **{key: meter.avg for key, meter in pca_term_meters.items()},
        "image_l1_loss": image_l1_loss_meter.avg,
        "image_tv_loss": image_tv_loss_meter.avg,
        "nme": nme_meter.avg if nme_meter.count else float("nan"),
        "visible_nme": (
            visible_nme_meter.avg if visible_nme_meter.count else float("nan")
        ),
        "batch_time": batch_time_meter.avg,
        "epoch_time": clock(device) - epoch_start_time,
        "validation_forward_images": validation_inference_images if not training else None,
        "validation_inference_seconds": validation_inference_seconds if not training else None,
        "validation_inference_seconds_per_image": (validation_inference_seconds/validation_inference_images
                                                   if validation_inference_images else None),
    }


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metrics: dict[str, Any],
    pca_loss_space: str | None = None,
    pca_mahalanobis_enabled: bool = False,
    pca_mahalanobis_weight: float = 1.0,
    pca_mahalanobis_c: float = 10.0,
    pca_mahalanobis_eps: float = 1e-8,
) -> None:
    """Save a model checkpoint with optimizer state and tracked metrics."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "metrics": metrics,
        "model_type": type(model).__name__,
    }
    if pca_loss_space is not None:
        payload["pca_loss_space"] = pca_loss_space
    payload["pca_mahalanobis_enabled"] = pca_mahalanobis_enabled
    payload["pca_mahalanobis_weight"] = pca_mahalanobis_weight
    payload["pca_mahalanobis_c"] = pca_mahalanobis_c
    payload["pca_mahalanobis_eps"] = pca_mahalanobis_eps

    if isinstance(model, NormalizedLandmarker) and model.normalizer is not None:
        payload["normalizer_architecture"] = model.normalizer.architecture_config()
    landmarker = model.landmarker if isinstance(model, NormalizedLandmarker) else model
    if hasattr(landmarker, "architecture_config"):
        payload["landmarker_architecture"] = landmarker.architecture_config()
    torch.save(payload, checkpoint_path)


def initialize_results_csv(csv_path: str | Path) -> None:
    """Create the metrics CSV file and header if it does not exist yet."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "split",
                "total_loss",
                "full_landmark_loss",
                "visible_landmark_loss",
                "visibility_loss",
                "pca_loss",
                "image_l1_loss",
                "image_tv_loss",
                "nme",
                "visible_nme",
                "lr",
                "landmark_loss",
                "coordinate_decoder",
                "epoch_time",
                "mean_heatmap_entropy",
                "mean_heatmap_spread",
            ]
        )


def append_results_row(
    csv_path: str | Path,
    epoch: int,
    split: str,
    metrics: dict[str, float],
    lr: float,
    landmark_loss: str,
    coordinate_decoder: str,
) -> None:
    """Append one train or validation metrics row to the experiment CSV."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = [key for key in ("mean_heatmap_entropy", "mean_heatmap_spread") if key not in fields]
        old_rows = list(reader) if missing else []
    if missing:
        fields.extend(missing)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)
    row = {**metrics, "epoch": epoch, "split": split, "lr": lr,
           "landmark_loss": landmark_loss, "coordinate_decoder": coordinate_decoder}
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerow(row)


def _device_display_name(device: torch.device) -> str:
    """Return a human-readable accelerator name without changing the device."""
    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return torch.cuda.get_device_name(index)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU"


def _quiet_wandb_settings(wandb_module: Any) -> Any | None:
    """Build W&B settings that suppress startup chatter but retain warnings."""
    try:
        return wandb_module.Settings(console="off", quiet=True)
    except (AttributeError, TypeError):
        return None


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    output_dir: str | Path,
    lambda_vis: float = 1.0,
    lambda_lmk_vis: float = 1.0,
    lambda_lmk_full: float = 1.0,
    lambda_pca_projection: float = 0.0,
    lambda_image_l1: float = 0.0,
    lambda_image_tv: float = 0.0,
    pca_prior_path: str | Path | None = None,
    patience: int = 15,
    project_name: str | None = None,
    run_name: str | None = None,
    use_wandb: bool = True,
    wandb_config: dict[str, Any] | None = None,
    finish_wandb: bool = True,
    use_amp: bool = True,
    landmark_loss: str = "mse",
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
    visualize_every_n_epochs: int = 5,
    num_visualization_images: int = 4,
    normalizer_monitoring_enabled: bool = False,
    normalizer_monitor_probes: int = 4,
    normalizer_monitor_steps: tuple[int, ...] = (0, 1, 5, 10, 20),
    normalizer_monitor_difference_max: float = 0.15,
    normalizer_monitor_registration_warning_px: float = 1.0,
    normalizer_monitor_edge_correlation_warning: float = 0.90,
    normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225),
    progress_reporter: TrainingProgressReporter | None = None,
    pca_loss_space: str = "image",
    pca_mahalanobis_enabled: bool = False,
    pca_mahalanobis_weight: float = 1.0,
    pca_mahalanobis_c: float = 10.0,
    pca_mahalanobis_eps: float = 1e-8,
) -> dict[str, Any]:
    """Execute the full training pipeline, including validation and checkpointing."""
    from .pca_shape_prior import validate_pca_mahalanobis
    validate_pca_mahalanobis(pca_mahalanobis_enabled, pca_mahalanobis_weight, pca_mahalanobis_c, pca_mahalanobis_eps)
    training_started = clock(device)
    if pca_loss_space not in {"aligned", "image"}:
        raise ValueError(f"Unsupported PCA loss space: {pca_loss_space}")
    wandb = None
    if use_wandb:
        import wandb as wandb_module

        wandb = wandb_module

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = output_dir / "results.csv"
    initialize_results_csv(results_csv_path)
    model.to(device)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if device.type == "cuda"
        else None
    )

    wandb_run = None
    if use_wandb and wandb is not None:
        init_kwargs: dict[str, Any] = {
            "project": project_name,
            "name": run_name,
            "config": wandb_config,
            "reinit": True,
        }
        quiet_settings = _quiet_wandb_settings(wandb)
        if quiet_settings is not None:
            init_kwargs["settings"] = quiet_settings
        wandb_run = wandb.init(
            **init_kwargs,
        )

    reporter = progress_reporter or TrainingProgressReporter()
    wandb_url = getattr(wandb_run, "url", None) if wandb_run is not None else None
    reporter.start_run(
        run_name=run_name,
        device=str(device),
        device_name=_device_display_name(device),
        train_samples=len(train_loader.dataset),
        validation_samples=len(val_loader.dataset),
        batch_size=train_loader.batch_size,
        epochs=num_epochs,
        optimizer_name=optimizer.__class__.__name__,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        wandb_project=project_name if use_wandb else None,
        wandb_url=wandb_url,
        checkpoint_dir=output_dir,
    )

    normalizer_monitor = None
    if normalizer_monitoring_enabled and isinstance(model, NormalizedLandmarker):
        normalizer_monitor = NormalizerProbeMonitor.from_dataloader(
            model=model,
            dataloader=val_loader,
            device=device,
            output_dir=output_dir / "normalizer_monitoring",
            mean=normalization_mean,
            std=normalization_std,
            coordinate_decoder=coordinate_decoder,
            softmax_temperature=wasserstein_softmax_temperature,
            max_images=normalizer_monitor_probes,
            difference_display_max=normalizer_monitor_difference_max,
            registration_warning_px=normalizer_monitor_registration_warning_px,
            edge_correlation_warning=normalizer_monitor_edge_correlation_warning,
            wandb_module=wandb if use_wandb else None,
        )
        if should_capture_source_step(0, normalizer_monitor_steps):
            normalizer_monitor.capture(stage="source_validation", step=0)

    pca_shape_prior = None
    if pca_prior_path is not None:
        pca_prior_path = Path(pca_prior_path)
        if not pca_prior_path.exists():
            raise FileNotFoundError(f"PCA prior file not found: {pca_prior_path}")
        pca_shape_prior = load_pca_shape_prior(pca_prior_path, device=device)
        if lambda_pca_projection > 0 and pca_mahalanobis_enabled:
            from .pca_shape_prior import validate_pca_mahalanobis_prior
            validate_pca_mahalanobis_prior(pca_shape_prior)
    if lambda_pca_projection > 0.0 and pca_shape_prior is None:
        raise ValueError("lambda_pca_projection > 0 requires a valid pca_prior_path.")

    best_val_loss = float("inf")
    best_val_nme = float("inf")
    best_nme_epoch = -1
    best_epoch = -1
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        # One loop iteration corresponds to one complete train/validation cycle.
        current_lr = optimizer.param_groups[0]["lr"]
        reporter.start_epoch(epoch + 1, num_epochs, current_lr)
        reporter.start_train(len(train_loader))
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            heatmap_loss_fn=heatmap_loss_fn,
            visibility_loss_fn=visibility_loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            lambda_vis=lambda_vis,
            lambda_lmk_vis=lambda_lmk_vis,
            lambda_lmk_full=lambda_lmk_full,
            lambda_pca_projection=lambda_pca_projection,
            pca_loss_space=pca_loss_space,
            pca_mahalanobis_enabled=pca_mahalanobis_enabled,
            pca_mahalanobis_weight=pca_mahalanobis_weight,
            pca_mahalanobis_c=pca_mahalanobis_c,
            pca_mahalanobis_eps=pca_mahalanobis_eps,
            lambda_image_l1=lambda_image_l1,
            lambda_image_tv=lambda_image_tv,
            pca_shape_prior=pca_shape_prior,
            training=True,
            use_subpixel_decode=True,
            use_amp=amp_enabled,
            coordinate_decoder=coordinate_decoder,
            wasserstein_softmax_temperature=wasserstein_softmax_temperature,
            progress_desc=f"Train {epoch + 1:03d}",
            progress_reporter=reporter,
            learning_rate=current_lr,
        )
        reporter.finish_train(train_metrics)
        reporter.start_validation(len(val_loader))
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            heatmap_loss_fn=heatmap_loss_fn,
            visibility_loss_fn=visibility_loss_fn,
            optimizer=None,
            scaler=None,
            lambda_vis=lambda_vis,
            lambda_lmk_vis=lambda_lmk_vis,
            lambda_lmk_full=lambda_lmk_full,
            lambda_pca_projection=lambda_pca_projection,
            pca_loss_space=pca_loss_space,
            pca_mahalanobis_enabled=pca_mahalanobis_enabled,
            pca_mahalanobis_weight=pca_mahalanobis_weight,
            pca_mahalanobis_c=pca_mahalanobis_c,
            pca_mahalanobis_eps=pca_mahalanobis_eps,
            lambda_image_l1=lambda_image_l1,
            lambda_image_tv=lambda_image_tv,
            pca_shape_prior=pca_shape_prior,
            training=False,
            use_subpixel_decode=True,
            use_amp=amp_enabled,
            coordinate_decoder=coordinate_decoder,
            wasserstein_softmax_temperature=wasserstein_softmax_temperature,
            progress_desc=f"Val   {epoch + 1:03d}",
            progress_reporter=reporter,
            learning_rate=current_lr,
        )
        reporter.finish_validation(val_metrics)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        save_training_curves(history, output_dir, plot=False)
        append_results_row(
            results_csv_path,
            epoch,
            "train",
            train_metrics,
            current_lr,
            landmark_loss,
            coordinate_decoder,
        )
        append_results_row(
            results_csv_path,
            epoch,
            "val",
            val_metrics,
            current_lr,
            landmark_loss,
            coordinate_decoder,
        )

        metrics_payload = {"train": train_metrics, "val": val_metrics, "lr": current_lr}
        save_checkpoint(
            output_dir / "last_model.pth", epoch, model, optimizer, metrics_payload,
            pca_loss_space=pca_loss_space,
            pca_mahalanobis_enabled=pca_mahalanobis_enabled,
            pca_mahalanobis_weight=pca_mahalanobis_weight,
            pca_mahalanobis_c=pca_mahalanobis_c,
            pca_mahalanobis_eps=pca_mahalanobis_eps,
        )
        reporter.report_checkpoint(output_dir / "last_model.pth", is_best=False)

        checkpoint_improved = val_metrics["total_loss"] < best_val_loss
        if checkpoint_improved:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                output_dir / "best_model.pth", epoch, model, optimizer, metrics_payload,
                pca_loss_space=pca_loss_space,
                pca_mahalanobis_enabled=pca_mahalanobis_enabled,
                pca_mahalanobis_weight=pca_mahalanobis_weight,
                pca_mahalanobis_c=pca_mahalanobis_c,
                pca_mahalanobis_eps=pca_mahalanobis_eps,
            )
            reporter.report_checkpoint(output_dir / "best_model.pth", is_best=True)
        else:
            patience_counter += 1

        current_val_nme = float(val_metrics.get("nme", float("nan")))
        if math.isfinite(current_val_nme) and current_val_nme < best_val_nme:
            best_val_nme = current_val_nme
            best_nme_epoch = epoch + 1

        reporter.finish_epoch(
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            learning_rate=current_lr,
            best_validation_nme=best_val_nme,
            best_nme_epoch=best_nme_epoch,
            checkpoint_improved=checkpoint_improved,
            early_stopping_counter=patience_counter,
            patience=patience,
        )

        if use_wandb and wandb is not None:
            wandb.log(
                {
                    **{f"{split}/{metric}": values[metric] for split, values in (("train", train_metrics), ("val", val_metrics)) for metric in ("mean_heatmap_entropy", "mean_heatmap_spread")},
                    "epoch": epoch,
                    "lr": current_lr,
                    "train/total_loss": train_metrics["total_loss"],
                    "train/full_landmark_loss": train_metrics["full_landmark_loss"],
                    "train/visible_landmark_loss": train_metrics[
                        "visible_landmark_loss"
                    ],
                    "train/visibility_loss": train_metrics["visibility_loss"],
                    "train/pca_loss": train_metrics["pca_loss"],
                    "train/pca_reconstruction_loss": train_metrics["pca_reconstruction_loss"],
                    "train/pca_mahalanobis_squared": train_metrics["pca_mahalanobis_squared"],
                    "train/pca_mahalanobis_penalty": train_metrics["pca_mahalanobis_penalty"],
                    "train/weighted_pca_mahalanobis_loss": train_metrics["weighted_pca_mahalanobis_loss"],

                    "train/image_l1_loss": train_metrics["image_l1_loss"],
                    "train/image_tv_loss": train_metrics["image_tv_loss"],
                    "train/nme": train_metrics["nme"],
                    "train/visible_nme": train_metrics["visible_nme"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/full_landmark_loss": val_metrics["full_landmark_loss"],
                    "val/visible_landmark_loss": val_metrics["visible_landmark_loss"],
                    "val/visibility_loss": val_metrics["visibility_loss"],
                    "val/pca_loss": val_metrics["pca_loss"],
                    "val/pca_reconstruction_loss": val_metrics["pca_reconstruction_loss"],
                    "val/pca_mahalanobis_squared": val_metrics["pca_mahalanobis_squared"],
                    "val/pca_mahalanobis_penalty": val_metrics["pca_mahalanobis_penalty"],
                    "val/weighted_pca_mahalanobis_loss": val_metrics["weighted_pca_mahalanobis_loss"],

                    "val/image_l1_loss": val_metrics["image_l1_loss"],
                    "val/image_tv_loss": val_metrics["image_tv_loss"],
                    "val/nme": val_metrics["nme"],
                    "val/visible_nme": val_metrics["visible_nme"],
                    "best/val_total_loss": best_val_loss,
                    "best/val_nme": best_val_nme,
                }
            )

        if visualize_every_n_epochs > 0 and (epoch + 1) % visualize_every_n_epochs == 0:
            from ..utils.visualization import (
                visualize_predicted_heatmaps_on_train_batch,
            )

            visualize_predicted_heatmaps_on_train_batch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
                output_dir=output_dir / "training_visualizations",
                num_images=num_visualization_images,
                grid_cols=min(num_visualization_images, 4),
                use_wandb=use_wandb and wandb is not None,
                coordinate_decoder=coordinate_decoder,
                wasserstein_softmax_temperature=wasserstein_softmax_temperature,
            )
            reporter.info(
                f"Saved predicted heatmap visualizations for epoch {epoch + 1}."
            )

        epoch_number = epoch + 1
        if normalizer_monitor is not None and should_capture_source_step(
            epoch_number, normalizer_monitor_steps
        ):
            normalizer_monitor.capture(stage="source_validation", step=epoch_number)

        scheduler.step()
        next_lr = float(optimizer.param_groups[0]["lr"])
        reporter.report_learning_rate_change(current_lr, next_lr)
        if patience_counter >= patience:
            reporter.warning(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    from .heatmap_diagnostics import save_training_heatmap_curves
    save_training_heatmap_curves(history, output_dir)
    save_training_curves(history, output_dir)
    final_epoch = len(history["train"])
    if normalizer_monitor is not None:
        normalizer_monitor.capture(
            stage="source_validation",
            step=final_epoch,
            is_final=True,
        )

    final_train_nme = history["train"][-1]["nme"] if history["train"] else None
    final_val_nme = history["val"][-1]["nme"] if history["val"] else None
    final_train_pca_loss = (
        history["train"][-1]["pca_loss"] if history["train"] else None
    )
    final_val_pca_loss = history["val"][-1]["pca_loss"] if history["val"] else None
    reporter.finish_run(
        best_epoch=best_epoch + 1 if best_epoch >= 0 else -1,
        best_validation_nme=best_val_nme,
        final_train_nme=final_train_nme,
        final_validation_nme=final_val_nme,
        final_train_pca_loss=final_train_pca_loss,
        final_validation_pca_loss=final_val_pca_loss,
        best_checkpoint_path=output_dir / "best_model.pth",
        wandb_url=wandb_url,
    )

    if use_wandb and wandb is not None and finish_wandb:
        wandb.finish()

    runtime = {"training_total_seconds": clock(device)-training_started,
               "train_epoch_seconds": sum(row["epoch_time"] for row in history["train"]),
               "val_epoch_seconds": sum(row["epoch_time"] for row in history["val"]),
               "completed_epochs": len(history["train"]),
               "validation_inference_seconds": sum(row["validation_inference_seconds"] for row in history["val"]),
               "validation_forward_images": sum(row["validation_forward_images"] for row in history["val"]),
               "boundary": "train_model entry through train/val, checkpoints, monitoring and final curves; excludes subsequent test evaluation"}
    (output_dir/"timing").mkdir(parents=True, exist_ok=True)
    (output_dir/"timing"/"training_summary.json").write_text(json.dumps(runtime, indent=2))
    print(f"[TIMING] Training total: {runtime['training_total_seconds']:.2f} s")
    return {
        "timing": runtime,
        "best_val_loss": best_val_loss,
        "pca_loss_space": pca_loss_space,
        "pca_mahalanobis_enabled": pca_mahalanobis_enabled,
        "pca_mahalanobis_weight": pca_mahalanobis_weight,
        "pca_mahalanobis_c": pca_mahalanobis_c,
        "pca_mahalanobis_eps": pca_mahalanobis_eps,

        "best_val_nme": best_val_nme,
        "best_nme_epoch": best_nme_epoch,
        "best_epoch": best_epoch,
        "history": history,
        "results_csv": str(results_csv_path),
        "wandb_url": wandb_url,
    }


def smoke_test_single_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_vis: float = 1.0,
    lambda_lmk_vis: float = 1.0,
    lambda_lmk_full: float = 1.0,
    lambda_pca_projection: float = 0.0,
    pca_prior_path: str | Path | None = None,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
    pca_loss_space: str = "image",
    pca_mahalanobis_enabled: bool = False,
    pca_mahalanobis_weight: float = 1.0,
    pca_mahalanobis_c: float = 10.0,
    pca_mahalanobis_eps: float = 1e-8,
) -> None:
    """Run a single optimization step to validate the end-to-end training path."""
    model.train()
    batch = next(iter(dataloader))
    images = batch["image"].to(device, non_blocking=True)
    heatmaps = batch["heatmaps"].to(device, non_blocking=True)
    visibility = batch["visibility"].to(device, non_blocking=True)
    landmarks = batch["landmarks"].to(device, non_blocking=True)
    pca_shape_prior = None
    if pca_prior_path is not None:
        pca_shape_prior = load_pca_shape_prior(pca_prior_path, device=device)
        if lambda_pca_projection > 0 and pca_mahalanobis_enabled:
            from .pca_shape_prior import validate_pca_mahalanobis_prior
            validate_pca_mahalanobis_prior(pca_shape_prior)
    if lambda_pca_projection > 0.0 and pca_shape_prior is None:
        raise ValueError("lambda_pca_projection > 0 requires a valid pca_prior_path.")
    outputs = model(images)
    loss_dict = compute_multitask_loss(
        outputs=outputs,
        batch={
            "heatmaps": heatmaps,
            "visibility": visibility,
        },
        heatmap_loss_fn=heatmap_loss_fn,
        visibility_loss_fn=visibility_loss_fn,
        lambda_vis=lambda_vis,
        lambda_lmk_vis=lambda_lmk_vis,
        lambda_lmk_full=lambda_lmk_full,
        lambda_pca_projection=lambda_pca_projection,
        pca_loss_space=pca_loss_space,
        pca_mahalanobis_enabled=pca_mahalanobis_enabled,
        pca_mahalanobis_weight=pca_mahalanobis_weight,
        pca_mahalanobis_c=pca_mahalanobis_c,
        pca_mahalanobis_eps=pca_mahalanobis_eps,
        pca_shape_prior=pca_shape_prior,
        image_height=images.shape[2],
        image_width=images.shape[3],
    )
    optimizer.zero_grad(set_to_none=True)
    loss_dict["total_loss"].backward()
    optimizer.step()
    pred_landmarks = decode_heatmaps_to_image_coords(
        heatmaps=outputs["heatmaps"].detach(),
        image_height=images.shape[2],
        image_width=images.shape[3],
        use_subpixel=True,
        decoder=coordinate_decoder,
        softmax_temperature=wasserstein_softmax_temperature,
    )
    nme_values = compute_box_normalized_nme(preds=pred_landmarks, targets=landmarks)
    print("Smoke test passed.")
    print(f"Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"Full landmark loss: {loss_dict['full_landmark_loss'].item():.6f}")
    print(f"Visible landmark loss: {loss_dict['visible_landmark_loss'].item():.6f}")
    print(f"Visibility loss: {loss_dict['visibility_loss'].item():.6f}")
    print(f"PCA loss: {loss_dict['pca_loss'].item():.6f}")
    print(f"NME: {float(nme_values.mean()):.6f}")
