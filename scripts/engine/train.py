from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .losses import compute_multitask_loss
from .metrics import (
    AverageMeter,
    compute_box_normalized_nme,
    decode_heatmaps_to_image_coords,
)
from .pca_shape_prior import load_pca_shape_prior
from ..utils.visualization import visualize_predicted_heatmaps_on_train_batch


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
    pca_shape_prior: dict[str, Any] | None = None,
    training: bool = True,
    use_subpixel_decode: bool = False,
    use_amp: bool = True,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Run one full training or validation epoch and aggregate split metrics."""
    if training and optimizer is None:
        raise ValueError("An optimizer is required when training=True.")

    model.train(training)
    total_loss_meter = AverageMeter()
    full_landmark_loss_meter = AverageMeter()
    visible_landmark_loss_meter = AverageMeter()
    visibility_loss_meter = AverageMeter()
    pca_projection_loss_meter = AverageMeter()
    nme_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    autocast_device = device.type
    epoch_start_time = time.time()
    progress_bar = tqdm(
        dataloader,
        desc=progress_desc or ("Train" if training else "Val"),
        dynamic_ncols=True,
        leave=False,
        ascii=False,
    )

    for batch in progress_bar:
        batch_start_time = time.time()
        images = batch["image"].to(device, non_blocking=True)
        heatmaps = batch["heatmaps"].to(device, non_blocking=True)
        visibility = batch["visibility"].to(device, non_blocking=True)
        class_idx = batch["class_idx"].to(device, non_blocking=True)
        batch_on_device: dict[str, torch.Tensor] = {
            "heatmaps": heatmaps,
            "visibility": visibility,
            "class_idx": class_idx,
        }
        if "landmarks" in batch:
            batch_on_device["landmarks"] = batch["landmarks"].to(
                device, non_blocking=True
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(images)
                loss_dict = compute_multitask_loss(
                    outputs=outputs,
                    batch=batch_on_device,
                    heatmap_loss_fn=heatmap_loss_fn,
                    visibility_loss_fn=visibility_loss_fn,
                    lambda_vis=lambda_vis,
                    lambda_lmk_vis=lambda_lmk_vis,
                    lambda_lmk_full=lambda_lmk_full,
                    lambda_pca_projection=lambda_pca_projection,
                    pca_shape_prior=pca_shape_prior,
                    image_height=images.shape[2],
                    image_width=images.shape[3],
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
                    outputs = model(images)
                    loss_dict = compute_multitask_loss(
                        outputs=outputs,
                        batch=batch_on_device,
                        heatmap_loss_fn=heatmap_loss_fn,
                        visibility_loss_fn=visibility_loss_fn,
                        lambda_vis=lambda_vis,
                        lambda_lmk_vis=lambda_lmk_vis,
                        lambda_lmk_full=lambda_lmk_full,
                        lambda_pca_projection=lambda_pca_projection,
                        pca_shape_prior=pca_shape_prior,
                        image_height=images.shape[2],
                        image_width=images.shape[3],
                    )
            total_loss = loss_dict["total_loss"]

        batch_size = images.size(0)
        total_loss_meter.update(total_loss.item(), batch_size)
        full_landmark_loss_meter.update(
            loss_dict["full_landmark_loss"].item(), batch_size
        )
        visible_landmark_loss_meter.update(
            loss_dict["visible_landmark_loss"].item(), batch_size
        )
        visibility_loss_meter.update(loss_dict["visibility_loss"].item(), batch_size)
        pca_projection_loss_meter.update(loss_dict["pca_loss"].item(), batch_size)

        if "landmarks" in batch_on_device:
            pred_landmarks = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"].detach(),
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=use_subpixel_decode,
            )
            nme_values = compute_box_normalized_nme(
                preds=pred_landmarks, targets=batch_on_device["landmarks"]
            )
            nme_meter.update(float(nme_values.mean()), batch_size)

        batch_time_meter.update(time.time() - batch_start_time)
        progress_bar.set_postfix(
            total=f"{total_loss_meter.avg:.4f}",
            full=f"{full_landmark_loss_meter.avg:.4f}",
            visible=f"{visible_landmark_loss_meter.avg:.4f}",
            vis=f"{visibility_loss_meter.avg:.4f}",
            pca=f"{pca_projection_loss_meter.avg:.4f}",
            nme=f"{nme_meter.avg:.4f}",
        )

    progress_bar.close()

    return {
        "total_loss": total_loss_meter.avg,
        "full_landmark_loss": full_landmark_loss_meter.avg,
        "visible_landmark_loss": visible_landmark_loss_meter.avg,
        "visibility_loss": visibility_loss_meter.avg,
        "pca_loss": pca_projection_loss_meter.avg,
        "nme": nme_meter.avg,
        "batch_time": batch_time_meter.avg,
        "epoch_time": time.time() - epoch_start_time,
    }


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metrics: dict[str, Any],
) -> None:
    """Save a model checkpoint with optimizer state and tracked metrics."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    patience_counter: int,
    patience: int,
) -> None:
    """Print a compact terminal summary for the current epoch."""
    print("=" * 120)
    print(f"Epoch {epoch + 1:03d}/{num_epochs:03d}")
    print("-" * 120)
    print(
        f"Train | total: {train_metrics['total_loss']:.6f} | "
        f"full: {train_metrics['full_landmark_loss']:.6f} | "
        f"visible: {train_metrics['visible_landmark_loss']:.6f} | "
        f"vis: {train_metrics['visibility_loss']:.6f} | "
        f"pca: {train_metrics['pca_loss']:.6f} | "
        f"NME: {train_metrics['nme']:.6f}"
    )
    print(
        f"Val   | total: {val_metrics['total_loss']:.6f} | "
        f"full: {val_metrics['full_landmark_loss']:.6f} | "
        f"visible: {val_metrics['visible_landmark_loss']:.6f} | "
        f"vis: {val_metrics['visibility_loss']:.6f} | "
        f"pca: {val_metrics['pca_loss']:.6f} | "
        f"NME: {val_metrics['nme']:.6f}"
    )
    print(
        f"Time  | train: {train_metrics['epoch_time']:.2f}s | "
        f"val: {val_metrics['epoch_time']:.2f}s | "
        f"epoch: {train_metrics['epoch_time'] + val_metrics['epoch_time']:.2f}s"
    )
    print(
        f"Best val total loss: {best_val_loss:.6f} | Early stopping: {patience_counter}/{patience}"
    )
    print("=" * 120)


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
                "nme",
                "lr",
                "epoch_time",
            ]
        )


def append_results_row(
    csv_path: str | Path, epoch: int, split: str, metrics: dict[str, float], lr: float
) -> None:
    """Append one train or validation metrics row to the experiment CSV."""
    csv_path = Path(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                split,
                metrics["total_loss"],
                metrics["full_landmark_loss"],
                metrics["visible_landmark_loss"],
                metrics["visibility_loss"],
                metrics["pca_loss"],
                metrics["nme"],
                lr,
                metrics["epoch_time"],
            ]
        )


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
    pca_prior_path: str | Path | None = None,
    patience: int = 15,
    project_name: str | None = None,
    run_name: str | None = None,
    use_wandb: bool = True,
    use_amp: bool = True,
    visualize_every_n_epochs: int = 5,
    num_visualization_images: int = 4,
) -> dict[str, Any]:
    """Execute the full training pipeline, including validation and checkpointing."""
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

    if use_wandb and wandb is not None:
        wandb.init(project=project_name, name=run_name, reinit=True)

    pca_shape_prior = None
    if pca_prior_path is not None:
        pca_prior_path = Path(pca_prior_path)
        if not pca_prior_path.exists():
            raise FileNotFoundError(f"PCA prior file not found: {pca_prior_path}")
        pca_shape_prior = load_pca_shape_prior(pca_prior_path, device=device)
    if lambda_pca_projection > 0.0 and pca_shape_prior is None:
        raise ValueError("lambda_pca_projection > 0 requires a valid pca_prior_path.")

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        # One loop iteration corresponds to one complete train/validation cycle.
        current_lr = optimizer.param_groups[0]["lr"]
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
            pca_shape_prior=pca_shape_prior,
            training=True,
            use_subpixel_decode=False,
            use_amp=amp_enabled,
            progress_desc=f"Train {epoch + 1:03d}",
        )
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
            pca_shape_prior=pca_shape_prior,
            training=False,
            use_subpixel_decode=False,
            use_amp=amp_enabled,
            progress_desc=f"Val   {epoch + 1:03d}",
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        append_results_row(results_csv_path, epoch, "train", train_metrics, current_lr)
        append_results_row(results_csv_path, epoch, "val", val_metrics, current_lr)

        metrics_payload = {"train": train_metrics, "val": val_metrics, "lr": current_lr}
        save_checkpoint(
            output_dir / "last_model.pth", epoch, model, optimizer, metrics_payload
        )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                output_dir / "best_model.pth", epoch, model, optimizer, metrics_payload
            )
        else:
            patience_counter += 1

        print_epoch_summary(
            epoch,
            num_epochs,
            train_metrics,
            val_metrics,
            best_val_loss,
            patience_counter,
            patience,
        )

        if use_wandb and wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train/total_loss": train_metrics["total_loss"],
                    "train/full_landmark_loss": train_metrics["full_landmark_loss"],
                    "train/visible_landmark_loss": train_metrics[
                        "visible_landmark_loss"
                    ],
                    "train/visibility_loss": train_metrics["visibility_loss"],
                    "train/pca_loss": train_metrics["pca_loss"],
                    "train/nme": train_metrics["nme"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/full_landmark_loss": val_metrics["full_landmark_loss"],
                    "val/visible_landmark_loss": val_metrics["visible_landmark_loss"],
                    "val/visibility_loss": val_metrics["visibility_loss"],
                    "val/pca_loss": val_metrics["pca_loss"],
                    "val/nme": val_metrics["nme"],
                    "best/val_total_loss": best_val_loss,
                }
            )

        if visualize_every_n_epochs > 0 and (epoch + 1) % visualize_every_n_epochs == 0:
            visualize_predicted_heatmaps_on_train_batch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
                output_dir=output_dir / "train_visualizations",
                num_images=num_visualization_images,
                grid_cols=min(num_visualization_images, 4),
                use_wandb=use_wandb and wandb is not None,
            )
            print(f"Saved predicted heatmap visualizations for epoch {epoch + 1}.")

        scheduler.step()
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    if use_wandb and wandb is not None:
        wandb.finish()

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "results_csv": str(results_csv_path),
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
) -> None:
    """Run a single optimization step to validate the end-to-end training path."""
    model.train()
    batch = next(iter(dataloader))
    images = batch["image"].to(device, non_blocking=True)
    heatmaps = batch["heatmaps"].to(device, non_blocking=True)
    visibility = batch["visibility"].to(device, non_blocking=True)
    landmarks = batch["landmarks"].to(device, non_blocking=True)
    class_idx = batch["class_idx"].to(device, non_blocking=True)
    pca_shape_prior = None
    if pca_prior_path is not None:
        pca_shape_prior = load_pca_shape_prior(pca_prior_path, device=device)
    if lambda_pca_projection > 0.0 and pca_shape_prior is None:
        raise ValueError("lambda_pca_projection > 0 requires a valid pca_prior_path.")
    outputs = model(images)
    loss_dict = compute_multitask_loss(
        outputs=outputs,
        batch={
            "heatmaps": heatmaps,
            "visibility": visibility,
            "class_idx": class_idx,
        },
        heatmap_loss_fn=heatmap_loss_fn,
        visibility_loss_fn=visibility_loss_fn,
        lambda_vis=lambda_vis,
        lambda_lmk_vis=lambda_lmk_vis,
        lambda_lmk_full=lambda_lmk_full,
        lambda_pca_projection=lambda_pca_projection,
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
        use_subpixel=False,
    )
    nme_values = compute_box_normalized_nme(preds=pred_landmarks, targets=landmarks)
    print("Smoke test passed.")
    print(f"Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"Full landmark loss: {loss_dict['full_landmark_loss'].item():.6f}")
    print(f"Visible landmark loss: {loss_dict['visible_landmark_loss'].item():.6f}")
    print(f"Visibility loss: {loss_dict['visibility_loss'].item():.6f}")
    print(f"PCA loss: {loss_dict['pca_loss'].item():.6f}")
    print(f"NME: {float(nme_values.mean()):.6f}")
