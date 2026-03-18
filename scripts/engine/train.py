from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from .losses import compute_multitask_loss
from .metrics import (
    AverageMeter,
    compute_box_normalized_nme,
    decode_heatmaps_to_image_coords,
)
from ..utils.visualization import visualize_predicted_heatmaps_on_train_batch


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    lambda_heatmap: float = 1.0,
    lambda_visibility: float = 1.0,
    training: bool = True,
    use_subpixel_decode: bool = False,
    use_amp: bool = True,
) -> dict[str, float]:
    if training and optimizer is None:
        raise ValueError("An optimizer is required when training=True.")

    model.train(training)
    total_loss_meter = AverageMeter()
    heatmap_loss_meter = AverageMeter()
    visibility_loss_meter = AverageMeter()
    nme_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    autocast_device = device.type
    epoch_start_time = time.time()

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
                outputs = model(images)
                loss_dict = compute_multitask_loss(
                    outputs=outputs,
                    batch=batch_on_device,
                    heatmap_loss_fn=heatmap_loss_fn,
                    visibility_loss_fn=visibility_loss_fn,
                    lambda_heatmap=lambda_heatmap,
                    lambda_visibility=lambda_visibility,
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
            with torch.no_grad():
                with torch.autocast(device_type=autocast_device, enabled=use_amp):
                    outputs = model(images)
                    loss_dict = compute_multitask_loss(
                        outputs=outputs,
                        batch=batch_on_device,
                        heatmap_loss_fn=heatmap_loss_fn,
                        visibility_loss_fn=visibility_loss_fn,
                        lambda_heatmap=lambda_heatmap,
                        lambda_visibility=lambda_visibility,
                    )
            total_loss = loss_dict["total_loss"]

        batch_size = images.size(0)
        total_loss_meter.update(total_loss.item(), batch_size)
        heatmap_loss_meter.update(loss_dict["heatmap_loss"].item(), batch_size)
        visibility_loss_meter.update(loss_dict["visibility_loss"].item(), batch_size)

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

    return {
        "total_loss": total_loss_meter.avg,
        "heatmap_loss": heatmap_loss_meter.avg,
        "visibility_loss": visibility_loss_meter.avg,
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
    print("=" * 120)
    print(f"Epoch {epoch + 1:03d}/{num_epochs:03d}")
    print("-" * 120)
    print(
        f"Train | total: {train_metrics['total_loss']:.6f} | heatmap: {train_metrics['heatmap_loss']:.6f} | "
        f"vis: {train_metrics['visibility_loss']:.6f} | NME: {train_metrics['nme']:.6f}"
    )
    print(
        f"Val   | total: {val_metrics['total_loss']:.6f} | heatmap: {val_metrics['heatmap_loss']:.6f} | "
        f"vis: {val_metrics['visibility_loss']:.6f} | NME: {val_metrics['nme']:.6f}"
    )
    print(
        f"Best val total loss: {best_val_loss:.6f} | Early stopping: {patience_counter}/{patience}"
    )
    print("=" * 120)


def initialize_results_csv(csv_path: str | Path) -> None:
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
                "heatmap_loss",
                "visibility_loss",
                "nme",
                "lr",
                "epoch_time",
            ]
        )


def append_results_row(
    csv_path: str | Path, epoch: int, split: str, metrics: dict[str, float], lr: float
) -> None:
    csv_path = Path(csv_path)
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                split,
                metrics["total_loss"],
                metrics["heatmap_loss"],
                metrics["visibility_loss"],
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
    lambda_heatmap: float = 1.0,
    lambda_visibility: float = 1.0,
    patience: int = 15,
    project_name: str | None = None,
    run_name: str | None = None,
    use_wandb: bool = True,
    use_amp: bool = True,
    visualize_every_n_epochs: int = 5,
    num_visualization_images: int = 4,
) -> dict[str, Any]:
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

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            heatmap_loss_fn=heatmap_loss_fn,
            visibility_loss_fn=visibility_loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            lambda_heatmap=lambda_heatmap,
            lambda_visibility=lambda_visibility,
            training=True,
            use_subpixel_decode=False,
            use_amp=amp_enabled,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            heatmap_loss_fn=heatmap_loss_fn,
            visibility_loss_fn=visibility_loss_fn,
            optimizer=None,
            scaler=None,
            lambda_heatmap=lambda_heatmap,
            lambda_visibility=lambda_visibility,
            training=False,
            use_subpixel_decode=False,
            use_amp=amp_enabled,
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
                    "train/heatmap_loss": train_metrics["heatmap_loss"],
                    "train/visibility_loss": train_metrics["visibility_loss"],
                    "train/nme": train_metrics["nme"],
                    "val/total_loss": val_metrics["total_loss"],
                    "val/heatmap_loss": val_metrics["heatmap_loss"],
                    "val/visibility_loss": val_metrics["visibility_loss"],
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
    lambda_heatmap: float = 1.0,
    lambda_visibility: float = 1.0,
) -> None:
    model.train()
    batch = next(iter(dataloader))
    images = batch["image"].to(device, non_blocking=True)
    heatmaps = batch["heatmaps"].to(device, non_blocking=True)
    visibility = batch["visibility"].to(device, non_blocking=True)
    landmarks = batch["landmarks"].to(device, non_blocking=True)
    outputs = model(images)
    loss_dict = compute_multitask_loss(
        outputs=outputs,
        batch={"heatmaps": heatmaps, "visibility": visibility},
        heatmap_loss_fn=heatmap_loss_fn,
        visibility_loss_fn=visibility_loss_fn,
        lambda_heatmap=lambda_heatmap,
        lambda_visibility=lambda_visibility,
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
    print(f"Heatmap loss: {loss_dict['heatmap_loss'].item():.6f}")
    print(f"Visibility loss: {loss_dict['visibility_loss'].item():.6f}")
    print(f"NME: {float(nme_values.mean()):.6f}")
