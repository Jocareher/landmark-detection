from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def denormalize_image_tensor(
    image: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Undo channel-wise normalization on one image tensor."""
    if image.ndim != 3:
        raise ValueError(
            f"Expected image with shape (C, H, W), got {tuple(image.shape)}."
        )
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(
        -1, 1, 1
    )
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(
        -1, 1, 1
    )
    return image.clone() * std_tensor + mean_tensor


def _resize_heatmap_to_image(
    heatmap: torch.Tensor, image_height: int, image_width: int
) -> np.ndarray:
    """Resize a heatmap tensor to match the spatial size of the source image."""
    resized_heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized_heatmap.squeeze(0).squeeze(0).cpu().numpy()


def visualize_predicted_heatmaps_on_train_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str | Path,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    num_images: int = 4,
    grid_cols: int = 4,
    overlay_alpha: float = 0.45,
    use_max_projection: bool = True,
    normalize_heatmap: bool = True,
    use_wandb: bool = False,
) -> Path:
    """Overlay predicted heatmaps on a fixed set of samples and save the figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()
    dataset = dataloader.dataset
    fixed_sample_indices = [
        index for index in range(min(len(dataset), max(num_images, 8)))
    ]
    if not fixed_sample_indices:
        raise RuntimeError("The dataset is empty. Cannot visualize fixed samples.")

    fixed_sample_indices = fixed_sample_indices[:num_images]
    selected_samples = [dataset[index] for index in fixed_sample_indices]
    images = torch.stack([sample["image"] for sample in selected_samples], dim=0).to(
        device, non_blocking=True
    )
    metadata_list = [sample.get("metadata", {}) for sample in selected_samples]

    with torch.no_grad():
        outputs = model(images)

    predicted_heatmaps = outputs["heatmaps"].detach().cpu()
    images_cpu = images.detach().cpu()
    num_images = min(num_images, images_cpu.shape[0])
    num_rows = math.ceil(num_images / grid_cols)
    fig, axes = plt.subplots(num_rows, grid_cols, figsize=(grid_cols * 4, num_rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for image_index in range(num_images):
        # Aggregate all landmark heatmaps into a single overlay for quick inspection.
        image_tensor = denormalize_image_tensor(
            images_cpu[image_index], mean=mean, std=std
        )
        image_np = image_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        image_height, image_width = image_np.shape[:2]
        current_heatmaps = predicted_heatmaps[image_index]
        aggregated_heatmap = (
            current_heatmaps.max(dim=0).values
            if use_max_projection
            else current_heatmaps.sum(dim=0)
        )
        heatmap_np = _resize_heatmap_to_image(
            aggregated_heatmap, image_height, image_width
        )

        if normalize_heatmap:
            heatmap_min = float(heatmap_np.min())
            heatmap_max = float(heatmap_np.max())
            if heatmap_max > heatmap_min:
                heatmap_np = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min)

        axes[image_index].imshow(image_np)
        axes[image_index].imshow(heatmap_np, cmap="jet", alpha=overlay_alpha)
        axes[image_index].axis("off")
        sample_title = metadata_list[image_index].get(
            "sample_id", f"sample_{fixed_sample_indices[image_index]}"
        )
        axes[image_index].set_title(sample_title, fontsize=10)

    for axis_index in range(num_images, len(axes)):
        axes[axis_index].axis("off")

    plt.tight_layout()
    save_path = output_dir / f"train_heatmaps_epoch_{epoch + 1:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if use_wandb:
        import wandb

        wandb.log(
            {
                "train/predicted_heatmaps_overlay": wandb.Image(
                    str(save_path), caption=f"Epoch {epoch + 1}"
                )
            }
        )

    if model_was_training:
        model.train()
    return save_path
