from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metrics import decode_heatmaps_to_image_coords


@dataclass
class TTAConsistencyResult:
    """Prediction variance measured across test-time augmentations."""

    variance: torch.Tensor
    predictions: torch.Tensor


def compute_tta_consistency(
    model: torch.nn.Module,
    images: torch.Tensor,
    num_samples: int,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
    brightness: float = 0.08,
    contrast: float = 0.08,
    blur_probability: float = 0.5,
    max_translation: float = 0.03,
    scale_min: float = 0.97,
    scale_max: float = 1.03,
    max_rotation_deg: float = 10.0,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> TTAConsistencyResult | None:
    """Compute per-landmark prediction variance under mild augmentations.

    Geometric predictions are mapped back to the original network-input
    coordinate system before variance is measured. Horizontal flips are not used
    here because this diagnostic only applies transforms that preserve landmark
    identity without requiring an index remapping table.
    """
    if num_samples <= 0:
        return None
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (B, C, H, W), got {images.shape}.")

    model_was_training = model.training
    model.eval()
    image_height, image_width = images.shape[2], images.shape[3]
    mapped_predictions: list[torch.Tensor] = []

    for _ in range(num_samples):
        augmented_images, theta = _augment_batch(
            images=images,
            brightness=brightness,
            contrast=contrast,
            blur_probability=blur_probability,
            max_translation=max_translation,
            scale_min=scale_min,
            scale_max=scale_max,
            max_rotation_deg=max_rotation_deg,
            mean=mean,
            std=std,
        )
        outputs = model(augmented_images)
        predictions = decode_heatmaps_to_image_coords(
            heatmaps=outputs["heatmaps"],
            image_height=image_height,
            image_width=image_width,
            use_subpixel=True,
            decoder=coordinate_decoder,
            softmax_temperature=wasserstein_softmax_temperature,
        )
        mapped_predictions.append(
            _map_predictions_to_original_input(
                predictions=predictions,
                theta=theta,
                image_height=image_height,
                image_width=image_width,
            )
        )

    if model_was_training:
        model.train()

    stacked = torch.stack(mapped_predictions, dim=0)
    coordinate_variance = stacked.var(dim=0, unbiased=False).sum(dim=-1)
    return TTAConsistencyResult(variance=coordinate_variance, predictions=stacked)


def _augment_batch(
    images: torch.Tensor,
    brightness: float,
    contrast: float,
    blur_probability: float,
    max_translation: float,
    scale_min: float,
    scale_max: float,
    max_rotation_deg: float,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = images.shape[0]
    theta = _sample_affine_theta(
        batch_size=batch_size,
        device=images.device,
        dtype=images.dtype,
        max_translation=max_translation,
        scale_min=scale_min,
        scale_max=scale_max,
        max_rotation_deg=max_rotation_deg,
    )
    grid = F.affine_grid(theta, size=images.shape, align_corners=False)
    augmented = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    augmented = _apply_photometric_jitter(
        images=augmented,
        brightness=brightness,
        contrast=contrast,
        mean=mean,
        std=std,
    )
    if blur_probability > 0.0:
        mask = torch.rand(batch_size, device=images.device) < float(blur_probability)
        if bool(mask.any().item()):
            augmented[mask] = _gaussian_blur(augmented[mask])
    return augmented, theta


def _sample_affine_theta(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_translation: float,
    scale_min: float,
    scale_max: float,
    max_rotation_deg: float,
) -> torch.Tensor:
    angles = (
        torch.rand(batch_size, device=device, dtype=dtype) * 2.0 - 1.0
    ) * math.radians(float(max_rotation_deg))
    scales = torch.empty(batch_size, device=device, dtype=dtype).uniform_(
        float(scale_min), float(scale_max)
    )
    translations = (
        torch.rand(batch_size, 2, device=device, dtype=dtype) * 2.0 - 1.0
    ) * float(max_translation)
    cos_values = torch.cos(angles) * scales
    sin_values = torch.sin(angles) * scales
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_values
    theta[:, 0, 1] = -sin_values
    theta[:, 1, 0] = sin_values
    theta[:, 1, 1] = cos_values
    theta[:, :, 2] = translations
    return theta


def _apply_photometric_jitter(
    images: torch.Tensor,
    brightness: float,
    contrast: float,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    denormalized = (images * std_tensor + mean_tensor).clamp(0.0, 1.0)
    batch_size = images.shape[0]
    if brightness > 0.0:
        factors = 1.0 + (
            torch.rand(batch_size, 1, 1, 1, device=images.device, dtype=images.dtype)
            * 2.0
            - 1.0
        ) * float(brightness)
        denormalized = denormalized * factors
    if contrast > 0.0:
        factors = 1.0 + (
            torch.rand(batch_size, 1, 1, 1, device=images.device, dtype=images.dtype)
            * 2.0
            - 1.0
        ) * float(contrast)
        image_mean = denormalized.mean(dim=(2, 3), keepdim=True)
        denormalized = (denormalized - image_mean) * factors + image_mean
    denormalized = denormalized.clamp(0.0, 1.0)
    return (denormalized - mean_tensor) / std_tensor


def _gaussian_blur(images: torch.Tensor) -> torch.Tensor:
    kernel_1d = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 1.0],
        device=images.device,
        dtype=images.dtype,
    )
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 5, 5)
    kernel = kernel_2d.repeat(images.shape[1], 1, 1, 1)
    return F.conv2d(images, kernel, padding=2, groups=images.shape[1])


def _map_predictions_to_original_input(
    predictions: torch.Tensor,
    theta: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    x_norm = (predictions[..., 0] + 0.5) * 2.0 / float(image_width) - 1.0
    y_norm = (predictions[..., 1] + 0.5) * 2.0 / float(image_height) - 1.0
    homogeneous = torch.stack(
        [x_norm, y_norm, torch.ones_like(x_norm)],
        dim=-1,
    )
    mapped = torch.einsum("bkj,bij->bki", homogeneous, theta)
    x_pixels = (mapped[..., 0] + 1.0) * float(image_width) * 0.5 - 0.5
    y_pixels = (mapped[..., 1] + 1.0) * float(image_height) * 0.5 - 0.5
    return torch.stack([x_pixels, y_pixels], dim=-1)
