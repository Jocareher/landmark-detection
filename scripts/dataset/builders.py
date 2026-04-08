from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..utils import seed_worker
from .natural_dataset import NaturalLandmarkEvaluationDataset
from .dataset import SyntheticLandmarkDataset
from .transforms import (
    Compose,
    Normalize,
    RandomAffineLandmarks,
    RandomColorJitter,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomJpegCompression,
    RandomRGBShift,
    Resize,
    ToTensor,
)


def build_transforms(config: ExperimentConfig) -> tuple[Compose, Compose]:
    """Build the train and evaluation transform pipelines from the config."""
    train_transforms = []

    if config.enable_geometric_augmentations:
        train_transforms.append(
            RandomAffineLandmarks(
                probability=config.geometric_probability,
                max_translation=config.geometric_max_translation,
                scale_min=config.geometric_scale_min,
                scale_max=config.geometric_scale_max,
                max_rotation_deg=config.geometric_max_rotation_deg,
            )
        )

    train_transforms.append(Resize(size=config.image_size))

    if config.enable_photometric_augmentations:
        train_transforms.extend(
            [
                RandomColorJitter(
                    brightness=config.color_jitter_brightness,
                    contrast=config.color_jitter_contrast,
                    saturation=config.color_jitter_saturation,
                    probability=config.color_jitter_probability,
                ),
                RandomGaussianBlur(
                    probability=config.blur_probability,
                    radius_min=config.blur_radius_min,
                    radius_max=config.blur_radius_max,
                ),
                RandomGaussianNoise(
                    probability=config.noise_probability,
                    std=config.noise_std,
                ),
                RandomJpegCompression(
                    probability=config.jpeg_probability,
                    quality_min=config.jpeg_quality_min,
                    quality_max=config.jpeg_quality_max,
                ),
                RandomRGBShift(
                    probability=config.rgb_shift_probability,
                    shift_limit=config.rgb_shift_limit,
                ),
            ]
        )

    train_transforms.extend(
        [
            ToTensor(),
            Normalize(mean=config.normalization_mean, std=config.normalization_std),
        ]
    )

    eval_transforms = [
        Resize(size=config.image_size),
        ToTensor(),
        Normalize(mean=config.normalization_mean, std=config.normalization_std),
    ]
    return Compose(train_transforms), Compose(eval_transforms)


def build_datasets(config: ExperimentConfig) -> dict[str, SyntheticLandmarkDataset]:
    """Instantiate dataset objects for the train, validation, and test splits."""
    train_transform, eval_transform = build_transforms(config)
    cache_dir = config.cache_dir
    return {
        "train": SyntheticLandmarkDataset(
            root_dir=config.dataset_root,
            split="train",
            transform=train_transform,
            target_mode=config.target_mode,
            num_landmarks=config.num_landmarks,
            heatmap_size=config.heatmap_size,
            sigma=config.heatmap_sigma,
            validate_labels=config.validate_labels,
            cache_file=cache_dir / "train_cache.pth" if cache_dir else None,
            use_cache=config.use_cache,
            show_progress=config.show_dataset_progress,
        ),
        "val": SyntheticLandmarkDataset(
            root_dir=config.dataset_root,
            split="val",
            transform=eval_transform,
            target_mode=config.target_mode,
            num_landmarks=config.num_landmarks,
            heatmap_size=config.heatmap_size,
            sigma=config.heatmap_sigma,
            validate_labels=config.validate_labels,
            cache_file=cache_dir / "val_cache.pth" if cache_dir else None,
            use_cache=config.use_cache,
            show_progress=config.show_dataset_progress,
        ),
        "test": SyntheticLandmarkDataset(
            root_dir=config.dataset_root,
            split="test",
            transform=eval_transform,
            target_mode=config.target_mode,
            num_landmarks=config.num_landmarks,
            heatmap_size=config.heatmap_size,
            sigma=config.heatmap_sigma,
            validate_labels=config.validate_labels,
            cache_file=cache_dir / "test_cache.pth" if cache_dir else None,
            use_cache=config.use_cache,
            show_progress=config.show_dataset_progress,
        ),
    }


def build_dataloaders(config: ExperimentConfig) -> dict[str, DataLoader]:
    """Create reproducible dataloaders for all configured dataset splits."""
    datasets = build_datasets(config)
    pin_memory = config.pin_memory and config.device in {"cuda", "auto"}
    generator = None
    if config.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.eval_batch_size or config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.eval_batch_size or config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
    }


def build_natural_evaluation_dataloader(
    export_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    config: ExperimentConfig,
    source_root: str | os.PathLike[str] | None = None,
) -> DataLoader:
    """Create a dataloader for natural-image evaluation on detector exports."""
    dataset = NaturalLandmarkEvaluationDataset(
        export_root=export_root,
        gt_root=gt_root,
        source_root=source_root,
        config=config,
        show_progress=config.show_dataset_progress,
    )
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and config.device in {"cuda", "auto"},
        worker_init_fn=seed_worker,
    )
