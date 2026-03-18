from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..utils import seed_worker
from .dataset import SyntheticLandmarkDataset
from .transforms import Compose, Normalize, Resize, ToTensor


def build_transforms(config: ExperimentConfig) -> tuple[Compose, Compose]:
    common = [
        Resize(size=config.image_size),
        ToTensor(),
        Normalize(mean=config.normalization_mean, std=config.normalization_std),
    ]
    return Compose(common), Compose(common.copy())


def build_datasets(config: ExperimentConfig) -> dict[str, SyntheticLandmarkDataset]:
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
