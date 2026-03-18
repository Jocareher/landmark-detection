from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

SampleDict = dict[str, Any]


@dataclass
class Compose:
    """Apply a list of sample-level transforms in sequence."""

    transforms: list[Callable[[SampleDict], SampleDict]]

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Transform one sample and return the updated mapping."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample


@dataclass
class Resize:
    """Resize the image and rescale landmark coordinates to the target size."""

    size: tuple[int, int]
    interpolation: int = Image.BILINEAR

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Resize one sample while keeping metadata and landmarks aligned."""
        image = sample["image"]
        landmarks = sample["landmarks"]
        metadata = dict(sample.get("metadata", {}))

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")

        original_width, original_height = image.size
        target_height, target_width = self.size
        scale_x = target_width / float(original_width)
        scale_y = target_height / float(original_height)

        resized_image = image.resize((target_width, target_height), self.interpolation)
        resized_landmarks = np.asarray(landmarks, dtype=np.float32).copy()
        resized_landmarks[:, 0] *= scale_x
        resized_landmarks[:, 1] *= scale_y

        metadata["original_size"] = (original_height, original_width)
        metadata["transformed_size"] = (target_height, target_width)
        metadata["resize_scale"] = (scale_x, scale_y)

        sample["image"] = resized_image
        sample["landmarks"] = resized_landmarks
        sample["metadata"] = metadata
        return sample


@dataclass
class ToTensor:
    """Convert image, landmarks, and visibility arrays to tensors."""

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Convert one sample in-place to tensor-based representations."""
        image = sample["image"]
        landmarks = sample["landmarks"]
        visibility = sample["visibility"]

        if isinstance(image, Image.Image):
            image = np.asarray(image).copy()

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected image as PIL image or numpy array, got {type(image)}."
            )

        sample["image"] = (
            torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        )
        sample["landmarks"] = torch.as_tensor(landmarks, dtype=torch.float32)
        sample["visibility"] = torch.as_tensor(visibility, dtype=torch.float32)
        return sample


@dataclass
class Normalize:
    """Normalize an image tensor using channel-wise mean and standard deviation."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Precompute broadcastable normalization tensors."""
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Normalize the `image` field of a sample."""
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image as torch.Tensor, got {type(image)}.")

        sample["image"] = (image - self.mean_tensor) / self.std_tensor
        return sample
