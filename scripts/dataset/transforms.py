from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

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
class RandomColorJitter:
    """Apply random brightness, contrast, and saturation changes to one image."""

    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    probability: float = 1.0

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Apply photometric jitter in-place with independent random factors."""
        if random.random() > self.probability:
            return sample

        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")

        if self.brightness > 0.0:
            factor = random.uniform(
                max(0.0, 1.0 - self.brightness), 1.0 + self.brightness
            )
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0.0:
            factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if self.saturation > 0.0:
            factor = random.uniform(
                max(0.0, 1.0 - self.saturation), 1.0 + self.saturation
            )
            image = ImageEnhance.Color(image).enhance(factor)

        sample["image"] = image
        return sample


@dataclass
class RandomGaussianBlur:
    """Apply a mild Gaussian blur with a random radius."""

    probability: float = 0.0
    radius_min: float = 0.1
    radius_max: float = 1.0

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Blur the image when the transform is triggered."""
        if random.random() > self.probability:
            return sample

        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")

        radius = random.uniform(self.radius_min, self.radius_max)
        sample["image"] = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return sample


@dataclass
class RandomGaussianNoise:
    """Add mild Gaussian noise to an RGB image."""

    probability: float = 0.0
    std: float = 0.0

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Inject noise into the image using the configured standard deviation."""
        if self.std <= 0.0 or random.random() > self.probability:
            return sample

        image = sample["image"]
        if isinstance(image, Image.Image):
            image_np = np.asarray(image, dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            image_np = image.astype(np.float32) / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}.")

        noise = np.random.normal(loc=0.0, scale=self.std, size=image_np.shape).astype(
            np.float32
        )
        noisy_image = np.clip(image_np + noise, 0.0, 1.0)
        sample["image"] = Image.fromarray((noisy_image * 255.0).astype(np.uint8))
        return sample


@dataclass
class RandomJpegCompression:
    """Simulate JPEG compression artifacts through encode/decode cycles."""

    probability: float = 0.0
    quality_min: int = 70
    quality_max: int = 95

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Apply JPEG compression with a random quality level."""
        if random.random() > self.probability:
            return sample

        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")

        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        sample["image"] = Image.open(buffer).convert("RGB")
        return sample


@dataclass
class RandomRGBShift:
    """Apply an additive per-channel color perturbation to an RGB image."""

    probability: float = 0.0
    shift_limit: float = 0.0

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Shift each color channel independently by a small random amount."""
        if self.shift_limit <= 0.0 or random.random() > self.probability:
            return sample

        image = sample["image"]
        if isinstance(image, Image.Image):
            image_np = np.asarray(image, dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            image_np = image.astype(np.float32) / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}.")

        channel_shift = np.random.uniform(
            low=-self.shift_limit,
            high=self.shift_limit,
            size=(1, 1, 3),
        ).astype(np.float32)
        shifted_image = np.clip(image_np + channel_shift, 0.0, 1.0)
        sample["image"] = Image.fromarray((shifted_image * 255.0).astype(np.uint8))
        return sample


@dataclass
class RandomAffineLandmarks:
    """Apply a mild affine transform and update landmarks consistently."""

    probability: float = 0.0
    max_translation: float = 0.05
    scale_min: float = 0.95
    scale_max: float = 1.05
    max_rotation_deg: float = 8.0

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Transform the image and visible landmarks using one shared affine matrix."""
        if random.random() > self.probability:
            return sample

        image = sample["image"]
        landmarks = np.asarray(sample["landmarks"], dtype=np.float32).copy()
        visibility = np.asarray(sample["visibility"], dtype=np.float32).copy()
        metadata = dict(sample.get("metadata", {}))

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")

        width, height = image.size
        rotation_deg = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        rotation_rad = math.radians(rotation_deg)
        scale = random.uniform(self.scale_min, self.scale_max)
        tx = random.uniform(-self.max_translation, self.max_translation) * float(width)
        ty = random.uniform(-self.max_translation, self.max_translation) * float(height)
        center_x = (width - 1.0) * 0.5
        center_y = (height - 1.0) * 0.5

        transform_matrix = self._build_forward_matrix(
            center_x=center_x,
            center_y=center_y,
            scale=scale,
            rotation_rad=rotation_rad,
            translation_x=tx,
            translation_y=ty,
        )
        inverse_matrix = np.linalg.inv(transform_matrix)

        transformed_image = image.transform(
            size=(width, height),
            method=Image.AFFINE,
            data=(
                float(inverse_matrix[0, 0]),
                float(inverse_matrix[0, 1]),
                float(inverse_matrix[0, 2]),
                float(inverse_matrix[1, 0]),
                float(inverse_matrix[1, 1]),
                float(inverse_matrix[1, 2]),
            ),
            resample=Image.BILINEAR,
        )

        transformed_landmarks = landmarks.copy()
        visible_indices = np.flatnonzero(visibility > 0.0)
        if visible_indices.size > 0:
            homogeneous_landmarks = np.concatenate(
                [
                    landmarks[visible_indices],
                    np.ones((visible_indices.size, 1), dtype=np.float32),
                ],
                axis=1,
            )
            transformed_points = homogeneous_landmarks @ transform_matrix.T
            transformed_points = transformed_points[:, :2] / np.maximum(
                transformed_points[:, 2:3], 1e-8
            )
            inside_mask = (
                (transformed_points[:, 0] >= 0.0)
                & (transformed_points[:, 0] < float(width))
                & (transformed_points[:, 1] >= 0.0)
                & (transformed_points[:, 1] < float(height))
            )

            transformed_landmarks[visible_indices] = transformed_points
            outside_indices = visible_indices[~inside_mask]
            if outside_indices.size > 0:
                visibility[outside_indices] = 0.0
                transformed_landmarks[outside_indices] = 0.0

        metadata["geometric_augmentation"] = {
            "rotation_deg": rotation_deg,
            "scale": scale,
            "translation_xy": (tx, ty),
        }
        sample["image"] = transformed_image
        sample["landmarks"] = transformed_landmarks
        sample["visibility"] = visibility
        sample["metadata"] = metadata
        return sample

    @staticmethod
    def _build_forward_matrix(
        center_x: float,
        center_y: float,
        scale: float,
        rotation_rad: float,
        translation_x: float,
        translation_y: float,
    ) -> np.ndarray:
        """Build the affine matrix that maps input pixels to output pixels."""
        cos_theta = math.cos(rotation_rad) * scale
        sin_theta = math.sin(rotation_rad) * scale

        translate_to_origin = np.array(
            [
                [1.0, 0.0, -center_x],
                [0.0, 1.0, -center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotate_and_scale = np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        translate_back = np.array(
            [
                [1.0, 0.0, center_x + translation_x],
                [0.0, 1.0, center_y + translation_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return translate_back @ rotate_and_scale @ translate_to_origin


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
