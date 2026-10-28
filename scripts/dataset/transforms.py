from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

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
class CaptureWhiteBackgroundReference:
    """Keep the resized synthetic RGB image for a late background-only transform."""

    def __call__(self, sample: SampleDict) -> SampleDict:
        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")
        sample["_white_background_reference"] = image.convert("RGB").copy()
        return sample


@dataclass
class RandomWhiteBackground:
    """Randomize near-white synthetic background without changing the face region.

    The available labels are landmarks rather than a foreground segmentation mask.
    A convex hull of visible landmarks and the non-white foreground are protected.
    Only neutral near-white pixels outside this protection are eligible. This is
    a conservative approximation, not pixel-accurate foreground segmentation.
    """

    probability: float = 0.7
    white_threshold: int = 225
    max_chroma: int = 20
    face_margin_px: int = 2
    texture_std: float = 0.06
    fine_noise_std: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Background augmentation probability must be in [0, 1].")
        if not 0 <= self.white_threshold < 255:
            raise ValueError("Background white threshold must be in [0, 254].")
        if not 0 <= self.max_chroma <= 255:
            raise ValueError("Background max chroma must be in [0, 255].")
        if not 0 <= self.face_margin_px <= 32:
            raise ValueError("Background face margin must be an integer in [0, 32].")
        if self.texture_std < 0 or self.fine_noise_std < 0:
            raise ValueError("Background noise standard deviations must be nonnegative.")

    def __call__(self, sample: SampleDict) -> SampleDict:
        """Replace eligible background with a random smooth color/texture field."""
        reference = sample.pop("_white_background_reference", None)
        metadata = dict(sample.get("metadata", {}))
        metadata["background_augmentation"] = {
            "applied": False,
            "landmark_box_xyxy": (0, 0, 0, 0),
            "eligible_fraction": 0.0,
        }
        sample["metadata"] = metadata
        if random.random() >= self.probability:
            return sample
        image = sample["image"]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"Unsupported image type: {type(image)}.")
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask_source = np.asarray(reference, dtype=np.uint8) if reference is not None else original
        height, width = original.shape[:2]
        if mask_source.shape != original.shape:
            raise ValueError("Background reference and augmented image sizes differ.")
        points = np.asarray(sample["landmarks"], dtype=np.float32)
        visibility = np.asarray(sample["visibility"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or visibility.shape != (len(points),):
            raise ValueError("Background augmentation requires Nx2 landmarks and N visibility flags.")
        usable = (
            (visibility > 0)
            & np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        if usable.sum() < 5:
            return sample  # fail closed when the face location is unreliable

        x_min, y_min = points[usable].min(axis=0)
        x_max, y_max = points[usable].max(axis=0)
        landmark_box = (
            max(0, int(math.floor(x_min))), max(0, int(math.floor(y_min))),
            min(width, int(math.ceil(x_max)) + 1),
            min(height, int(math.ceil(y_max)) + 1),
        )

        # A pixel must be bright and close to neutral RGB. The silhouette and
        # landmark hull protect face/skin highlights without leaving a white
        # rectangular box in the randomized background.
        image_f = original.astype(np.float32) / 255.0
        low = mask_source.min(axis=2).astype(np.float32)
        chroma = (mask_source.max(axis=2).astype(np.int16)
                  - mask_source.min(axis=2).astype(np.int16))
        whiteness = np.clip((low - self.white_threshold) / (255.0 - self.white_threshold), 0.0, 1.0)
        whiteness[chroma > self.max_chroma] = 0.0
        foreground = ((whiteness == 0) | (chroma > self.max_chroma)).astype(np.uint8) * 255
        hull_points = self._convex_hull(points[usable])
        if len(hull_points) < 3:
            return sample
        hull = Image.new('L', (width, height), 0)
        ImageDraw.Draw(hull).polygon(hull_points, fill=255)
        margin = self.face_margin_px
        foreground_image = Image.fromarray(foreground)
        if margin:
            filter_size = 2 * margin + 1
            foreground_image = foreground_image.filter(ImageFilter.MaxFilter(filter_size))
            hull = hull.filter(ImageFilter.MaxFilter(filter_size))
        protection = np.maximum(np.asarray(foreground_image, dtype=np.uint8),
                                np.asarray(hull, dtype=np.uint8))
        if margin:
            soft = np.asarray(Image.fromarray(protection).filter(
                ImageFilter.GaussianBlur(radius=max(1, margin / 2))), dtype=np.uint8)
            protection = np.maximum(protection, soft)
        alpha = (whiteness * (1.0 - protection.astype(np.float32) / 255.0)).astype(np.float32)
        if not np.any(alpha > 0):
            return sample

        # A varied color plus smooth texture removes the constant-white cue.
        # Fine noise alone would leave the background nearly white and create
        # an artificial high-frequency texture.
        luminance = random.uniform(0.30, 0.94)
        base_color = np.clip(
            luminance + np.random.uniform(-0.10, 0.10, size=3), 0.15, 1.0
        ).astype(np.float32)
        coarse_luminance = np.random.normal(
            0.0, self.texture_std, size=(6, 6, 1)).astype(np.float32)
        coarse_color = np.random.normal(
            0.0, self.texture_std * 0.2, size=(6, 6, 3)).astype(np.float32)
        coarse = coarse_luminance + coarse_color
        smooth = np.stack([
            np.asarray(Image.fromarray(coarse[:, :, channel]).resize(
                (width, height), Image.BILINEAR), dtype=np.float32)
            for channel in range(3)
        ], axis=2)
        fine = np.random.normal(0.0, self.fine_noise_std,
                                size=(height, width, 1)).astype(np.float32)
        background = np.clip(base_color + smooth + fine, 0.0, 1.0)
        result = np.clip((1.0 - alpha[..., None]) * image_f
                         + alpha[..., None] * background, 0.0, 1.0)
        output = np.rint(result * 255.0).astype(np.uint8)
        # Exact preservation for all non-white foreground and the landmark hull.
        output[(protection == 255) | (whiteness == 0)] = original[
            (protection == 255) | (whiteness == 0)]
        sample["image"] = Image.fromarray(output)
        metadata["background_augmentation"] = {
            "applied": True,
            "landmark_box_xyxy": landmark_box,
            "eligible_fraction": float((alpha > 0).mean()),
        }
        sample["metadata"] = metadata
        return sample

    @staticmethod
    def _convex_hull(points: np.ndarray) -> list[tuple[int, int]]:
        """Return a simple polygon enclosing the visible landmark points."""
        coords = sorted({(int(round(x)), int(round(y))) for x, y in points})
        if len(coords) < 3:
            return coords

        def cross(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        lower: list[tuple[int, int]] = []
        for point in coords:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[int, int]] = []
        for point in reversed(coords):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]


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
        metadata = dict(sample.get("metadata", {}))
        metadata["geometric_augmentation"] = {
            "applied": False,
            "rotation_deg": 0.0,
            "scale": 1.0,
            "translation_xy": (0.0, 0.0),
        }

        if random.random() > self.probability:
            sample["metadata"] = metadata
            return sample

        image = sample["image"]
        landmarks = np.asarray(sample["landmarks"], dtype=np.float32).copy()
        visibility = np.asarray(sample["visibility"], dtype=np.float32).copy()

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
            "applied": True,
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
