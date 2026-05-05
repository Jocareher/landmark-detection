from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..utils.synthetic_labels import parse_synthetic_landmark_label

SampleDict = dict[str, Any]
TargetMode = Literal["regression", "heatmap", "both"]


@lru_cache(maxsize=32)
def build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a cached 2D Gaussian kernel used to render landmark heatmaps."""
    x_axis = np.arange(0, size, dtype=np.float32)
    y_axis = x_axis[:, np.newaxis]
    center = size // 2
    kernel = np.exp(
        -((x_axis - center) ** 2 + (y_axis - center) ** 2) / (2.0 * sigma**2)
    )
    return kernel.astype(np.float32)


def draw_gaussian(heatmap: np.ndarray, point: np.ndarray, sigma: float) -> np.ndarray:
    """Draw a truncated Gaussian centered at `point` on the provided heatmap."""
    tmp_size = int(sigma * 3)
    x_coord, y_coord = float(point[0]), float(point[1])

    upper_left = [int(x_coord - tmp_size), int(y_coord - tmp_size)]
    bottom_right = [int(x_coord + tmp_size + 1), int(y_coord + tmp_size + 1)]

    if (
        upper_left[0] >= heatmap.shape[1]
        or upper_left[1] >= heatmap.shape[0]
        or bottom_right[0] < 0
        or bottom_right[1] < 0
    ):
        return heatmap

    size = 2 * tmp_size + 1
    gaussian = build_gaussian_kernel(size=size, sigma=sigma)

    gaussian_x = (
        max(0, -upper_left[0]),
        min(bottom_right[0], heatmap.shape[1]) - upper_left[0],
    )
    gaussian_y = (
        max(0, -upper_left[1]),
        min(bottom_right[1], heatmap.shape[0]) - upper_left[1],
    )
    image_x = max(0, upper_left[0]), min(bottom_right[0], heatmap.shape[1])
    image_y = max(0, upper_left[1]), min(bottom_right[1], heatmap.shape[0])

    heatmap[image_y[0] : image_y[1], image_x[0] : image_x[1]] = gaussian[
        gaussian_y[0] : gaussian_y[1],
        gaussian_x[0] : gaussian_x[1],
    ]
    return heatmap


class SyntheticLandmarkDataset(Dataset):
    """Dataset for synthetic face images with landmarks, visibility, and heatmaps."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transform: Callable[[SampleDict], SampleDict] | None = None,
        target_mode: TargetMode = "heatmap",
        num_landmarks: int = 72,
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 2.0,
        return_metadata: bool = True,
        validate_labels: bool = False,
        cache_file: str | Path | None = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> None:
        """Index one dataset split and configure how targets should be returned."""
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_mode = target_mode
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.return_metadata = return_metadata
        self.validate_labels = validate_labels
        self.use_cache = use_cache
        self.show_progress = show_progress

        if self.target_mode not in {"regression", "heatmap", "both"}:
            raise ValueError(f"Invalid target_mode '{self.target_mode}'.")

        self.images_dir = self.root_dir / self.split / "images"
        self.labels_dir = self.root_dir / self.split / "labels"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        if cache_file is None:
            self.cache_file = self.root_dir / f"{self.split}_samples_cache.pth"
        else:
            cache_path = Path(cache_file)
            if cache_path.suffix == "":
                cache_path = cache_path.with_suffix(".pth")
            self.cache_file = cache_path

        self.samples = self._load_or_build_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in split '{self.split}' under {self.root_dir}."
            )

    def __len__(self) -> int:
        """Return the number of indexed samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleDict:
        """Load one sample, apply transforms, and generate the configured targets."""
        sample_info = self.samples[index]
        image_path = Path(sample_info["image_path"])
        label_path = Path(sample_info["label_path"])

        image = self._load_image(image_path)
        original_width, original_height = image.size
        landmarks, visibility, class_idx, class_name = self._load_label(
            label_path, original_width, original_height
        )

        metadata = {
            "sample_id": image_path.stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "original_size": (original_height, original_width),
            "transformed_size": (original_height, original_width),
            "split": self.split,
            "num_landmarks": self.num_landmarks,
            "class_idx": -1 if class_idx is None else int(class_idx),
            "class_name": class_name,
            "geometric_augmentation": {
                "applied": False,
                "rotation_deg": 0.0,
                "scale": 1.0,
                "translation_xy": (0.0, 0.0),
            },
        }
        sample: SampleDict = {
            "image": image,
            "landmarks": landmarks,
            "visibility": visibility,
            "class_idx": -1 if class_idx is None else int(class_idx),
            "metadata": metadata,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        image_tensor = self._ensure_image_tensor(sample["image"])
        landmarks_tensor = self._ensure_landmark_tensor(sample["landmarks"])
        visibility_tensor = self._ensure_visibility_tensor(sample["visibility"])

        output: SampleDict = {
            "image": image_tensor,
            "visibility": visibility_tensor,
            "class_idx": torch.as_tensor(sample["class_idx"], dtype=torch.int64),
        }
        if self.target_mode in {"regression", "both"}:
            output["landmarks"] = landmarks_tensor
        if self.target_mode in {"heatmap", "both"}:
            output["heatmaps"] = self._generate_heatmaps(
                landmarks_tensor, sample["metadata"]["transformed_size"]
            )
        if self.return_metadata:
            output["metadata"] = sample.get("metadata", metadata)
        return output

    def _load_or_build_samples(self) -> list[dict[str, str]]:
        """Load the cached index when available or rebuild it from disk."""
        if self.use_cache and self.cache_file.exists():
            try:
                samples = self._load_cache()
                if samples:
                    return samples
            except Exception:
                pass

        samples = self._index_samples()
        if self.use_cache:
            self._save_cache(samples)
        return samples

    def _load_cache(self) -> list[dict[str, str]]:
        """Deserialize a previously saved sample index from disk."""
        payload = torch.load(self.cache_file, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid cache format in {self.cache_file}")
        if payload.get("split") != self.split:
            raise ValueError("Cache split mismatch.")
        if payload.get("num_landmarks") != self.num_landmarks:
            raise ValueError("Cache num_landmarks mismatch.")
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"Invalid 'samples' field in cache {self.cache_file}")
        return samples

    def _save_cache(self, samples: list[dict[str, str]]) -> None:
        """Persist the indexed samples using an atomic temporary file."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": self.split,
            "num_landmarks": self.num_landmarks,
            "samples": samples,
        }
        temp_cache_file = self.cache_file.with_suffix(self.cache_file.suffix + ".tmp")
        torch.save(payload, temp_cache_file)
        temp_cache_file.replace(self.cache_file)

    def _index_samples(self) -> list[dict[str, str]]:
        """Scan one split directory and keep valid image/label pairs only."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in valid_extensions
        )
        samples: list[dict[str, str]] = []
        iterator = (
            tqdm(
                image_paths,
                desc=f"Indexing {self.split}",
                unit="img",
                dynamic_ncols=True,
            )
            if self.show_progress
            else image_paths
        )

        for image_path in iterator:
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            if self.validate_labels and not self._is_minimally_valid_label(label_path):
                continue
            samples.append(
                {"image_path": str(image_path), "label_path": str(label_path)}
            )
        return samples

    def _is_minimally_valid_label(self, label_path: Path) -> bool:
        """Check that a label file has the expected `(num_landmarks, 3)` shape."""
        try:
            parse_synthetic_landmark_label(
                label_path=label_path,
                expected_num_landmarks=self.num_landmarks,
            )
        except Exception:
            return False
        return True

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        """Load an image from disk as an RGB PIL image."""
        return Image.open(image_path).convert("RGB")

    def _load_label(
        self, label_path: Path, image_width: int, image_height: int
    ) -> tuple[np.ndarray, np.ndarray, int | None, str]:
        """Load normalized labels and convert landmark coordinates to pixel space."""
        parsed_label = parse_synthetic_landmark_label(
            label_path=label_path,
            expected_num_landmarks=self.num_landmarks,
        )
        landmarks = parsed_label.landmarks.astype(np.float32, copy=True)
        visibility = parsed_label.visibility.astype(np.float32, copy=True)
        landmarks[:, 0] *= float(image_width)
        landmarks[:, 1] *= float(image_height)
        return landmarks, visibility, parsed_label.class_idx, parsed_label.class_name

    @staticmethod
    def _ensure_image_tensor(image: Any) -> torch.Tensor:
        """Convert the image field to a float tensor in CHW layout."""
        if isinstance(image, torch.Tensor):
            return image.float()
        if isinstance(image, Image.Image):
            image = np.asarray(image).copy()
        if isinstance(image, np.ndarray):
            return torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        raise TypeError(f"Unsupported image type: {type(image)}.")

    @staticmethod
    def _ensure_landmark_tensor(landmarks: Any) -> torch.Tensor:
        """Validate and convert landmarks to a float tensor of shape `(N, 2)`."""
        tensor = torch.as_tensor(landmarks, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError(
                f"Expected landmarks with shape (N, 2), got {tuple(tensor.shape)}."
            )
        return tensor

    def _ensure_visibility_tensor(self, visibility: Any) -> torch.Tensor:
        """Validate and convert visibility flags to a float tensor of shape `(N,)`."""
        tensor = torch.as_tensor(visibility, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.shape[0] != self.num_landmarks:
            raise ValueError(
                f"Expected visibility with shape ({self.num_landmarks},), got {tuple(tensor.shape)}."
            )
        return tensor

    def _generate_heatmaps(
        self, landmarks: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        """Render one Gaussian heatmap per landmark in the target heatmap resolution."""
        image_height, image_width = image_size
        heatmap_height, heatmap_width = self.heatmap_size
        scale_x = heatmap_width / float(image_width)
        scale_y = heatmap_height / float(image_height)

        landmarks_np = landmarks.detach().cpu().numpy().astype(np.float32, copy=True)
        heatmaps = np.zeros(
            (self.num_landmarks, heatmap_height, heatmap_width), dtype=np.float32
        )
        for landmark_index in range(self.num_landmarks):
            point = landmarks_np[landmark_index].copy()
            point[0] *= scale_x
            point[1] *= scale_y
            heatmaps[landmark_index] = draw_gaussian(
                heatmaps[landmark_index], point, self.sigma
            )
        return torch.from_numpy(heatmaps)
