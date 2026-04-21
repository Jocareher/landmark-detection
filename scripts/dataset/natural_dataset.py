from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..config import ExperimentConfig
from ..utils.natural_labels import parse_natural_landmark_label

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class NaturalLandmarkEvaluationDataset(Dataset):
    """Dataset for evaluating detector-aligned crops against original-image GT."""

    def __init__(
        self,
        export_root: str | Path,
        gt_root: str | Path,
        config: ExperimentConfig,
        source_root: str | Path | None = None,
        show_progress: bool = True,
    ) -> None:
        """Index detector-export crops and load original-image GT labels."""
        self.export_root = Path(export_root)
        self.gt_root = Path(gt_root)
        self.source_root = Path(source_root) if source_root is not None else None
        self.config = config
        self.show_progress = show_progress

        self.images_dir = self.export_root / "images"
        self.metadata_dir = self.export_root / "metadata"

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Detector export images directory not found: {self.images_dir}"
            )
        if not self.metadata_dir.exists():
            raise FileNotFoundError(
                f"Detector export metadata directory not found: {self.metadata_dir}"
            )
        if not self.gt_root.exists():
            raise FileNotFoundError(f"Natural GT directory not found: {self.gt_root}")

        self.mean = torch.tensor(
            self.config.normalization_mean,
            dtype=torch.float32,
        ).view(3, 1, 1)
        self.std = torch.tensor(
            self.config.normalization_std,
            dtype=torch.float32,
        ).view(3, 1, 1)

        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(
                f"No valid natural evaluation samples found under {self.export_root}."
            )

    def __len__(self) -> int:
        """Return the number of indexed detector-export crops."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one detector crop together with metadata and original-image GT."""
        sample_info = self.samples[index]

        crop_image = Image.open(sample_info["crop_image_path"]).convert("RGB")
        crop_width, crop_height = crop_image.size
        target_height, target_width = self.config.image_size

        resized_crop = crop_image.resize((target_width, target_height), Image.BILINEAR)
        crop_tensor = (
            torch.from_numpy(np.asarray(resized_crop, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .contiguous()
        )
        crop_tensor = (crop_tensor - self.mean) / self.std

        original_width, original_height = self._load_image_size(
            sample_info["source_image_path"]
        )
        gt_landmarks, gt_visibility, class_idx, orientation = self._load_original_gt(
            label_path=sample_info["gt_label_path"],
            image_width=original_width,
            image_height=original_height,
        )

        metadata = {
            "sample_id": sample_info["crop_id"],
            "crop_id": sample_info["crop_id"],
            "image_path": str(sample_info["crop_image_path"]),
            "crop_image_path": str(sample_info["crop_image_path"]),
            "source_image_path": str(sample_info["source_image_path"]),
            "source_image_name": sample_info["source_image_path"].name,
            "label_path": str(sample_info["gt_label_path"]),
            "original_size": (original_height, original_width),
            "crop_size": (crop_height, crop_width),
            "transformed_size": (target_height, target_width),
            "transform_crop_to_orig": torch.tensor(
                sample_info["transform_crop_to_orig"],
                dtype=torch.float32,
            ),
            "transform_orig_to_crop": torch.tensor(
                sample_info["transform_orig_to_crop"],
                dtype=torch.float32,
            ),
            "detection_index": sample_info["metadata"].get("detection_index"),
            "predicted_class_id": sample_info["metadata"].get("predicted_class_id"),
            "predicted_class_name": sample_info["metadata"].get("predicted_class_name"),
            "class_idx": -1 if class_idx is None else int(class_idx),
            "orientation": orientation,
        }

        return {
            "image": crop_tensor,
            "landmarks": torch.from_numpy(gt_landmarks),
            "visibility": torch.from_numpy(gt_visibility),
            "metadata": metadata,
        }

    def _index_samples(self) -> list[dict[str, Any]]:
        """Scan the detector export and keep image/metadata pairs with resolvable GT."""
        image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

        samples: list[dict[str, Any]] = []
        for image_path in image_paths:
            metadata_path = self.metadata_dir / f"{image_path.stem}.json"
            if not metadata_path.exists():
                continue

            metadata = self._load_metadata(metadata_path)
            crop_id = str(metadata.get("crop_id") or image_path.stem)
            source_image_path = self._resolve_source_image_path(
                metadata=metadata,
                metadata_path=metadata_path,
            )
            gt_label_path = self._resolve_gt_label_path(
                metadata=metadata,
                source_image_path=source_image_path,
            )

            transform_crop_to_orig = self._parse_transform_matrix(
                metadata=metadata,
                field_name="transform_crop_to_orig",
            )
            transform_orig_to_crop = self._parse_transform_matrix(
                metadata=metadata,
                field_name="transform_orig_to_crop",
            )

            if not gt_label_path.exists():
                continue

            samples.append(
                {
                    "crop_id": crop_id,
                    "crop_image_path": image_path,
                    "metadata_path": metadata_path,
                    "source_image_path": source_image_path,
                    "gt_label_path": gt_label_path,
                    "transform_crop_to_orig": transform_crop_to_orig,
                    "transform_orig_to_crop": transform_orig_to_crop,
                    "metadata": metadata,
                }
            )

        return samples

    @staticmethod
    def _load_metadata(metadata_path: Path) -> dict[str, Any]:
        """Load one detector-export metadata JSON file."""
        with metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {metadata_path}.")
        return payload

    def _resolve_source_image_path(
        self,
        metadata: dict[str, Any],
        metadata_path: Path,
    ) -> Path:
        """Resolve the original source image path referenced by detector metadata."""
        raw_path = metadata.get("source_image_path")
        if not raw_path:
            raise KeyError(f"Missing 'source_image_path' in {metadata_path}.")

        source_path = Path(str(raw_path))
        candidate_paths = []
        if source_path.is_absolute():
            candidate_paths.append(source_path)
        else:
            if self.source_root is not None:
                candidate_paths.append(self.source_root / source_path)
            candidate_paths.append(self.export_root / source_path)
            candidate_paths.append(metadata_path.parent / source_path)

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path.resolve()

        raise FileNotFoundError(
            f"Could not resolve source image path '{raw_path}' from {metadata_path}."
        )

    def _resolve_gt_label_path(
        self,
        metadata: dict[str, Any],
        source_image_path: Path,
    ) -> Path:
        """Resolve the GT txt path that corresponds to the original source image."""
        raw_source_path = str(metadata.get("source_image_path") or source_image_path)
        source_name = str(metadata.get("source_image_name") or source_image_path.name)
        source_stem = Path(source_name).stem

        candidate_paths = []
        raw_source_path_obj = Path(raw_source_path)

        if not raw_source_path_obj.is_absolute():
            candidate_paths.append(
                self.gt_root / raw_source_path_obj.with_suffix(".txt")
            )

        if self.source_root is not None:
            try:
                source_relative = source_image_path.relative_to(
                    self.source_root.resolve()
                )
                candidate_paths.append(
                    self.gt_root / source_relative.with_suffix(".txt")
                )
            except Exception:
                pass

        candidate_paths.append(self.gt_root / f"{source_stem}.txt")
        candidate_paths.append(
            self.gt_root / source_image_path.with_suffix(".txt").name
        )

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        return self.gt_root / f"{source_stem}.txt"

    @staticmethod
    def _parse_transform_matrix(
        metadata: dict[str, Any],
        field_name: str,
    ) -> list[list[float]]:
        """Validate one 3x3 transform matrix stored in detector metadata."""
        matrix = np.asarray(metadata.get(field_name), dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Expected '{field_name}' to be a 3x3 matrix, got shape {matrix.shape}."
            )
        return matrix.tolist()

    @staticmethod
    def _load_image_size(image_path: Path) -> tuple[int, int]:
        """Read an image size without keeping the image open longer than needed."""
        with Image.open(image_path) as image:
            return image.size

    def _load_original_gt(
        self,
        label_path: Path,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray, int | None, str]:
        """Load normalized original-image GT and convert visible points to pixels."""
        parsed_label = parse_natural_landmark_label(
            label_path=label_path,
            expected_num_landmarks=self.config.num_landmarks,
        )
        landmarks = parsed_label.landmarks
        visibility = parsed_label.visibility

        visible_mask = visibility == 1.0
        landmarks[visible_mask, 0] *= float(image_width)
        landmarks[visible_mask, 1] *= float(image_height)
        landmarks[~visible_mask] = 0.0
        return landmarks, visibility, parsed_label.class_idx, parsed_label.orientation
