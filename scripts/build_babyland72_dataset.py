from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image, ImageOps

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils.visualization import (
    render_landmark_preview_image,
    save_overlay_image,
)


NUM_LANDMARKS = 72
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DuplicatePolicy = Literal["report_and_skip_image", "keep_first", "keep_last"]

ORIENTATION_LABEL_TO_CLASS_IDX = {
    "Left_sideview": 0,
    "3/4_left_sideview": 1,
    "Frontal": 2,
    "3/4_rigth_sideview": 3,
    "Right_sideview": 4,
}
FLIP_CLASS_IDX_MAPPING = {
    0: 4,
    1: 3,
    2: 2,
    3: 1,
    4: 0,
}


@dataclass(frozen=True)
class BabyLand72GenerationConfig:
    """Configuration for regenerating a BabyLand-72 test dataset."""

    source_images_dir: Path
    landmarks_json_path: Path
    orientation_json_path: Path | None
    output_dataset_root: Path
    report_dir: Path | None = None
    duplicate_policy: DuplicatePolicy = "report_and_skip_image"
    jpeg_quality: int = 95


@dataclass
class OrientationRecord:
    """Resolved orientation metadata for one image stem."""

    stem: str
    class_idx: int | None
    status: str
    labels: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    annotation_ids: list[str] = field(default_factory=list)


@dataclass
class LandmarkTask:
    """Label Studio landmark task normalized for generation."""

    stem: str
    task_id: str
    annotation_id: str | None
    image_field: str
    landmarks: list[dict[str, Any]]


@dataclass
class GenerationResult:
    """Detailed result rows and counters produced by dataset regeneration."""

    generated_samples: list[dict[str, Any]] = field(default_factory=list)
    unmatched_source_images: list[dict[str, Any]] = field(default_factory=list)
    tasks_without_matching_image: list[dict[str, Any]] = field(default_factory=list)
    samples_without_orientation: list[dict[str, Any]] = field(default_factory=list)
    orientation_conflicts: list[dict[str, Any]] = field(default_factory=list)
    duplicate_landmarks: list[dict[str, Any]] = field(default_factory=list)
    unknown_keypoint_labels: list[dict[str, Any]] = field(default_factory=list)
    missing_landmarks_by_image: list[dict[str, Any]] = field(default_factory=list)
    missing_landmarks_by_index: list[dict[str, Any]] = field(default_factory=list)
    plot_failures: list[dict[str, Any]] = field(default_factory=list)
    summary_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_counts_original: Counter[int] = field(default_factory=Counter)
    class_counts_flipped: Counter[int] = field(default_factory=Counter)
    source_image_count: int = 0
    landmark_task_count: int = 0
    orientation_entry_count: int = 0
    skipped_sample_count: int = 0


def extract_stem_from_labelstudio_image_path(image_field: str) -> str:
    """Extract the filename stem from one Label Studio image field."""
    normalized = str(image_field).replace("\\/", "/")
    parsed = urlparse(normalized)
    query_path = parse_qs(parsed.query).get("d", [None])[0]
    candidate = unquote(query_path or parsed.path or normalized)
    return Path(candidate).stem


def parse_landmark_index_from_label_name(label_name: str) -> int | None:
    """Parse a 1-based landmark index from a Label Studio keypoint label."""
    digits = []
    for character in reversed(str(label_name)):
        if character.isdigit():
            digits.append(character)
        elif digits:
            break
    if not digits:
        return None
    index = int("".join(reversed(digits)))
    if 1 <= index <= NUM_LANDMARKS:
        return index
    return None


def convert_labelstudio_percent_to_normalized(value: Any) -> float:
    """Convert a Label Studio percentage coordinate to a normalized coordinate."""
    normalized = float(value) / 100.0
    return min(max(normalized, 0.0), 1.0)


def _load_json_list(json_path: Path) -> list[dict[str, Any]]:
    """Load a Label Studio JSON export and validate that it is a task list."""
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {json_path}.")
    return [task for task in payload if isinstance(task, dict)]


def _extract_task_image_field(task: dict[str, Any]) -> str | None:
    """Return the image field from common Label Studio export shapes."""
    if task.get("image"):
        return str(task["image"])
    data = task.get("data")
    if isinstance(data, dict) and data.get("image"):
        return str(data["image"])
    return None


def _iter_annotation_results(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect annotation result dictionaries from common Label Studio shapes."""
    results: list[dict[str, Any]] = []
    if isinstance(task.get("landmarks"), list):
        results.extend(item for item in task["landmarks"] if isinstance(item, dict))
    if isinstance(task.get("label"), list):
        results.extend(item for item in task["label"] if isinstance(item, dict))
    annotations = task.get("annotations")
    if isinstance(annotations, list):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            annotation_results = annotation.get("result")
            if isinstance(annotation_results, list):
                results.extend(
                    item for item in annotation_results if isinstance(item, dict)
                )
    return results


def _extract_annotation_id(task: dict[str, Any]) -> str | None:
    """Extract one annotation identifier when present."""
    if task.get("annotation_id") is not None:
        return str(task["annotation_id"])
    annotations = task.get("annotations")
    if isinstance(annotations, list) and annotations:
        annotation = annotations[0]
        if isinstance(annotation, dict) and annotation.get("id") is not None:
            return str(annotation["id"])
    return None


def _extract_keypoint_label(result: dict[str, Any]) -> str | None:
    """Extract a keypoint label from a Label Studio result object."""
    labels = result.get("keypointlabels")
    if not labels and isinstance(result.get("value"), dict):
        labels = result["value"].get("keypointlabels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return None


def _extract_xy_percent(result: dict[str, Any]) -> tuple[float, float] | None:
    """Extract Label Studio x/y percentage coordinates from a result object."""
    if "x" in result and "y" in result:
        return float(result["x"]), float(result["y"])
    value = result.get("value")
    if isinstance(value, dict) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    return None


def _extract_orientation_labels(task: dict[str, Any]) -> list[str]:
    """Extract orientation labels from common Label Studio rectangle outputs."""
    labels: list[str] = []
    for result in _iter_annotation_results(task):
        candidates = result.get("rectanglelabels")
        if not candidates and isinstance(result.get("value"), dict):
            candidates = result["value"].get("rectanglelabels")
        if isinstance(candidates, list):
            labels.extend(str(label) for label in candidates)
    return labels


def build_labelstudio_orientation_index(
    json_path: Path | None,
) -> tuple[dict[str, OrientationRecord], list[dict[str, Any]], int]:
    """Build an orientation index keyed by image stem from a Label Studio export."""
    if json_path is None:
        return {}, [], 0
    tasks = _load_json_list(json_path)
    grouped: dict[str, list[tuple[str, str | None, list[str]]]] = defaultdict(list)
    conflicts: list[dict[str, Any]] = []

    for task in tasks:
        image_field = _extract_task_image_field(task)
        if not image_field:
            conflicts.append(
                {
                    "source_stem": "",
                    "conflict_type": "missing_image_field",
                    "detected_labels": "",
                    "selected_class_idx_if_any": "",
                    "generated_without_class_idx": True,
                }
            )
            continue
        stem = extract_stem_from_labelstudio_image_path(image_field)
        labels = _extract_orientation_labels(task)
        grouped[stem].append(
            (str(task.get("id", "")), _extract_annotation_id(task), labels)
        )

    index: dict[str, OrientationRecord] = {}
    for stem, entries in grouped.items():
        valid_labels = [
            label
            for _, _, labels in entries
            for label in labels
            if label in ORIENTATION_LABEL_TO_CLASS_IDX
        ]
        invalid_labels = [
            label
            for _, _, labels in entries
            for label in labels
            if label not in ORIENTATION_LABEL_TO_CLASS_IDX
        ]
        class_indices = sorted(
            {ORIENTATION_LABEL_TO_CLASS_IDX[label] for label in valid_labels}
        )
        task_ids = [task_id for task_id, _, _ in entries if task_id]
        annotation_ids = [
            annotation_id for _, annotation_id, _ in entries if annotation_id
        ]
        if len(class_indices) == 1:
            status = "valid"
            class_idx = class_indices[0]
            if len(valid_labels) > 1:
                conflicts.append(
                    {
                        "source_stem": stem,
                        "conflict_type": "duplicate_same_orientation",
                        "detected_labels": "|".join(valid_labels + invalid_labels),
                        "selected_class_idx_if_any": class_idx,
                        "generated_without_class_idx": False,
                    }
                )
        elif len(class_indices) > 1:
            status = "conflicting_orientation"
            class_idx = None
            conflicts.append(
                {
                    "source_stem": stem,
                    "conflict_type": status,
                    "detected_labels": "|".join(valid_labels + invalid_labels),
                    "selected_class_idx_if_any": "",
                    "generated_without_class_idx": True,
                }
            )
        else:
            status = "no_valid_orientation_label"
            class_idx = None
            conflicts.append(
                {
                    "source_stem": stem,
                    "conflict_type": status,
                    "detected_labels": "|".join(invalid_labels),
                    "selected_class_idx_if_any": "",
                    "generated_without_class_idx": True,
                }
            )
        index[stem] = OrientationRecord(
            stem=stem,
            class_idx=class_idx,
            status=status,
            labels=valid_labels + invalid_labels,
            task_ids=task_ids,
            annotation_ids=annotation_ids,
        )
    return index, conflicts, len(tasks)


def build_landmark_index_from_labelstudio(json_path: Path) -> dict[str, LandmarkTask]:
    """Build one landmark task per image stem from a Label Studio landmark export."""
    tasks = _load_json_list(json_path)
    landmark_tasks: dict[str, LandmarkTask] = {}
    for task in tasks:
        image_field = _extract_task_image_field(task)
        if not image_field:
            continue
        stem = extract_stem_from_labelstudio_image_path(image_field)
        landmarks = [
            result
            for result in _iter_annotation_results(task)
            if _extract_keypoint_label(result) is not None
            and _extract_xy_percent(result) is not None
        ]
        landmark_tasks[stem] = LandmarkTask(
            stem=stem,
            task_id=str(task.get("id", "")),
            annotation_id=_extract_annotation_id(task),
            image_field=image_field,
            landmarks=landmarks,
        )
    return landmark_tasks


def find_matching_source_image(
    source_images: dict[str, Path], stem: str
) -> Path | None:
    """Find one source image by filename stem."""
    return source_images.get(stem)


def create_landmark_array_72(
    task: LandmarkTask,
    duplicate_policy: DuplicatePolicy,
    result: GenerationResult,
) -> tuple[np.ndarray | None, str]:
    """Create a 72x3 normalized landmark array from one Label Studio task."""
    landmarks = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    landmarks[:, 2] = 0.0
    seen: dict[int, str] = {}
    duplicate_found = False

    for entry in task.landmarks:
        label_name = _extract_keypoint_label(entry)
        if label_name is None:
            continue
        landmark_index = parse_landmark_index_from_label_name(label_name)
        if landmark_index is None:
            result.unknown_keypoint_labels.append(
                {
                    "source_stem": task.stem,
                    "task_id": task.task_id,
                    "unknown_label_name": label_name,
                    "action_taken": "skipped_landmark_entry",
                }
            )
            continue
        xy_percent = _extract_xy_percent(entry)
        if xy_percent is None:
            continue
        if landmark_index in seen:
            duplicate_found = True
            action = (
                "skipped_image"
                if duplicate_policy == "report_and_skip_image"
                else duplicate_policy
            )
            result.duplicate_landmarks.append(
                {
                    "source_stem": task.stem,
                    "task_id": task.task_id,
                    "duplicated_landmark_index": landmark_index,
                    "duplicated_label_name": label_name,
                    "duplicate_policy": duplicate_policy,
                    "action_taken": action,
                }
            )
            if duplicate_policy == "report_and_skip_image":
                continue
            if duplicate_policy == "keep_first":
                continue
        seen[landmark_index] = label_name
        row_index = landmark_index - 1
        landmarks[row_index, 0] = convert_labelstudio_percent_to_normalized(
            xy_percent[0]
        )
        landmarks[row_index, 1] = convert_labelstudio_percent_to_normalized(
            xy_percent[1]
        )
        landmarks[row_index, 2] = 1.0

    if duplicate_found and duplicate_policy == "report_and_skip_image":
        return None, "skipped_duplicate_landmarks"
    return landmarks, "valid"


def write_landmark_txt(
    label_path: Path, landmarks: np.ndarray, class_idx: int | None
) -> None:
    """Write one BabyLand-72 label file with an optional class_idx header."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        if class_idx is not None:
            handle.write(f"{int(class_idx)}\n")
        for x_coord, y_coord, visibility in landmarks:
            if int(visibility) == 0 or not (
                math.isfinite(float(x_coord)) and math.isfinite(float(y_coord))
            ):
                handle.write("nan nan 0\n")
            else:
                handle.write(f"{float(x_coord):.8f} {float(y_coord):.8f} 1\n")


def save_image_as_jpg(
    source_path: Path, output_path: Path, jpeg_quality: int = 95
) -> None:
    """Save one image as lowercase .jpg RGB output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        ImageOps.exif_transpose(image).convert("RGB").save(
            output_path, format="JPEG", quality=jpeg_quality
        )


def flip_image_horizontally(
    source_path: Path, output_path: Path, jpeg_quality: int = 95
) -> None:
    """Save a horizontally flipped copy of an image as JPEG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        flipped = ImageOps.mirror(ImageOps.exif_transpose(image).convert("RGB"))
        flipped.save(output_path, format="JPEG", quality=jpeg_quality)


def draw_babyland72_landmark_overlay(
    image_path: Path,
    output_path: Path,
    landmarks: np.ndarray,
    point_radius: int = 5,
    line_width: int = 2,
    line_color: str = "#00C853",
) -> None:
    """Draw visible BabyLand-72 landmarks over one generated dataset image."""
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        image_width, image_height = image_rgb.size
        pixel_landmarks = landmarks[:, :2].astype(np.float32, copy=True)
        finite_mask = np.isfinite(pixel_landmarks).all(axis=1)
        pixel_landmarks[finite_mask, 0] *= float(image_width)
        pixel_landmarks[finite_mask, 1] *= float(image_height)
        visibility = landmarks[:, 2].astype(np.int64, copy=True)
        visibility[~finite_mask] = 0
        rendered = render_landmark_preview_image(
            image=image_rgb,
            landmarks=pixel_landmarks,
            visibility=visibility,
            point_radius=point_radius,
            line_width=line_width,
            line_color=line_color,
        )
    save_overlay_image(rendered, output_path)


def get_horizontal_flip_index_mapping_72() -> list[int]:
    """Return the explicit 1-based BabyLand-72 horizontal flip permutation."""
    pairs = [
        (1, 17),
        (2, 16),
        (3, 15),
        (4, 14),
        (5, 13),
        (6, 12),
        (7, 11),
        (8, 10),
        (18, 27),
        (19, 26),
        (20, 25),
        (21, 24),
        (22, 23),
        (32, 36),
        (33, 35),
        (37, 46),
        (38, 45),
        (39, 44),
        (40, 43),
        (41, 48),
        (42, 47),
        (49, 55),
        (50, 54),
        (51, 53),
        (56, 60),
        (57, 59),
        (61, 65),
        (62, 64),
        (66, 68),
        (71, 72),
    ]
    mapping = list(range(1, NUM_LANDMARKS + 1))
    for left, right in pairs:
        mapping[left - 1] = right
        mapping[right - 1] = left
    if len(mapping) != NUM_LANDMARKS or sorted(mapping) != list(
        range(1, NUM_LANDMARKS + 1)
    ):
        raise AssertionError("Invalid BabyLand-72 flip mapping permutation.")
    for index, mapped_index in enumerate(mapping, start=1):
        if mapping[mapped_index - 1] != index:
            raise AssertionError("BabyLand-72 flip mapping is not symmetric.")
    return mapping


def flip_landmark_array_72(landmarks: np.ndarray) -> np.ndarray:
    """Flip normalized BabyLand-72 landmarks horizontally and remap indices."""
    mapping = get_horizontal_flip_index_mapping_72()
    flipped = np.full_like(landmarks, np.nan)
    flipped[:, 2] = 0.0
    for original_index, mapped_index in enumerate(mapping, start=1):
        source_row = landmarks[original_index - 1]
        target_row = mapped_index - 1
        visibility = int(source_row[2])
        flipped[target_row, 2] = visibility
        if (
            visibility == 1
            and math.isfinite(float(source_row[0]))
            and math.isfinite(float(source_row[1]))
        ):
            flipped[target_row, 0] = 1.0 - float(source_row[0])
            flipped[target_row, 1] = float(source_row[1])
    return flipped


def flip_class_idx(class_idx: int | None) -> int | None:
    """Return the horizontally flipped class_idx when a header exists."""
    if class_idx is None:
        return None
    return FLIP_CLASS_IDX_MAPPING[int(class_idx)]


def _index_source_images(source_images_dir: Path) -> dict[str, Path]:
    """Index supported source images recursively by stem."""
    indexed: dict[str, Path] = {}
    for image_path in sorted(source_images_dir.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ):
            indexed.setdefault(image_path.stem, image_path)
    return indexed


def _append_generated_sample(
    result: GenerationResult,
    source_stem: str,
    source_image_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    output_plot_path: Path,
    plot_generation_status: str,
    is_flipped: bool,
    class_idx: int | None,
    orientation_status: str,
    landmarks: np.ndarray,
) -> None:
    """Append sample and missing-landmark report rows."""
    present = int(np.sum(landmarks[:, 2] == 1.0))
    missing_indices = [
        str(index)
        for index in range(1, NUM_LANDMARKS + 1)
        if landmarks[index - 1, 2] == 0.0
    ]
    result.generated_samples.append(
        {
            "source_stem": source_stem,
            "source_image_path": str(source_image_path),
            "output_image_path": str(output_image_path),
            "output_label_path": str(output_label_path),
            "output_plot_path": str(output_plot_path),
            "is_flipped": is_flipped,
            "has_class_idx": class_idx is not None,
            "class_idx": "" if class_idx is None else int(class_idx),
            "orientation_status": orientation_status,
            "num_present_landmarks": present,
            "num_missing_landmarks": NUM_LANDMARKS - present,
            "generation_status": "generated",
            "plot_generation_status": plot_generation_status,
        }
    )
    result.missing_landmarks_by_image.append(
        {
            "source_stem": source_stem,
            "output_label_path": str(output_label_path),
            "missing_landmark_indices": "|".join(missing_indices),
            "num_missing_landmarks": NUM_LANDMARKS - present,
            "has_class_idx": class_idx is not None,
        }
    )


def generate_babyland72_test_dataset_from_labelstudio(
    config: BabyLand72GenerationConfig,
) -> GenerationResult:
    """Regenerate a BabyLand-72 test dataset from Label Studio JSON exports."""
    if config.duplicate_policy not in {
        "report_and_skip_image",
        "keep_first",
        "keep_last",
    }:
        raise ValueError(f"Unsupported duplicate policy: {config.duplicate_policy}")
    get_horizontal_flip_index_mapping_72()

    output_images_dir = config.output_dataset_root / "test" / "images"
    output_labels_dir = config.output_dataset_root / "test" / "labels"
    output_plots_dir = config.output_dataset_root / "test" / "plots"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    result = GenerationResult()
    source_images = _index_source_images(config.source_images_dir)
    landmark_tasks = build_landmark_index_from_labelstudio(config.landmarks_json_path)
    (
        orientation_index,
        orientation_conflicts,
        orientation_entry_count,
    ) = build_labelstudio_orientation_index(config.orientation_json_path)
    result.source_image_count = len(source_images)
    result.landmark_task_count = len(landmark_tasks)
    result.orientation_entry_count = orientation_entry_count
    result.orientation_conflicts.extend(orientation_conflicts)

    for stem, source_path in source_images.items():
        if stem not in landmark_tasks:
            result.unmatched_source_images.append(
                {
                    "source_stem": stem,
                    "source_image_path": str(source_path),
                    "reason": "no_landmark_task",
                }
            )

    for stem, task in landmark_tasks.items():
        source_path = find_matching_source_image(source_images, stem)
        if source_path is None:
            result.tasks_without_matching_image.append(
                {
                    "source_stem": stem,
                    "landmark_task_id": task.task_id,
                    "image_field": task.image_field,
                    "reason": "no_matching_source_image",
                }
            )
            result.skipped_sample_count += 1
            continue

        landmarks, status = create_landmark_array_72(
            task, config.duplicate_policy, result
        )
        if landmarks is None:
            result.skipped_sample_count += 1
            continue

        orientation = orientation_index.get(stem)
        class_idx = orientation.class_idx if orientation is not None else None
        orientation_status = (
            orientation.status
            if orientation is not None
            else "missing_orientation_match"
        )
        if class_idx is None:
            result.samples_without_orientation.append(
                {
                    "source_stem": stem,
                    "source_image_path": str(source_path),
                    "landmark_task_id": task.task_id,
                    "reason": orientation_status,
                    "generated_without_class_idx": True,
                }
            )
        else:
            result.class_counts_original[int(class_idx)] += 1

        image_output_path = output_images_dir / f"{stem}.jpg"
        label_output_path = output_labels_dir / f"{stem}.txt"
        plot_output_path = output_plots_dir / f"{stem}.jpg"
        save_image_as_jpg(
            source_path, image_output_path, jpeg_quality=config.jpeg_quality
        )
        write_landmark_txt(label_output_path, landmarks, class_idx)
        plot_status = "generated"
        try:
            draw_babyland72_landmark_overlay(
                image_path=image_output_path,
                output_path=plot_output_path,
                landmarks=landmarks,
            )
        except Exception as error:
            plot_status = f"failed: {error}"
            result.plot_failures.append(
                {
                    "source_stem": stem,
                    "output_image_path": str(image_output_path),
                    "output_plot_path": str(plot_output_path),
                    "is_flipped": False,
                    "error": str(error),
                }
            )
        _append_generated_sample(
            result,
            stem,
            source_path,
            image_output_path,
            label_output_path,
            plot_output_path,
            plot_status,
            False,
            class_idx,
            orientation_status,
            landmarks,
        )

        flipped_stem = f"flip_{stem}"
        flipped_class_idx = flip_class_idx(class_idx)
        if flipped_class_idx is not None:
            result.class_counts_flipped[int(flipped_class_idx)] += 1
        flipped_landmarks = flip_landmark_array_72(landmarks)
        flipped_image_output_path = output_images_dir / f"{flipped_stem}.jpg"
        flipped_label_output_path = output_labels_dir / f"{flipped_stem}.txt"
        flipped_plot_output_path = output_plots_dir / f"{flipped_stem}.jpg"
        flip_image_horizontally(
            source_path, flipped_image_output_path, jpeg_quality=config.jpeg_quality
        )
        write_landmark_txt(
            flipped_label_output_path, flipped_landmarks, flipped_class_idx
        )
        flipped_plot_status = "generated"
        try:
            draw_babyland72_landmark_overlay(
                image_path=flipped_image_output_path,
                output_path=flipped_plot_output_path,
                landmarks=flipped_landmarks,
            )
        except Exception as error:
            flipped_plot_status = f"failed: {error}"
            result.plot_failures.append(
                {
                    "source_stem": stem,
                    "output_image_path": str(flipped_image_output_path),
                    "output_plot_path": str(flipped_plot_output_path),
                    "is_flipped": True,
                    "error": str(error),
                }
            )
        _append_generated_sample(
            result,
            stem,
            source_path,
            flipped_image_output_path,
            flipped_label_output_path,
            flipped_plot_output_path,
            flipped_plot_status,
            True,
            flipped_class_idx,
            orientation_status,
            flipped_landmarks,
        )

    original_rows = [row for row in result.generated_samples if not row["is_flipped"]]
    for landmark_index in range(1, NUM_LANDMARKS + 1):
        missing_count = sum(
            1
            for row in result.missing_landmarks_by_image
            if not Path(str(row["output_label_path"])).stem.startswith("flip_")
            and str(landmark_index) in str(row["missing_landmark_indices"]).split("|")
        )
        total = len(original_rows)
        visible_count = total - missing_count
        result.missing_landmarks_by_index.append(
            {
                "landmark_index": landmark_index,
                "missing_count": missing_count,
                "visible_count": visible_count,
                "total_generated_original_samples": total,
                "missing_percentage": (missing_count / total * 100.0) if total else 0.0,
            }
        )

    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV, always creating a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_generation_reports(
    config: BabyLand72GenerationConfig,
    result: GenerationResult,
) -> Path:
    """Write CSV diagnostics and a Markdown summary report."""
    report_dir = config.report_dir or config.output_dataset_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    generated = result.generated_samples
    original = [row for row in generated if not row["is_flipped"]]
    flipped = [row for row in generated if row["is_flipped"]]
    original_with_class = [row for row in original if row["has_class_idx"]]
    original_without_class = [row for row in original if not row["has_class_idx"]]
    flipped_with_class = [row for row in flipped if row["has_class_idx"]]
    flipped_without_class = [row for row in flipped if not row["has_class_idx"]]
    original_plots = [
        row for row in original if row["plot_generation_status"] == "generated"
    ]
    flipped_plots = [
        row for row in flipped if row["plot_generation_status"] == "generated"
    ]
    visible_counts = [int(row["num_present_landmarks"]) for row in generated]
    missing_distribution = Counter(
        int(row["num_missing_landmarks"]) for row in generated
    )
    combined_class_counts = result.class_counts_original + result.class_counts_flipped

    summary = {
        "source_images_found": result.source_image_count,
        "landmark_tasks": result.landmark_task_count,
        "orientation_entries_indexed": result.orientation_entry_count,
        "original_samples_generated": len(original),
        "flipped_samples_generated": len(flipped),
        "total_images_written": len(generated),
        "total_labels_written": len(generated),
        "plot_overlays_written": len(original_plots) + len(flipped_plots),
        "original_plot_overlays_written": len(original_plots),
        "flipped_plot_overlays_written": len(flipped_plots),
        "plot_generation_failures": len(result.plot_failures),
        "original_labels_with_class_idx": len(original_with_class),
        "original_labels_without_class_idx": len(original_without_class),
        "flipped_labels_with_class_idx": len(flipped_with_class),
        "flipped_labels_without_class_idx": len(flipped_without_class),
        "total_labels_with_class_idx": len(original_with_class)
        + len(flipped_with_class),
        "total_labels_without_class_idx": len(original_without_class)
        + len(flipped_without_class),
        "skipped_samples": result.skipped_sample_count,
        "duplicate_policy": config.duplicate_policy,
        "warnings_or_conflicts": len(result.orientation_conflicts)
        + len(result.duplicate_landmarks)
        + len(result.unknown_keypoint_labels)
        + len(result.plot_failures),
    }
    result.summary_rows = [
        {"metric": key, "value": value} for key, value in summary.items()
    ]

    _write_csv(
        report_dir / "dataset_generation_summary.csv",
        result.summary_rows,
        ["metric", "value"],
    )
    _write_csv(
        report_dir / "generated_samples.csv",
        result.generated_samples,
        [
            "source_stem",
            "source_image_path",
            "output_image_path",
            "output_label_path",
            "output_plot_path",
            "is_flipped",
            "has_class_idx",
            "class_idx",
            "orientation_status",
            "num_present_landmarks",
            "num_missing_landmarks",
            "generation_status",
            "plot_generation_status",
        ],
    )
    _write_csv(
        report_dir / "unmatched_source_images.csv",
        result.unmatched_source_images,
        ["source_stem", "source_image_path", "reason"],
    )
    _write_csv(
        report_dir / "tasks_without_matching_image.csv",
        result.tasks_without_matching_image,
        ["source_stem", "landmark_task_id", "image_field", "reason"],
    )
    _write_csv(
        report_dir / "samples_without_orientation.csv",
        result.samples_without_orientation,
        [
            "source_stem",
            "source_image_path",
            "landmark_task_id",
            "reason",
            "generated_without_class_idx",
        ],
    )
    _write_csv(
        report_dir / "orientation_conflicts.csv",
        result.orientation_conflicts,
        [
            "source_stem",
            "conflict_type",
            "detected_labels",
            "selected_class_idx_if_any",
            "generated_without_class_idx",
        ],
    )
    _write_csv(
        report_dir / "duplicate_landmarks.csv",
        result.duplicate_landmarks,
        [
            "source_stem",
            "task_id",
            "duplicated_landmark_index",
            "duplicated_label_name",
            "duplicate_policy",
            "action_taken",
        ],
    )
    _write_csv(
        report_dir / "unknown_keypoint_labels.csv",
        result.unknown_keypoint_labels,
        ["source_stem", "task_id", "unknown_label_name", "action_taken"],
    )
    _write_csv(
        report_dir / "missing_landmarks_by_image.csv",
        result.missing_landmarks_by_image,
        [
            "source_stem",
            "output_label_path",
            "missing_landmark_indices",
            "num_missing_landmarks",
            "has_class_idx",
        ],
    )
    _write_csv(
        report_dir / "missing_landmarks_by_index.csv",
        result.missing_landmarks_by_index,
        [
            "landmark_index",
            "missing_count",
            "visible_count",
            "total_generated_original_samples",
            "missing_percentage",
        ],
    )
    _write_csv(
        report_dir / "plot_generation_failures.csv",
        result.plot_failures,
        ["source_stem", "output_image_path", "output_plot_path", "is_flipped", "error"],
    )

    def class_lines(counter: Counter[int]) -> list[str]:
        return [f"- class_idx {idx}: {counter.get(idx, 0)}" for idx in range(5)]

    markdown = [
        "# BabyLand-72 Dataset Regeneration Report",
        "",
        "## Inputs",
        f"- source images directory: `{config.source_images_dir}`",
        f"- landmarks JSON path: `{config.landmarks_json_path}`",
        f"- orientation JSON path: `{config.orientation_json_path}`",
        f"- output dataset root: `{config.output_dataset_root}`",
        f"- duplicate policy: `{config.duplicate_policy}`",
        f"- generation timestamp: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Dataset generation summary",
        *[f"- {key}: {value}" for key, value in summary.items()],
        "",
        "## Class_idx availability",
        f"- samples with a valid class_idx match: {len(original_with_class)}",
        f"- samples generated without class_idx because no orientation match was found: {sum(row['reason'] == 'missing_orientation_match' for row in result.samples_without_orientation)}",
        f"- samples generated without class_idx because orientation metadata was ambiguous or conflicting: {sum(row['reason'] != 'missing_orientation_match' for row in result.samples_without_orientation)}",
        "",
        "## Class distribution",
        "Original samples with class_idx:",
        *class_lines(result.class_counts_original),
        "",
        "Flipped samples with class_idx:",
        *class_lines(result.class_counts_flipped),
        "",
        "Combined output with class_idx:",
        *class_lines(combined_class_counts),
        f"- original samples without class_idx: {len(original_without_class)}",
        f"- flipped samples without class_idx: {len(flipped_without_class)}",
        "",
        "## Landmark completeness",
        f"- total number of 72-landmark labels generated: {len(generated)}",
        f"- mean visible landmarks per image: {float(np.mean(visible_counts)) if visible_counts else 0.0:.2f}",
        f"- min visible landmarks per image: {min(visible_counts) if visible_counts else 0}",
        f"- max visible landmarks per image: {max(visible_counts) if visible_counts else 0}",
        "- distribution of missing landmark counts:",
        *[
            f"  - {missing_count} missing: {count}"
            for missing_count, count in sorted(missing_distribution.items())
        ],
        "- per-landmark missing frequency: see `missing_landmarks_by_index.csv`",
        "",
        "## Overlay plots",
        f"- total plots generated: {len(original_plots) + len(flipped_plots)}",
        f"- original plots generated: {len(original_plots)}",
        f"- flipped plots generated: {len(flipped_plots)}",
        f"- plot generation failures: {len(result.plot_failures)}",
        "- plot failures: see `plot_generation_failures.csv`",
        "",
        "## Matching diagnostics",
        f"- tasks without source image: {len(result.tasks_without_matching_image)}",
        f"- source images without landmark task: {len(result.unmatched_source_images)}",
        f"- samples without orientation match: {len(result.samples_without_orientation)}",
        f"- orientation conflicts: {len(result.orientation_conflicts)}",
        f"- duplicate landmark issues: {len(result.duplicate_landmarks)}",
        f"- unknown keypoint label issues: {len(result.unknown_keypoint_labels)}",
        "",
        "## Flipping diagnostics",
        f"- flipped images created: {len(flipped)}",
        f"- flipped labels created: {len(flipped)}",
        "- class_idx was remapped when present: yes",
        "- labels without class_idx remained without class_idx after flipping: yes",
        "- horizontal landmark index permutation passed validation: yes",
        "",
        "## Warnings",
        *(
            [f"- {warning}" for warning in result.warnings]
            if result.warnings
            else ["- No fatal warnings. Review CSV diagnostics for non-fatal issues."]
        ),
        "",
    ]
    report_path = report_dir / "dataset_generation_summary.md"
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BabyLand-72 regeneration."""
    parser = argparse.ArgumentParser(
        description="Regenerate a BabyLand-72 test dataset from Label Studio exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-images-dir", type=Path, required=True)
    parser.add_argument("--landmarks-json-path", type=Path, required=True)
    parser.add_argument("--orientation-json-path", type=Path, default=None)
    parser.add_argument("--output-dataset-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--duplicate-policy",
        choices=["report_and_skip_image", "keep_first", "keep_last"],
        default="report_and_skip_image",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    """Run BabyLand-72 dataset regeneration from the command line."""
    args = parse_args()
    config = BabyLand72GenerationConfig(
        source_images_dir=args.source_images_dir,
        landmarks_json_path=args.landmarks_json_path,
        orientation_json_path=args.orientation_json_path,
        output_dataset_root=args.output_dataset_root,
        report_dir=args.report_dir,
        duplicate_policy=args.duplicate_policy,
        jpeg_quality=args.jpeg_quality,
    )
    result = generate_babyland72_test_dataset_from_labelstudio(config)
    report_path = write_generation_reports(config, result)

    generated_original = sum(
        1 for row in result.generated_samples if not row["is_flipped"]
    )
    generated_flipped = sum(1 for row in result.generated_samples if row["is_flipped"])
    labels_with_class = sum(
        1 for row in result.generated_samples if row["has_class_idx"]
    )
    labels_without_class = sum(
        1 for row in result.generated_samples if not row["has_class_idx"]
    )
    plot_count = sum(
        1
        for row in result.generated_samples
        if row["plot_generation_status"] == "generated"
    )
    warning_count = (
        len(result.orientation_conflicts)
        + len(result.duplicate_landmarks)
        + len(result.unknown_keypoint_labels)
        + len(result.plot_failures)
    )
    print("[INFO] BabyLand-72 regeneration finished.")
    print(f"[INFO] Output dataset root: {config.output_dataset_root}")
    print(f"[INFO] Generated original samples: {generated_original}")
    print(f"[INFO] Generated flipped samples: {generated_flipped}")
    print(f"[INFO] Skipped samples: {result.skipped_sample_count}")
    print(f"[INFO] Labels with class_idx: {labels_with_class}")
    print(f"[INFO] Labels without class_idx: {labels_without_class}")
    print(f"[INFO] Overlay plots written: {plot_count}")
    print(f"[INFO] Warning/conflict cases: {warning_count}")
    print(f"[INFO] Markdown report: {report_path}")


if __name__ == "__main__":
    main()
