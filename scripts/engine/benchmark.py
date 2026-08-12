from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .evaluate import (
    _build_boxplot_title,
    compute_box_normalization_factor,
    compute_normalized_hausdorff_distance,
    format_metric_value,
    round_metric_value,
    save_per_image_nme_csv,
    save_per_image_per_landmark_nme_csv,
    save_per_landmark_nme_csv,
)
from .geometry_metrics import compute_per_landmark_point_to_line_distances
from .visibility_metrics import (
    compute_visibility_analysis,
    save_visibility_metrics_csv,
    save_visibility_plots,
    visibility_summary_fields,
)
from ..utils.natural_labels import (
    NATURAL_ORIENTATION_NAMES,
    UNKNOWN_ORIENTATION,
    orientation_from_class_idx,
    parse_natural_landmark_label,
)
from ..utils.orientation import ORIENTATION_ORDER, normalize_orientation_label
from ..utils.synthetic_labels import (
    format_synthetic_yaw_group,
    parse_synthetic_landmark_label,
)
from ..utils.visualization import (
    compute_global_linear_y_limits,
    compute_global_log_y_limits,
    plot_confusion_matrix,
    plot_per_landmark_boxplot,
    plot_yaw_view_boxplots,
)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DETECTOR_EXPORT_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)__det_(?P<index>\d+)$")


def format_yaw_filename_key(yaw_angle: float) -> str:
    """Return a stable filename/summary key for a synthetic yaw angle."""
    yaw_value = float(yaw_angle)
    if yaw_value.is_integer():
        return f"yaw_{int(yaw_value):+d}deg".replace("+", "plus_").replace(
            "-", "minus_"
        )
    return (
        f"yaw_{yaw_value:+g}deg".replace("+", "plus_")
        .replace("-", "minus_")
        .replace(".", "p")
    )


def normalize_benchmark_orientation(value: Any) -> str:
    """Normalize benchmark orientation labels and preserve true yaw-angle labels."""
    normalized = normalize_orientation_label(value)
    if normalized is not None:
        return normalized
    return str(value) if value is not None else UNKNOWN_ORIENTATION


@dataclass
class ParsedPrediction:
    """Parsed prediction content for one image."""

    landmarks: np.ndarray
    visibility: np.ndarray | None
    num_landmarks: int
    parse_mode: str


def strip_detector_export_suffix(prediction_stem: str) -> str | None:
    """Strip one trailing detector-export suffix like ``__det_000`` when present."""
    match = DETECTOR_EXPORT_SUFFIX_PATTERN.match(prediction_stem)
    if match is None:
        return None
    return str(match.group("base"))


def resolve_gt_stem_for_prediction(
    prediction_stem: str,
    available_gt_stems: set[str],
) -> str | None:
    """Resolve the GT stem for one prediction stem using exact match then suffix fallback."""
    if prediction_stem in available_gt_stems:
        return prediction_stem

    stripped_base_stem = strip_detector_export_suffix(prediction_stem)
    if stripped_base_stem is not None and stripped_base_stem in available_gt_stems:
        return stripped_base_stem

    return None


def resolve_text_labels_dir(
    root_path: str | Path,
    labels_subdir_name: str = "labels",
) -> tuple[Path, Path]:
    """Resolve either a direct txt directory or a root that contains one txt subdir."""
    provided_root = Path(root_path)
    if not provided_root.exists():
        raise FileNotFoundError(f"Root not found: {provided_root}")

    labels_dir = provided_root / labels_subdir_name
    if labels_dir.is_dir() and any(labels_dir.glob("*.txt")):
        return provided_root.resolve(), labels_dir.resolve()

    if provided_root.is_dir() and any(provided_root.glob("*.txt")):
        return provided_root.resolve(), provided_root.resolve()

    raise FileNotFoundError(
        "Could not find txt files. Supported layouts are either "
        f"'<root>/*.txt' or '<root>/{labels_subdir_name}/*.txt'."
    )


def resolve_prediction_labels_dir(prediction_root: str | Path) -> tuple[Path, Path]:
    """Resolve the actual directory that stores prediction txt files."""
    return resolve_text_labels_dir(prediction_root, labels_subdir_name="labels")


def iter_dataset_samples(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Index image/label pairs from one benchmark split root."""
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    samples: list[dict[str, Any]] = []
    for image_path in sorted(images_dir.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS
        ):
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        samples.append(
            {
                "sample_id": image_path.stem,
                "image_path": image_path,
                "label_path": label_path,
            }
        )
    if not samples:
        raise RuntimeError(f"No valid image/label pairs found under {dataset_root}.")
    return samples


def load_ground_truth_landmarks(
    label_path: str | Path,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, int | None, str]:
    """Load normalized synthetic GT landmarks and convert them to absolute image coordinates."""
    label_path = Path(label_path)
    parsed_label = parse_synthetic_landmark_label(
        label_path=label_path,
        expected_num_landmarks=72,
    )

    landmarks = parsed_label.landmarks
    visibility = parsed_label.visibility.astype(np.int64)
    landmarks[:, 0] *= float(image_width)
    landmarks[:, 1] *= float(image_height)
    return (
        landmarks.astype(np.float32),
        visibility,
        None,
        normalize_benchmark_orientation(
            format_yaw_filename_key(float(parsed_label.yaw_angle))
            if parsed_label.yaw_angle is not None
            else UNKNOWN_ORIENTATION
        ),
    )


def load_infantface_ground_truth_landmarks(
    label_path: str | Path,
) -> tuple[np.ndarray, int | None, str]:
    """Load one InfantFace GT file with absolute 68-point landmarks.

    Supported formats:

    New format:
        class_idx
        x1 y1
        ...
        x68 y68

    Legacy format:
        x1 y1
        ...
        x68 y68

    The class header is used only for orientation-based analysis. Legacy files are
    still accepted as ``unknown`` orientation so older annotation exports do not
    get silently misread as one landmark row.
    """
    label_path = Path(label_path)
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty InfantFace GT file: {label_path}")

    first_tokens = lines[0].split()
    if len(first_tokens) == 1:
        class_value = float(first_tokens[0])
        class_idx = int(class_value)
        if class_value != float(class_idx):
            raise ValueError(
                f"Expected integer class_idx in first line of {label_path}, "
                f"got {first_tokens[0]!r}."
            )
        orientation = normalize_benchmark_orientation(class_idx)
        landmark_lines = lines[1:]
    elif len(first_tokens) == 2:
        class_idx = None
        orientation = UNKNOWN_ORIENTATION
        landmark_lines = lines
    else:
        raise ValueError(
            f"Could not parse first line of {label_path}. Expected either "
            "a one-value class_idx header or a two-value landmark row."
        )

    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(
        landmark_lines,
        start=(2 if class_idx is not None else 1),
    ):
        tokens = raw_line.split()
        if len(tokens) != 2:
            raise ValueError(
                f"Invalid InfantFace landmark row in {label_path} at line "
                f"{line_number}. Expected 'x y', got: {raw_line!r}."
            )
        rows.append([float(token) for token in tokens])

    data = np.asarray(rows, dtype=np.float32)
    if data.shape != (68, 2):
        raise ValueError(
            f"Expected InfantFace GT shape (68, 2) in {label_path}, got {tuple(data.shape)}."
        )
    return data.astype(np.float32), class_idx, orientation


def _parse_prediction_lines(lines: list[str]) -> list[np.ndarray]:
    """Parse numeric prediction rows while preserving blank-line detection splits."""
    groups: list[list[np.ndarray]] = []
    current_group: list[np.ndarray] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        parts = stripped.split()
        if len(parts) not in {2, 3}:
            raise ValueError(f"Unsupported prediction line: '{raw_line.rstrip()}'.")
        current_group.append(
            np.asarray([float(value) for value in parts], dtype=np.float32)
        )

    if current_group:
        groups.append(current_group)
    return [np.stack(group, axis=0) for group in groups if group]


def _candidate_detections_from_block(block: np.ndarray) -> list[ParsedPrediction]:
    """Generate valid first-detection candidates from one parsed prediction block."""
    if block.ndim != 2 or block.shape[1] not in {2, 3}:
        return []

    candidates: list[ParsedPrediction] = []
    num_rows, num_columns = block.shape

    if num_columns == 2 and num_rows >= 68:
        candidates.append(
            ParsedPrediction(
                landmarks=block[:68, :2].astype(np.float32),
                visibility=None,
                num_landmarks=68,
                parse_mode="first_68_lines",
            )
        )

    if num_columns == 3 and num_rows >= 72:
        candidates.append(
            ParsedPrediction(
                landmarks=block[:72, :2].astype(np.float32),
                visibility=block[:72, 2].astype(np.int64),
                num_landmarks=72,
                parse_mode="first_72_lines",
            )
        )

    return candidates


def load_prediction_file(
    prediction_path: str | Path,
    expected_num_landmarks: int | None = None,
) -> ParsedPrediction:
    """Load the first deterministic detection from one prediction txt file."""
    prediction_path = Path(prediction_path)
    lines = prediction_path.read_text(encoding="utf-8").splitlines()
    blocks = _parse_prediction_lines(lines)
    if not blocks:
        raise ValueError(f"No valid prediction rows found in {prediction_path}.")

    candidates: list[ParsedPrediction] = []
    for block in blocks:
        candidates.extend(_candidate_detections_from_block(block))

    if not candidates:
        raise ValueError(
            "Could not parse a supported prediction format from "
            f"{prediction_path}. Expected 68x2 or 72x3 rows."
        )

    if expected_num_landmarks is not None:
        for candidate in candidates:
            if candidate.num_landmarks == expected_num_landmarks:
                return candidate
        raise ValueError(
            f"Prediction file {prediction_path} does not match the expected "
            f"{expected_num_landmarks}-landmark format."
        )

    return candidates[0]


def compute_masked_per_landmark_metrics(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    valid_mask: np.ndarray,
    normalization_landmarks: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[dict[int, float], dict[int, float], float | None, float | None]:
    """Compute point-to-point and point-to-line NME for one masked subset."""
    finite_target_mask = np.isfinite(target_landmarks[:, 0]) & np.isfinite(
        target_landmarks[:, 1]
    )
    finite_prediction_mask = np.isfinite(predicted_landmarks[:, 0]) & np.isfinite(
        predicted_landmarks[:, 1]
    )
    valid_mask = valid_mask & finite_target_mask & finite_prediction_mask
    if valid_mask.sum() == 0:
        return {}, {}, None, None

    safe_predicted_landmarks = np.nan_to_num(
        predicted_landmarks.astype(np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_target_landmarks = np.nan_to_num(
        target_landmarks.astype(np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if normalization_landmarks is None:
        normalization_landmarks = target_landmarks[valid_mask]
    finite_normalization_mask = np.isfinite(
        normalization_landmarks[:, 0]
    ) & np.isfinite(normalization_landmarks[:, 1])
    safe_normalization_landmarks = np.nan_to_num(
        normalization_landmarks[finite_normalization_mask].astype(
            np.float32,
            copy=True,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if len(safe_normalization_landmarks) == 0:
        return {}, {}, None, None

    normalization = compute_box_normalization_factor(
        target_landmarks=safe_normalization_landmarks,
        eps=eps,
    )
    point_errors = np.linalg.norm(
        safe_predicted_landmarks - safe_target_landmarks, axis=1
    )
    normalized_errors = point_errors / normalization
    point_to_line_errors = (
        compute_per_landmark_point_to_line_distances(
            predicted_landmarks=safe_predicted_landmarks,
            target_landmarks=safe_target_landmarks,
        )
        / normalization
    )
    per_landmark_errors = {
        int(index): float(normalized_errors[index])
        for index in np.flatnonzero(valid_mask)
    }
    per_landmark_point_to_line_errors = {
        int(index): float(point_to_line_errors[index])
        for index in np.flatnonzero(valid_mask)
    }
    return (
        per_landmark_errors,
        per_landmark_point_to_line_errors,
        float(np.mean(list(per_landmark_errors.values()))),
        float(np.mean(list(per_landmark_point_to_line_errors.values()))),
    )


def summarize_finite_values(values: list[float]) -> dict[str, float | None]:
    """Return mean, median, and tail percentiles for finite image-level values."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def prefixed_summary(prefix: str, values: list[float]) -> dict[str, float | None]:
    """Return summary values using a metric-name prefix."""
    summary = summarize_finite_values(values)
    return {
        f"mean_{prefix}": summary["mean"],
        f"median_{prefix}": summary["median"],
        f"p90_{prefix}": summary["p90"],
        f"p95_{prefix}": summary["p95"],
        f"p99_{prefix}": summary["p99"],
    }


def save_benchmark_summary_csv(output_path: Path, summary: dict[str, Any]) -> None:
    """Save the benchmark summary as a simple key-value CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("model_name", summary.get("model_name")),
        ("prediction_root", summary.get("prediction_root")),
        ("prediction_labels_dir", summary.get("prediction_labels_dir")),
        ("model_landmark_format", summary.get("model_landmark_format")),
        ("evaluated_landmark_count", summary.get("evaluated_landmark_count")),
        ("total_images", summary.get("total_images")),
        ("images_with_prediction", summary.get("images_with_prediction")),
        ("images_without_prediction", summary.get("images_without_prediction")),
        (
            "images_with_invalid_prediction",
            summary.get("images_with_invalid_prediction"),
        ),
        (
            "unmatched_prediction_files_count",
            summary.get("unmatched_prediction_files_count"),
        ),
        ("detection_rate", summary.get("detection_rate")),
        ("valid_landmarks_used", summary.get("valid_landmarks_used")),
        (
            "valid_non_contour_landmarks_used",
            summary.get("valid_non_contour_landmarks_used"),
        ),
    ]
    metric_keys = [
        "mean_nme_box",
        "median_nme_box",
        "mean_nme_box_point_to_line",
        "median_nme_box_point_to_line",
        "mean_hausdorff_box",
        "median_hausdorff_box",
        "p90_hausdorff_box",
        "p95_hausdorff_box",
        "p99_hausdorff_box",
        "mean_nme_box_visible_intersection",
        "median_nme_box_visible_intersection",
        "mean_nme_box_point_to_line_visible_intersection",
        "median_nme_box_point_to_line_visible_intersection",
        "mean_hausdorff_box_visible_intersection",
        "median_hausdorff_box_visible_intersection",
        "p90_hausdorff_box_visible_intersection",
        "p95_hausdorff_box_visible_intersection",
        "p99_hausdorff_box_visible_intersection",
        "mean_nme_box_gt_valid",
        "median_nme_box_gt_valid",
        "mean_nme_box_point_to_line_gt_valid",
        "median_nme_box_point_to_line_gt_valid",
        "mean_hausdorff_box_gt_valid",
        "median_hausdorff_box_gt_valid",
        "p90_hausdorff_box_gt_valid",
        "p95_hausdorff_box_gt_valid",
        "p99_hausdorff_box_gt_valid",
        "mean_nme_box_non_contour",
        "median_nme_box_non_contour",
        "mean_nme_box_point_to_line_non_contour",
        "median_nme_box_point_to_line_non_contour",
        "mean_hausdorff_box_non_contour",
        "median_hausdorff_box_non_contour",
        "p90_hausdorff_box_non_contour",
        "p95_hausdorff_box_non_contour",
        "p99_hausdorff_box_non_contour",
    ]
    rows.extend((key, summary.get(key)) for key in metric_keys if key in summary)
    orientation_sample_counts = summary.get("orientation_sample_counts")
    if orientation_sample_counts is not None:
        for orientation_name, count in orientation_sample_counts.items():
            rows.append((f"samples_{orientation_name}", count))

    orientation_metrics = summary.get("orientation_metrics")
    if orientation_metrics is not None:
        for orientation_name, metrics in orientation_metrics.items():
            rows.extend(
                (f"{key}_{orientation_name}", metrics.get(key))
                for key in metric_keys
                if key in metrics
            )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in rows:
            writer.writerow(
                [
                    metric_name,
                    format_metric_value(round_metric_value(metric_value)),
                ]
            )


def benchmark_prediction_directory(
    dataset_root: str | Path,
    prediction_root: str | Path,
    output_dir: str | Path,
    use_landmark_names_in_boxplot: bool = True,
    fixed_log_y_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Benchmark one prediction directory against one GT split."""
    dataset_root = Path(dataset_root)
    provided_prediction_root, prediction_labels_dir = resolve_prediction_labels_dir(
        prediction_root
    )
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    samples = iter_dataset_samples(dataset_root)
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    available_gt_stems = set(sample_by_id)
    prediction_paths = sorted(prediction_labels_dir.glob("*.txt"))
    prediction_paths_by_gt_stem: dict[str, list[Path]] = {
        sample_id: [] for sample_id in sample_by_id
    }
    unmatched_prediction_files: list[str] = []

    for prediction_path in prediction_paths:
        resolved_gt_stem = resolve_gt_stem_for_prediction(
            prediction_stem=prediction_path.stem,
            available_gt_stems=available_gt_stems,
        )
        if resolved_gt_stem is None:
            unmatched_prediction_files.append(
                f"{prediction_path.name}: could not resolve a GT file by exact stem "
                "or by stripping a trailing '__det_<index>' suffix."
            )
            continue
        prediction_paths_by_gt_stem[resolved_gt_stem].append(prediction_path)

    for current_prediction_paths in prediction_paths_by_gt_stem.values():
        current_prediction_paths.sort(
            key=lambda path: (
                1 if strip_detector_export_suffix(path.stem) is not None else 0,
                path.name,
            )
        )

    per_landmark_errors_68: list[list[float]] = [[] for _ in range(68)]
    per_landmark_errors_72: list[list[float]] = [[] for _ in range(72)]
    per_landmark_point_to_line_errors_68: list[list[float]] = [[] for _ in range(68)]
    per_landmark_point_to_line_errors_72: list[list[float]] = [[] for _ in range(72)]
    orientation_names: list[str] = []
    orientation_to_errors_68: dict[str, list[list[float]]] = {}
    orientation_to_errors_72: dict[str, list[list[float]]] = {}
    orientation_to_box_nme_values: dict[str, list[float]] = {}
    orientation_to_box_nme_point_to_line_values: dict[str, list[float]] = {}
    orientation_to_hausdorff_box_values: dict[str, list[float]] = {}
    orientation_to_box_nme_gt_valid_values: dict[str, list[float]] = {}
    orientation_to_box_nme_point_to_line_gt_valid_values: dict[str, list[float]] = {}
    orientation_to_hausdorff_box_gt_valid_values: dict[str, list[float]] = {}
    orientation_sample_counts: dict[str, int] = {}
    orientation_display_labels: dict[str, str] = {}
    per_image_nme: list[dict[str, Any]] = []
    per_image_per_landmark_nme: list[dict[str, Any]] = []
    invalid_prediction_files: list[str] = []
    inferred_num_landmarks: int | None = None
    images_with_prediction = 0
    images_without_prediction = 0
    images_with_invalid_prediction = 0
    valid_landmarks_used = 0
    valid_non_contour_landmarks_used = 0
    non_contour_image_nme_values: list[float] = []
    non_contour_image_point_to_line_values: list[float] = []
    gt_valid_landmarks_used = 0
    gt_valid_image_nme_values: list[float] = []
    gt_valid_image_point_to_line_values: list[float] = []
    visible_image_hausdorff_values: list[float] = []
    gt_valid_image_hausdorff_values: list[float] = []
    all_visibility_targets: list[np.ndarray] = []
    all_visibility_predictions: list[np.ndarray] = []
    all_visibility_pose_labels: list[np.ndarray] = []
    all_visibility_landmark_indices: list[np.ndarray] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        with Image.open(sample["image_path"]) as image:
            image_width, image_height = image.size

        (
            gt_landmarks_all,
            gt_visibility_all,
            gt_class_idx,
            gt_orientation,
        ) = load_ground_truth_landmarks(
            label_path=sample["label_path"],
            image_width=image_width,
            image_height=image_height,
        )
        if gt_orientation not in orientation_sample_counts:
            orientation_names.append(gt_orientation)
            orientation_to_errors_68[gt_orientation] = [[] for _ in range(68)]
            orientation_to_errors_72[gt_orientation] = [[] for _ in range(72)]
            orientation_to_box_nme_values[gt_orientation] = []
            orientation_to_box_nme_point_to_line_values[gt_orientation] = []
            orientation_to_hausdorff_box_values[gt_orientation] = []
            orientation_to_box_nme_gt_valid_values[gt_orientation] = []
            orientation_to_box_nme_point_to_line_gt_valid_values[gt_orientation] = []
            orientation_to_hausdorff_box_gt_valid_values[gt_orientation] = []
            orientation_sample_counts[gt_orientation] = 0
            if gt_orientation.startswith("yaw_"):
                yaw_text = gt_orientation.removeprefix("yaw_").removesuffix("deg")
                yaw_text = yaw_text.replace("plus_", "+").replace("minus_", "-")
                orientation_display_labels[gt_orientation] = f"{yaw_text}°"
            else:
                orientation_display_labels[gt_orientation] = gt_orientation
        orientation_sample_counts[gt_orientation] += 1

        candidate_prediction_paths = prediction_paths_by_gt_stem.get(sample_id, [])
        if not candidate_prediction_paths:
            images_without_prediction += 1
            per_image_nme.append(
                {
                    "sample_id": sample_id,
                    "orientation": gt_orientation,
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
                    "hausdorff_pixel_visible_intersection": None,
                    "hausdorff_box_visible_intersection": None,
                    "mean_nme_box_gt_valid": None,
                    "mean_nme_box_point_to_line_gt_valid": None,
                    "hausdorff_pixel_gt_valid": None,
                    "hausdorff_box_gt_valid": None,
                    "mean_nme_interocular": None,
                }
            )
            continue

        sample_has_valid_prediction = False
        for prediction_path in candidate_prediction_paths:
            try:
                parsed_prediction = load_prediction_file(
                    prediction_path=prediction_path,
                    expected_num_landmarks=inferred_num_landmarks,
                )
            except Exception as error:
                images_with_invalid_prediction += 1
                invalid_prediction_files.append(
                    f"{prediction_path.stem} -> {sample_id}: {error}"
                )
                continue

            inferred_num_landmarks = parsed_prediction.num_landmarks
            sample_has_valid_prediction = True

            landmark_count = parsed_prediction.num_landmarks
            gt_landmarks = gt_landmarks_all[:landmark_count]
            gt_visibility = gt_visibility_all[:landmark_count]
            gt_visible_mask = gt_visibility == 1
            if parsed_prediction.visibility is not None:
                pred_visibility = parsed_prediction.visibility[:landmark_count]
                valid_mask = gt_visible_mask & (pred_visibility == 1)
                all_visibility_targets.append(gt_visibility.astype(np.int64))
                all_visibility_predictions.append(pred_visibility.astype(np.int64))
                all_visibility_pose_labels.append(
                    np.repeat(gt_orientation, landmark_count)
                )
                all_visibility_landmark_indices.append(
                    np.arange(landmark_count, dtype=np.int64)
                )
            else:
                pred_visibility = None
                valid_mask = gt_visible_mask
            gt_valid_mask = np.isfinite(gt_landmarks[:, 0]) & np.isfinite(
                gt_landmarks[:, 1]
            )
            gt_landmarks_visible_mode = gt_landmarks.astype(np.float32, copy=True)
            gt_landmarks_visible_mode[~gt_visible_mask] = 0.0
            per_landmark_errors = (
                per_landmark_errors_68
                if landmark_count == 68
                else per_landmark_errors_72
            )
            per_landmark_point_to_line_errors = (
                per_landmark_point_to_line_errors_68
                if landmark_count == 68
                else per_landmark_point_to_line_errors_72
            )
            orientation_to_errors = (
                orientation_to_errors_68
                if landmark_count == 68
                else orientation_to_errors_72
            )

            (
                visible_errors,
                visible_point_to_line_errors,
                mean_box_nme,
                mean_box_nme_point_to_line,
            ) = compute_masked_per_landmark_metrics(
                predicted_landmarks=parsed_prediction.landmarks,
                target_landmarks=gt_landmarks_visible_mode,
                valid_mask=valid_mask,
            )
            (
                hausdorff_pixel_visible,
                hausdorff_box_visible,
            ) = compute_normalized_hausdorff_distance(
                predicted_landmarks=parsed_prediction.landmarks,
                target_landmarks=gt_landmarks_visible_mode,
                valid_mask=valid_mask,
            )
            (
                gt_valid_errors,
                gt_valid_point_to_line_errors,
                mean_box_nme_gt_valid,
                mean_box_nme_point_to_line_gt_valid,
            ) = compute_masked_per_landmark_metrics(
                predicted_landmarks=parsed_prediction.landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=gt_valid_mask,
            )
            (
                hausdorff_pixel_gt_valid,
                hausdorff_box_gt_valid,
            ) = compute_normalized_hausdorff_distance(
                predicted_landmarks=parsed_prediction.landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=gt_valid_mask,
            )
            gt_valid_landmarks_used += len(gt_valid_errors)
            if mean_box_nme_gt_valid is not None:
                gt_valid_image_nme_values.append(mean_box_nme_gt_valid)
            if mean_box_nme_point_to_line_gt_valid is not None:
                gt_valid_image_point_to_line_values.append(
                    mean_box_nme_point_to_line_gt_valid
                )
            if np.isfinite(hausdorff_box_gt_valid):
                gt_valid_image_hausdorff_values.append(hausdorff_box_gt_valid)
                orientation_to_hausdorff_box_gt_valid_values[gt_orientation].append(
                    hausdorff_box_gt_valid
                )
            valid_landmarks_used += len(visible_errors)
            if np.isfinite(hausdorff_box_visible):
                visible_image_hausdorff_values.append(hausdorff_box_visible)
                orientation_to_hausdorff_box_values[gt_orientation].append(
                    hausdorff_box_visible
                )
            for landmark_index, error_value in visible_errors.items():
                per_landmark_errors[landmark_index].append(error_value)
            for landmark_index, error_value in visible_point_to_line_errors.items():
                per_landmark_point_to_line_errors[landmark_index].append(error_value)
            for landmark_index, error_value in visible_errors.items():
                pred_visibility_value = (
                    int(parsed_prediction.visibility[landmark_index])
                    if parsed_prediction.visibility is not None
                    else None
                )
                per_image_per_landmark_nme.append(
                    {
                        "image_id": sample_id,
                        "prediction_id": prediction_path.stem,
                        "evaluation_mode": "sota",
                        "split": "benchmark",
                        "orientation": gt_orientation,
                        "class_idx": gt_class_idx,
                        "landmark_idx": int(landmark_index),
                        "point_to_point_nme_box": float(error_value),
                        "point_to_line_nme_box": float(
                            visible_point_to_line_errors[landmark_index]
                        ),
                        "evaluation_landmark_inclusion": "visible_intersection",
                        "gt_visibility": int(gt_visibility[landmark_index]),
                        "pred_visibility": pred_visibility_value,
                        "landmark_count": int(landmark_count),
                    }
                )
            for landmark_index, error_value in gt_valid_errors.items():
                pred_visibility_value = (
                    int(parsed_prediction.visibility[landmark_index])
                    if parsed_prediction.visibility is not None
                    else None
                )
                per_image_per_landmark_nme.append(
                    {
                        "image_id": sample_id,
                        "prediction_id": prediction_path.stem,
                        "evaluation_mode": "sota",
                        "split": "benchmark",
                        "orientation": gt_orientation,
                        "class_idx": gt_class_idx,
                        "landmark_idx": int(landmark_index),
                        "point_to_point_nme_box": float(error_value),
                        "point_to_line_nme_box": float(
                            gt_valid_point_to_line_errors[landmark_index]
                        ),
                        "evaluation_landmark_inclusion": "gt_valid",
                        "gt_visibility": int(gt_visibility[landmark_index]),
                        "pred_visibility": pred_visibility_value,
                        "landmark_count": int(landmark_count),
                    }
                )
            for landmark_index, error_value in visible_errors.items():
                orientation_to_errors[gt_orientation][landmark_index].append(
                    error_value
                )
            if mean_box_nme is not None:
                orientation_to_box_nme_values[gt_orientation].append(mean_box_nme)
            if mean_box_nme_point_to_line is not None:
                orientation_to_box_nme_point_to_line_values[gt_orientation].append(
                    mean_box_nme_point_to_line
                )
            if mean_box_nme_gt_valid is not None:
                orientation_to_box_nme_gt_valid_values[gt_orientation].append(
                    mean_box_nme_gt_valid
                )
            if mean_box_nme_point_to_line_gt_valid is not None:
                orientation_to_box_nme_point_to_line_gt_valid_values[
                    gt_orientation
                ].append(mean_box_nme_point_to_line_gt_valid)

            per_image_nme.append(
                {
                    "sample_id": prediction_path.stem,
                    "orientation": gt_orientation,
                    "mean_nme_box": mean_box_nme,
                    "mean_nme_box_point_to_line": mean_box_nme_point_to_line,
                    "hausdorff_pixel_visible_intersection": hausdorff_pixel_visible,
                    "hausdorff_box_visible_intersection": hausdorff_box_visible,
                    "mean_nme_box_gt_valid": mean_box_nme_gt_valid,
                    "mean_nme_box_point_to_line_gt_valid": mean_box_nme_point_to_line_gt_valid,
                    "hausdorff_pixel_gt_valid": hausdorff_pixel_gt_valid,
                    "hausdorff_box_gt_valid": hausdorff_box_gt_valid,
                    "mean_nme_interocular": None,
                }
            )

        if sample_has_valid_prediction:
            images_with_prediction += 1
        else:
            images_without_prediction += 1
            per_image_nme.append(
                {
                    "sample_id": sample_id,
                    "orientation": gt_orientation,
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
                    "hausdorff_pixel_visible_intersection": None,
                    "hausdorff_box_visible_intersection": None,
                    "mean_nme_box_gt_valid": None,
                    "mean_nme_box_point_to_line_gt_valid": None,
                    "hausdorff_pixel_gt_valid": None,
                    "hausdorff_box_gt_valid": None,
                    "mean_nme_interocular": None,
                }
            )

    if inferred_num_landmarks is None:
        raise RuntimeError(
            "Could not infer the model landmark format because no valid prediction file was found."
        )
    orientation_names = sorted(
        orientation_names,
        key=lambda name: (
            ORIENTATION_ORDER.index(name)
            if name in ORIENTATION_ORDER
            else (
                float(
                    name.removeprefix("yaw_")
                    .removesuffix("deg")
                    .replace("plus_", "+")
                    .replace("minus_", "-")
                )
                if name.startswith("yaw_")
                else float("inf")
            )
        ),
    )

    selected_per_landmark_errors = (
        per_landmark_errors_68
        if inferred_num_landmarks == 68
        else per_landmark_errors_72
    )
    selected_per_landmark_point_to_line_errors = (
        per_landmark_point_to_line_errors_68
        if inferred_num_landmarks == 68
        else per_landmark_point_to_line_errors_72
    )
    selected_orientation_to_errors = (
        orientation_to_errors_68
        if inferred_num_landmarks == 68
        else orientation_to_errors_72
    )

    valid_image_nme_values = [
        row["mean_nme_box"] for row in per_image_nme if row["mean_nme_box"] is not None
    ]
    valid_image_point_to_line_values = [
        row["mean_nme_box_point_to_line"]
        for row in per_image_nme
        if row["mean_nme_box_point_to_line"] is not None
    ]

    save_per_landmark_nme_csv(
        per_landmark_errors=selected_per_landmark_errors,
        output_path=output_dir / "per_landmark_nme.csv",
    )
    save_per_landmark_nme_csv(
        per_landmark_errors=selected_per_landmark_point_to_line_errors,
        output_path=output_dir / "per_landmark_nme_point_to_line.csv",
    )
    save_per_image_nme_csv(
        per_image_nme=per_image_nme,
        output_path=output_dir / "per_image_nme.csv",
    )
    save_per_image_per_landmark_nme_csv(
        rows=per_image_per_landmark_nme,
        output_path=output_dir / "per_image_per_landmark_nme.csv",
    )

    if all_visibility_targets:
        visibility_analysis = compute_visibility_analysis(
            targets=np.concatenate(all_visibility_targets),
            predictions=np.concatenate(all_visibility_predictions),
            pose_labels=np.concatenate(all_visibility_pose_labels),
            landmark_indices=np.concatenate(all_visibility_landmark_indices),
            pose_display_labels=orientation_display_labels,
        )
        general_visibility = visibility_analysis["general"]
        plot_confusion_matrix(
            matrix=np.asarray(general_visibility["confusion_matrix_raw"]),
            output_path=figures_dir / "confusion_matrix_raw.png",
            title="Visibility confusion matrix",
            value_format="d",
        )
        plot_confusion_matrix(
            matrix=np.asarray(general_visibility["confusion_matrix_normalized"]),
            output_path=figures_dir / "confusion_matrix_normalized.png",
            title="Visibility confusion matrix normalized",
            value_format=".4f",
        )
    else:
        visibility_analysis = compute_visibility_analysis(
            targets=np.asarray([], dtype=np.int64),
            predictions=np.asarray([], dtype=np.int64),
            pose_labels=np.asarray([], dtype=str),
            landmark_indices=np.asarray([], dtype=np.int64),
        )
        visibility_analysis["reason"] = (
            "Prediction files do not contain visibility labels (68x2 format)."
        )
    save_visibility_metrics_csv(
        output_path=output_dir / "visibility_metrics.csv",
        analysis=visibility_analysis,
    )
    save_visibility_plots(
        figures_dir,
        visibility_analysis,
        include_babyland_region_protocols=inferred_num_landmarks == 72,
    )

    if any(values for values in selected_per_landmark_errors):
        grouped_errors = [selected_per_landmark_errors] + list(
            selected_orientation_to_errors.values()
        )
        title = _build_boxplot_title(
            label=f"Benchmark ({inferred_num_landmarks} landmarks)",
            mean_nme_box=(
                float(np.mean(valid_image_nme_values))
                if valid_image_nme_values
                else None
            ),
        )
        y_limits_log = (
            fixed_log_y_limits
            if fixed_log_y_limits is not None
            else compute_global_log_y_limits(grouped_errors)
        )
        clip_log_for_comparison = fixed_log_y_limits is not None
        clipping_note = (
            f"Values clipped to [{y_limits_log[0]:.0e}, {y_limits_log[1]:.0e}] for display only"
            if clip_log_for_comparison
            else None
        )
        orientation_metrics = {
            orientation: {
                "mean_nme_box_visible_intersection": (
                    float(np.mean(values)) if values else None
                ),
                "median_nme_box_visible_intersection": (
                    float(np.median(values)) if values else None
                ),
                "mean_nme_box_point_to_line_visible_intersection": (
                    float(
                        np.mean(
                            orientation_to_box_nme_point_to_line_values[orientation]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_values[orientation]
                    else None
                ),
                "median_nme_box_point_to_line_visible_intersection": (
                    float(
                        np.median(
                            orientation_to_box_nme_point_to_line_values[orientation]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_values[orientation]
                    else None
                ),
                "mean_hausdorff_box_visible_intersection": (
                    float(np.mean(orientation_to_hausdorff_box_values[orientation]))
                    if orientation_to_hausdorff_box_values[orientation]
                    else None
                ),
                "median_hausdorff_box_visible_intersection": (
                    float(np.median(orientation_to_hausdorff_box_values[orientation]))
                    if orientation_to_hausdorff_box_values[orientation]
                    else None
                ),
                "mean_nme_box_gt_valid": (
                    float(np.mean(orientation_to_box_nme_gt_valid_values[orientation]))
                    if orientation_to_box_nme_gt_valid_values[orientation]
                    else None
                ),
                "median_nme_box_gt_valid": (
                    float(
                        np.median(orientation_to_box_nme_gt_valid_values[orientation])
                    )
                    if orientation_to_box_nme_gt_valid_values[orientation]
                    else None
                ),
                "mean_nme_box_point_to_line_gt_valid": (
                    float(
                        np.mean(
                            orientation_to_box_nme_point_to_line_gt_valid_values[
                                orientation
                            ]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_gt_valid_values[orientation]
                    else None
                ),
                "median_nme_box_point_to_line_gt_valid": (
                    float(
                        np.median(
                            orientation_to_box_nme_point_to_line_gt_valid_values[
                                orientation
                            ]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_gt_valid_values[orientation]
                    else None
                ),
                "mean_hausdorff_box_gt_valid": (
                    float(
                        np.mean(
                            orientation_to_hausdorff_box_gt_valid_values[orientation]
                        )
                    )
                    if orientation_to_hausdorff_box_gt_valid_values[orientation]
                    else None
                ),
                "median_hausdorff_box_gt_valid": (
                    float(
                        np.median(
                            orientation_to_hausdorff_box_gt_valid_values[orientation]
                        )
                    )
                    if orientation_to_hausdorff_box_gt_valid_values[orientation]
                    else None
                ),
                "mean_nme_interocular": None,
            }
            for orientation, values in orientation_to_box_nme_values.items()
        }
        y_limits_linear = compute_global_linear_y_limits(grouped_errors)
        plot_yaw_view_boxplots(
            orientation_to_errors=selected_orientation_to_errors,
            output_dir=figures_dir,
            use_landmark_names=use_landmark_names_in_boxplot,
            y_limits=y_limits_log,
            y_scale="log",
            filename_suffix="log",
            orientation_metrics=orientation_metrics,
            clip_values_to_y_limits=clip_log_for_comparison,
            clipping_note=clipping_note,
            ordered_orientations=orientation_names,
            display_labels=orientation_display_labels,
            filename_labels={
                orientation: orientation for orientation in orientation_names
            },
        )
        plot_yaw_view_boxplots(
            orientation_to_errors=selected_orientation_to_errors,
            output_dir=figures_dir,
            use_landmark_names=use_landmark_names_in_boxplot,
            y_limits=y_limits_linear,
            y_scale="linear",
            filename_suffix="linear",
            orientation_metrics=orientation_metrics,
            ordered_orientations=orientation_names,
            display_labels=orientation_display_labels,
            filename_labels={
                orientation: orientation for orientation in orientation_names
            },
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=selected_per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_log.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=y_limits_log,
            y_scale="log",
            clip_values_to_y_limits=clip_log_for_comparison,
            clipping_note=clipping_note,
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=selected_per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_linear.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=y_limits_linear,
            y_scale="linear",
        )
    else:
        orientation_metrics = {
            orientation: {
                "mean_nme_box_visible_intersection": None,
                "median_nme_box_visible_intersection": None,
                "mean_nme_box_point_to_line_visible_intersection": None,
                "median_nme_box_point_to_line_visible_intersection": None,
                "mean_hausdorff_box_visible_intersection": None,
                "median_hausdorff_box_visible_intersection": None,
                "mean_nme_box_gt_valid": None,
                "median_nme_box_gt_valid": None,
                "mean_nme_box_point_to_line_gt_valid": None,
                "median_nme_box_point_to_line_gt_valid": None,
                "mean_hausdorff_box_gt_valid": None,
                "median_hausdorff_box_gt_valid": None,
                "mean_nme_interocular": None,
            }
            for orientation in orientation_names
        }

    summary = {
        "model_name": provided_prediction_root.name,
        "prediction_root": str(provided_prediction_root),
        "prediction_labels_dir": str(prediction_labels_dir),
        "model_landmark_format": int(inferred_num_landmarks),
        "evaluated_landmark_count": int(inferred_num_landmarks),
        "total_images": int(len(samples)),
        "images_with_prediction": int(images_with_prediction),
        "images_without_prediction": int(images_without_prediction),
        "images_with_invalid_prediction": int(images_with_invalid_prediction),
        "unmatched_prediction_files_count": int(len(unmatched_prediction_files)),
        "detection_rate": float(images_with_prediction / max(len(samples), 1)),
        **visibility_summary_fields(visibility_analysis),
        "mean_nme_box_visible_intersection": (
            float(np.mean(valid_image_nme_values)) if valid_image_nme_values else None
        ),
        "median_nme_box_visible_intersection": (
            float(np.median(valid_image_nme_values)) if valid_image_nme_values else None
        ),
        "mean_nme_box_point_to_line_visible_intersection": (
            float(np.mean(valid_image_point_to_line_values))
            if valid_image_point_to_line_values
            else None
        ),
        "median_nme_box_point_to_line_visible_intersection": (
            float(np.median(valid_image_point_to_line_values))
            if valid_image_point_to_line_values
            else None
        ),
        **prefixed_summary(
            "hausdorff_box_visible_intersection",
            visible_image_hausdorff_values,
        ),
        "mean_nme_box_gt_valid": (
            float(np.mean(gt_valid_image_nme_values))
            if gt_valid_image_nme_values
            else None
        ),
        "median_nme_box_gt_valid": (
            float(np.median(gt_valid_image_nme_values))
            if gt_valid_image_nme_values
            else None
        ),
        "mean_nme_box_point_to_line_gt_valid": (
            float(np.mean(gt_valid_image_point_to_line_values))
            if gt_valid_image_point_to_line_values
            else None
        ),
        "median_nme_box_point_to_line_gt_valid": (
            float(np.median(gt_valid_image_point_to_line_values))
            if gt_valid_image_point_to_line_values
            else None
        ),
        **prefixed_summary("hausdorff_box_gt_valid", gt_valid_image_hausdorff_values),
        "valid_landmarks_used": int(valid_landmarks_used),
        "gt_valid_landmarks_used": int(gt_valid_landmarks_used),
        "evaluation_modes": {
            "visible_intersection": "gt_visibility == 1 and pred_visibility == 1 when prediction visibility is available; otherwise GT-visible landmarks",
            "gt_valid": "finite GT coordinates, regardless of predicted visibility",
        },
        "invalid_prediction_files": invalid_prediction_files,
        "unmatched_prediction_files": unmatched_prediction_files,
        "orientation_sample_counts": {
            orientation: int(count)
            for orientation, count in orientation_sample_counts.items()
            if count > 0 or orientation != UNKNOWN_ORIENTATION
        },
        "orientation_metrics": {
            orientation: metrics
            for orientation, metrics in orientation_metrics.items()
            if orientation != UNKNOWN_ORIENTATION
            or orientation_sample_counts.get(orientation, 0) > 0
        },
        "comparable_log_boxplot": {
            "y_limits": list(fixed_log_y_limits)
            if fixed_log_y_limits is not None
            else None,
            "values_clipped_for_visualization_only": bool(
                fixed_log_y_limits is not None
            ),
        },
    }

    summary = round_metric_value(summary)

    save_benchmark_summary_csv(
        output_path=output_dir / "metrics_summary.csv",
        summary=summary,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def benchmark_infantface_prediction_directory(
    gt_root: str | Path,
    prediction_root: str | Path,
    output_dir: str | Path,
    use_landmark_names_in_boxplot: bool = True,
    fixed_log_y_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Benchmark one prediction directory against class-aware InfantFace 68-point GT."""
    provided_gt_root, gt_labels_dir = resolve_text_labels_dir(
        gt_root, labels_subdir_name="labels"
    )
    provided_prediction_root, prediction_labels_dir = resolve_prediction_labels_dir(
        prediction_root
    )
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    gt_paths = sorted(gt_labels_dir.glob("*.txt"))
    if not gt_paths:
        raise RuntimeError(f"No InfantFace GT txt files found under {gt_labels_dir}.")

    gt_paths_by_stem = {gt_path.stem: gt_path for gt_path in gt_paths}
    available_gt_stems = set(gt_paths_by_stem)
    prediction_paths = sorted(prediction_labels_dir.glob("*.txt"))
    prediction_paths_by_gt_stem: dict[str, list[Path]] = {
        gt_stem: [] for gt_stem in gt_paths_by_stem
    }
    unmatched_prediction_files: list[str] = []

    for prediction_path in prediction_paths:
        resolved_gt_stem = resolve_gt_stem_for_prediction(
            prediction_stem=prediction_path.stem,
            available_gt_stems=available_gt_stems,
        )
        if resolved_gt_stem is None:
            unmatched_prediction_files.append(
                f"{prediction_path.name}: could not resolve a GT file by exact stem "
                "or by stripping a trailing '__det_<index>' suffix."
            )
            continue
        prediction_paths_by_gt_stem[resolved_gt_stem].append(prediction_path)

    for current_prediction_paths in prediction_paths_by_gt_stem.values():
        current_prediction_paths.sort(
            key=lambda path: (
                1 if strip_detector_export_suffix(path.stem) is not None else 0,
                path.name,
            )
        )

    per_landmark_errors: list[list[float]] = [[] for _ in range(68)]
    per_landmark_point_to_line_errors: list[list[float]] = [[] for _ in range(68)]
    orientation_names = [*NATURAL_ORIENTATION_NAMES, UNKNOWN_ORIENTATION]
    orientation_to_errors: dict[str, list[list[float]]] = {
        orientation: [[] for _ in range(68)] for orientation in orientation_names
    }
    orientation_to_box_nme_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_to_box_nme_point_to_line_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_to_hausdorff_box_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_to_box_nme_non_contour_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_to_box_nme_point_to_line_non_contour_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_to_hausdorff_box_non_contour_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_sample_counts = {orientation: 0 for orientation in orientation_names}
    per_image_nme: list[dict[str, Any]] = []
    per_image_per_landmark_nme: list[dict[str, Any]] = []
    invalid_prediction_files: list[str] = []
    inferred_num_landmarks: int | None = None
    images_with_prediction = 0
    images_without_prediction = 0
    images_with_invalid_prediction = 0
    valid_landmarks_used = 0
    valid_non_contour_landmarks_used = 0
    non_contour_image_nme_values: list[float] = []
    non_contour_image_point_to_line_values: list[float] = []
    image_hausdorff_values: list[float] = []
    non_contour_image_hausdorff_values: list[float] = []

    for gt_path in gt_paths:
        sample_id = gt_path.stem
        (
            gt_landmarks,
            gt_class_idx,
            gt_orientation,
        ) = load_infantface_ground_truth_landmarks(
            gt_path,
        )
        if gt_orientation not in orientation_sample_counts:
            gt_orientation = UNKNOWN_ORIENTATION
        orientation_sample_counts[gt_orientation] += 1

        candidate_prediction_paths = prediction_paths_by_gt_stem.get(sample_id, [])
        if not candidate_prediction_paths:
            images_without_prediction += 1
            per_image_nme.append(
                {
                    "sample_id": sample_id,
                    "orientation": gt_orientation,
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
                    "hausdorff_pixel": None,
                    "hausdorff_box": None,
                    "hausdorff_pixel_non_contour": None,
                    "hausdorff_box_non_contour": None,
                    "mean_nme_interocular": None,
                }
            )
            continue

        sample_has_valid_prediction = False
        for prediction_path in candidate_prediction_paths:
            try:
                parsed_prediction = load_prediction_file(
                    prediction_path=prediction_path,
                    expected_num_landmarks=inferred_num_landmarks,
                )
            except Exception as error:
                images_with_invalid_prediction += 1
                invalid_prediction_files.append(
                    f"{prediction_path.stem} -> {sample_id}: {error}"
                )
                continue

            inferred_num_landmarks = parsed_prediction.num_landmarks
            sample_has_valid_prediction = True

            predicted_landmarks = parsed_prediction.landmarks[:68].astype(np.float32)
            valid_mask = np.ones(68, dtype=bool)

            (
                current_errors,
                current_point_to_line_errors,
                mean_box_nme,
                mean_box_nme_point_to_line,
            ) = compute_masked_per_landmark_metrics(
                predicted_landmarks=predicted_landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=valid_mask,
            )
            hausdorff_pixel, hausdorff_box = compute_normalized_hausdorff_distance(
                predicted_landmarks=predicted_landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=valid_mask,
            )
            non_contour_mask = np.zeros(68, dtype=bool)
            non_contour_mask[17:68] = True
            (
                non_contour_errors,
                non_contour_point_to_line_errors,
                mean_box_nme_non_contour,
                mean_box_nme_point_to_line_non_contour,
            ) = compute_masked_per_landmark_metrics(
                predicted_landmarks=predicted_landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=non_contour_mask,
                normalization_landmarks=gt_landmarks,
            )
            (
                hausdorff_pixel_non_contour,
                hausdorff_box_non_contour,
            ) = compute_normalized_hausdorff_distance(
                predicted_landmarks=predicted_landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=non_contour_mask,
                normalization_landmarks=gt_landmarks,
            )
            valid_landmarks_used += len(current_errors)
            valid_non_contour_landmarks_used += len(non_contour_errors)
            if np.isfinite(hausdorff_box):
                image_hausdorff_values.append(hausdorff_box)
                orientation_to_hausdorff_box_values[gt_orientation].append(
                    hausdorff_box
                )
            if mean_box_nme_non_contour is not None:
                non_contour_image_nme_values.append(mean_box_nme_non_contour)
            if mean_box_nme_point_to_line_non_contour is not None:
                non_contour_image_point_to_line_values.append(
                    mean_box_nme_point_to_line_non_contour
                )
            if np.isfinite(hausdorff_box_non_contour):
                non_contour_image_hausdorff_values.append(hausdorff_box_non_contour)
                orientation_to_hausdorff_box_non_contour_values[gt_orientation].append(
                    hausdorff_box_non_contour
                )
            if mean_box_nme_non_contour is not None:
                orientation_to_box_nme_non_contour_values[gt_orientation].append(
                    mean_box_nme_non_contour
                )
            if mean_box_nme_point_to_line_non_contour is not None:
                orientation_to_box_nme_point_to_line_non_contour_values[
                    gt_orientation
                ].append(mean_box_nme_point_to_line_non_contour)
            for landmark_index, error_value in current_errors.items():
                per_landmark_errors[landmark_index].append(error_value)
                orientation_to_errors[gt_orientation][landmark_index].append(
                    error_value
                )
            for landmark_index, error_value in current_point_to_line_errors.items():
                per_landmark_point_to_line_errors[landmark_index].append(error_value)
            if mean_box_nme is not None:
                orientation_to_box_nme_values[gt_orientation].append(mean_box_nme)
            if mean_box_nme_point_to_line is not None:
                orientation_to_box_nme_point_to_line_values[gt_orientation].append(
                    mean_box_nme_point_to_line
                )
            for landmark_index, error_value in current_errors.items():
                pred_visibility_value = (
                    int(parsed_prediction.visibility[landmark_index])
                    if parsed_prediction.visibility is not None
                    and landmark_index < len(parsed_prediction.visibility)
                    else None
                )
                per_image_per_landmark_nme.append(
                    {
                        "image_id": sample_id,
                        "prediction_id": prediction_path.stem,
                        "evaluation_mode": "infantface",
                        "split": "benchmark",
                        "orientation": gt_orientation,
                        "class_idx": gt_class_idx,
                        "landmark_idx": int(landmark_index),
                        "point_to_point_nme_box": float(error_value),
                        "point_to_line_nme_box": float(
                            current_point_to_line_errors[landmark_index]
                        ),
                        "gt_visibility": 1,
                        "pred_visibility": pred_visibility_value,
                        "landmark_count": 68,
                    }
                )

            per_image_nme.append(
                {
                    "sample_id": prediction_path.stem,
                    "orientation": gt_orientation,
                    "mean_nme_box": mean_box_nme,
                    "mean_nme_box_point_to_line": mean_box_nme_point_to_line,
                    "hausdorff_pixel": hausdorff_pixel,
                    "hausdorff_box": hausdorff_box,
                    "mean_nme_box_non_contour": mean_box_nme_non_contour,
                    "mean_nme_box_point_to_line_non_contour": mean_box_nme_point_to_line_non_contour,
                    "hausdorff_pixel_non_contour": hausdorff_pixel_non_contour,
                    "hausdorff_box_non_contour": hausdorff_box_non_contour,
                    "mean_nme_interocular": None,
                }
            )

        if sample_has_valid_prediction:
            images_with_prediction += 1
        else:
            images_without_prediction += 1
            per_image_nme.append(
                {
                    "sample_id": sample_id,
                    "orientation": gt_orientation,
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
                    "hausdorff_pixel": None,
                    "hausdorff_box": None,
                    "hausdorff_pixel_non_contour": None,
                    "hausdorff_box_non_contour": None,
                    "mean_nme_interocular": None,
                }
            )

    if inferred_num_landmarks is None:
        raise RuntimeError(
            "Could not infer the model landmark format because no valid prediction file was found."
        )

    valid_image_nme_values = [
        row["mean_nme_box"] for row in per_image_nme if row["mean_nme_box"] is not None
    ]
    valid_image_point_to_line_values = [
        row["mean_nme_box_point_to_line"]
        for row in per_image_nme
        if row["mean_nme_box_point_to_line"] is not None
    ]

    save_per_landmark_nme_csv(
        per_landmark_errors=per_landmark_errors,
        output_path=output_dir / "per_landmark_nme.csv",
    )
    save_per_landmark_nme_csv(
        per_landmark_errors=per_landmark_point_to_line_errors,
        output_path=output_dir / "per_landmark_nme_point_to_line.csv",
    )
    save_per_image_nme_csv(
        per_image_nme=per_image_nme,
        output_path=output_dir / "per_image_nme.csv",
    )
    save_per_image_per_landmark_nme_csv(
        rows=per_image_per_landmark_nme,
        output_path=output_dir / "per_image_per_landmark_nme.csv",
    )

    if any(values for values in per_landmark_errors):
        grouped_errors = [per_landmark_errors] + list(orientation_to_errors.values())
        title = _build_boxplot_title(
            label="InfantFace (68 landmarks)",
            mean_nme_box=(
                float(np.mean(valid_image_nme_values))
                if valid_image_nme_values
                else None
            ),
        )
        y_limits_log = (
            fixed_log_y_limits
            if fixed_log_y_limits is not None
            else compute_global_log_y_limits(grouped_errors)
        )
        clip_log_for_comparison = fixed_log_y_limits is not None
        clipping_note = (
            f"Values clipped to [{y_limits_log[0]:.0e}, {y_limits_log[1]:.0e}] for display only"
            if clip_log_for_comparison
            else None
        )
        orientation_metrics = {
            orientation: {
                "mean_nme_box": (float(np.mean(values)) if values else None),
                "median_nme_box": (float(np.median(values)) if values else None),
                "mean_nme_box_point_to_line": (
                    float(
                        np.mean(
                            orientation_to_box_nme_point_to_line_values[orientation]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_values[orientation]
                    else None
                ),
                "median_nme_box_point_to_line": (
                    float(
                        np.median(
                            orientation_to_box_nme_point_to_line_values[orientation]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_values[orientation]
                    else None
                ),
                "mean_hausdorff_box": (
                    float(np.mean(orientation_to_hausdorff_box_values[orientation]))
                    if orientation_to_hausdorff_box_values[orientation]
                    else None
                ),
                "median_hausdorff_box": (
                    float(np.median(orientation_to_hausdorff_box_values[orientation]))
                    if orientation_to_hausdorff_box_values[orientation]
                    else None
                ),
                "mean_nme_box_non_contour": (
                    float(
                        np.mean(orientation_to_box_nme_non_contour_values[orientation])
                    )
                    if orientation_to_box_nme_non_contour_values[orientation]
                    else None
                ),
                "median_nme_box_non_contour": (
                    float(
                        np.median(
                            orientation_to_box_nme_non_contour_values[orientation]
                        )
                    )
                    if orientation_to_box_nme_non_contour_values[orientation]
                    else None
                ),
                "mean_nme_box_point_to_line_non_contour": (
                    float(
                        np.mean(
                            orientation_to_box_nme_point_to_line_non_contour_values[
                                orientation
                            ]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_non_contour_values[
                        orientation
                    ]
                    else None
                ),
                "median_nme_box_point_to_line_non_contour": (
                    float(
                        np.median(
                            orientation_to_box_nme_point_to_line_non_contour_values[
                                orientation
                            ]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_non_contour_values[
                        orientation
                    ]
                    else None
                ),
                "mean_hausdorff_box_non_contour": (
                    float(
                        np.mean(
                            orientation_to_hausdorff_box_non_contour_values[orientation]
                        )
                    )
                    if orientation_to_hausdorff_box_non_contour_values[orientation]
                    else None
                ),
                "median_hausdorff_box_non_contour": (
                    float(
                        np.median(
                            orientation_to_hausdorff_box_non_contour_values[orientation]
                        )
                    )
                    if orientation_to_hausdorff_box_non_contour_values[orientation]
                    else None
                ),
                "mean_nme_interocular": None,
            }
            for orientation, values in orientation_to_box_nme_values.items()
        }
        y_limits_linear = compute_global_linear_y_limits(grouped_errors)
        plot_yaw_view_boxplots(
            orientation_to_errors=orientation_to_errors,
            output_dir=figures_dir,
            use_landmark_names=use_landmark_names_in_boxplot,
            y_limits=y_limits_log,
            y_scale="log",
            filename_suffix="log",
            orientation_metrics=orientation_metrics,
            clip_values_to_y_limits=clip_log_for_comparison,
            clipping_note=clipping_note,
        )
        plot_yaw_view_boxplots(
            orientation_to_errors=orientation_to_errors,
            output_dir=figures_dir,
            use_landmark_names=use_landmark_names_in_boxplot,
            y_limits=y_limits_linear,
            y_scale="linear",
            filename_suffix="linear",
            orientation_metrics=orientation_metrics,
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_log.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=y_limits_log,
            y_scale="log",
            clip_values_to_y_limits=clip_log_for_comparison,
            clipping_note=clipping_note,
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_linear.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=y_limits_linear,
            y_scale="linear",
        )
    else:
        orientation_metrics = {
            orientation: {
                "mean_nme_box": None,
                "median_nme_box": None,
                "mean_nme_box_point_to_line": None,
                "median_nme_box_point_to_line": None,
                "mean_hausdorff_box": None,
                "median_hausdorff_box": None,
                "mean_nme_box_non_contour": None,
                "median_nme_box_non_contour": None,
                "mean_nme_box_point_to_line_non_contour": None,
                "median_nme_box_point_to_line_non_contour": None,
                "mean_hausdorff_box_non_contour": None,
                "median_hausdorff_box_non_contour": None,
                "mean_nme_interocular": None,
            }
            for orientation in orientation_names
        }

    visibility_analysis = compute_visibility_analysis(
        targets=np.asarray([], dtype=np.int64),
        predictions=np.asarray([], dtype=np.int64),
        pose_labels=np.asarray([], dtype=str),
        landmark_indices=np.asarray([], dtype=np.int64),
    )
    visibility_analysis["reason"] = (
        "InfantFace ground-truth files do not contain visibility labels."
    )
    save_visibility_metrics_csv(
        output_path=output_dir / "visibility_metrics.csv",
        analysis=visibility_analysis,
    )

    summary = {
        "model_name": provided_prediction_root.name,
        "gt_root": str(provided_gt_root),
        "prediction_root": str(provided_prediction_root),
        "prediction_labels_dir": str(prediction_labels_dir),
        "model_landmark_format": int(inferred_num_landmarks),
        "evaluated_landmark_count": 68,
        "total_images": int(len(gt_paths)),
        "images_with_prediction": int(images_with_prediction),
        "images_without_prediction": int(images_without_prediction),
        "images_with_invalid_prediction": int(images_with_invalid_prediction),
        "unmatched_prediction_files_count": int(len(unmatched_prediction_files)),
        "detection_rate": float(images_with_prediction / max(len(gt_paths), 1)),
        **visibility_summary_fields(visibility_analysis),
        "mean_nme_box": (
            float(np.mean(valid_image_nme_values)) if valid_image_nme_values else None
        ),
        "median_nme_box": (
            float(np.median(valid_image_nme_values)) if valid_image_nme_values else None
        ),
        "mean_nme_box_point_to_line": (
            float(np.mean(valid_image_point_to_line_values))
            if valid_image_point_to_line_values
            else None
        ),
        "median_nme_box_point_to_line": (
            float(np.median(valid_image_point_to_line_values))
            if valid_image_point_to_line_values
            else None
        ),
        **prefixed_summary("hausdorff_box", image_hausdorff_values),
        "valid_landmarks_used": int(valid_landmarks_used),
        "valid_non_contour_landmarks_used": int(valid_non_contour_landmarks_used),
        "excluded_non_contour_landmarks": "0-16 excluded; 17-67 evaluated",
        "mean_nme_box_non_contour": (
            float(np.mean(non_contour_image_nme_values))
            if non_contour_image_nme_values
            else None
        ),
        "median_nme_box_non_contour": (
            float(np.median(non_contour_image_nme_values))
            if non_contour_image_nme_values
            else None
        ),
        "mean_nme_box_point_to_line_non_contour": (
            float(np.mean(non_contour_image_point_to_line_values))
            if non_contour_image_point_to_line_values
            else None
        ),
        "median_nme_box_point_to_line_non_contour": (
            float(np.median(non_contour_image_point_to_line_values))
            if non_contour_image_point_to_line_values
            else None
        ),
        **prefixed_summary(
            "hausdorff_box_non_contour",
            non_contour_image_hausdorff_values,
        ),
        "invalid_prediction_files": invalid_prediction_files,
        "unmatched_prediction_files": unmatched_prediction_files,
        "orientation_sample_counts": {
            orientation: int(count)
            for orientation, count in orientation_sample_counts.items()
            if count > 0 or orientation != UNKNOWN_ORIENTATION
        },
        "orientation_metrics": {
            orientation: metrics
            for orientation, metrics in orientation_metrics.items()
            if orientation != UNKNOWN_ORIENTATION
            or orientation_sample_counts.get(orientation, 0) > 0
        },
        "comparable_log_boxplot": {
            "y_limits": list(fixed_log_y_limits)
            if fixed_log_y_limits is not None
            else None,
            "values_clipped_for_visualization_only": bool(
                fixed_log_y_limits is not None
            ),
        },
    }

    summary = round_metric_value(summary)

    save_benchmark_summary_csv(
        output_path=output_dir / "metrics_summary.csv",
        summary=summary,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
