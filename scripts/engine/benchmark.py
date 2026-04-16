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
    save_per_image_nme_csv,
    save_per_landmark_nme_csv,
)
from .geometry_metrics import compute_per_landmark_point_to_line_distances
from ..utils.visualization import (
    compute_global_linear_y_limits,
    compute_global_log_y_limits,
    plot_per_landmark_boxplot,
)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DETECTOR_EXPORT_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+)__det_(?P<index>\d+)$")


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
) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized GT landmarks and convert them to absolute image coordinates."""
    label_path = Path(label_path)
    data = np.loadtxt(label_path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape != (72, 3):
        raise ValueError(
            f"Expected GT shape (72, 3) in {label_path}, got {tuple(data.shape)}."
        )

    landmarks = data[:, :2].copy()
    visibility = data[:, 2].astype(np.int64)
    landmarks[:, 0] *= float(image_width)
    landmarks[:, 1] *= float(image_height)
    invisible_mask = visibility == 0
    landmarks[invisible_mask] = 0.0
    return landmarks.astype(np.float32), visibility


def load_infantface_ground_truth_landmarks(
    label_path: str | Path,
) -> np.ndarray:
    """Load one InfantFace GT file containing 68 absolute landmark coordinates."""
    label_path = Path(label_path)
    data = np.loadtxt(label_path, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape != (68, 2):
        raise ValueError(
            f"Expected InfantFace GT shape (68, 2) in {label_path}, got {tuple(data.shape)}."
        )
    return data.astype(np.float32)


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
    eps: float = 1e-6,
) -> tuple[dict[int, float], dict[int, float], float | None, float | None]:
    """Compute point-to-point and point-to-line NME for one masked subset."""
    if valid_mask.sum() == 0:
        return {}, {}, None, None

    normalization = compute_box_normalization_factor(
        target_landmarks=target_landmarks[valid_mask],
        eps=eps,
    )
    point_errors = np.linalg.norm(predicted_landmarks - target_landmarks, axis=1)
    normalized_errors = point_errors / normalization
    point_to_line_errors = (
        compute_per_landmark_point_to_line_distances(
            predicted_landmarks=predicted_landmarks,
            target_landmarks=target_landmarks,
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
        ("mean_nme_box", summary.get("mean_nme_box")),
        ("median_nme_box", summary.get("median_nme_box")),
        ("mean_nme_box_point_to_line", summary.get("mean_nme_box_point_to_line")),
        (
            "median_nme_box_point_to_line",
            summary.get("median_nme_box_point_to_line"),
        ),
        ("valid_landmarks_used", summary.get("valid_landmarks_used")),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in rows:
            writer.writerow([metric_name, metric_value])


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
    per_image_nme: list[dict[str, Any]] = []
    invalid_prediction_files: list[str] = []
    inferred_num_landmarks: int | None = None
    images_with_prediction = 0
    images_without_prediction = 0
    images_with_invalid_prediction = 0
    valid_landmarks_used = 0

    for sample in samples:
        sample_id = sample["sample_id"]
        candidate_prediction_paths = prediction_paths_by_gt_stem.get(sample_id, [])
        if not candidate_prediction_paths:
            images_without_prediction += 1
            per_image_nme.append(
                {
                    "sample_id": sample_id,
                    "orientation": "benchmark",
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
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

            with Image.open(sample["image_path"]) as image:
                image_width, image_height = image.size

            gt_landmarks_all, gt_visibility_all = load_ground_truth_landmarks(
                label_path=sample["label_path"],
                image_width=image_width,
                image_height=image_height,
            )

            landmark_count = parsed_prediction.num_landmarks
            gt_landmarks = gt_landmarks_all[:landmark_count]
            gt_visibility = gt_visibility_all[:landmark_count]
            valid_mask = gt_visibility == 1
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

            (
                visible_errors,
                visible_point_to_line_errors,
                mean_box_nme,
                mean_box_nme_point_to_line,
            ) = compute_masked_per_landmark_metrics(
                predicted_landmarks=parsed_prediction.landmarks,
                target_landmarks=gt_landmarks,
                valid_mask=valid_mask,
            )
            valid_landmarks_used += len(visible_errors)
            for landmark_index, error_value in visible_errors.items():
                per_landmark_errors[landmark_index].append(error_value)
            for landmark_index, error_value in visible_point_to_line_errors.items():
                per_landmark_point_to_line_errors[landmark_index].append(error_value)

            per_image_nme.append(
                {
                    "sample_id": prediction_path.stem,
                    "orientation": "benchmark",
                    "mean_nme_box": mean_box_nme,
                    "mean_nme_box_point_to_line": mean_box_nme_point_to_line,
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
                    "orientation": "benchmark",
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
                    "mean_nme_interocular": None,
                }
            )

    if inferred_num_landmarks is None:
        raise RuntimeError(
            "Could not infer the model landmark format because no valid prediction file was found."
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

    if any(values for values in selected_per_landmark_errors):
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
            else compute_global_log_y_limits([selected_per_landmark_errors])
        )
        y_limits_linear = compute_global_linear_y_limits([selected_per_landmark_errors])
        plot_per_landmark_boxplot(
            per_landmark_errors=selected_per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_log.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=y_limits_log,
            y_scale="log",
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=selected_per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_linear.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=y_limits_linear,
            y_scale="linear",
        )

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
        "valid_landmarks_used": int(valid_landmarks_used),
        "invalid_prediction_files": invalid_prediction_files,
        "unmatched_prediction_files": unmatched_prediction_files,
    }

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
) -> dict[str, Any]:
    """Benchmark one prediction directory against InfantFace 68-point GT."""
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
    per_image_nme: list[dict[str, Any]] = []
    invalid_prediction_files: list[str] = []
    inferred_num_landmarks: int | None = None
    images_with_prediction = 0
    images_without_prediction = 0
    images_with_invalid_prediction = 0
    valid_landmarks_used = 0

    for gt_path in gt_paths:
        sample_id = gt_path.stem
        candidate_prediction_paths = prediction_paths_by_gt_stem.get(sample_id, [])
        if not candidate_prediction_paths:
            images_without_prediction += 1
            per_image_nme.append(
                {
                    "sample_id": sample_id,
                    "orientation": "infantface",
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
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

            gt_landmarks = load_infantface_ground_truth_landmarks(gt_path)
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
            valid_landmarks_used += len(current_errors)
            for landmark_index, error_value in current_errors.items():
                per_landmark_errors[landmark_index].append(error_value)
            for landmark_index, error_value in current_point_to_line_errors.items():
                per_landmark_point_to_line_errors[landmark_index].append(error_value)

            per_image_nme.append(
                {
                    "sample_id": prediction_path.stem,
                    "orientation": "infantface",
                    "mean_nme_box": mean_box_nme,
                    "mean_nme_box_point_to_line": mean_box_nme_point_to_line,
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
                    "orientation": "infantface",
                    "mean_nme_box": None,
                    "mean_nme_box_point_to_line": None,
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

    if any(values for values in per_landmark_errors):
        title = _build_boxplot_title(
            label="InfantFace (68 landmarks)",
            mean_nme_box=(
                float(np.mean(valid_image_nme_values))
                if valid_image_nme_values
                else None
            ),
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_log.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=compute_global_log_y_limits([per_landmark_errors]),
            y_scale="log",
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_linear.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=compute_global_linear_y_limits([per_landmark_errors]),
            y_scale="linear",
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
        "valid_landmarks_used": int(valid_landmarks_used),
        "invalid_prediction_files": invalid_prediction_files,
        "unmatched_prediction_files": unmatched_prediction_files,
    }

    save_benchmark_summary_csv(
        output_path=output_dir / "metrics_summary.csv",
        summary=summary,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
