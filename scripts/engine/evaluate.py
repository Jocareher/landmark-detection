from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .geometry_metrics import compute_per_landmark_point_to_line_distances
from .metrics import decode_heatmaps_to_image_coords
from .postprocessing import extract_batched_size, project_landmarks_to_original_size
from ..utils.predictions import save_prediction_file
from ..utils.visualization import (
    compute_global_linear_y_limits,
    compute_global_log_y_limits,
    get_default_landmark_names,
    get_landmark_anatomical_group,
    get_landmark_anatomical_label,
    plot_confusion_matrix,
    plot_grouped_nme_boxplot,
    plot_per_landmark_boxplot,
    plot_yaw_view_boxplots,
    save_landmark_overlay_image,
)
from ..utils.synthetic_labels import format_synthetic_yaw_group


def compute_box_normalization_factor(
    target_landmarks: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute the normalization factor based on the GT landmark bounding box.

    Parameters
    ----------
    target_landmarks : np.ndarray
        Ground-truth landmarks of shape (K, 2).
    eps : float, optional
        Numerical stability term. Defaults to 1e-6.

    Returns
    -------
    float
        Box-based normalization factor.
    """
    min_xy = target_landmarks.min(axis=0)
    max_xy = target_landmarks.max(axis=0)

    box_width = max_xy[0] - min_xy[0]
    box_height = max_xy[1] - min_xy[1]

    return float(np.sqrt(max(box_width * box_height, eps)))


def compute_per_landmark_nme(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute normalized error per landmark for one sample.

    Parameters
    ----------
    predicted_landmarks : np.ndarray
        Predicted landmarks of shape (K, 2).
    target_landmarks : np.ndarray
        Ground-truth landmarks of shape (K, 2).
    eps : float, optional
        Numerical stability term. Defaults to 1e-6.

    Returns
    -------
    np.ndarray
        Per-landmark normalized error of shape (K,).
    """
    normalization = compute_box_normalization_factor(target_landmarks, eps=eps)
    point_errors = np.linalg.norm(predicted_landmarks - target_landmarks, axis=1)
    return point_errors / normalization


def compute_per_landmark_point_to_line_nme(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute normalized point-to-line error per landmark for one sample."""
    normalization = compute_box_normalization_factor(target_landmarks, eps=eps)
    point_to_line_errors = compute_per_landmark_point_to_line_distances(
        predicted_landmarks=predicted_landmarks,
        target_landmarks=target_landmarks,
    )
    return point_to_line_errors / normalization


def compute_interocular_normalization_factor(
    target_landmarks: np.ndarray,
    left_eye_corner_index: int = 36,
    right_eye_corner_index: int = 45,
    eps: float = 1e-6,
) -> float:
    """Compute the interocular distance using the outer eye corners."""
    left_corner = target_landmarks[left_eye_corner_index]
    right_corner = target_landmarks[right_eye_corner_index]
    return float(max(np.linalg.norm(right_corner - left_corner), eps))


def compute_per_landmark_interocular_nme(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute per-landmark NME normalized by interocular distance."""
    normalization = compute_interocular_normalization_factor(
        target_landmarks=target_landmarks,
        eps=eps,
    )
    point_errors = np.linalg.norm(predicted_landmarks - target_landmarks, axis=1)
    return point_errors / normalization


def compute_binary_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Compute the binary confusion matrix.

    Parameters
    ----------
    targets : np.ndarray
        Ground-truth binary labels of shape (N,).
    predictions : np.ndarray
        Predicted binary labels of shape (N,).

    Returns
    -------
    np.ndarray
        Confusion matrix with shape (2, 2).
    """
    targets = targets.astype(np.int64).reshape(-1)
    predictions = predictions.astype(np.int64).reshape(-1)

    true_negative = int(((targets == 0) & (predictions == 0)).sum())
    false_positive = int(((targets == 0) & (predictions == 1)).sum())
    false_negative = int(((targets == 1) & (predictions == 0)).sum())
    true_positive = int(((targets == 1) & (predictions == 1)).sum())

    return np.array(
        [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        dtype=np.int64,
    )


def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Row-normalize a confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Raw confusion matrix of shape (2, 2).

    Returns
    -------
    np.ndarray
        Row-normalized confusion matrix of shape (2, 2).
    """
    row_sums = confusion_matrix.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0.0] = 1.0
    return confusion_matrix.astype(np.float64) / row_sums


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a stable ratio for metric computation."""
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _compute_f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall with zero-division protection."""
    return _safe_divide(2.0 * precision * recall, precision + recall)


def compute_visibility_classification_metrics(
    confusion_matrix: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute visibility precision, recall, and F1 from a binary confusion matrix."""
    true_negative, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]

    visible_precision = _safe_divide(true_positive, true_positive + false_positive)
    visible_recall = _safe_divide(true_positive, true_positive + false_negative)
    visible_f1 = _compute_f1(visible_precision, visible_recall)

    invisible_precision = _safe_divide(true_negative, true_negative + false_negative)
    invisible_recall = _safe_divide(true_negative, true_negative + false_positive)
    invisible_f1 = _compute_f1(invisible_precision, invisible_recall)

    return {
        "global": {
            "precision": (visible_precision + invisible_precision) / 2.0,
            "recall": (visible_recall + invisible_recall) / 2.0,
            "f1": (visible_f1 + invisible_f1) / 2.0,
        },
        "visible": {
            "precision": visible_precision,
            "recall": visible_recall,
            "f1": visible_f1,
        },
        "invisible": {
            "precision": invisible_precision,
            "recall": invisible_recall,
            "f1": invisible_f1,
        },
    }


def round_metric_value(value: Any, decimals: int = 4) -> Any:
    """Round float-like metric values recursively while preserving counts and paths."""
    if isinstance(value, dict):
        return {
            key: round_metric_value(nested_value, decimals=decimals)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [
            round_metric_value(nested_value, decimals=decimals)
            for nested_value in value
        ]
    if isinstance(value, tuple):
        return tuple(
            round_metric_value(nested_value, decimals=decimals)
            for nested_value in value
        )
    if isinstance(value, np.ndarray):
        return round_metric_value(value.tolist(), decimals=decimals)
    if isinstance(value, (np.floating, float)):
        return round(float(value), decimals)
    return value


def format_metric_value(value: Any, decimals: int = 4) -> Any:
    """Format float-like metric values for fixed-decimal CSV/terminal output."""
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.{decimals}f}"
    return value


def save_metrics_summary_csv(
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    """
    Save a unified evaluation summary CSV.

    Parameters
    ----------
    output_path : Path
        Destination CSV path.
    summary : dict[str, Any]
        Evaluation summary dictionary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Any]] = [
        ("num_samples", summary.get("num_samples")),
        (
            "num_samples_with_geometric_metrics",
            summary.get("num_samples_with_geometric_metrics"),
        ),
        (
            "num_visible_visible_landmarks",
            summary.get("num_visible_visible_landmarks"),
        ),
        ("num_landmarks", summary.get("num_landmarks")),
        ("mean_nme_box", summary.get("mean_nme_box")),
        ("median_nme_box", summary.get("median_nme_box")),
        ("mean_nme_box_point_to_line", summary.get("mean_nme_box_point_to_line")),
        (
            "median_nme_box_point_to_line",
            summary.get("median_nme_box_point_to_line"),
        ),
        ("mean_nme_interocular", summary.get("mean_nme_interocular")),
        ("landmark_loss", summary.get("landmark_loss")),
        ("coordinate_decoder", summary.get("coordinate_decoder")),
        (
            "visibility_global_precision",
            summary.get("visibility_metrics", {}).get("global", {}).get("precision"),
        ),
        (
            "visibility_global_recall",
            summary.get("visibility_metrics", {}).get("global", {}).get("recall"),
        ),
        (
            "visibility_global_f1",
            summary.get("visibility_metrics", {}).get("global", {}).get("f1"),
        ),
        (
            "visibility_visible_precision",
            summary.get("visibility_metrics", {}).get("visible", {}).get("precision"),
        ),
        (
            "visibility_visible_recall",
            summary.get("visibility_metrics", {}).get("visible", {}).get("recall"),
        ),
        (
            "visibility_visible_f1",
            summary.get("visibility_metrics", {}).get("visible", {}).get("f1"),
        ),
        (
            "visibility_invisible_precision",
            summary.get("visibility_metrics", {}).get("invisible", {}).get("precision"),
        ),
        (
            "visibility_invisible_recall",
            summary.get("visibility_metrics", {}).get("invisible", {}).get("recall"),
        ),
        (
            "visibility_invisible_f1",
            summary.get("visibility_metrics", {}).get("invisible", {}).get("f1"),
        ),
        ("visibility_threshold", summary.get("visibility_threshold")),
    ]

    confusion_matrix_raw = summary.get("confusion_matrix_raw")
    confusion_matrix_normalized = summary.get("confusion_matrix_normalized")

    if confusion_matrix_raw is not None:
        rows.extend(
            [
                ("cm_raw_tn_invisible_invisible", confusion_matrix_raw[0][0]),
                ("cm_raw_fp_invisible_visible", confusion_matrix_raw[0][1]),
                ("cm_raw_fn_visible_invisible", confusion_matrix_raw[1][0]),
                ("cm_raw_tp_visible_visible", confusion_matrix_raw[1][1]),
            ]
        )

    if confusion_matrix_normalized is not None:
        rows.extend(
            [
                ("cm_norm_tn_invisible_invisible", confusion_matrix_normalized[0][0]),
                ("cm_norm_fp_invisible_visible", confusion_matrix_normalized[0][1]),
                ("cm_norm_fn_visible_invisible", confusion_matrix_normalized[1][0]),
                ("cm_norm_tp_visible_visible", confusion_matrix_normalized[1][1]),
            ]
        )

    yaw_sample_counts = summary.get("yaw_sample_counts")
    if yaw_sample_counts is not None:
        for yaw_key, count in yaw_sample_counts.items():
            rows.append((f"samples_{yaw_key}", count))

    yaw_metrics = summary.get("yaw_metrics")
    if yaw_metrics is not None:
        for yaw_key, metrics in yaw_metrics.items():
            rows.append((f"mean_nme_box_{yaw_key}", metrics.get("mean_nme_box")))
            rows.append(
                (
                    f"mean_nme_box_point_to_line_{yaw_key}",
                    metrics.get("mean_nme_box_point_to_line"),
                )
            )
            rows.append(
                (
                    f"mean_nme_interocular_{yaw_key}",
                    metrics.get("mean_nme_interocular"),
                )
            )

    orientation_sample_counts = summary.get("orientation_sample_counts")
    if orientation_sample_counts is not None:
        for orientation_name, count in orientation_sample_counts.items():
            rows.append((f"samples_{orientation_name}", count))

    orientation_metrics = summary.get("orientation_metrics")
    if orientation_metrics is not None:
        for orientation_name, metrics in orientation_metrics.items():
            rows.append(
                (f"mean_nme_box_{orientation_name}", metrics.get("mean_nme_box"))
            )
            rows.append(
                (
                    f"mean_nme_box_point_to_line_{orientation_name}",
                    metrics.get("mean_nme_box_point_to_line"),
                )
            )
            rows.append(
                (
                    f"mean_nme_interocular_{orientation_name}",
                    metrics.get("mean_nme_interocular"),
                )
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


def save_per_landmark_nme_csv(
    per_landmark_errors: list[list[float]],
    output_path: Path,
) -> None:
    """
    Save per-landmark NME values to CSV.

    Parameters
    ----------
    per_landmark_errors : list[list[float]]
        Error values grouped per landmark.
    output_path : Path
        Destination CSV path.
    """
    landmark_names = get_default_landmark_names()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["landmark_index", "landmark_name", "sample_index", "nme"])

        for landmark_index, errors in enumerate(per_landmark_errors):
            for sample_index, error_value in enumerate(errors):
                writer.writerow(
                    [
                        landmark_index,
                        landmark_names[landmark_index],
                        sample_index,
                        format_metric_value(round_metric_value(float(error_value))),
                    ]
                )


def save_per_image_nme_csv(
    per_image_nme: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save mean NME per image to CSV.

    Parameters
    ----------
    per_image_nme : list[dict[str, Any]]
        Per-image NME metrics.
    output_path : Path
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_id",
                "orientation",
                "yaw_angle",
                "yaw_group",
                "mean_nme_box",
                "mean_nme_box_point_to_line",
                "mean_nme_interocular",
            ]
        )
        for row in per_image_nme:
            writer.writerow(
                [
                    row["sample_id"],
                    row.get("orientation"),
                    row.get("yaw_angle"),
                    row.get("yaw_group"),
                    (
                        format_metric_value(
                            round_metric_value(float(row["mean_nme_box"]))
                        )
                        if row["mean_nme_box"] is not None
                        else None
                    ),
                    (
                        format_metric_value(
                            round_metric_value(float(row["mean_nme_box_point_to_line"]))
                        )
                        if row.get("mean_nme_box_point_to_line") is not None
                        else None
                    ),
                    (
                        format_metric_value(
                            round_metric_value(float(row["mean_nme_interocular"]))
                        )
                        if row["mean_nme_interocular"] is not None
                        else None
                    ),
                ]
            )


def save_per_image_per_landmark_nme_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save one long-format row per evaluated image/prediction and landmark."""
    fieldnames = [
        "image_id",
        "prediction_id",
        "evaluation_mode",
        "split",
        "orientation",
        "class_idx",
        "yaw_angle",
        "yaw_group",
        "landmark_idx",
        "anatomical_group",
        "anatomical_label",
        "point_to_point_nme_box",
        "point_to_line_nme_box",
        "gt_visibility",
        "pred_visibility",
        "landmark_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            landmark_idx = row.get("landmark_idx")
            anatomical_group = (
                get_landmark_anatomical_group(int(landmark_idx))
                if landmark_idx is not None
                else None
            )
            anatomical_label = (
                get_landmark_anatomical_label(int(landmark_idx))
                if landmark_idx is not None
                else None
            )
            writer.writerow(
                {
                    field_name: format_metric_value(
                        round_metric_value(
                            {
                                "anatomical_group": anatomical_group,
                                "anatomical_label": anatomical_label,
                            }.get(field_name, row.get(field_name))
                        )
                    )
                    for field_name in fieldnames
                }
            )


def extract_face_orientation(sample_id: str) -> str:
    """
    Infer face orientation from the filename stem.

    Expected filename patterns:
    - xxxxx_left
    - xxxxx_quarter_left
    - xxxxx_frontal
    - xxxxx_quarter_right
    - xxxxx_right

    Parameters
    ----------
    sample_id : str
        Sample identifier or filename stem.

    Returns
    -------
    str
        Orientation label without leading underscore.

    Raises
    ------
    ValueError
        If the orientation cannot be inferred.
    """
    normalized_name = Path(sample_id).stem

    suffix_to_orientation = {
        "_quarter_left": "quarter_left",
        "_quarter_right": "quarter_right",
        "_frontal": "frontal",
        "_left": "left",
        "_right": "right",
    }

    for suffix, orientation in suffix_to_orientation.items():
        if normalized_name.endswith(suffix):
            return orientation

    raise ValueError(f"Could not infer orientation from sample_id='{sample_id}'.")


def _extract_batched_scalar(value: Any, sample_index: int) -> Any:
    """Extract one scalar from default-collated metadata."""
    if isinstance(value, torch.Tensor):
        return value[sample_index].item()
    if isinstance(value, np.ndarray):
        return value[sample_index].item()
    if isinstance(value, (list, tuple)):
        return value[sample_index]
    return value


def _format_yaw_key(yaw_angle: float) -> str:
    """Return a stable filename/CSV-safe key for a yaw angle."""
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


def _build_boxplot_title(
    label: str,
    mean_nme_box: float | None,
    mean_nme_interocular: float | None = None,
) -> str:
    """Build a boxplot title that includes the requested mean NME summaries."""
    title = f"Per-landmark NME distribution - {label}"
    summary_parts = []
    if mean_nme_box is not None:
        summary_parts.append(f"Mean NME box: {mean_nme_box:.4f}")
    if mean_nme_interocular is not None:
        summary_parts.append(f"Mean NME interocular: {mean_nme_interocular:.4f}")
    if summary_parts:
        title = f"{title}\n" + " | ".join(summary_parts)
    return title


def evaluate_checkpoint(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    visibility_threshold: float = 0.5,
    save_predictions: bool = False,
    save_overlays: bool = True,
    show_indices: bool = False,
    use_landmark_names_in_boxplot: bool = True,
    point_radius: int = 10,
    line_width: int = 4,
    line_color: str = "#FFD400",
    landmark_loss: str | None = None,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
) -> dict[str, Any]:
    """
    Evaluate a trained checkpoint on a dataset and save all requested artifacts.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : torch.utils.data.DataLoader
        Evaluation dataloader.
    device : torch.device
        Computation device.
    output_dir : str | Path
        Output directory.
    visibility_threshold : float, optional
        Threshold for binarizing visibility predictions. Defaults to 0.5.
    save_overlays : bool, optional
        Whether to save image overlays. Defaults to True.
    show_indices : bool, optional
        Whether to draw landmark indices in overlays. Defaults to False.
    use_landmark_names_in_boxplot : bool, optional
        Whether to use landmark names on the x-axis. Defaults to True.

    Returns
    -------
    dict[str, Any]
        Evaluation summary.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = output_dir / "predictions" if save_predictions else None
    prediction_overlays_dir = predictions_dir / "images" if predictions_dir else None
    prediction_labels_dir = predictions_dir / "labels" if predictions_dir else None
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        assert prediction_overlays_dir is not None
        assert prediction_labels_dir is not None
        prediction_overlays_dir.mkdir(parents=True, exist_ok=True)
        prediction_labels_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    per_landmark_errors: list[list[float]] | None = None
    per_landmark_point_to_line_errors: list[list[float]] | None = None
    per_image_nme: list[dict[str, Any]] = []
    per_image_per_landmark_nme: list[dict[str, Any]] = []
    all_visibility_targets: list[np.ndarray] = []
    all_visibility_predictions: list[np.ndarray] = []

    yaw_to_errors: dict[str, list[list[float]]] = {}
    yaw_to_box_nme_values: dict[str, list[float]] = {}
    yaw_to_box_nme_point_to_line_values: dict[str, list[float]] = {}
    yaw_to_interocular_nme_values: dict[str, list[float]] = {}
    yaw_display_labels: dict[str, str] = {}
    yaw_sort_values: dict[str, float] = {}
    global_interocular_nme_values: list[float] = []

    yaw_sample_counts: dict[str, int] = {}

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)

            predicted_landmarks_batch = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=True,
                decoder=coordinate_decoder,
                softmax_temperature=wasserstein_softmax_temperature,
            ).cpu()

            predicted_visibility_logits = outputs["visibility_logits"].cpu()
            predicted_visibility_scores = torch.sigmoid(predicted_visibility_logits)
            predicted_visibility_batch = (
                predicted_visibility_scores >= visibility_threshold
            ).to(torch.int64)

            target_landmarks_batch = batch["landmarks"].cpu()
            target_visibility_batch = batch["visibility"].cpu()
            metadata_batch = batch["metadata"]

            batch_size = images.shape[0]

            if per_landmark_errors is None:
                number_of_landmarks = predicted_landmarks_batch.shape[1]
                per_landmark_errors = [[] for _ in range(number_of_landmarks)]
                per_landmark_point_to_line_errors = [
                    [] for _ in range(number_of_landmarks)
                ]

            for sample_index in range(batch_size):
                sample_id = str(metadata_batch["sample_id"][sample_index])
                image_path = Path(metadata_batch["image_path"][sample_index])

                yaw_angle = float(
                    _extract_batched_scalar(metadata_batch["yaw_angle"], sample_index)
                )
                yaw_group = format_synthetic_yaw_group(yaw_angle)
                yaw_key = _format_yaw_key(yaw_angle)
                if yaw_key not in yaw_to_errors:
                    yaw_to_errors[yaw_key] = [[] for _ in range(number_of_landmarks)]
                    yaw_to_box_nme_values[yaw_key] = []
                    yaw_to_box_nme_point_to_line_values[yaw_key] = []
                    yaw_to_interocular_nme_values[yaw_key] = []
                    yaw_sample_counts[yaw_key] = 0
                    yaw_display_labels[yaw_key] = yaw_group
                    yaw_sort_values[yaw_key] = yaw_angle
                yaw_sample_counts[yaw_key] += 1

                original_size = extract_batched_size(
                    batched_size=metadata_batch["original_size"],
                    sample_index=sample_index,
                )
                transformed_size = extract_batched_size(
                    batched_size=metadata_batch["transformed_size"],
                    sample_index=sample_index,
                )
                predicted_landmarks_original = project_landmarks_to_original_size(
                    landmarks=predicted_landmarks_batch[sample_index],
                    transformed_size=transformed_size,
                    original_size=original_size,
                ).numpy()

                target_landmarks_original = project_landmarks_to_original_size(
                    landmarks=target_landmarks_batch[sample_index],
                    transformed_size=transformed_size,
                    original_size=original_size,
                ).numpy()

                predicted_visibility = (
                    predicted_visibility_batch[sample_index].numpy().astype(np.int64)
                )
                target_visibility = (
                    target_visibility_batch[sample_index].numpy().astype(np.int64)
                )
                split = str(metadata_batch.get("split", [""])[sample_index])

                if prediction_labels_dir is not None:
                    save_prediction_file(
                        output_path=prediction_labels_dir / f"{sample_id}.txt",
                        landmarks=predicted_landmarks_original,
                        visibility=predicted_visibility,
                    )
                    if save_overlays and prediction_overlays_dir is not None:
                        save_landmark_overlay_image(
                            image_path=image_path,
                            output_path=prediction_overlays_dir / f"{sample_id}.png",
                            predicted_landmarks=predicted_landmarks_original,
                            predicted_visibility=predicted_visibility,
                            show_indices=show_indices,
                            point_radius=point_radius,
                            line_width=line_width,
                            line_color=line_color,
                        )

                current_errors = compute_per_landmark_nme(
                    predicted_landmarks=predicted_landmarks_original,
                    target_landmarks=target_landmarks_original,
                )
                current_mean_box_nme = float(current_errors.mean())
                current_point_to_line_errors = compute_per_landmark_point_to_line_nme(
                    predicted_landmarks=predicted_landmarks_original,
                    target_landmarks=target_landmarks_original,
                )
                current_mean_box_nme_point_to_line = float(
                    current_point_to_line_errors.mean()
                )

                current_mean_interocular_nme: float | None = None
                if abs(yaw_angle) < 55.0:
                    current_interocular_errors = compute_per_landmark_interocular_nme(
                        predicted_landmarks=predicted_landmarks_original,
                        target_landmarks=target_landmarks_original,
                    )
                    current_mean_interocular_nme = float(
                        current_interocular_errors.mean()
                    )
                    yaw_to_interocular_nme_values[yaw_key].append(
                        current_mean_interocular_nme
                    )
                    global_interocular_nme_values.append(current_mean_interocular_nme)

                for landmark_index, error_value in enumerate(current_errors):
                    per_landmark_errors[landmark_index].append(float(error_value))
                assert per_landmark_point_to_line_errors is not None
                for landmark_index, error_value in enumerate(
                    current_point_to_line_errors
                ):
                    per_landmark_point_to_line_errors[landmark_index].append(
                        float(error_value)
                    )
                for landmark_index in range(len(current_errors)):
                    per_image_per_landmark_nme.append(
                        {
                            "image_id": sample_id,
                            "prediction_id": sample_id,
                            "evaluation_mode": "synthetic",
                            "split": split,
                            "yaw_angle": yaw_angle,
                            "yaw_group": yaw_group,
                            "landmark_idx": int(landmark_index),
                            "point_to_point_nme_box": float(
                                current_errors[landmark_index]
                            ),
                            "point_to_line_nme_box": float(
                                current_point_to_line_errors[landmark_index]
                            ),
                            "gt_visibility": int(target_visibility[landmark_index]),
                            "pred_visibility": int(
                                predicted_visibility[landmark_index]
                            ),
                            "landmark_count": int(len(current_errors)),
                        }
                    )

                for landmark_index, error_value in enumerate(current_errors):
                    yaw_to_errors[yaw_key][landmark_index].append(float(error_value))

                yaw_to_box_nme_values[yaw_key].append(current_mean_box_nme)
                yaw_to_box_nme_point_to_line_values[yaw_key].append(
                    current_mean_box_nme_point_to_line
                )
                per_image_nme.append(
                    {
                        "sample_id": sample_id,
                        "yaw_angle": yaw_angle,
                        "yaw_group": yaw_group,
                        "mean_nme_box": current_mean_box_nme,
                        "mean_nme_box_point_to_line": current_mean_box_nme_point_to_line,
                        "mean_nme_interocular": current_mean_interocular_nme,
                    }
                )

                all_visibility_targets.append(target_visibility.reshape(-1))
                all_visibility_predictions.append(predicted_visibility.reshape(-1))

    if per_landmark_errors is None or per_landmark_point_to_line_errors is None:
        raise RuntimeError("No evaluation samples were processed.")

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
    ordered_yaw_keys = sorted(
        yaw_to_errors, key=lambda yaw_key: yaw_sort_values[yaw_key]
    )
    yaw_filename_labels = {yaw_key: yaw_key for yaw_key in ordered_yaw_keys}

    all_grouped_errors = [per_landmark_errors] + [
        yaw_to_errors[yaw_key] for yaw_key in ordered_yaw_keys
    ]
    global_y_limits = compute_global_log_y_limits(all_grouped_errors)
    global_linear_y_limits = compute_global_linear_y_limits(all_grouped_errors)

    plot_per_landmark_boxplot(
        per_landmark_errors=per_landmark_errors,
        output_path=figures_dir / "boxplot_nme_per_landmark_global_log.png",
        use_landmark_names=use_landmark_names_in_boxplot,
        title=_build_boxplot_title(
            label="Global",
            mean_nme_box=float(np.mean([row["mean_nme_box"] for row in per_image_nme])),
            mean_nme_interocular=(
                float(np.mean(global_interocular_nme_values))
                if global_interocular_nme_values
                else None
            ),
        ),
        y_limits=global_y_limits,
        y_scale="log",
    )
    plot_per_landmark_boxplot(
        per_landmark_errors=per_landmark_errors,
        output_path=figures_dir / "boxplot_nme_per_landmark_global_linear.png",
        use_landmark_names=use_landmark_names_in_boxplot,
        title=_build_boxplot_title(
            label="Global",
            mean_nme_box=float(np.mean([row["mean_nme_box"] for row in per_image_nme])),
            mean_nme_interocular=(
                float(np.mean(global_interocular_nme_values))
                if global_interocular_nme_values
                else None
            ),
        ),
        y_limits=global_linear_y_limits,
        y_scale="linear",
    )

    plot_yaw_view_boxplots(
        orientation_to_errors=yaw_to_errors,
        output_dir=figures_dir,
        use_landmark_names=use_landmark_names_in_boxplot,
        y_limits=global_y_limits,
        y_scale="log",
        filename_suffix="log",
        ordered_orientations=ordered_yaw_keys,
        display_labels=yaw_display_labels,
        filename_labels=yaw_filename_labels,
        orientation_metrics={
            yaw_key: {
                "mean_nme_box": (float(np.mean(values)) if values else None),
                "mean_nme_box_point_to_line": (
                    float(np.mean(yaw_to_box_nme_point_to_line_values[yaw_key]))
                    if yaw_to_box_nme_point_to_line_values[yaw_key]
                    else None
                ),
                "mean_nme_interocular": (
                    float(np.mean(yaw_to_interocular_nme_values[yaw_key]))
                    if yaw_to_interocular_nme_values[yaw_key]
                    else None
                ),
            }
            for yaw_key, values in yaw_to_box_nme_values.items()
        },
    )
    plot_yaw_view_boxplots(
        orientation_to_errors=yaw_to_errors,
        output_dir=figures_dir,
        use_landmark_names=use_landmark_names_in_boxplot,
        y_limits=global_linear_y_limits,
        y_scale="linear",
        filename_suffix="linear",
        ordered_orientations=ordered_yaw_keys,
        display_labels=yaw_display_labels,
        filename_labels=yaw_filename_labels,
        orientation_metrics={
            yaw_key: {
                "mean_nme_box": (float(np.mean(values)) if values else None),
                "mean_nme_box_point_to_line": (
                    float(np.mean(yaw_to_box_nme_point_to_line_values[yaw_key]))
                    if yaw_to_box_nme_point_to_line_values[yaw_key]
                    else None
                ),
                "mean_nme_interocular": (
                    float(np.mean(yaw_to_interocular_nme_values[yaw_key]))
                    if yaw_to_interocular_nme_values[yaw_key]
                    else None
                ),
            }
            for yaw_key, values in yaw_to_box_nme_values.items()
        },
    )
    plot_grouped_nme_boxplot(
        group_to_values=yaw_to_box_nme_values,
        ordered_groups=ordered_yaw_keys,
        display_labels=yaw_display_labels,
        output_path=figures_dir / "boxplot_mean_nme_by_yaw_angle_linear.png",
        title="Synthetic mean NME by yaw angle",
        y_scale="linear",
    )
    plot_grouped_nme_boxplot(
        group_to_values=yaw_to_box_nme_values,
        ordered_groups=ordered_yaw_keys,
        display_labels=yaw_display_labels,
        output_path=figures_dir / "boxplot_mean_nme_by_yaw_angle_log.png",
        title="Synthetic mean NME by yaw angle",
        y_scale="log",
    )

    visibility_targets = np.concatenate(all_visibility_targets, axis=0)
    visibility_predictions = np.concatenate(all_visibility_predictions, axis=0)

    confusion_matrix_raw = compute_binary_confusion_matrix(
        targets=visibility_targets,
        predictions=visibility_predictions,
    )
    confusion_matrix_normalized = normalize_confusion_matrix(confusion_matrix_raw)
    visibility_metrics = compute_visibility_classification_metrics(confusion_matrix_raw)

    plot_confusion_matrix(
        matrix=confusion_matrix_raw,
        output_path=figures_dir / "confusion_matrix_raw.png",
        title="Visibility confusion matrix",
        value_format="d",
    )
    plot_confusion_matrix(
        matrix=confusion_matrix_normalized,
        output_path=figures_dir / "confusion_matrix_normalized.png",
        title="Visibility confusion matrix normalized",
        value_format=".4f",
    )

    summary = {
        "num_samples": int(len(per_image_nme)),
        "num_landmarks": int(len(per_landmark_errors)),
        "mean_nme_box": float(np.mean([row["mean_nme_box"] for row in per_image_nme])),
        "median_nme_box": float(
            np.median([row["mean_nme_box"] for row in per_image_nme])
        ),
        "mean_nme_box_point_to_line": float(
            np.mean([row["mean_nme_box_point_to_line"] for row in per_image_nme])
        ),
        "median_nme_box_point_to_line": float(
            np.median([row["mean_nme_box_point_to_line"] for row in per_image_nme])
        ),
        "mean_nme_interocular": (
            float(np.mean(global_interocular_nme_values))
            if global_interocular_nme_values
            else None
        ),
        "visibility_metrics": visibility_metrics,
        "visibility_threshold": float(visibility_threshold),
        "landmark_loss": landmark_loss,
        "coordinate_decoder": coordinate_decoder,
        "confusion_matrix_raw": confusion_matrix_raw.tolist(),
        "confusion_matrix_normalized": confusion_matrix_normalized.tolist(),
        "predictions_dir": str(predictions_dir)
        if predictions_dir is not None
        else None,
        "prediction_labels_dir": str(prediction_labels_dir)
        if prediction_labels_dir is not None
        else None,
        "prediction_overlays_dir": str(prediction_overlays_dir)
        if prediction_overlays_dir is not None
        else None,
        "yaw_sample_counts": {
            yaw_key: yaw_sample_counts[yaw_key] for yaw_key in ordered_yaw_keys
        },
        "yaw_group_labels": {
            yaw_key: yaw_display_labels[yaw_key] for yaw_key in ordered_yaw_keys
        },
        "yaw_metrics": {
            yaw_key: {
                "mean_nme_box": float(np.mean(box_values)) if box_values else None,
                "mean_nme_box_point_to_line": (
                    float(np.mean(yaw_to_box_nme_point_to_line_values[yaw_key]))
                    if yaw_to_box_nme_point_to_line_values[yaw_key]
                    else None
                ),
                "mean_nme_interocular": (
                    float(np.mean(yaw_to_interocular_nme_values[yaw_key]))
                    if yaw_to_interocular_nme_values[yaw_key]
                    else None
                ),
            }
            for yaw_key, box_values in yaw_to_box_nme_values.items()
        },
    }

    summary = round_metric_value(summary)

    save_metrics_summary_csv(
        output_path=output_dir / "metrics_summary.csv",
        summary=summary,
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
