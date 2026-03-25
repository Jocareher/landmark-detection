from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .metrics import decode_heatmaps_to_image_coords
from .postprocessing import extract_batched_size, project_landmarks_to_original_size
from ..utils.predictions import save_prediction_file
from ..utils.visualization import (
    compute_global_log_y_limits,
    get_default_landmark_names,
    plot_confusion_matrix,
    plot_per_landmark_boxplot,
    plot_yaw_view_boxplots,
    save_landmark_overlay_image,
)


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
        ("num_landmarks", summary.get("num_landmarks")),
        ("mean_nme", summary.get("mean_nme")),
        ("median_nme", summary.get("median_nme")),
        ("visibility_accuracy", summary.get("visibility_accuracy")),
        ("visibility_threshold", summary.get("visibility_threshold")),
    ]

    confusion_matrix_raw = summary.get("confusion_matrix_raw")
    confusion_matrix_normalized = summary.get("confusion_matrix_normalized")

    if confusion_matrix_raw is not None:
        rows.extend(
            [
                ("cm_raw_tn_visible_visible", confusion_matrix_raw[0][0]),
                ("cm_raw_fp_visible_invisible", confusion_matrix_raw[0][1]),
                ("cm_raw_fn_invisible_visible", confusion_matrix_raw[1][0]),
                ("cm_raw_tp_invisible_invisible", confusion_matrix_raw[1][1]),
            ]
        )

    if confusion_matrix_normalized is not None:
        rows.extend(
            [
                ("cm_norm_tn_visible_visible", confusion_matrix_normalized[0][0]),
                ("cm_norm_fp_visible_invisible", confusion_matrix_normalized[0][1]),
                ("cm_norm_fn_invisible_visible", confusion_matrix_normalized[1][0]),
                ("cm_norm_tp_invisible_invisible", confusion_matrix_normalized[1][1]),
            ]
        )

    orientation_sample_counts = summary.get("orientation_sample_counts")
    if orientation_sample_counts is not None:
        for orientation_name, count in orientation_sample_counts.items():
            rows.append((f"samples_{orientation_name}", count))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in rows:
            writer.writerow([metric_name, metric_value])


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
                        float(error_value),
                    ]
                )


def save_per_image_nme_csv(
    per_image_nme: list[tuple[str, float]],
    output_path: Path,
) -> None:
    """
    Save mean NME per image to CSV.

    Parameters
    ----------
    per_image_nme : list[tuple[str, float]]
        List of (sample_id, mean_nme).
    output_path : Path
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "mean_nme"])
        for sample_id, mean_nme in per_image_nme:
            writer.writerow([sample_id, float(mean_nme)])


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
    per_image_nme: list[tuple[str, float]] = []
    all_visibility_targets: list[np.ndarray] = []
    all_visibility_predictions: list[np.ndarray] = []

    orientation_names = ["left", "quarter_left", "frontal", "quarter_right", "right"]

    orientation_to_errors: dict[str, list[list[float]]] = {}

    orientation_sample_counts = {
        "left": 0,
        "quarter_left": 0,
        "frontal": 0,
        "quarter_right": 0,
        "right": 0,
    }

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)

            predicted_landmarks_batch = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=False,
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

                orientation_to_errors = {
                    orientation: [[] for _ in range(number_of_landmarks)]
                    for orientation in orientation_names
                }

            for sample_index in range(batch_size):
                sample_id = str(metadata_batch["sample_id"][sample_index])
                image_path = Path(metadata_batch["image_path"][sample_index])

                orientation = extract_face_orientation(sample_id)
                orientation_sample_counts[orientation] += 1

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
                        )

                current_errors = compute_per_landmark_nme(
                    predicted_landmarks=predicted_landmarks_original,
                    target_landmarks=target_landmarks_original,
                )

                for landmark_index, error_value in enumerate(current_errors):
                    per_landmark_errors[landmark_index].append(float(error_value))

                for landmark_index, error_value in enumerate(current_errors):
                    orientation_to_errors[orientation][landmark_index].append(
                        float(error_value)
                    )

                per_image_nme.append((sample_id, float(current_errors.mean())))

                all_visibility_targets.append(target_visibility.reshape(-1))
                all_visibility_predictions.append(predicted_visibility.reshape(-1))

    if per_landmark_errors is None:
        raise RuntimeError("No evaluation samples were processed.")

    save_per_landmark_nme_csv(
        per_landmark_errors=per_landmark_errors,
        output_path=output_dir / "per_landmark_nme.csv",
    )
    save_per_image_nme_csv(
        per_image_nme=per_image_nme,
        output_path=output_dir / "per_image_nme.csv",
    )
    all_grouped_errors = [per_landmark_errors] + list(orientation_to_errors.values())
    global_y_limits = compute_global_log_y_limits(all_grouped_errors)

    plot_per_landmark_boxplot(
        per_landmark_errors=per_landmark_errors,
        output_path=figures_dir / "boxplot_nme_per_landmark_global.png",
        use_landmark_names=use_landmark_names_in_boxplot,
        title="Per-landmark NME distribution - Global",
        y_limits=global_y_limits,
    )

    plot_yaw_view_boxplots(
        orientation_to_errors=orientation_to_errors,
        output_dir=figures_dir,
        use_landmark_names=use_landmark_names_in_boxplot,
        y_limits=global_y_limits,
    )

    visibility_targets = np.concatenate(all_visibility_targets, axis=0)
    visibility_predictions = np.concatenate(all_visibility_predictions, axis=0)

    confusion_matrix_raw = compute_binary_confusion_matrix(
        targets=visibility_targets,
        predictions=visibility_predictions,
    )
    confusion_matrix_normalized = normalize_confusion_matrix(confusion_matrix_raw)

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
        value_format=".3f",
    )

    summary = {
        "num_samples": int(len(per_image_nme)),
        "num_landmarks": int(len(per_landmark_errors)),
        "mean_nme": float(np.mean([value for _, value in per_image_nme])),
        "median_nme": float(np.median([value for _, value in per_image_nme])),
        "visibility_accuracy": float(
            (visibility_targets == visibility_predictions).mean()
        ),
        "visibility_threshold": float(visibility_threshold),
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
        "orientation_sample_counts": orientation_sample_counts,
    }

    save_metrics_summary_csv(
        output_path=output_dir / "metrics_summary.csv",
        summary=summary,
    )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
