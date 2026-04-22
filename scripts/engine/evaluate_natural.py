from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .evaluate import (
    _build_boxplot_title,
    compute_binary_confusion_matrix,
    compute_box_normalization_factor,
    compute_visibility_classification_metrics,
    normalize_confusion_matrix,
    round_metric_value,
    save_metrics_summary_csv,
    save_per_image_nme_csv,
    save_per_image_per_landmark_nme_csv,
    save_per_landmark_nme_csv,
)
from .geometry_metrics import compute_per_landmark_point_to_line_distances
from .metrics import decode_heatmaps_to_image_coords
from .postprocessing import (
    apply_homogeneous_transform,
    extract_batched_size,
    project_landmarks_between_sizes,
)
from ..utils.predictions import save_prediction_file
from ..utils.natural_labels import NATURAL_ORIENTATION_NAMES, UNKNOWN_ORIENTATION
from ..utils.visualization import (
    compute_global_linear_y_limits,
    compute_global_log_y_limits,
    plot_confusion_matrix,
    plot_per_landmark_boxplot,
    plot_yaw_view_boxplots,
    save_landmark_comparison_overlay_image,
    save_landmark_overlay_image,
)


def compute_visible_only_per_landmark_nme(
    predicted_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    target_visibility: np.ndarray,
    predicted_visibility: np.ndarray,
    eps: float = 1e-6,
) -> tuple[dict[int, float], dict[int, float], float | None, float | None]:
    """Compute visible-visible point-to-point and point-to-line NME values."""
    visible_gt_mask = target_visibility == 1
    valid_mask = visible_gt_mask & (predicted_visibility == 1)

    if visible_gt_mask.sum() == 0:
        return {}, {}, None, None

    normalization = compute_box_normalization_factor(
        target_landmarks=target_landmarks[visible_gt_mask],
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
        int(landmark_index): float(normalized_errors[landmark_index])
        for landmark_index in np.flatnonzero(valid_mask)
    }
    per_landmark_point_to_line_errors = {
        int(landmark_index): float(point_to_line_errors[landmark_index])
        for landmark_index in np.flatnonzero(valid_mask)
    }
    if not per_landmark_errors:
        return {}, {}, None, None

    return (
        per_landmark_errors,
        per_landmark_point_to_line_errors,
        float(np.mean(list(per_landmark_errors.values()))),
        float(np.mean(list(per_landmark_point_to_line_errors.values()))),
    )


def evaluate_natural_checkpoint(
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
    save_crop_overlays: bool = False,
) -> dict[str, Any]:
    """Evaluate a checkpoint on detector-export crops and original-image GT."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = output_dir / "predictions" if save_predictions else None
    prediction_overlays_dir = predictions_dir / "images" if predictions_dir else None
    prediction_labels_dir = predictions_dir / "labels" if predictions_dir else None
    prediction_crops_dir = (
        predictions_dir / "crops"
        if predictions_dir is not None and save_crop_overlays
        else None
    )
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        assert prediction_overlays_dir is not None
        assert prediction_labels_dir is not None
        prediction_overlays_dir.mkdir(parents=True, exist_ok=True)
        prediction_labels_dir.mkdir(parents=True, exist_ok=True)
        if prediction_crops_dir is not None:
            prediction_crops_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    per_landmark_errors: list[list[float]] | None = None
    per_landmark_point_to_line_errors: list[list[float]] | None = None
    per_image_nme: list[dict[str, Any]] = []
    per_image_per_landmark_nme: list[dict[str, Any]] = []
    all_visibility_targets: list[np.ndarray] = []
    all_visibility_predictions: list[np.ndarray] = []
    num_samples_with_geometric_metrics = 0
    num_visible_visible_landmarks = 0
    orientation_names = [*NATURAL_ORIENTATION_NAMES, UNKNOWN_ORIENTATION]
    orientation_to_errors: dict[str, list[list[float]]] = {}
    orientation_to_box_nme_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_to_box_nme_point_to_line_values: dict[str, list[float]] = {
        orientation: [] for orientation in orientation_names
    }
    orientation_sample_counts = {orientation: 0 for orientation in orientation_names}

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)

            predicted_landmarks_batch = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=True,
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
                orientation_to_errors = {
                    orientation: [[] for _ in range(number_of_landmarks)]
                    for orientation in orientation_names
                }

            for sample_index in range(batch_size):
                sample_id = str(metadata_batch["sample_id"][sample_index])
                orientation = str(metadata_batch["orientation"][sample_index])
                if orientation not in orientation_sample_counts:
                    orientation = UNKNOWN_ORIENTATION
                orientation_sample_counts[orientation] += 1
                source_image_path = Path(
                    metadata_batch["source_image_path"][sample_index]
                )
                crop_image_path = Path(metadata_batch["crop_image_path"][sample_index])

                network_input_size = extract_batched_size(
                    batched_size=metadata_batch["transformed_size"],
                    sample_index=sample_index,
                )
                crop_size = extract_batched_size(
                    batched_size=metadata_batch["crop_size"],
                    sample_index=sample_index,
                )
                transform_crop_to_orig = metadata_batch["transform_crop_to_orig"][
                    sample_index
                ]

                predicted_landmarks_crop = project_landmarks_between_sizes(
                    landmarks=predicted_landmarks_batch[sample_index],
                    source_size=network_input_size,
                    target_size=crop_size,
                )
                predicted_landmarks_original = apply_homogeneous_transform(
                    landmarks=predicted_landmarks_crop,
                    transform_matrix=transform_crop_to_orig,
                )

                target_landmarks_original = (
                    target_landmarks_batch[sample_index].numpy().astype(np.float32)
                )
                predicted_visibility = (
                    predicted_visibility_batch[sample_index].numpy().astype(np.int64)
                )
                target_visibility = (
                    target_visibility_batch[sample_index].numpy().astype(np.int64)
                )
                class_idx = metadata_batch["class_idx"][sample_index]
                if hasattr(class_idx, "item"):
                    class_idx = int(class_idx.item())
                else:
                    class_idx = int(class_idx)
                source_image_name = str(
                    metadata_batch["source_image_name"][sample_index]
                )

                if prediction_labels_dir is not None:
                    save_prediction_file(
                        output_path=prediction_labels_dir / f"{sample_id}.txt",
                        landmarks=predicted_landmarks_original,
                        visibility=predicted_visibility,
                    )
                    if save_overlays and prediction_overlays_dir is not None:
                        save_landmark_comparison_overlay_image(
                            image_path=source_image_path,
                            output_path=prediction_overlays_dir / f"{sample_id}.jpg",
                            predicted_landmarks=predicted_landmarks_original,
                            predicted_visibility=predicted_visibility,
                            target_landmarks=target_landmarks_original,
                            target_visibility=target_visibility,
                            show_indices=show_indices,
                            point_radius=point_radius,
                            line_width=line_width,
                            predicted_line_color=line_color,
                        )
                        if prediction_crops_dir is not None:
                            save_landmark_overlay_image(
                                image_path=crop_image_path,
                                output_path=prediction_crops_dir / f"{sample_id}.png",
                                predicted_landmarks=predicted_landmarks_crop.numpy(),
                                predicted_visibility=predicted_visibility,
                                show_indices=show_indices,
                                point_radius=point_radius,
                                line_width=line_width,
                                line_color=line_color,
                            )

                (
                    visible_errors,
                    visible_point_to_line_errors,
                    mean_box_nme,
                    mean_box_nme_point_to_line,
                ) = compute_visible_only_per_landmark_nme(
                    predicted_landmarks=predicted_landmarks_original,
                    target_landmarks=target_landmarks_original,
                    target_visibility=target_visibility,
                    predicted_visibility=predicted_visibility,
                )

                if mean_box_nme is not None:
                    num_samples_with_geometric_metrics += 1
                    num_visible_visible_landmarks += len(visible_errors)
                    for landmark_index, error_value in visible_errors.items():
                        per_landmark_errors[landmark_index].append(error_value)
                    assert per_landmark_point_to_line_errors is not None
                    for (
                        landmark_index,
                        error_value,
                    ) in visible_point_to_line_errors.items():
                        per_landmark_point_to_line_errors[landmark_index].append(
                            error_value
                        )
                    for landmark_index, error_value in visible_errors.items():
                        per_image_per_landmark_nme.append(
                            {
                                "image_id": source_image_name,
                                "prediction_id": sample_id,
                                "evaluation_mode": "natural",
                                "split": "natural",
                                "orientation": orientation,
                                "class_idx": class_idx if class_idx >= 0 else None,
                                "landmark_idx": int(landmark_index),
                                "point_to_point_nme_box": float(error_value),
                                "point_to_line_nme_box": float(
                                    visible_point_to_line_errors[landmark_index]
                                ),
                                "gt_visibility": int(target_visibility[landmark_index]),
                                "pred_visibility": int(
                                    predicted_visibility[landmark_index]
                                ),
                                "landmark_count": int(len(target_visibility)),
                            }
                        )
                    for landmark_index, error_value in visible_errors.items():
                        orientation_to_errors[orientation][landmark_index].append(
                            error_value
                        )
                    orientation_to_box_nme_values[orientation].append(mean_box_nme)
                    if mean_box_nme_point_to_line is not None:
                        orientation_to_box_nme_point_to_line_values[orientation].append(
                            mean_box_nme_point_to_line
                        )

                per_image_nme.append(
                    {
                        "sample_id": sample_id,
                        "orientation": orientation,
                        "mean_nme_box": mean_box_nme,
                        "mean_nme_box_point_to_line": mean_box_nme_point_to_line,
                        "mean_nme_interocular": None,
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

    valid_image_nme_values = [
        row["mean_nme_box"] for row in per_image_nme if row["mean_nme_box"] is not None
    ]
    valid_image_point_to_line_values = [
        row["mean_nme_box_point_to_line"]
        for row in per_image_nme
        if row["mean_nme_box_point_to_line"] is not None
    ]

    all_grouped_errors = [per_landmark_errors] + list(orientation_to_errors.values())
    if any(len(values) > 0 for values in per_landmark_errors):
        global_y_limits = compute_global_log_y_limits(all_grouped_errors)
        global_linear_y_limits = compute_global_linear_y_limits(all_grouped_errors)
        title = _build_boxplot_title(
            label="Natural mode",
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
            y_limits=global_y_limits,
            y_scale="log",
        )
        plot_per_landmark_boxplot(
            per_landmark_errors=per_landmark_errors,
            output_path=figures_dir / "boxplot_nme_per_landmark_global_linear.png",
            use_landmark_names=use_landmark_names_in_boxplot,
            title=title,
            y_limits=global_linear_y_limits,
            y_scale="linear",
        )
        orientation_metrics = {
            orientation: {
                "mean_nme_box": (float(np.mean(values)) if values else None),
                "mean_nme_box_point_to_line": (
                    float(
                        np.mean(
                            orientation_to_box_nme_point_to_line_values[orientation]
                        )
                    )
                    if orientation_to_box_nme_point_to_line_values[orientation]
                    else None
                ),
                "mean_nme_interocular": None,
            }
            for orientation, values in orientation_to_box_nme_values.items()
        }
        plot_yaw_view_boxplots(
            orientation_to_errors=orientation_to_errors,
            output_dir=figures_dir,
            use_landmark_names=use_landmark_names_in_boxplot,
            y_limits=global_y_limits,
            y_scale="log",
            filename_suffix="log",
            orientation_metrics=orientation_metrics,
        )
        plot_yaw_view_boxplots(
            orientation_to_errors=orientation_to_errors,
            output_dir=figures_dir,
            use_landmark_names=use_landmark_names_in_boxplot,
            y_limits=global_linear_y_limits,
            y_scale="linear",
            filename_suffix="linear",
            orientation_metrics=orientation_metrics,
        )
    else:
        orientation_metrics = {
            orientation: {
                "mean_nme_box": None,
                "mean_nme_box_point_to_line": None,
                "mean_nme_interocular": None,
            }
            for orientation in orientation_names
        }

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
        "num_samples_with_geometric_metrics": int(num_samples_with_geometric_metrics),
        "num_visible_visible_landmarks": int(num_visible_visible_landmarks),
        "num_landmarks": int(len(per_landmark_errors)),
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
        "mean_nme_interocular": None,
        "visibility_metrics": visibility_metrics,
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
        "prediction_crop_overlays_dir": str(prediction_crops_dir)
        if prediction_crops_dir is not None
        else None,
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
