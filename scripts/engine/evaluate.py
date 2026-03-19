from __future__ import annotations
from collections.abc import Sequence

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .metrics import decode_heatmaps_to_image_coords


def get_default_landmark_names() -> list[str]:
    """
    Return the default list of 72 landmark names.

    Returns
    -------
    list[str]
        Ordered landmark names.
    """
    return [
        "exR",
        "enR",
        "n",
        "enL",
        "exL",
        "acR",
        "aR",
        "prn",
        "aL",
        "acL",
        "sn",
        "chR",
        "cphR",
        "ls",
        "cphL",
        "chL",
        "li",
        "sl",
        "pg",
        "tR",
        "oiR",
        "tL",
        "oiL",
        "faceO_23",
        "faceO_24",
        "faceO_25",
        "faceO_26",
        "faceO_27",
        "faceO_28",
        "chin_29",
        "faceO_30",
        "faceO_31",
        "faceO_32",
        "faceO_33",
        "faceO_34",
        "faceO_35",
        "rightEB_36",
        "rightEB_37",
        "rightEB_38",
        "rightEB_39",
        "rightEB_40",
        "leftEB_41",
        "leftEB_42",
        "leftEB_43",
        "leftEB_44",
        "leftEB_45",
        "nose_46",
        "nose_47",
        "rightE_48",
        "rightE_49",
        "rightE_50",
        "rightE_51",
        "leftE_52",
        "leftE_53",
        "leftE_54",
        "leftE_55",
        "upperL_56",
        "upperL_57",
        "lowerL_58",
        "lowerL_59",
        "lowerL_60",
        "lowerL_61",
        "lipE_62",
        "upperL_63",
        "upperL_64",
        "upperL_65",
        "lipE_66",
        "lowerL_67",
        "lowerL_68",
        "lowerL_69",
        "chin_70",
        "chin_71",
    ]


def project_landmarks_to_original_size(
    landmarks: torch.Tensor,
    transformed_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """
    Project landmark coordinates from transformed image space to original image space.

    Parameters
    ----------
    landmarks : torch.Tensor
        Landmark coordinates of shape (K, 2) in transformed image space.
    transformed_size : tuple[int, int]
        Transformed image size as (height, width).
    original_size : tuple[int, int]
        Original image size as (height, width).

    Returns
    -------
    torch.Tensor
        Landmark coordinates of shape (K, 2) in original image space.
    """
    transformed_height, transformed_width = transformed_size
    original_height, original_width = original_size

    scale_x = original_width / float(transformed_width)
    scale_y = original_height / float(transformed_height)

    projected_landmarks = landmarks.clone()
    projected_landmarks[:, 0] *= scale_x
    projected_landmarks[:, 1] *= scale_y

    return projected_landmarks


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


def save_prediction_file(
    output_path: Path,
    landmarks: np.ndarray,
    visibility: np.ndarray,
) -> None:
    """
    Save predicted landmarks and visibility to a text file.

    Parameters
    ----------
    output_path : Path
        Destination file path.
    landmarks : np.ndarray
        Landmark coordinates of shape (K, 2).
    visibility : np.ndarray
        Predicted binary visibility of shape (K,).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for landmark_index in range(landmarks.shape[0]):
            x_coord = float(landmarks[landmark_index, 0])
            y_coord = float(landmarks[landmark_index, 1])
            visibility_value = int(visibility[landmark_index])
            file.write(f"{x_coord:.6f} {y_coord:.6f} {visibility_value}\n")


def save_overlay_image(
    image_path: Path,
    output_path: Path,
    predicted_landmarks: np.ndarray,
    predicted_visibility: np.ndarray,
    show_indices: bool = False,
    point_radius: int = 8,
) -> None:
    """
    Save an image overlay with predicted landmarks.

    Visibility convention:
    - 0: visible   -> red
    - 1: invisible -> blue

    Parameters
    ----------
    image_path : Path
        Original image path.
    output_path : Path
        Output image path.
    predicted_landmarks : np.ndarray
        Predicted landmark coordinates of shape (K, 2).
    predicted_visibility : np.ndarray
        Predicted binary visibility of shape (K,).
    show_indices : bool, optional
        Whether to draw landmark indices. Defaults to False.
    point_radius : int, optional
        Landmark circle radius in pixels. Defaults to 5.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for landmark_index, (x_coord, y_coord) in enumerate(predicted_landmarks):
        visibility_value = int(predicted_visibility[landmark_index])
        color = "red" if visibility_value == 1 else "blue"

        left = x_coord - point_radius
        top = y_coord - point_radius
        right = x_coord + point_radius
        bottom = y_coord + point_radius

        draw.ellipse((left, top, right, bottom), fill=color, outline="white", width=1)

        if show_indices:
            draw.text(
                (x_coord + point_radius + 2, y_coord + point_radius + 2),
                str(landmark_index),
                fill=color,
            )

    image.save(output_path)


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


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    value_format: str,
    normalized: bool = False,
) -> None:
    """
    Plot and save a confusion matrix with adaptive text color.

    Visibility convention:
    - 0: visible
    - 1: invisible

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix of shape (2, 2).
    output_path : Path
        Output image path.
    title : str
        Figure title.
    value_format : str
        Formatting string for cell values.
    normalized : bool, optional
        Whether the matrix is normalized. Defaults to False.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")

    class_labels = ["Visible (0)", "Invisible (1)"]

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(class_labels, fontsize=11)
    axis.set_yticklabels(class_labels, fontsize=11)
    axis.set_xlabel("Predicted label", fontsize=12)
    axis.set_ylabel("Ground-truth label", fontsize=12)
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)

    threshold = float(matrix.max()) * 0.55 if matrix.size > 0 else 0.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if value >= threshold else "black"

            axis.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=10)

    for spine in axis.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_per_landmark_boxplot(
    per_landmark_errors: list[list[float]],
    output_path: Path,
    use_landmark_names: bool = True,
    title: str = "Per-landmark NME distribution",
) -> None:
    """
    Plot one boxplot per landmark with improved aesthetics.

    Parameters
    ----------
    per_landmark_errors : list[list[float]]
        Error values grouped per landmark.
    output_path : Path
        Output image path.
    use_landmark_names : bool, optional
        Whether to use landmark names on the x-axis. Defaults to True.
    title : str, optional
        Figure title. Defaults to "Per-landmark NME distribution".
    """
    landmark_names = get_default_landmark_names()
    number_of_landmarks = len(per_landmark_errors)

    figure_width = max(20.0, number_of_landmarks * 0.34)
    figure, axis = plt.subplots(figsize=(figure_width, 7))

    boxplot = axis.boxplot(
        per_landmark_errors,
        showfliers=True,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#d62728", "linewidth": 2.0},
        meanprops={
            "marker": "D",
            "markerfacecolor": "#1f77b4",
            "markeredgecolor": "black",
            "markersize": 4,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#9467bd",
            "markeredgecolor": "#9467bd",
            "markersize": 2.8,
            "alpha": 0.35,
        },
        whiskerprops={"color": "#4d4d4d", "linewidth": 1.2},
        capprops={"color": "#4d4d4d", "linewidth": 1.2},
    )

    box_colors = plt.cm.tab20(np.linspace(0, 1, number_of_landmarks))
    for patch, color in zip(boxplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    axis.set_xticks(np.arange(1, number_of_landmarks + 1))

    if use_landmark_names:
        axis.set_xticklabels(landmark_names, rotation=90, fontsize=8)
    else:
        axis.set_xticklabels(
            [str(index) for index in range(number_of_landmarks)],
            rotation=90,
            fontsize=8,
        )

    axis.set_xlabel("Landmark", fontsize=12)
    axis.set_ylabel("Normalized error", fontsize=12)
    axis.set_title(title, fontsize=15, fontweight="bold", pad=12)
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#999999", edgecolor="#333333", alpha=0.55, label="IQR"),
        Line2D([0], [0], color="#d62728", lw=2.0, label="Median"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markersize=6,
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9467bd",
            markeredgecolor="#9467bd",
            markersize=5,
            label="Outlier",
        ),
    ]
    axis.legend(handles=legend_handles, loc="upper right", fontsize=10, frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


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

    raise ValueError(
        f"Could not infer orientation from sample_id='{sample_id}'."
    )


def plot_yaw_view_boxplots(
    orientation_to_errors: dict[str, list[list[float]]],
    output_dir: Path,
    use_landmark_names: bool = True,
) -> None:
    """
    Generate one per-landmark boxplot for each face orientation.

    Parameters
    ----------
    orientation_to_errors : dict[str, list[list[float]]]
        Mapping from orientation name to per-landmark error lists.
    output_dir : Path
        Directory where the plots will be saved.
    use_landmark_names : bool, optional
        Whether to use landmark names on the x-axis. Defaults to True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered_orientations = [
        "left",
        "quarter_left",
        "frontal",
        "quarter_right",
        "right",
    ]

    for orientation in ordered_orientations:
        if orientation not in orientation_to_errors:
            continue

        current_errors = orientation_to_errors[orientation]

        if all(len(values) == 0 for values in current_errors):
            continue

        plot_per_landmark_boxplot(
            per_landmark_errors=current_errors,
            output_path=output_dir / f"boxplot_nme_per_landmark_{orientation}.png",
            use_landmark_names=use_landmark_names,
            title=f"Per-landmark NME distribution - {orientation.replace('_', ' ').title()}",
        )


def evaluate_checkpoint(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    visibility_threshold: float = 0.5,
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
    predictions_dir = output_dir / "predictions"
    overlays_dir = figures_dir / "overlays"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

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

                original_size = _extract_batched_size(
                    batched_size=metadata_batch["original_size"],
                    sample_index=sample_index,
                )
                transformed_size = _extract_batched_size(
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

                save_prediction_file(
                    output_path=predictions_dir / f"{sample_id}.txt",
                    landmarks=predicted_landmarks_original,
                    visibility=predicted_visibility,
                )

                if save_overlays:
                    save_overlay_image(
                        image_path=image_path,
                        output_path=overlays_dir / f"{sample_id}.png",
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
    plot_per_landmark_boxplot(
        per_landmark_errors=per_landmark_errors,
        output_path=figures_dir / "boxplot_nme_per_landmark_global.png",
        use_landmark_names=use_landmark_names_in_boxplot,
        title="Per-landmark NME distribution - Global",
    )

    plot_yaw_view_boxplots(
        orientation_to_errors=orientation_to_errors,
        output_dir=figures_dir,
        use_landmark_names=use_landmark_names_in_boxplot,
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
        normalized=False,
    )
    plot_confusion_matrix(
        matrix=confusion_matrix_normalized,
        output_path=figures_dir / "confusion_matrix_normalized.png",
        title="Visibility confusion matrix normalized",
        value_format=".3f",
        normalized=True,
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
        "predictions_dir": str(predictions_dir),
        "overlays_dir": str(overlays_dir),
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


def _extract_batched_size(
    batched_size: Any,
    sample_index: int,
) -> tuple[int, int]:
    """
    Extract one `(height, width)` pair from a collated metadata size field.

    PyTorch's default collation may transform a per-sample tuple like
    `(height, width)` into one of several batched structures:
    - list[tuple[int, int]]
    - tuple[list[int], list[int]]
    - tuple[torch.Tensor, torch.Tensor]
    - list[torch.Tensor]
    - torch.Tensor with shape (B, 2)

    Parameters
    ----------
    batched_size : Any
        Collated metadata field corresponding to `original_size` or
        `transformed_size`.
    sample_index : int
        Batch sample index.

    Returns
    -------
    tuple[int, int]
        Size pair as `(height, width)`.

    Raises
    ------
    ValueError
        If the input structure cannot be interpreted as a batched size field.
    """
    if isinstance(batched_size, torch.Tensor):
        if batched_size.ndim == 2 and batched_size.shape[1] == 2:
            return int(batched_size[sample_index, 0].item()), int(
                batched_size[sample_index, 1].item()
            )
        raise ValueError(
            f"Unsupported tensor shape for batched size: {tuple(batched_size.shape)}."
        )

    if isinstance(batched_size, Sequence) and not isinstance(
        batched_size, (str, bytes)
    ):
        if len(batched_size) == 2:
            first, second = batched_size

            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                if first.ndim == 1 and second.ndim == 1:
                    return int(first[sample_index].item()), int(
                        second[sample_index].item()
                    )

            if isinstance(first, Sequence) and isinstance(second, Sequence):
                return int(first[sample_index]), int(second[sample_index])

        current_value = batched_size[sample_index]

        if isinstance(current_value, torch.Tensor):
            if current_value.numel() == 2:
                flat_value = current_value.view(-1)
                return int(flat_value[0].item()), int(flat_value[1].item())

        if isinstance(current_value, Sequence) and len(current_value) == 2:
            return int(current_value[0]), int(current_value[1])

    raise ValueError(
        f"Could not parse batched size field of type {type(batched_size)}."
    )
