from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.visualization import get_landmark_region_definitions


def compute_binary_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    """Return ``[[TN, FP], [FN, TP]]`` for binary visibility labels."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("Visibility targets and predictions must have equal shapes.")
    return np.asarray(
        [
            [
                int(((targets == 0) & (predictions == 0)).sum()),
                int(((targets == 0) & (predictions == 1)).sum()),
            ],
            [
                int(((targets == 1) & (predictions == 0)).sum()),
                int(((targets == 1) & (predictions == 1)).sum()),
            ],
        ],
        dtype=np.int64,
    )


def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """Normalize a binary confusion matrix by ground-truth row."""
    confusion_matrix = np.asarray(confusion_matrix)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0.0] = 1.0
    return confusion_matrix.astype(np.float64) / row_sums


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _compute_f1(precision: float, recall: float) -> float:
    return _safe_divide(2.0 * precision * recall, precision + recall)


def compute_visibility_classification_metrics(
    confusion_matrix: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute macro, visible-class, and invisible-class visibility metrics."""
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


def _summarize_group(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion_matrix = compute_binary_confusion_matrix(targets, predictions)
    return {
        "num_observations": int(len(targets)),
        "support_invisible": int((targets == 0).sum()),
        "support_visible": int((targets == 1).sum()),
        "predicted_invisible": int((predictions == 0).sum()),
        "predicted_visible": int((predictions == 1).sum()),
        "metrics": compute_visibility_classification_metrics(confusion_matrix),
        "confusion_matrix_raw": confusion_matrix.tolist(),
        "confusion_matrix_normalized": normalize_confusion_matrix(
            confusion_matrix
        ).tolist(),
    }


def compute_visibility_analysis(
    targets: np.ndarray,
    predictions: np.ndarray,
    pose_labels: np.ndarray,
    landmark_indices: np.ndarray,
    pose_display_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate visibility classification globally, by pose, and by anatomy."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    pose_labels = np.asarray(pose_labels, dtype=str).reshape(-1)
    landmark_indices = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    if not (
        targets.shape
        == predictions.shape
        == pose_labels.shape
        == landmark_indices.shape
    ):
        raise ValueError("All visibility analysis inputs must have equal shapes.")
    if len(targets) == 0:
        return {
            "available": False,
            "reason": "No paired ground-truth and predicted visibility labels.",
            "general": None,
            "by_pose": {},
            "by_anatomical_region": {},
        }
    if not np.isin(targets, [0, 1]).all() or not np.isin(
        predictions, [0, 1]
    ).all():
        raise ValueError("Visibility labels must contain only 0 and 1.")

    pose_display_labels = pose_display_labels or {}
    by_pose: dict[str, dict[str, Any]] = {}
    for pose in dict.fromkeys(pose_labels.tolist()):
        mask = pose_labels == pose
        group = _summarize_group(targets[mask], predictions[mask])
        group["display_label"] = pose_display_labels.get(pose, pose)
        by_pose[pose] = group

    by_region: dict[str, dict[str, Any]] = {}
    for region_name, landmark_range, _ in get_landmark_region_definitions():
        region_indices = np.asarray(list(landmark_range), dtype=np.int64)
        mask = np.isin(landmark_indices, region_indices)
        if not mask.any():
            continue
        region_key = region_name.lower().replace(" ", "_")
        group = _summarize_group(targets[mask], predictions[mask])
        group["display_label"] = region_name
        group["landmark_indices"] = [
            int(index) for index in region_indices if (landmark_indices == index).any()
        ]
        by_region[region_key] = group

    return {
        "available": True,
        "reason": None,
        "general": _summarize_group(targets, predictions),
        "by_pose": by_pose,
        "by_anatomical_region": by_region,
    }


def visibility_summary_fields(analysis: dict[str, Any]) -> dict[str, Any]:
    """Expose visibility analysis through stable top-level summary fields."""
    if not analysis.get("available"):
        return {
            "visibility_metrics_available": False,
            "visibility_metrics_unavailable_reason": analysis.get("reason"),
            "visibility_metrics": None,
            "visibility_metrics_by_pose": {},
            "visibility_metrics_by_anatomical_region": {},
            "visibility_analysis": analysis,
            "confusion_matrix_raw": None,
            "confusion_matrix_normalized": None,
        }
    general = analysis["general"]
    return {
        "visibility_metrics_available": True,
        "visibility_metrics_unavailable_reason": None,
        "visibility_metrics": general["metrics"],
        "visibility_metrics_by_pose": {
            key: value["metrics"] for key, value in analysis["by_pose"].items()
        },
        "visibility_metrics_by_anatomical_region": {
            key: value["metrics"]
            for key, value in analysis["by_anatomical_region"].items()
        },
        "visibility_analysis": analysis,
        "confusion_matrix_raw": general["confusion_matrix_raw"],
        "confusion_matrix_normalized": general["confusion_matrix_normalized"],
    }


def _visibility_csv_rows(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not analysis.get("available"):
        return [
            {
                "available": False,
                "unavailable_reason": analysis.get("reason"),
                "scope": "metadata",
                "group": "general",
                "display_label": "General",
            }
        ]
    scopes = [
        ("general", {"general": analysis["general"]}),
        ("pose", analysis["by_pose"]),
        ("anatomical_region", analysis["by_anatomical_region"]),
    ]
    for scope, groups in scopes:
        for group_name, group in groups.items():
            matrix = group["confusion_matrix_raw"]
            for class_name, metrics in group["metrics"].items():
                support = (
                    group["num_observations"]
                    if class_name == "global"
                    else group[f"support_{class_name}"]
                )
                rows.append(
                    {
                        "available": True,
                        "unavailable_reason": None,
                        "scope": scope,
                        "group": group_name,
                        "display_label": group.get("display_label", "General"),
                        "class": class_name,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "support": support,
                        "num_observations": group["num_observations"],
                        "predicted_invisible": group["predicted_invisible"],
                        "predicted_visible": group["predicted_visible"],
                        "tn": matrix[0][0],
                        "fp": matrix[0][1],
                        "fn": matrix[1][0],
                        "tp": matrix[1][1],
                    }
                )
    return rows


def save_visibility_metrics_csv(
    output_path: str | Path,
    analysis: dict[str, Any],
) -> None:
    """Save long-format visibility metrics for all aggregation scopes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "available",
        "unavailable_reason",
        "scope",
        "group",
        "display_label",
        "class",
        "precision",
        "recall",
        "f1",
        "support",
        "num_observations",
        "predicted_invisible",
        "predicted_visible",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_visibility_csv_rows(analysis))


def _plot_grouped_f1(
    groups: dict[str, dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    if not groups:
        return
    from ..utils.visualization import plt

    if plt is None:
        raise RuntimeError("Matplotlib is required to save visibility plots.")
    labels = [group.get("display_label", key) for key, group in groups.items()]
    x_positions = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    figure_width = max(8.0, 0.75 * len(labels))
    figure, axis = plt.subplots(figsize=(figure_width, 5.8))
    for offset, class_name, display_name, color in (
        (-width, "global", "Macro", "#4C78A8"),
        (0.0, "visible", "Visible", "#54A24B"),
        (width, "invisible", "Invisible", "#E45756"),
    ):
        values = [group["metrics"][class_name]["f1"] for group in groups.values()]
        axis.bar(
            x_positions + offset,
            values,
            width=width,
            label=display_name,
            color=color,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("F1")
    axis.set_title(title, fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_visibility_plots(
    output_dir: str | Path,
    analysis: dict[str, Any],
    include_babyland_region_protocols: bool = False,
) -> None:
    """Save compact F1 comparisons by pose and anatomical region."""
    if not analysis.get("available"):
        return
    output_dir = Path(output_dir)
    _plot_grouped_f1(
        analysis["by_pose"],
        output_dir / "visibility_f1_by_pose.png",
        "Visibility classification F1 by pose",
    )
    region_groups = analysis["by_anatomical_region"]
    if include_babyland_region_protocols:
        _plot_grouped_f1(
            region_groups,
            output_dir / "visibility_f1_by_anatomical_region_72.png",
            "Visibility classification F1 by anatomical region (72 landmarks)",
        )
        common68_groups = {
            key: group
            for key, group in region_groups.items()
            if group.get("landmark_indices")
            and max(group["landmark_indices"]) < 68
        }
        _plot_grouped_f1(
            common68_groups,
            output_dir / "visibility_f1_by_anatomical_region_common68.png",
            "Visibility classification F1 by anatomical region (common 68 landmarks)",
        )
    else:
        _plot_grouped_f1(
            region_groups,
            output_dir / "visibility_f1_by_anatomical_region.png",
            "Visibility classification F1 by anatomical region",
        )
