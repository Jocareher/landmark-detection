from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


REPORT_CATEGORIES = (
    "Coverage",
    "NME",
    "Hausdorff",
    "Orientation",
    "Visibility",
    "Setup",
)


def evaluation_metrics_view(summary: Any) -> dict[str, Any]:
    """Return official metrics, unwrapping inference-plus-metrics containers."""
    if not isinstance(summary, dict):
        return {}
    metrics = summary.get("metrics")
    return metrics if isinstance(metrics, dict) else summary


def collect_official_metric_rows(
    evaluation_summaries: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect ordered scalar evaluation metrics without truncating fields."""
    coverage_keys = {
        "num_samples",
        "total_images",
        "images_with_prediction",
        "images_without_prediction",
        "images_with_invalid_prediction",
        "num_samples_with_geometric_metrics",
        "num_samples_with_gt_valid_metrics",
        "num_visible_visible_landmarks",
        "num_gt_valid_landmarks",
        "valid_landmarks_used",
        "gt_valid_landmarks_used",
        "valid_non_contour_landmarks_used",
        "detection_rate",
    }
    setup_keys = {
        "landmark_loss",
        "coordinate_decoder",
        "visibility_threshold",
        "model_landmark_format",
        "evaluated_landmark_count",
    }
    rows: list[dict[str, Any]] = []
    for dataset_name, raw_summary in evaluation_summaries.items():
        summary = evaluation_metrics_view(raw_summary)
        for metric_name, value in summary.items():
            if isinstance(value, (dict, list, tuple)):
                continue
            if metric_name in coverage_keys:
                category = "Coverage"
            elif "hausdorff" in metric_name.lower():
                category = "Hausdorff"
            elif "nme" in metric_name.lower():
                category = "NME"
            elif metric_name in setup_keys:
                category = "Setup"
            else:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "category": category,
                    "metric": metric_name,
                    "value": value,
                }
            )

        visibility_metrics = summary.get("visibility_metrics", {})
        if isinstance(visibility_metrics, dict):
            for class_name in ("global", "visible", "invisible"):
                class_metrics = visibility_metrics.get(class_name, {})
                if not isinstance(class_metrics, dict):
                    continue
                for metric_name in ("precision", "recall", "f1"):
                    if metric_name in class_metrics:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "category": "Visibility",
                                "metric": f"visibility_{class_name}_{metric_name}",
                                "value": class_metrics[metric_name],
                            }
                        )
        orientation_metrics = summary.get("orientation_metrics", {})
        if isinstance(orientation_metrics, dict):
            for orientation_name, metrics in orientation_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                for metric_name, value in metrics.items():
                    if isinstance(value, (dict, list, tuple)):
                        continue
                    if (
                        "nme" not in metric_name.lower()
                        and "hausdorff" not in metric_name.lower()
                    ):
                        continue
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "category": "Orientation",
                            "metric": f"orientation_{orientation_name}_{metric_name}",
                            "value": value,
                        }
                    )
    return rows


def write_official_metric_exports(
    reports_dir: str | Path,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Write long, wide, and direct-copy TSV tables for spreadsheet use."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    long_path = reports_dir / "official_metrics_long.csv"
    wide_path = reports_dir / "official_metrics_wide.csv"
    tsv_path = reports_dir / "official_metrics_copy_paste.tsv"
    _write_rows_csv(long_path, rows)

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    metric_names = list(dict.fromkeys(str(row["metric"]) for row in rows))
    values = {(str(row["dataset"]), str(row["metric"])): row["value"] for row in rows}
    wide_rows = [
        {
            "dataset": dataset,
            **{
                metric_name: values.get((dataset, metric_name))
                for metric_name in metric_names
            },
        }
        for dataset in datasets
    ]
    _write_rows_csv(wide_path, wide_rows)

    if not wide_rows:
        tsv_path.write_text("", encoding="utf-8")
    else:
        with tsv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset", *metric_names],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(wide_rows)
    return {
        "long_csv": str(long_path),
        "wide_csv": str(wide_path),
        "copy_paste_tsv": str(tsv_path),
    }


def write_evaluation_report(
    reports_dir: str | Path,
    evaluation_summaries: dict[str, Any],
) -> dict[str, str]:
    """Write the repository-wide consolidated evaluation report and tables."""
    reports_dir = Path(reports_dir)
    rows = collect_official_metric_rows(evaluation_summaries)
    paths = write_official_metric_exports(reports_dir, rows)
    report_path = reports_dir / "evaluation_report.md"
    lines = [
        "# Evaluation report",
        "",
        "This report is generated for every full evaluation, independently of the experiment mode.",
        "",
    ]
    for dataset_name in evaluation_summaries:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        lines.extend([f"## {dataset_name}", ""])
        if not dataset_rows:
            lines.extend(["No scalar official metrics were available.", ""])
            continue
        for category in REPORT_CATEGORIES:
            category_rows = [row for row in dataset_rows if row["category"] == category]
            if not category_rows:
                continue
            lines.extend([f"### {category}", "", "| Metric | Value |", "|---|---:|"])
            lines.extend(
                f"| {row['metric']} | {format_report_value(row['value'])} |"
                for row in category_rows
            )
            lines.append("")
    lines.extend(
        [
            "## Spreadsheet exports",
            "",
            "- `official_metrics_long.csv`: one row per dataset and metric.",
            "- `official_metrics_wide.csv`: one dataset per row.",
            "- `official_metrics_copy_paste.tsv`: tab-separated wide table for direct copy/paste.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"markdown": str(report_path), **paths}


def format_report_value(value: Any) -> str:
    """Format report scalars consistently while preserving counts and text."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionary rows using their shared ordered schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
