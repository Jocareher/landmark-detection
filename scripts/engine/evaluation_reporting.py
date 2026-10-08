from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


REPORT_CATEGORIES = (
    "Coverage",
    "NME",
    "Hausdorff",
    "Heatmaps",
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
            elif "heatmap" in metric_name.lower():
                category = "Heatmaps"
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
                        and "heatmap" not in metric_name.lower()
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


def clean_metrics(data: dict[str, Any], protocol: str) -> dict[str, Any]:
    """Keep the dataset's official geometry protocol; preserve classification metrics."""
    result = {}
    for key, value in data.items():
        lower = key.lower()
        if protocol in {"synbaby", "babyland"} and "non_contour" in lower:
            continue
        if protocol == "synbaby" and "gt_valid" in lower:
            continue
        if protocol == "babyland":
            if "visible_intersection" in lower or "interocular" in lower:
                continue
            if ("nme" in lower or "hausdorff" in lower) and "gt_valid" not in lower and not isinstance(value, dict):
                continue
            if key in {"num_visible_visible_landmarks", "num_samples_with_geometric_metrics"}:
                continue
        result[key] = clean_metrics(value, protocol) if isinstance(value, dict) else value
    return result


def add_distribution_statistics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Population std of finite per-image errors (ddof=0), globally and by pose."""
    metric_keys = sorted({key for row in rows for key in row if key.startswith("mean_nme_") or key.startswith("hausdorff_box")})
    def add(target, selected):
        for key in metric_keys:
            values = np.asarray([row[key] for row in selected if row.get(key) is not None], dtype=float)
            values = values[np.isfinite(values)]
            name = key.removeprefix("mean_")
            target[f"std_{name}"] = float(values.std()) if values.size else None
    add(summary, rows)
    orientations = summary.get("orientation_metrics", {})
    for pose, metrics in orientations.items():
        add(metrics, [row for row in rows if row.get("orientation") == pose])
    summary["std_definition"] = "population standard deviation across finite per-image errors; ddof=0"


def write_dataset_metric_summary(path: Path, summary: dict[str, Any]) -> None:
    """Ordered sections, populated fields only, with a separate pose comparison table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = {"Coverage": [], "NME": [], "Hausdorff": [], "Heatmaps": [], "Visibility": [], "Orientation": [], "Setup": []}
    def section(key):
        if "heatmap" in key: return "Heatmaps"
        if "nme" in key: return "NME"
        if "hausdorff" in key: return "Hausdorff"
        if key.startswith(("num_", "images_", "total_")) or "landmarks_used" in key or key == "detection_rate": return "Coverage"
        if key.startswith("visibility"): return "Visibility"
        return "Setup"
    def visit(data, prefix="", override=None):
        for key, value in data.items():
            if value is None: continue
            full = f"{prefix}{key}"
            category = override or section(key)
            if isinstance(value, dict):
                nested_category = "Orientation" if key in {"orientation_metrics", "yaw_metrics"} else category
                if key in {"orientation_metrics", "yaw_metrics"}:
                    for pose, metrics in value.items():
                        for name, scalar in metrics.items():
                            if scalar is not None and not isinstance(scalar, (dict, list, tuple)):
                                sections["Orientation"].append({"section": "Orientation", "metric": f"{name}_{pose}", "value": f"{scalar:.4f}" if isinstance(scalar, float) else scalar})
                else:
                    visit(value, full + "_", nested_category)
            elif not isinstance(value, (list, tuple)):
                sections[category].append({"section": category, "metric": full, "value": f"{value:.4f}" if isinstance(value, float) else value})
    visit(summary)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        for rows in sections.values():
            writer.writerows(rows)
    poses = summary.get("orientation_metrics", {})
    if poses:
        fields = list(dict.fromkeys(key for metrics in poses.values() for key, value in metrics.items() if value is not None and not isinstance(value, (dict, list, tuple))))
        with (path.parent / "metrics_by_orientation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["orientation", *fields])
            writer.writeheader()
            for pose, metrics in poses.items():
                writer.writerow({"orientation": pose, **{key: metrics.get(key) for key in fields}})
