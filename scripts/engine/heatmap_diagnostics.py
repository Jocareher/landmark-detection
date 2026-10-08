"""Label-free heatmap monitoring, shared by training evaluation and TTA."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import compute_heatmap_uncertainty

DEFINITIONS = {
    "temperature": 1,
    "landmarks": "all output heatmaps; no GT or predicted visibility mask",
    "entropy": "spatial Shannon entropy divided by log(H*W)",
    "spread": "2 * (Var(x) + Var(y)); x,y normalized to [0,1]",
    "std": "population std of per-image means; ddof=0",
}


def heatmap_image_rows(heatmaps, sample_ids) -> list[dict[str, Any]]:
    uncertainty = compute_heatmap_uncertainty(heatmaps)
    means = {key: value.mean(1).cpu().tolist() for key, value in uncertainty.items()}
    return [{"sample_id": str(sample_id), "mean_heatmap_entropy": means["entropy"][i], "mean_heatmap_spread": means["spread"][i]} for i, sample_id in enumerate(sample_ids)]


def save_heatmap_diagnostics(rows, output_dir, summary):
    directory = Path(output_dir) / "heatmaps"
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "orientation", "mean_heatmap_entropy", "mean_heatmap_spread"]
    with (directory / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    def stats(selected):
        result = {"num_heatmap_samples": len(selected)}
        for metric in ("entropy", "spread"):
            values = np.asarray([row[f"mean_heatmap_{metric}"] for row in selected], dtype=float)
            values = values[np.isfinite(values)]
            result[f"mean_heatmap_{metric}"] = float(values.mean()) if values.size else None
            result[f"std_heatmap_{metric}"] = float(values.std()) if values.size else None
        return result
    summary.update(stats(rows))
    pose_metrics = summary.setdefault("orientation_metrics", {})
    poses = sorted({str(row.get("orientation", "unknown")) for row in rows})
    for pose in poses:
        pose_metrics.setdefault(pose, {}).update(stats([row for row in rows if str(row.get("orientation", "unknown")) == pose]))
    summary["heatmap_definitions"] = DEFINITIONS
    (directory / "summary.json").write_text(json.dumps({"definitions": DEFINITIONS, "global": stats(rows), "by_orientation": {pose: stats([row for row in rows if str(row.get("orientation", "unknown")) == pose]) for pose in poses}}, indent=2), encoding="utf-8")
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = Path(output_dir) / "figures" / "heatmaps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, metric in zip(axes, ("entropy", "spread")):
        values = np.asarray([row[f"mean_heatmap_{metric}"] for row in rows], dtype=float)
        axis.hist(values[np.isfinite(values)], bins=np.linspace(0, 1, 31), color="#159895")
        axis.set(xlim=(0, 1), xlabel=f"Mean normalized heatmap {metric}", ylabel="Images")
    fig.savefig(figure_dir / "distributions.png", dpi=180)
    plt.close(fig)
    order = [pose for pose in ("left", "quarter_left", "frontal", "quarter_right", "right") if pose in poses]
    order += sorted(set(poses) - set(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, metric in zip(axes, ("entropy", "spread")):
        for i, pose in enumerate(order):
            values = np.asarray([row[f"mean_heatmap_{metric}"] for row in rows if str(row.get("orientation", "unknown")) == pose], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                axis.errorbar(i, values.mean(), yerr=values.std(), fmt="o", capsize=4, color="#159895")
        axis.set(ylim=(0, 1), ylabel=f"Normalized {metric} (mean ± std)", xticks=range(len(order)), xticklabels=[pose.replace("_", " ") for pose in order])
        axis.tick_params(axis="x", rotation=25)
    fig.savefig(figure_dir / "by_orientation.png", dpi=180)
    plt.close(fig)


def save_tta_heatmap_curves(rows, output_dir):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(output_dir) / "figures" / "heatmaps"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    steps = [row["step"] for row in rows]
    for axis, metric in zip(axes, ("entropy", "spread")):
        prefix = f"heatmap_{metric}"
        axis.plot(steps, [row[f"{prefix}_mean"] for row in rows], color="#159895", label="Mean")
        axis.fill_between(steps, [row[f"{prefix}_p10"] for row in rows], [row[f"{prefix}_p90"] for row in rows], color="#159895", alpha=.2, label="10–90th percentile")
        axis.set(ylim=(0, 1), xlabel="Adaptation step", ylabel=f"Normalized heatmap {metric}")
        axis.legend()
    fig.savefig(directory / "adaptation_curves.png", dpi=180)
    plt.close(fig)


def save_training_heatmap_curves(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(output_dir) / "figures" / "heatmaps"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, metric in zip(axes, ("entropy", "spread")):
        for split, color in (("train", "#90a4bb"), ("val", "#159895")):
            rows = history[split]
            axis.plot(range(1, len(rows) + 1), [row.get(f"mean_heatmap_{metric}", np.nan) for row in rows], label=split, color=color)
        axis.set(ylim=(0, 1), xlabel="Epoch", ylabel=f"Normalized heatmap {metric}")
        axis.legend()
    fig.savefig(directory / "training_curves.png", dpi=180)
    plt.close(fig)
