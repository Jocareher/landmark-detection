"""Post-hoc BabyLand evaluation at the monitored TTA steps."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .runtime_reporting import write_rows
from ..utils.visualization import plt


def save_monitored_nme_report(rows, steps, output_dir: Path):
    """Write raw observations, matched trajectory summaries, and plots."""
    raw_path = output_dir / "nme_by_image_and_step.csv"
    write_rows(raw_path, rows)

    by_image = defaultdict(list)
    for row in rows:
        by_image[row["sample_id"]].append(row)

    complete = []
    image_summaries = []
    for sample_id, observations in by_image.items():
        source = observations[0]
        source_nme = source["nme_box_gt_valid"]
        by_step = {row["step"]: row["nme_box_gt_valid"] for row in observations[1:]}
        values = np.asarray([by_step.get(step) for step in steps], dtype=float)
        eligible = (
            not source["episode_failed"] and source_nme is not None
            and np.isfinite(source_nme) and np.isfinite(values).all()
        )
        image = {
            "sample_id": sample_id, "orientation": source["orientation"],
            "episode_failed": source["episode_failed"], "complete_trajectory": bool(eligible),
            "source_nme": source_nme,
        }
        if eligible:
            best_index = int(values.argmin())
            best_nme = float(values[best_index])
            image.update({
                "step_0_nme": float(values[0]),
                "final_nme": float(values[-1]),
                "best_monitored_step": steps[best_index],
                "best_monitored_nme": best_nme,
                "first_step_worse_than_source": next(
                    (step for step, value in zip(steps, values) if value > source_nme), None
                ),
                "source_preferred_or_tied": bool(source_nme <= best_nme),
                "final_minus_best_monitored_nme": float(values[-1] - best_nme),
            })
            complete.append((image, values))
        image_summaries.append(image)

    image_path = output_dir / "nme_trajectory_image_summary.csv"
    write_rows(image_path, image_summaries)
    report = {
        "per_image_and_step_csv": str(raw_path),
        "per_image_summary_csv": str(image_path),
        "monitored_steps": steps,
        "num_complete_trajectories": len(complete),
        "num_excluded_from_step_comparisons": len(by_image) - len(complete),
        "source_definition": "Official unadapted reference. With batch TTA, the step-0 forward may use adaptation-mode normalization.",
        "analysis_definition": "Ground-truth diagnostics only. Pose curves use the same successful images with finite NME at source and every monitored step. Best steps are retrospective among the recorded predictions.",
        "nme_units": "fraction (multiply by 100 for percent)",
    }
    if not complete:
        return report

    groups = defaultdict(list)
    for image, values in complete:
        groups[image["orientation"]].append((image, values))
    pose_rows = []
    for orientation, members in groups.items():
        values = np.stack([values for _, values in members])
        baseline = np.asarray([image["source_nme"] for image, _ in members])
        for step, column in zip(steps, values.T):
            deltas = column - baseline
            pose_rows.append({
                "orientation": orientation, "step": step, "count": len(members),
                "mean_source_nme": float(baseline.mean()),
                "mean_nme": float(column.mean()),
                "median_nme": float(np.median(column)),
                "std_nme": float(column.std()),
                "mean_delta_from_source": float(deltas.mean()),
                "median_delta_from_source": float(np.median(deltas)),
                "fraction_worse_than_source": float((deltas > 0).mean()),
                "fraction_worse_than_step_0": float((column > values[:, 0]).mean()),
            })
    pose_path = output_dir / "nme_by_orientation_and_step.csv"
    write_rows(pose_path, pose_rows)
    report["by_orientation_csv"] = str(pose_path)
    if plt is not None:
        report["figures"] = _plot_nme_trajectories(pose_rows, groups, steps, output_dir / "figures")
    return report


def _plot_nme_trajectories(pose_rows, groups, steps, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for orientation, members in groups.items():
        current = [row for row in pose_rows if row["orientation"] == orientation]
        label = f"{orientation.replace('_', ' ').title()} (n={len(members)})"
        for axis, metric in zip(axes, ("mean_nme", "median_nme", "fraction_worse_than_source")):
            axis.plot(steps, [100 * row[metric] for row in current], marker=".", label=label)
    for axis, title in zip(axes, ("Mean NME (%)", "Median NME (%)", "Images worse than source (%)")):
        axis.set(xlabel="TTA updates", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    pose_path = figures_dir / "nme_trajectory_by_orientation.png"
    figure.savefig(pose_path, dpi=160)
    plt.close(figure)

    blocks = []
    boundaries = [0]
    for members in groups.values():
        ordered = sorted(members, key=lambda member: (member[0]["source_nme"], member[0]["sample_id"]))
        blocks.extend(100 * (values - image["source_nme"]) for image, values in ordered)
        boundaries.append(len(blocks))
    matrix = np.stack(blocks)
    limit = max(float(np.percentile(np.abs(matrix), 95)), 1e-6)
    figure, axis = plt.subplots(figsize=(12, 7), layout="constrained")
    heatmap = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    for boundary in boundaries[1:-1]:
        axis.axhline(boundary - 0.5, color="black", linewidth=0.5)
    axis.set_yticks(
        [(start + end - 1) / 2 for start, end in zip(boundaries, boundaries[1:])],
        [orientation.replace("_", " ").title() for orientation in groups],
    )
    axis.set_xticks(range(len(steps)), steps, rotation=45)
    axis.set(xlabel="Monitored TTA step", title="One row per image, sorted by source NME within each pose")
    figure.colorbar(heatmap, ax=axis, label="NME change from source (percentage points)", extend="both")
    image_path = figures_dir / "nme_trajectory_by_image.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    return [str(pose_path), str(image_path)]
