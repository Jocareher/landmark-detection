from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_TTA_STEPS = (0, 1, 5, 10, 20, 50, 75, 100, 150, 200)
DEFAULT_MONITOR_STEPS = (0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200)
ORIENTATION_ORDER = ("left", "quarter_left", "frontal", "quarter_right", "right")


def parse_args() -> argparse.Namespace:
    """Parse the PCA-TTA step-sweep launcher arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run independent PCA-guided TTA evaluations for several fixed numbers "
            "of per-image adaptation steps."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML accepted by scripts/evaluate.py.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_TTA_STEPS),
        help="Independent fixed TTA episode lengths to evaluate.",
    )
    parser.add_argument(
        "--monitor-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_MONITOR_STEPS),
        help="Probe snapshots requested for every evaluation run.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=10,
        help="Number of images with detailed TTA probe grids in every run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override pca_tta_learning_rate from the base YAML for every run.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override pca_tta_weight_decay from the base YAML for every run.",
    )
    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=None,
        help="Override pca_tta_adam_beta1 from the base YAML for every run.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=None,
        help="Override pca_tta_adam_beta2 from the base YAML for every run.",
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=None,
        help="Override pca_tta_adam_epsilon from the base YAML for every run.",
    )
    parser.add_argument(
        "--max-gradient-norm",
        type=float,
        default=None,
        help="Override TTA gradient clipping; zero disables clipping.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "cosine"),
        default=None,
        help="Override the per-image TTA learning-rate schedule.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=None,
        help="Override the final learning rate of the cosine schedule.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Sweep root. By default, the base YAML output_dir is used and each "
            "run is written below it as tta-<steps>/ ."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later step counts if one evaluation fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without running evaluations.",
    )
    return parser.parse_args()


def load_evaluation_arguments(config_path: Path) -> dict[str, Any]:
    """Load the `arguments` mapping used by the standalone evaluator."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("The sweep YAML root must be a mapping.")
    arguments = payload.get("arguments", payload)
    if not isinstance(arguments, dict):
        raise ValueError("The sweep YAML 'arguments' entry must be a mapping.")
    return dict(arguments)


def validate_sweep_arguments(
    args: argparse.Namespace, base_arguments: dict[str, Any]
) -> None:
    """Validate the sweep without modifying the base evaluation YAML."""
    if not args.steps:
        raise ValueError("At least one TTA step count is required.")
    if any(step < 0 for step in args.steps):
        raise ValueError("TTA step counts cannot be negative.")
    if len(set(args.steps)) != len(args.steps):
        raise ValueError("TTA step counts must be unique.")
    if any(step < 0 for step in args.monitor_steps):
        raise ValueError("Monitor steps cannot be negative.")
    if args.probe_count < 0:
        raise ValueError("--probe-count cannot be negative.")
    if getattr(args, "learning_rate", None) is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if getattr(args, "weight_decay", None) is not None and args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative.")
    for name in ("adam_beta1", "adam_beta2"):
        value = getattr(args, name, None)
        if value is not None and not 0 <= value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1).")
    if getattr(args, "adam_epsilon", None) is not None and args.adam_epsilon <= 0:
        raise ValueError("--adam-epsilon must be positive.")
    if (
        getattr(args, "max_gradient_norm", None) is not None
        and args.max_gradient_norm < 0
    ):
        raise ValueError("--max-gradient-norm cannot be negative.")
    if (
        getattr(args, "min_learning_rate", None) is not None
        and args.min_learning_rate < 0
    ):
        raise ValueError("--min-learning-rate cannot be negative.")
    if base_arguments.get("checkpoint") is None:
        raise ValueError("The base evaluation YAML must define checkpoint.")
    if base_arguments.get("output_dir") is None and args.output_root is None:
        raise ValueError(
            "Set output_dir in the base YAML or pass --output-root to the sweep."
        )


def build_evaluation_command(
    *,
    config_path: Path,
    run_output_dir: Path,
    steps: int,
    monitor_steps: Sequence[int],
    probe_count: int,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    adam_beta1: float | None = None,
    adam_beta2: float | None = None,
    adam_epsilon: float | None = None,
    max_gradient_norm: float | None = None,
    lr_scheduler: str | None = None,
    min_learning_rate: float | None = None,
) -> list[str]:
    """Build one independent evaluator command for a fixed episode length."""
    run_name = f"tta-{steps}"
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("evaluate.py")),
        "--config",
        str(config_path.resolve()),
        "--pca-tta",
        "--pca-tta-steps",
        str(steps),
        "--pca-tta-monitor-steps",
        *(str(step) for step in monitor_steps),
        "--pca-tta-probe-count",
        str(probe_count),
        "--output-dir",
        str(run_output_dir.resolve()),
        "--wandb-run-name",
        run_name,
    ]
    optional_overrides = (
        ("--pca-tta-learning-rate", learning_rate),
        ("--pca-tta-weight-decay", weight_decay),
        ("--pca-tta-adam-beta1", adam_beta1),
        ("--pca-tta-adam-beta2", adam_beta2),
        ("--pca-tta-adam-epsilon", adam_epsilon),
        ("--pca-tta-max-gradient-norm", max_gradient_norm),
        ("--pca-tta-lr-scheduler", lr_scheduler),
        ("--pca-tta-min-learning-rate", min_learning_rate),
    )
    for option, value in optional_overrides:
        if value is not None:
            command.extend((option, str(value)))
    return command


def collect_run_result(steps: int, run_output_dir: Path) -> dict[str, Any]:
    """Collect official final metrics from one completed fixed-step run."""
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", summary)
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics payload in {summary_path}.")

    row = {
        "steps": int(steps),
        "run_name": f"tta-{steps}",
        "output_dir": str(run_output_dir),
        "num_samples": metrics.get("num_samples"),
        "mean_nme": _first_available(
            metrics,
            "mean_nme_box_gt_valid",
            "mean_nme_box_visible_intersection",
            "mean_nme_box",
        ),
        "median_nme": _first_available(
            metrics,
            "median_nme_box_gt_valid",
            "median_nme_box_visible_intersection",
            "median_nme_box",
        ),
        "mean_hausdorff": _first_available(
            metrics,
            "mean_hausdorff_box_gt_valid",
            "mean_hausdorff_box_visible_intersection",
            "mean_hausdorff_box",
        ),
        "median_hausdorff": _first_available(
            metrics,
            "median_hausdorff_box_gt_valid",
            "median_hausdorff_box_visible_intersection",
            "median_hausdorff_box",
        ),
        "status": "completed",
        "error": "",
    }
    return row


def collect_orientation_results(
    steps: int,
    run_output_dir: Path,
) -> list[dict[str, Any]]:
    """Collect orientation-specific NME and Hausdorff from one run."""
    summary_path = run_output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", summary)
    orientation_metrics = metrics.get("orientation_metrics", {})
    orientation_counts = metrics.get("orientation_sample_counts", {})
    rows: list[dict[str, Any]] = []
    for orientation, values in orientation_metrics.items():
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "steps": int(steps),
                "run_name": f"tta-{steps}",
                "orientation": orientation,
                "num_samples": orientation_counts.get(orientation),
                "mean_nme": _first_available(
                    values,
                    "mean_nme_box_gt_valid",
                    "mean_nme_box_visible_intersection",
                    "mean_nme_box",
                ),
                "median_nme": _first_available(
                    values,
                    "median_nme_box_gt_valid",
                    "median_nme_box_visible_intersection",
                    "median_nme_box",
                ),
                "mean_hausdorff": _first_available(
                    values,
                    "mean_hausdorff_box_gt_valid",
                    "mean_hausdorff_box_visible_intersection",
                    "mean_hausdorff_box",
                ),
                "median_hausdorff": _first_available(
                    values,
                    "median_hausdorff_box_gt_valid",
                    "median_hausdorff_box_visible_intersection",
                    "median_hausdorff_box",
                ),
            }
        )
    return rows


def _first_available(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write dictionaries to CSV using a stable union of their columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plots(
    result_rows: Sequence[dict[str, Any]],
    orientation_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Create meeting-ready comparisons without selecting an optimal step."""
    from scripts.utils.visualization import plt

    if plt is None:
        print("[WARNING] Matplotlib unavailable; comparison CSVs were still saved.")
        return []
    completed = sorted(
        (row for row in result_rows if row.get("status") == "completed"),
        key=lambda row: int(row["steps"]),
    )
    if not completed:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    steps = [int(row["steps"]) for row in completed]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for axis, mean_key, median_key, title, y_label in (
        (
            axes[0],
            "mean_nme",
            "median_nme",
            "NME by fixed TTA episode length",
            "NME (%)",
        ),
        (
            axes[1],
            "mean_hausdorff",
            "median_hausdorff",
            "Hausdorff distance by fixed TTA episode length",
            "Normalized Hausdorff distance (%)",
        ),
    ):
        mean_values = [100.0 * _numeric_or_nan(row.get(mean_key)) for row in completed]
        median_values = [
            100.0 * _numeric_or_nan(row.get(median_key)) for row in completed
        ]
        axis.plot(steps, mean_values, marker="o", linewidth=2.5, label="Mean")
        axis.plot(steps, median_values, marker="s", linewidth=2.5, label="Median")
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("TTA updates per image")
        axis.set_ylabel(y_label)
        axis.set_xticks(steps)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "PCA-guided TTA: final accuracy at each fixed number of updates",
        fontsize=19,
        fontweight="bold",
    )
    overall_path = output_dir / "tta_steps_metric_comparison.png"
    figure.savefig(overall_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    saved = [overall_path]
    usable_orientations = [
        orientation
        for orientation in ORIENTATION_ORDER
        if any(row.get("orientation") == orientation for row in orientation_rows)
    ]
    if usable_orientations:
        figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        colors = {
            "left": "#7B2CBF",
            "quarter_left": "#3A86FF",
            "frontal": "#2A9D8F",
            "quarter_right": "#F4A261",
            "right": "#E63946",
        }
        for orientation in usable_orientations:
            rows = sorted(
                (row for row in orientation_rows if row["orientation"] == orientation),
                key=lambda row: int(row["steps"]),
            )
            orientation_steps = [int(row["steps"]) for row in rows]
            label = orientation.replace("_", " ").title()
            axes[0].plot(
                orientation_steps,
                [100.0 * _numeric_or_nan(row.get("mean_nme")) for row in rows],
                marker="o",
                linewidth=2.2,
                label=label,
                color=colors[orientation],
            )
            axes[1].plot(
                orientation_steps,
                [100.0 * _numeric_or_nan(row.get("mean_hausdorff")) for row in rows],
                marker="o",
                linewidth=2.2,
                label=label,
                color=colors[orientation],
            )
        for axis, title, ylabel in (
            (axes[0], "Mean NME", "NME (%)"),
            (
                axes[1],
                "Mean Hausdorff distance",
                "Normalized Hausdorff distance (%)",
            ),
        ):
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("TTA updates per image")
            axis.set_ylabel(ylabel)
            axis.set_xticks(steps)
            axis.tick_params(axis="x", rotation=35)
            axis.grid(alpha=0.25)
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=max(1, len(labels)),
            frameon=False,
        )
        figure.suptitle(
            "PCA-guided TTA sensitivity by head orientation",
            fontsize=19,
            fontweight="bold",
        )
        orientation_path = output_dir / "tta_steps_by_orientation.png"
        figure.savefig(orientation_path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        saved.append(orientation_path)
    return saved


def _numeric_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_sweep_readme(
    output_root: Path,
    steps: Sequence[int],
    monitor_steps: Sequence[int],
    probe_count: int,
    optimizer_overrides: dict[str, Any] | None = None,
) -> None:
    """Document the fixed-step comparison and its output layout."""
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# PCA-TTA fixed-step sweep",
                "",
                f"Evaluated episode lengths: `{list(steps)}`.",
                f"Requested monitor steps: `{list(monitor_steps)}`.",
                f"Probe images per run: `{probe_count}`.",
                f"Optimizer overrides: `{optimizer_overrides or {}}`.",
                "",
                "Each `tta-<steps>/` directory is a complete independent evaluation. ",
                "The normalizer is reset from the same source checkpoint for every image and run.",
                "",
                "`sweep_results.csv` compares final NME and Hausdorff at each fixed ",
                "episode length. No run is selected using PCA reconstruction loss, ",
                "and no automatic best-step selection is performed.",
                "",
                "`sweep_orientation_results.csv` contains the same comparison by head orientation.",
                "The `figures/` directory contains meeting-ready plots derived from these tables.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run every fixed-step evaluation sequentially and consolidate its metrics."""
    args = parse_args()
    base_arguments = load_evaluation_arguments(args.config)
    validate_sweep_arguments(args, base_arguments)
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(base_arguments["output_dir"])
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    optimizer_overrides = {
        name: getattr(args, name)
        for name in (
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_gradient_norm",
            "lr_scheduler",
            "min_learning_rate",
        )
        if getattr(args, name) is not None
    }
    write_sweep_readme(
        output_root,
        steps=args.steps,
        monitor_steps=args.monitor_steps,
        probe_count=args.probe_count,
        optimizer_overrides=optimizer_overrides,
    )

    result_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    print(f"[SWEEP] Output root: {output_root}")
    print(f"[SWEEP] Fixed episode lengths: {args.steps}")
    print("[SWEEP] No PCA-loss-based best-step selection will be performed.")

    for steps in args.steps:
        run_name = f"tta-{steps}"
        run_output_dir = output_root / run_name
        command = build_evaluation_command(
            config_path=args.config,
            run_output_dir=run_output_dir,
            steps=steps,
            monitor_steps=args.monitor_steps,
            probe_count=args.probe_count,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            max_gradient_norm=args.max_gradient_norm,
            lr_scheduler=args.lr_scheduler,
            min_learning_rate=args.min_learning_rate,
        )
        print(f"\n[SWEEP] Starting {run_name}")
        print(f"[SWEEP] Command: {shlex.join(command)}")
        if args.dry_run:
            continue
        try:
            subprocess.run(command, check=True)
            result_rows.append(collect_run_result(steps, run_output_dir))
            orientation_rows.extend(collect_orientation_results(steps, run_output_dir))
            print(f"[SWEEP] Completed {run_name}")
        except (subprocess.CalledProcessError, OSError, ValueError) as error:
            result_rows.append(
                {
                    "steps": int(steps),
                    "run_name": run_name,
                    "output_dir": str(run_output_dir),
                    "status": "failed",
                    "error": str(error),
                }
            )
            print(f"[SWEEP] Failed {run_name}: {error}", file=sys.stderr)
            write_csv(output_root / "sweep_results.csv", result_rows)
            if not args.continue_on_error:
                raise

        write_csv(output_root / "sweep_results.csv", result_rows)
        write_csv(
            output_root / "sweep_orientation_results.csv",
            orientation_rows,
        )
        save_comparison_plots(
            result_rows,
            orientation_rows,
            output_root / "figures",
        )

    if args.dry_run:
        print("\n[SWEEP] Dry run complete; no evaluation was executed.")
        return
    print(f"\n[SWEEP] Finished. Comparison: {output_root / 'sweep_results.csv'}")
    print(f"[SWEEP] Figures: {output_root / 'figures'}")


if __name__ == "__main__":
    main()
