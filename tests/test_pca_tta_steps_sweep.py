from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_pca_tta_steps_sweep as sweep


def test_default_sweep_values_match_fixed_step_experiment() -> None:
    assert sweep.DEFAULT_TTA_STEPS == (0, 1, 5, 10, 20, 50, 75, 100, 150, 200)
    assert sweep.DEFAULT_MONITOR_STEPS == (
        0,
        1,
        5,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        150,
        200,
    )


def test_command_sets_independent_output_and_wandb_name(tmp_path: Path) -> None:
    command = sweep.build_evaluation_command(
        config_path=tmp_path / "base.yaml",
        run_output_dir=tmp_path / "runs" / "tta-75",
        steps=75,
        monitor_steps=sweep.DEFAULT_MONITOR_STEPS,
        probe_count=10,
    )

    assert command[0] == sys.executable
    assert command[command.index("--pca-tta-steps") + 1] == "75"
    assert command[command.index("--wandb-run-name") + 1] == "tta-75"
    assert command[command.index("--output-dir") + 1].endswith("/tta-75")
    assert command[command.index("--pca-tta-probe-count") + 1] == "10"
    monitor_start = command.index("--pca-tta-monitor-steps") + 1
    monitor_end = command.index("--pca-tta-probe-count")
    assert tuple(map(int, command[monitor_start:monitor_end])) == (
        sweep.DEFAULT_MONITOR_STEPS
    )


def test_command_forwards_optional_optimizer_overrides(tmp_path: Path) -> None:
    command = sweep.build_evaluation_command(
        config_path=tmp_path / "base.yaml",
        run_output_dir=tmp_path / "runs" / "tta-1250",
        steps=1250,
        monitor_steps=(0, 1250),
        probe_count=2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        max_gradient_norm=1.0,
        lr_scheduler="cosine",
        min_learning_rate=1e-5,
    )

    assert command[command.index("--pca-tta-learning-rate") + 1] == "0.001"
    assert command[command.index("--pca-tta-weight-decay") + 1] == "1e-05"
    assert command[command.index("--pca-tta-max-gradient-norm") + 1] == "1.0"
    assert command[command.index("--pca-tta-lr-scheduler") + 1] == "cosine"
    assert command[command.index("--pca-tta-min-learning-rate") + 1] == "1e-05"


def test_collect_run_result_uses_gt_valid_official_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "tta-20"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_samples": 622,
                "mean_nme_box": 0.20,
                "mean_nme_box_gt_valid": 0.1041,
                "median_nme_box_gt_valid": 0.0627,
                "mean_hausdorff_box_gt_valid": 0.2043,
                "median_hausdorff_box_gt_valid": 0.18,
            }
        ),
        encoding="utf-8",
    )

    row = sweep.collect_run_result(20, run_dir)

    assert row["steps"] == 20
    assert row["run_name"] == "tta-20"
    assert row["mean_nme"] == 0.1041
    assert row["mean_hausdorff"] == 0.2043


def test_collect_orientation_results_uses_orientation_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "tta-50"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "orientation_sample_counts": {"frontal": 100},
                "orientation_metrics": {
                    "frontal": {
                        "mean_nme_box_gt_valid": 0.05,
                        "median_nme_box_gt_valid": 0.04,
                        "mean_hausdorff_box_gt_valid": 0.12,
                        "median_hausdorff_box_gt_valid": 0.10,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows = sweep.collect_orientation_results(50, run_dir)

    assert rows == [
        {
            "steps": 50,
            "run_name": "tta-50",
            "orientation": "frontal",
            "num_samples": 100,
            "mean_nme": 0.05,
            "median_nme": 0.04,
            "mean_hausdorff": 0.12,
            "median_hausdorff": 0.10,
        }
    ]


def test_validation_rejects_duplicate_step_counts() -> None:
    args = SimpleNamespace(
        steps=[0, 1, 1],
        monitor_steps=[0, 1],
        probe_count=10,
        output_root=Path("runs"),
    )

    with pytest.raises(ValueError, match="unique"):
        sweep.validate_sweep_arguments(args, {"checkpoint": "model.pth"})
