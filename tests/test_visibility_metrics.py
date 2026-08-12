from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.engine.evaluate import save_metrics_summary_csv
from scripts.engine.visibility_metrics import (
    compute_visibility_analysis,
    save_visibility_metrics_csv,
    save_visibility_plots,
    visibility_summary_fields,
)


def test_visibility_analysis_aggregates_general_pose_and_anatomy() -> None:
    targets = np.asarray([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    predictions = np.asarray([1, 0, 0, 1, 1, 0, 1, 1], dtype=np.int64)
    poses = np.asarray(["left"] * 4 + ["right"] * 4)
    landmark_indices = np.asarray([0, 17, 36, 68, 0, 17, 36, 68])

    analysis = compute_visibility_analysis(
        targets=targets,
        predictions=predictions,
        pose_labels=poses,
        landmark_indices=landmark_indices,
    )

    assert analysis["available"] is True
    assert analysis["general"]["confusion_matrix_raw"] == [[2, 2], [1, 3]]
    assert set(analysis["by_pose"]) == {"left", "right"}
    assert set(analysis["by_anatomical_region"]) == {
        "face_contour",
        "right_eyebrow",
        "right_eye",
        "under_lip",
    }
    assert analysis["by_pose"]["left"]["num_observations"] == 4
    assert analysis["by_anatomical_region"]["face_contour"][
        "num_observations"
    ] == 2
    assert np.isclose(analysis["general"]["metrics"]["visible"]["recall"], 0.75)


def test_visibility_csv_contains_all_scopes(tmp_path) -> None:
    analysis = compute_visibility_analysis(
        targets=np.asarray([0, 1]),
        predictions=np.asarray([0, 1]),
        pose_labels=np.asarray(["frontal", "frontal"]),
        landmark_indices=np.asarray([0, 17]),
    )
    output_path = tmp_path / "visibility_metrics.csv"

    save_visibility_metrics_csv(output_path, analysis)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert {row["scope"] for row in rows} == {
        "general",
        "pose",
        "anatomical_region",
    }
    assert {row["class"] for row in rows} == {"global", "visible", "invisible"}


def test_unavailable_visibility_is_explicit_and_summary_safe(tmp_path) -> None:
    analysis = compute_visibility_analysis(
        targets=np.asarray([], dtype=np.int64),
        predictions=np.asarray([], dtype=np.int64),
        pose_labels=np.asarray([], dtype=str),
        landmark_indices=np.asarray([], dtype=np.int64),
    )
    analysis["reason"] = "Predictions have no visibility column."

    fields = visibility_summary_fields(analysis)
    output_path = tmp_path / "visibility_metrics.csv"
    save_visibility_metrics_csv(output_path, analysis)

    assert fields["visibility_metrics_available"] is False
    assert fields["visibility_metrics"] is None
    with output_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == 1
    assert csv_rows[0]["available"] == "False"
    assert csv_rows[0]["unavailable_reason"] == analysis["reason"]


def test_metrics_summary_excludes_visibility_information(tmp_path) -> None:
    analysis = compute_visibility_analysis(
        targets=np.asarray([0, 1]),
        predictions=np.asarray([0, 1]),
        pose_labels=np.asarray(["frontal", "frontal"]),
        landmark_indices=np.asarray([0, 17]),
    )
    output_path = tmp_path / "metrics_summary.csv"

    save_metrics_summary_csv(
        output_path,
        {**visibility_summary_fields(analysis), "visibility_threshold": 0.5},
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = dict(csv.reader(file))
    assert not any(key.startswith("visibility_") for key in rows)
    assert not any(key.startswith("cm_") for key in rows)


def test_babyland_region_plots_split_72_and_common68(monkeypatch, tmp_path) -> None:
    targets = np.zeros(72, dtype=np.int64)
    analysis = compute_visibility_analysis(
        targets=targets,
        predictions=targets,
        pose_labels=np.asarray(["frontal"] * 72),
        landmark_indices=np.arange(72),
    )
    calls: list[tuple[Path, set[str]]] = []

    def record_plot(groups, output_path, title):
        calls.append((Path(output_path), set(groups)))

    monkeypatch.setattr(
        "scripts.engine.visibility_metrics._plot_grouped_f1",
        record_plot,
    )
    save_visibility_plots(
        tmp_path,
        analysis,
        include_babyland_region_protocols=True,
    )

    calls_by_name = {path.name: groups for path, groups in calls}
    assert "visibility_f1_by_anatomical_region_72.png" in calls_by_name
    assert "visibility_f1_by_anatomical_region_common68.png" in calls_by_name
    assert "under_lip" in calls_by_name["visibility_f1_by_anatomical_region_72.png"]
    assert (
        "under_lip"
        not in calls_by_name["visibility_f1_by_anatomical_region_common68.png"]
    )
