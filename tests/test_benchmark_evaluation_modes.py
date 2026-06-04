from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.benchmark_landmark_models import (
    BenchmarkAnalysisConfig,
    BenchmarkMetricConfig,
    DatasetBenchmarkConfig,
    EvaluationProtocolConfig,
    LoadedModelResults,
    ModelBenchmarkConfig,
    compute_best_worst_cases,
    expand_models_for_analysis,
    filter_per_landmark_by_protocol,
    split_best_worst_case_tables,
)
from scripts.engine.evaluation_modes import compute_masked_natural_per_landmark_nme
from scripts.utils.orientation import normalize_orientation_label


def simple_box_normalization(target_landmarks: np.ndarray, eps: float = 1e-6) -> float:
    """Small test-local box normalization function."""
    width = float(np.max(target_landmarks[:, 0]) - np.min(target_landmarks[:, 0]))
    height = float(np.max(target_landmarks[:, 1]) - np.min(target_landmarks[:, 1]))
    return max(width, height, eps)


def test_orientation_class_labels_are_normalized() -> None:
    """SOTA yaw_plus labels are class ids, not degree values."""
    assert normalize_orientation_label("yaw_plus_0deg") == "left"
    assert normalize_orientation_label("yaw_plus_1deg") == "quarter_left"
    assert normalize_orientation_label("yaw_plus_2deg") == "frontal"
    assert normalize_orientation_label("yaw_plus_3deg") == "quarter_right"
    assert normalize_orientation_label("yaw_plus_4deg") == "right"


def test_gt_valid_includes_predicted_invisible_and_excludes_nan_gt() -> None:
    """The gt_valid mode ignores predicted visibility but excludes missing GT."""
    predicted = np.asarray([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.0], [2.0, 1.0], [np.nan, np.nan]], dtype=np.float32)
    target_visibility = np.asarray([1, 1, 0], dtype=np.int64)
    predicted_visibility = np.asarray([1, 0, 1], dtype=np.int64)

    visible_errors, _, _, _ = compute_masked_natural_per_landmark_nme(
        predicted,
        target,
        target_visibility,
        predicted_visibility,
        normalization_fn=simple_box_normalization,
        inclusion_mode="visible_intersection",
    )
    gt_valid_errors, _, _, _ = compute_masked_natural_per_landmark_nme(
        predicted,
        target,
        target_visibility,
        predicted_visibility,
        normalization_fn=simple_box_normalization,
        inclusion_mode="gt_valid",
    )

    assert set(visible_errors) == {0}
    assert set(gt_valid_errors) == {0, 1}


def test_best_worst_cases_limits_to_ten_per_side() -> None:
    """Best/worst extraction returns ten rows per side when enough images exist."""
    per_image = pd.DataFrame(
        {
            "model": "model_a",
            "image_id": [f"img_{index:02d}" for index in range(12)],
            "image_key": [f"img_{index:02d}" for index in range(12)],
            "orientation": ["frontal"] * 12,
            "nme": np.linspace(0.0, 0.11, 12),
            "detected": [True] * 12,
        }
    )
    result = LoadedModelResults(
        config=ModelBenchmarkConfig(name="model_a"),
        per_image=per_image,
        per_landmark=None,
    )
    cases = compute_best_worst_cases(
        [result],
        DatasetBenchmarkConfig(name="dataset", output_dir="unused"),
        BenchmarkMetricConfig(
            name="point_to_point",
            display_name="Point-to-point",
            per_image_column_candidates=("nme",),
            per_landmark_column_candidates=("nme",),
        ),
    )
    assert len(cases[cases["rank_type"] == "best"]) == 10
    assert len(cases[cases["rank_type"] == "worst"]) == 10


def test_best_worst_cases_are_split_into_two_consolidated_tables() -> None:
    """Best/worst CSV tables keep all models together instead of splitting by model."""
    cases = pd.DataFrame(
        {
            "model_name": ["model_a", "model_b", "model_a", "model_b"],
            "rank_type": ["best", "best", "worst", "worst"],
            "rank": [1, 1, 1, 1],
            "image_level_nme": [0.01, 0.02, 0.50, 0.60],
        }
    )

    tables = split_best_worst_case_tables(cases)

    assert set(tables) == {"best_cases", "worst_cases"}
    assert list(tables["best_cases"]["model_name"]) == ["model_a", "model_b"]
    assert list(tables["worst_cases"]["model_name"]) == ["model_a", "model_b"]


def test_babyland_analysis_uses_visibility_flag_capability() -> None:
    """BabyLand main comparison falls back to standard SOTA metrics without visibility."""
    config = BenchmarkAnalysisConfig(
        dataset=DatasetBenchmarkConfig(
            name="BabyLand-72",
            output_dir="unused",
            dataset_type="babyland72",
        ),
        models=[
            ModelBenchmarkConfig(name="visibility_model", predicts_visibility=True),
            ModelBenchmarkConfig(name="sota_model", predicts_visibility=False),
        ],
    )

    main_jobs, main_warnings = expand_models_for_analysis(
        config,
        EvaluationProtocolConfig(name="main_comparison", display_name="Main comparison"),
    )
    assert main_warnings == []
    assert [(model.name, source.name, source_name) for model, source, source_name in main_jobs] == [
        ("visibility_model", "gt_valid", "gt_valid"),
        ("sota_model", "standard", "standard"),
    ]

    visibility_jobs, visibility_warnings = expand_models_for_analysis(
        config,
        EvaluationProtocolConfig(
            name="visibility_protocol_analysis",
            display_name="Visibility protocol analysis",
        ),
    )
    assert len(visibility_jobs) == 2
    assert [source_name for _, _, source_name in visibility_jobs] == [
        "visibility_intersection",
        "gt_valid",
    ]
    assert any("sota_model excluded" in warning for warning in visibility_warnings)


def test_babyland_standard_keeps_sota_rows_with_empty_inclusion_column() -> None:
    """Standard SOTA metrics should not require visibility-inclusion labels."""
    raw = pd.DataFrame(
        {
            "landmark_idx": [0, 1, 2],
            "evaluation_landmark_inclusion": [np.nan, np.nan, np.nan],
        }
    )
    mask = filter_per_landmark_by_protocol(
        raw,
        DatasetBenchmarkConfig(
            name="BabyLand-72",
            output_dir="unused",
            dataset_type="babyland72",
        ),
        EvaluationProtocolConfig(name="standard", display_name="Standard"),
        raw["landmark_idx"],
    )

    assert mask.tolist() == [True, True, True]
