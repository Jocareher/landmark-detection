from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from scripts.analysis.benchmark_landmark_models import (
    BenchmarkAnalysisConfig,
    BenchmarkMetricConfig,
    DatasetBenchmarkConfig,
    DetectionRateInfo,
    EvaluationProtocolConfig,
    LoadedModelResults,
    ModelBenchmarkConfig,
    compute_best_worst_cases,
    expand_models_for_analysis,
    filter_per_landmark_by_protocol,
    infer_detection_rate_info,
    load_model_results,
    split_best_worst_case_tables,
)
from scripts.engine.evaluate import (
    compute_normalized_hausdorff_distance,
    compute_symmetric_hausdorff_distance,
)
from scripts.engine.evaluation_modes import compute_masked_natural_per_landmark_nme
from scripts.utils.visualization import save_landmark_overlay_image
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


def test_symmetric_hausdorff_distance_handles_basic_cases() -> None:
    """Hausdorff is zero for identical sets, shifted for translated sets, and nan for empty sets."""
    points = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    shifted = points + np.asarray([2.0, 0.0], dtype=np.float32)

    assert compute_symmetric_hausdorff_distance(points, points) == 0.0
    assert compute_symmetric_hausdorff_distance(shifted, points) == 2.0
    assert np.isnan(compute_symmetric_hausdorff_distance(points[:0], points))


def test_normalized_hausdorff_uses_box_denominator() -> None:
    """Normalized Hausdorff uses the same box denominator as NME."""
    target = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    predicted = target + np.asarray([2.0, 0.0], dtype=np.float32)
    valid_mask = np.ones(3, dtype=bool)

    pixel, normalized = compute_normalized_hausdorff_distance(
        predicted,
        target,
        valid_mask,
    )

    assert pixel == 2.0
    assert np.isclose(normalized, 0.5)


def test_infanface_non_contour_hausdorff_excludes_contour_and_uses_full_denominator() -> None:
    """Non-contour Hausdorff excludes contour points but keeps the full-face denominator."""
    target = np.zeros((68, 2), dtype=np.float32)
    target[0] = [0.0, 0.0]
    target[16] = [10.0, 10.0]
    target[17:] = np.stack(
        [
            np.linspace(2.0, 8.0, 51),
            np.linspace(2.0, 8.0, 51),
        ],
        axis=1,
    )
    predicted = target.copy()
    predicted[:17] += 100.0
    predicted[17:] += np.asarray([2.0, 0.0], dtype=np.float32)
    non_contour_mask = np.zeros(68, dtype=bool)
    non_contour_mask[17:] = True

    pixel, normalized = compute_normalized_hausdorff_distance(
        predicted,
        target,
        non_contour_mask,
        normalization_landmarks=target,
    )

    assert np.isclose(pixel, 2.0)
    assert np.isclose(normalized, 0.2)


def test_overlay_connections_ignore_visibility_for_finite_landmarks(tmp_path) -> None:
    """Invisible finite landmarks are blue points, but connections still pass through them."""
    image_path = tmp_path / "source.png"
    output_path = tmp_path / "overlay.png"
    Image.new("RGB", (80, 80), color="white").save(image_path)
    landmarks = np.zeros((68, 2), dtype=np.float32)
    landmarks[:17] = np.asarray([[10.0 + index * 3.0, 20.0] for index in range(17)])
    landmarks[17:] = [10.0, 50.0]
    visibility = np.ones(68, dtype=np.int64)
    visibility[1] = 0

    save_landmark_overlay_image(
        image_path=image_path,
        output_path=output_path,
        predicted_landmarks=landmarks,
        predicted_visibility=visibility,
        point_radius=2,
        line_width=2,
        line_color="#00C853",
    )

    rendered = Image.open(output_path).convert("RGB")
    line_pixel = rendered.getpixel((12, 20))
    visible_pixel = rendered.getpixel((10, 20))
    invisible_pixel = rendered.getpixel((13, 20))
    assert line_pixel != (255, 255, 255)
    assert visible_pixel[0] > visible_pixel[2]
    assert invisible_pixel[2] > invisible_pixel[0]


def test_detection_rate_uses_detected_column() -> None:
    """Detected columns are the preferred detection-rate source."""
    raw = pd.DataFrame({"image_id": ["a", "b", "c"], "nme": [0.1, np.nan, 0.2], "detected": [1, 0, 1]})

    info = infer_detection_rate_info(
        raw=raw,
        image_col="image_id",
        nme_col="nme",
        detected_col="detected",
        model_config=ModelBenchmarkConfig(name="model"),
        dataset_config=DatasetBenchmarkConfig(name="dataset", output_dir="unused", total_images=10),
    )

    assert info.detection_rate == 2 / 3
    assert info.detection_rate_source == "detected_column"
    assert info.n_detected == 2
    assert info.n_total == 3


def test_detection_rate_uses_dataset_total_without_defaulting_to_full_coverage() -> None:
    """Evaluated-only CSVs use n_detected / dataset.total_images instead of assuming 100%."""
    raw = pd.DataFrame({"image_id": ["a", "b"], "nme": [0.1, 0.2]})

    info = infer_detection_rate_info(
        raw=raw,
        image_col="image_id",
        nme_col="nme",
        detected_col=None,
        model_config=ModelBenchmarkConfig(name="model"),
        dataset_config=DatasetBenchmarkConfig(name="dataset", output_dir="unused", total_images=5),
    )

    assert info.detection_rate == 0.4
    assert info.detection_rate_source == "n_detected_over_dataset_total"
    assert info.n_detected == 2
    assert info.n_total == 5


def test_detection_rate_unavailable_without_source() -> None:
    """Detection rate is marked unavailable when no source can infer it."""
    raw = pd.DataFrame({"image_id": ["a", "b"], "nme": [0.1, 0.2]})

    info = infer_detection_rate_info(
        raw=raw,
        image_col="image_id",
        nme_col="nme",
        detected_col=None,
        model_config=ModelBenchmarkConfig(name="model"),
        dataset_config=DatasetBenchmarkConfig(name="dataset", output_dir="unused"),
    )

    assert isinstance(info, DetectionRateInfo)
    assert info.detection_rate is None
    assert info.detection_rate_source == "unavailable"


def test_babyland_standard_hausdorff_prefers_gt_valid_column(tmp_path) -> None:
    """Main-comparison SOTA Hausdorff should not select the empty generic column."""
    per_image_csv = tmp_path / "per_image_nme.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "hausdorff_box": [np.nan, np.nan],
            "hausdorff_box_gt_valid": [0.2, 0.4],
        }
    ).to_csv(per_image_csv, index=False)
    config = BenchmarkAnalysisConfig(
        dataset=DatasetBenchmarkConfig(
            name="BabyLand-72",
            output_dir=tmp_path,
            dataset_type="babyland72",
            total_images=2,
        ),
        models=[],
        column_mappings={
            "per_image": {
                "standard": {
                    "hausdorff": {
                        "candidates": ["hausdorff_box_gt_valid", "hausdorff_box"]
                    }
                }
            }
        },
    )
    protocol = EvaluationProtocolConfig(name="standard", display_name="Standard")
    metric = BenchmarkMetricConfig(
        name="hausdorff",
        display_name="Hausdorff",
        per_image_column_candidates=("hausdorff_box",),
        per_landmark_column_candidates=(),
    )

    result = load_model_results(
        ModelBenchmarkConfig(name="sota", per_image_csv=per_image_csv),
        config,
        protocol,
        metric,
        selected_metric_source="standard",
    )

    assert result.per_image is not None
    assert result.per_image["selected_metric_column"].iloc[0] == "hausdorff_box_gt_valid"
    assert result.per_image["nme"].tolist() == [0.2, 0.4]


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
