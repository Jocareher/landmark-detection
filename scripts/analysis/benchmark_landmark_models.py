from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.orientation import ORIENTATION_ORDER, normalize_orientation_label

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent.
    yaml = None

_SCIPY_STATS: Any | None = None
_SCIPY_IMPORT_ATTEMPTED = False


IMAGE_ID_ALIASES = ("image_id", "sample_id", "stem", "filename", "file", "image")
POINT_TO_POINT_IMAGE_NME_ALIASES = (
    "nme",
    "image_nme",
    "mean_nme",
    "mean_nme_box",
    "box_nme",
    "nme_box",
)
POINT_TO_POINT_LANDMARK_NME_ALIASES = (
    "nme",
    "point_to_point_nme_box",
    "landmark_nme",
    "nme_box",
)
POINT_TO_LINE_IMAGE_NME_ALIASES = (
    "mean_nme_box_point_to_line",
    "point_to_line_nme_box",
    "point_to_line_nme",
)
POINT_TO_LINE_LANDMARK_NME_ALIASES = ("point_to_line_nme_box", "point_to_line_nme")
HAUSDORFF_IMAGE_NME_ALIASES = (
    "hausdorff_box",
    "mean_hausdorff_box",
    "image_hausdorff_box",
    "hausdorff_nme_box",
    "hausdorff_box_all_landmarks",
    "hausdorff_box_non_contour",
    "hausdorff_box_visibility_intersection",
    "hausdorff_box_visible_intersection",
    "hausdorff_box_gt_valid",
)
LANDMARK_INDEX_ALIASES = (
    "landmark_index",
    "landmark_idx",
    "landmark_id",
    "landmark",
    "point_index",
)
ORIENTATION_ALIASES = ("orientation", "yaw_group", "yaw_angle", "class_idx", "pose")
DETECTED_ALIASES = ("detected", "is_detected", "success", "valid_detection")

DEFAULT_MODEL_ORDER = ["vggheads", "Exp11", "dslpt", "mediapipe", "dlib"]
DEFAULT_MODEL_COLORS = {
    "vggheads": "#4C78A8",
    "Exp11": "#F58518",
    "BabyLand-72 Exp11": "#F58518",
    "dslpt": "#54A24B",
    "mediapipe": "#B279A2",
    "dlib": "#E45756",
}
FALLBACK_COLORS = [
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#A0CBE8",
    "#FFBE7D",
    "#8CD17D",
    "#D4A6C8",
]

DEFAULT_BABYLAND_ORIENTATION_MAPPING = {
    "0": "left",
    "1": "quarter_left",
    "2": "frontal",
    "3": "quarter_right",
    "4": "right",
}

DEFAULT_BABYLAND72_REGIONS = {
    "face_contour": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_base": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
    "under_lip": [68],
    "upper_chin": [69],
    "left_chin": [70],
    "right_chin": [71],
}

DEFAULT_MULTIPIE68_REGIONS = {
    region: indices
    for region, indices in DEFAULT_BABYLAND72_REGIONS.items()
    if max(indices) < 68
}


@dataclass
class BenchmarkMetricConfig:
    """Configuration for one geometric benchmark metric."""

    name: str
    display_name: str
    per_image_column_candidates: tuple[str, ...]
    per_landmark_column_candidates: tuple[str, ...]


@dataclass
class EvaluationProtocolConfig:
    """Configuration for one dataset-specific evaluation protocol."""

    name: str
    display_name: str
    description: str = ""


@dataclass
class ModelBenchmarkConfig:
    """Configuration for one model in the landmark benchmark."""

    name: str
    per_image_csv: Path | None = None
    per_landmark_csv: Path | None = None
    detection_rate: float | None = None
    detection_rate_fallback: float | None = None
    landmark_format: str | None = None
    display_name: str | None = None
    predicts_visibility: bool = False
    columns: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class DatasetBenchmarkConfig:
    """Dataset-level benchmark configuration."""

    name: str
    output_dir: Path
    dataset_type: str = ""
    primary_model: str | None = None
    reference_models: list[str] = field(default_factory=list)
    orientation_mapping: dict[str, str] = field(default_factory=dict)
    anatomical_regions: dict[str, list[int]] = field(default_factory=dict)
    total_images: int | None = None
    landmark_count: int | None = None
    landmark_index_base: str = "auto"
    image_id_strip_regexes: list[str] = field(default_factory=lambda: [r"__det_\\d+$"])
    suspicious_nme_threshold: float = 1.0
    model_order: list[str] = field(default_factory=list)
    model_display_names: dict[str, str] = field(default_factory=dict)
    model_colors: dict[str, str] = field(default_factory=dict)
    orientation_order: list[str] = field(
        default_factory=lambda: list(ORIENTATION_ORDER)
    )
    include_unknown_orientations: bool = False
    ced_zoom_max_nme: float = 0.40
    plot_dpi: int = 300
    use_percent_axis: bool = False
    annotate_bars: bool = True
    annotate_boxplot_means: bool = True
    show_violin_plots: bool = True
    generate_log_scale_variants: bool = True
    log_scale_epsilon: float = 1.0e-8
    evaluation_protocol: str = ""
    evaluation_protocol_display_name: str = ""


@dataclass
class BenchmarkAnalysisConfig:
    """Complete benchmark-analysis configuration."""

    dataset: DatasetBenchmarkConfig
    models: list[ModelBenchmarkConfig]
    evaluation_protocols: list[EvaluationProtocolConfig] = field(default_factory=list)
    drop_invalid_nme: bool = False
    bootstrap_iterations: int = 2000
    random_seed: int = 12345
    save_pdf: bool = False
    metrics: list[BenchmarkMetricConfig] = field(default_factory=list)
    column_mappings: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionRateInfo:
    """Resolved detection-rate metadata for one model/result table."""

    detection_rate: float | None = None
    detection_rate_source: str = "unavailable"
    n_detected: int | None = None
    n_total: int | None = None
    detected_column_name: str = ""
    warning_message: str = ""


@dataclass
class LoadedModelResults:
    """Loaded and normalized result tables for one model."""

    config: ModelBenchmarkConfig
    per_image: pd.DataFrame | None
    per_landmark: pd.DataFrame | None
    detection_rate_info: DetectionRateInfo = field(default_factory=DetectionRateInfo)
    warnings: list[str] = field(default_factory=list)
    orientation_warnings: list[dict[str, Any]] = field(default_factory=list)
    column_audit: list[dict[str, Any]] = field(default_factory=list)


DEFAULT_BENCHMARK_METRICS = [
    BenchmarkMetricConfig(
        name="point_to_point",
        display_name="Point-to-point",
        per_image_column_candidates=POINT_TO_POINT_IMAGE_NME_ALIASES,
        per_landmark_column_candidates=POINT_TO_POINT_LANDMARK_NME_ALIASES,
    ),
    BenchmarkMetricConfig(
        name="point_to_line",
        display_name="Point-to-line",
        per_image_column_candidates=POINT_TO_LINE_IMAGE_NME_ALIASES,
        per_landmark_column_candidates=POINT_TO_LINE_LANDMARK_NME_ALIASES,
    ),
    BenchmarkMetricConfig(
        name="hausdorff",
        display_name="Hausdorff",
        per_image_column_candidates=HAUSDORFF_IMAGE_NME_ALIASES,
        per_landmark_column_candidates=(),
    ),
]

DEFAULT_PROTOCOLS_BY_DATASET_TYPE = {
    "babyland72": [
        EvaluationProtocolConfig(
            name="main_comparison",
            display_name="Main SOTA comparison",
            description=(
                "Uses GT-valid metrics for visibility-aware models and standard "
                "valid-landmark metrics for models without predicted visibility."
            ),
        ),
        EvaluationProtocolConfig(
            name="visibility_protocol_analysis",
            display_name="Visibility protocol analysis",
            description=(
                "Compares visibility_intersection and gt_valid for models that "
                "explicitly predict visibility."
            ),
        ),
    ],
    "infanface": [
        EvaluationProtocolConfig(
            name="all_landmarks",
            display_name="All landmarks",
            description="Uses all evaluated landmarks.",
        ),
        EvaluationProtocolConfig(
            name="non_contour",
            display_name="Non-contour landmarks",
            description="Excludes face-contour landmarks from the averaged error.",
        ),
    ],
}

DEFAULT_COLUMN_MAPPINGS = {
    "babyland72": {
        "per_image": {
            "standard": {
                "point_to_point": {
                    "candidates": [
                        "mean_nme_box",
                        "image_nme",
                        "point_to_point_nme_box",
                        "point_to_point_nme",
                        "nme",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "mean_nme_box_point_to_line",
                        "point_to_line_nme_box",
                        "point_to_line_nme",
                    ]
                },
                "hausdorff": {
                    "candidates": [
                        "hausdorff_box_gt_valid",
                        "mean_hausdorff_box_gt_valid",
                        "hausdorff_box",
                        "mean_hausdorff_box",
                        "image_hausdorff_box",
                        "hausdorff_nme_box",
                    ]
                },
            },
            "visibility_intersection": {
                "point_to_point": {
                    "candidates": [
                        "mean_nme_box_visibility_intersection",
                        "mean_nme_box_visible_intersection",
                        "mean_nme_visibility_intersection",
                        "mean_nme_box",
                        "image_nme",
                        "nme",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "mean_nme_box_point_to_line_visibility_intersection",
                        "mean_nme_box_point_to_line_visible_intersection",
                        "point_to_line_nme_box_visibility_intersection",
                        "mean_nme_box_point_to_line",
                    ]
                },
                "hausdorff": {
                    "candidates": [
                        "hausdorff_box_visibility_intersection",
                        "hausdorff_box_visible_intersection",
                        "mean_hausdorff_box_visibility_intersection",
                        "mean_hausdorff_box_visible_intersection",
                    ]
                },
            },
            "gt_valid": {
                "point_to_point": {
                    "candidates": [
                        "mean_nme_box_gt_valid",
                        "point_to_point_nme_box_gt_valid",
                        "mean_nme_gt_valid",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "mean_nme_box_point_to_line_gt_valid",
                        "point_to_line_nme_box_gt_valid",
                        "mean_nme_point_to_line_gt_valid",
                    ]
                },
                "hausdorff": {
                    "candidates": [
                        "hausdorff_box_gt_valid",
                        "mean_hausdorff_box_gt_valid",
                    ]
                },
            },
        },
        "per_landmark": {
            "standard": {
                "point_to_point": {
                    "candidates": [
                        "point_to_point_nme_box",
                        "point_to_point_nme",
                        "nme",
                    ]
                },
                "point_to_line": {
                    "candidates": ["point_to_line_nme_box", "point_to_line_nme"]
                },
                "hausdorff": {"candidates": []},
            },
            "visibility_intersection": {
                "point_to_point": {
                    "candidates": [
                        "point_to_point_nme_box_visibility_intersection",
                        "point_to_point_nme_visibility_intersection",
                        "point_to_point_nme_box",
                        "nme",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "point_to_line_nme_box_visibility_intersection",
                        "point_to_line_nme_box",
                    ]
                },
                "hausdorff": {"candidates": []},
            },
            "gt_valid": {
                "point_to_point": {
                    "candidates": [
                        "point_to_point_nme_box_gt_valid",
                        "point_to_point_nme_gt_valid",
                        "point_to_point_nme_box",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "point_to_line_nme_box_gt_valid",
                        "point_to_line_nme_gt_valid",
                        "point_to_line_nme_box",
                    ]
                },
                "hausdorff": {"candidates": []},
            },
        },
    },
    "infanface": {
        "per_image": {
            "all_landmarks": {
                "point_to_point": {
                    "candidates": [
                        "mean_nme_box_all",
                        "mean_nme_box_all_landmarks",
                        "mean_nme_box",
                        "image_nme",
                        "nme",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "mean_nme_box_point_to_line_all",
                        "mean_nme_box_point_to_line_all_landmarks",
                        "mean_nme_box_point_to_line",
                    ]
                },
                "hausdorff": {
                    "candidates": [
                        "hausdorff_box_all_landmarks",
                        "hausdorff_box_all",
                        "hausdorff_box",
                        "mean_hausdorff_box",
                    ]
                },
            },
            "non_contour": {
                "point_to_point": {
                    "candidates": [
                        "mean_nme_box_non_contour",
                        "point_to_point_nme_box_non_contour",
                    ]
                },
                "point_to_line": {
                    "candidates": [
                        "mean_nme_box_point_to_line_non_contour",
                        "point_to_line_nme_box_non_contour",
                    ]
                },
                "hausdorff": {
                    "candidates": [
                        "hausdorff_box_non_contour",
                        "mean_hausdorff_box_non_contour",
                    ]
                },
            },
        },
        "per_landmark": {
            "all_landmarks": {
                "point_to_point": {"candidates": ["point_to_point_nme_box", "nme"]},
                "point_to_line": {"candidates": ["point_to_line_nme_box"]},
                "hausdorff": {"candidates": []},
            },
            "non_contour": {
                "point_to_point": {"candidates": ["point_to_point_nme_box", "nme"]},
                "point_to_line": {"candidates": ["point_to_line_nme_box"]},
                "hausdorff": {"candidates": []},
            },
        },
    },
}


def load_config(
    config_path: str | Path, drop_invalid_nme: bool | None = None
) -> BenchmarkAnalysisConfig:
    """Load a benchmark config from YAML or JSON."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "PyYAML is required to read YAML configs. Install pyyaml or use JSON."
            )
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    else:
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

    dataset_raw = raw.get("dataset", {})
    dataset_name = str(dataset_raw.get("name", "benchmark"))
    dataset_type = str(dataset_raw.get("dataset_type", "")).strip().lower()
    if not dataset_type:
        dataset_name_lower = dataset_name.lower()
        if "infanface" in dataset_name_lower or "infantface" in dataset_name_lower:
            dataset_type = "infanface"
        elif "babyland" in dataset_name_lower:
            dataset_type = "babyland72"
    regions = dataset_raw.get("anatomical_regions") or {}
    if not regions and dataset_type == "babyland72":
        regions = DEFAULT_BABYLAND72_REGIONS
    elif not regions and dataset_type == "infanface":
        regions = DEFAULT_MULTIPIE68_REGIONS

    orientation_mapping = dataset_raw.get("orientation_mapping") or {}
    if not orientation_mapping and dataset_type in {"babyland72", "infanface"}:
        orientation_mapping = DEFAULT_BABYLAND_ORIENTATION_MAPPING

    plotting_raw = raw.get("plotting", {})
    dataset = DatasetBenchmarkConfig(
        name=dataset_name,
        output_dir=Path(dataset_raw["output_dir"]),
        dataset_type=dataset_type,
        primary_model=dataset_raw.get("primary_model"),
        reference_models=list(dataset_raw.get("reference_models", [])),
        orientation_mapping={
            str(key): str(value) for key, value in orientation_mapping.items()
        },
        anatomical_regions={str(key): list(value) for key, value in regions.items()},
        total_images=(
            None
            if dataset_raw.get("total_images") is None
            else int(dataset_raw["total_images"])
        ),
        landmark_count=dataset_raw.get("landmark_count"),
        landmark_index_base=str(dataset_raw.get("landmark_index_base", "auto")),
        image_id_strip_regexes=list(
            dataset_raw.get("image_id_strip_regexes", [r"__det_\\d+$"])
        ),
        suspicious_nme_threshold=float(
            dataset_raw.get("suspicious_nme_threshold", 1.0)
        ),
        model_order=list(dataset_raw.get("model_order", raw.get("model_order", []))),
        model_display_names={
            str(key): str(value)
            for key, value in dataset_raw.get(
                "model_display_names", raw.get("model_display_names", {})
            ).items()
        },
        model_colors={
            str(key): str(value)
            for key, value in dataset_raw.get(
                "model_colors", raw.get("model_colors", {})
            ).items()
        },
        orientation_order=list(
            dataset_raw.get(
                "orientation_order", raw.get("orientation_order", ORIENTATION_ORDER)
            )
        ),
        include_unknown_orientations=bool(
            dataset_raw.get(
                "include_unknown_orientations",
                raw.get("include_unknown_orientations", False),
            )
        ),
        ced_zoom_max_nme=float(
            dataset_raw.get(
                "ced_zoom_max_nme",
                plotting_raw.get("ced_zoom_max_nme", raw.get("ced_zoom_max_nme", 0.40)),
            )
        ),
        plot_dpi=int(
            dataset_raw.get(
                "plot_dpi", plotting_raw.get("plot_dpi", raw.get("plot_dpi", 300))
            )
        ),
        use_percent_axis=bool(
            dataset_raw.get("use_percent_axis", raw.get("use_percent_axis", False))
        ),
        annotate_bars=bool(
            dataset_raw.get("annotate_bars", raw.get("annotate_bars", True))
        ),
        annotate_boxplot_means=bool(
            dataset_raw.get(
                "annotate_boxplot_means", raw.get("annotate_boxplot_means", True)
            )
        ),
        show_violin_plots=bool(
            dataset_raw.get("show_violin_plots", raw.get("show_violin_plots", True))
        ),
        generate_log_scale_variants=bool(
            dataset_raw.get(
                "generate_log_scale_variants",
                plotting_raw.get(
                    "generate_log_scale_variants",
                    raw.get("generate_log_scale_variants", True),
                ),
            )
        ),
        log_scale_epsilon=float(
            dataset_raw.get(
                "log_scale_epsilon",
                plotting_raw.get(
                    "log_scale_epsilon", raw.get("log_scale_epsilon", 1.0e-8)
                ),
            )
        ),
    )

    models = []
    for item in raw.get("models", []):
        models.append(
            ModelBenchmarkConfig(
                name=str(item["name"]),
                detection_rate=(
                    None
                    if item.get("detection_rate") is None
                    else float(item["detection_rate"])
                ),
                detection_rate_fallback=(
                    None
                    if item.get("detection_rate_fallback") is None
                    else float(item["detection_rate_fallback"])
                ),
                per_image_csv=Path(item["per_image_csv"])
                if item.get("per_image_csv")
                else None,
                per_landmark_csv=Path(item["per_landmark_csv"])
                if item.get("per_landmark_csv")
                else None,
                landmark_format=None
                if item.get("landmark_format") is None
                else str(item["landmark_format"]),
                display_name=item.get("display_name"),
                predicts_visibility=bool(item.get("predicts_visibility", False)),
                columns=dict(item.get("columns", {})),
            )
        )

    if not models:
        raise ValueError("Config must contain at least one model.")

    protocol_configs = []
    for item in raw.get("evaluation_protocols", []):
        protocol_configs.append(
            EvaluationProtocolConfig(
                name=str(item["name"]),
                display_name=str(item.get("display_name", item["name"])),
                description=str(item.get("description", "")),
            )
        )
    if not protocol_configs:
        protocol_configs = list(DEFAULT_PROTOCOLS_BY_DATASET_TYPE.get(dataset_type, []))
    if not protocol_configs:
        protocol_configs = [
            EvaluationProtocolConfig(
                name="default",
                display_name="Default evaluation",
                description="Default CSV metric columns.",
            )
        ]

    metric_configs = []
    for item in raw.get("metrics", []):
        metric_configs.append(
            BenchmarkMetricConfig(
                name=str(item["name"]),
                display_name=str(item.get("display_name", item["name"])),
                per_image_column_candidates=tuple(
                    str(value) for value in item.get("per_image_column_candidates", [])
                ),
                per_landmark_column_candidates=tuple(
                    str(value)
                    for value in item.get("per_landmark_column_candidates", [])
                ),
            )
        )
    if not metric_configs:
        metric_configs = DEFAULT_BENCHMARK_METRICS

    default_column_mappings = DEFAULT_COLUMN_MAPPINGS.get(dataset_type, {})
    column_mappings = raw.get("column_mappings") or default_column_mappings

    return BenchmarkAnalysisConfig(
        dataset=dataset,
        models=models,
        evaluation_protocols=protocol_configs,
        drop_invalid_nme=bool(
            raw.get("drop_invalid_nme", False)
            if drop_invalid_nme is None
            else drop_invalid_nme
        ),
        bootstrap_iterations=int(raw.get("bootstrap_iterations", 2000)),
        random_seed=int(raw.get("random_seed", 12345)),
        save_pdf=bool(plotting_raw.get("save_pdf", raw.get("save_pdf", False))),
        metrics=metric_configs,
        column_mappings=column_mappings,
    )


def normalize_image_id(value: Any, strip_regexes: list[str] | None = None) -> str:
    """Normalize image identifiers so rows from different model CSVs can be paired."""
    text = "" if pd.isna(value) else str(value)
    text = Path(text).name
    text = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", text, flags=re.IGNORECASE)
    for pattern in strip_regexes or []:
        text = re.sub(pattern, "", text)
    return text.strip()


def get_model_display_name(
    model_name: str, config: BenchmarkAnalysisConfig | DatasetBenchmarkConfig
) -> str:
    """Return the configured display name for a raw model name."""
    dataset = config.dataset if isinstance(config, BenchmarkAnalysisConfig) else config
    return dataset.model_display_names.get(model_name, model_name)


def get_ordered_model_names(
    models: list[str],
    dataset_config: DatasetBenchmarkConfig,
) -> list[str]:
    """Order raw model names using configured raw or display-name order."""
    available = list(dict.fromkeys(models))
    configured = dataset_config.model_order or DEFAULT_MODEL_ORDER
    display_to_raw = {
        get_model_display_name(name, dataset_config): name for name in available
    }
    ordered = []
    for item in configured:
        if item in available and item not in ordered:
            ordered.append(item)
        elif item in display_to_raw and display_to_raw[item] not in ordered:
            ordered.append(display_to_raw[item])
    ordered.extend([name for name in available if name not in ordered])
    return ordered


def build_model_style_maps(
    model_names: list[str],
    dataset_config: DatasetBenchmarkConfig,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Build display-name and color maps for all models, with deterministic fallbacks."""
    ordered = get_ordered_model_names(model_names, dataset_config)
    display_names = {
        name: get_model_display_name(name, dataset_config) for name in ordered
    }
    colors = dict(DEFAULT_MODEL_COLORS)
    colors.update(dataset_config.model_colors)
    fallback_warnings = []
    fallback_index = 0
    color_map = {}
    for raw_name in ordered:
        display_name = display_names[raw_name]
        color = colors.get(raw_name, colors.get(display_name))
        if color is None:
            color = FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]
            fallback_index += 1
            fallback_warnings.append(
                f"No configured color for model {raw_name!r}; assigned fallback color {color}."
            )
        color_map[raw_name] = color
    return display_names, color_map, fallback_warnings


def label_for_model(model_name: str, display_names: dict[str, str]) -> str:
    """Return display label for a raw model name."""
    return display_names.get(model_name, model_name)


def resolve_column(
    df: pd.DataFrame,
    canonical_name: str,
    aliases: tuple[str, ...],
    explicit_mapping: dict[str, str] | None = None,
    required: bool = True,
) -> str | None:
    """Resolve a canonical column name using explicit config or known aliases."""
    explicit_mapping = explicit_mapping or {}
    if canonical_name in explicit_mapping:
        column = explicit_mapping[canonical_name]
        if column not in df.columns:
            raise ValueError(
                f"Configured column {column!r} for {canonical_name!r} is not present."
            )
        return column

    normalized = {str(column).lower(): column for column in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]

    if required:
        raise ValueError(
            f"Could not resolve required column {canonical_name!r}. "
            f"Tried aliases: {', '.join(aliases)}. Available columns: {list(df.columns)}"
        )
    return None


def get_protocol_metric_candidates(
    config: BenchmarkAnalysisConfig,
    csv_type: str,
    protocol_config: EvaluationProtocolConfig,
    metric_config: BenchmarkMetricConfig,
) -> tuple[str, ...]:
    """Return configured candidate columns for one CSV/protocol/metric combination."""
    mapping = (
        config.column_mappings.get(csv_type, {})
        .get(protocol_config.name, {})
        .get(metric_config.name, {})
    )
    candidates = mapping.get("candidates", []) if isinstance(mapping, dict) else []
    if candidates:
        return tuple(str(candidate) for candidate in candidates)
    if csv_type == "per_image":
        return metric_config.per_image_column_candidates
    return metric_config.per_landmark_column_candidates


def resolve_column_with_warning(
    df: pd.DataFrame,
    canonical_name: str,
    aliases: tuple[str, ...],
    explicit_mapping: dict[str, str] | None,
    model_name: str,
    csv_type: str,
    warnings: list[str],
    required: bool = True,
) -> str | None:
    """Resolve a column and append a non-fatal warning if it is unavailable."""
    try:
        return resolve_column(
            df=df,
            canonical_name=canonical_name,
            aliases=aliases,
            explicit_mapping=explicit_mapping,
            required=required,
        )
    except ValueError as error:
        warnings.append(f"{model_name} {csv_type}: {error}")
        return None


def build_column_audit_row(
    model_config: ModelBenchmarkConfig,
    csv_type: str,
    csv_path: Path | None,
    raw: pd.DataFrame | None,
    dataset_config: DatasetBenchmarkConfig,
    protocol_config: EvaluationProtocolConfig,
    metric_config: BenchmarkMetricConfig,
    selected_nme_column: str | None,
    missing_required_columns: list[str],
    analysis_name: str | None = None,
    selected_metric_source: str | None = None,
) -> dict[str, Any]:
    """Build one CSV column-audit row for debugging benchmark inputs."""
    if raw is None:
        return {
            "dataset_name": dataset_config.name,
            "analysis_name": analysis_name or protocol_config.name,
            "evaluation_protocol": protocol_config.name,
            "metric_name": metric_config.name,
            "model_name": model_config.name,
            "predicts_visibility": bool(model_config.predicts_visibility),
            "selected_metric_source": selected_metric_source or protocol_config.name,
            "csv_type": csv_type,
            "csv_path": str(csv_path) if csv_path is not None else "",
            "available_columns": "",
            "selected_nme_column": selected_nme_column or "",
            "missing_required_columns": "csv_file",
            "number_of_rows": 0,
            "number_of_unique_images": 0,
            "number_of_unique_landmarks": 0,
        }
    image_column = next(
        (column for column in IMAGE_ID_ALIASES if column in raw.columns), None
    )
    landmark_column = next(
        (column for column in LANDMARK_INDEX_ALIASES if column in raw.columns),
        None,
    )
    return {
        "dataset_name": dataset_config.name,
        "analysis_name": analysis_name or protocol_config.name,
        "evaluation_protocol": protocol_config.name,
        "metric_name": metric_config.name,
        "model_name": model_config.name,
        "predicts_visibility": bool(model_config.predicts_visibility),
        "selected_metric_source": selected_metric_source or protocol_config.name,
        "csv_type": csv_type,
        "csv_path": str(csv_path) if csv_path is not None else "",
        "available_columns": " | ".join(str(column) for column in raw.columns),
        "selected_nme_column": selected_nme_column or "",
        "missing_required_columns": " | ".join(missing_required_columns),
        "number_of_rows": int(len(raw)),
        "number_of_unique_images": int(raw[image_column].nunique())
        if image_column
        else 0,
        "number_of_unique_landmarks": int(raw[landmark_column].nunique())
        if landmark_column
        else 0,
    }


def filter_per_landmark_by_protocol(
    raw: pd.DataFrame,
    dataset_config: DatasetBenchmarkConfig,
    protocol_config: EvaluationProtocolConfig,
    landmark_index: pd.Series,
) -> pd.Series:
    """Return the per-landmark row mask for one dataset-specific protocol."""
    mask = pd.Series(True, index=raw.index)
    protocol_name = protocol_config.name
    if dataset_config.dataset_type == "babyland72":
        if protocol_name == "standard":
            return mask
        inclusion_col = next(
            (
                column
                for column in ("evaluation_landmark_inclusion", "landmark_inclusion")
                if column in raw.columns
            ),
            None,
        )
        if inclusion_col is None:
            return mask
        expected = (
            "visible_intersection"
            if protocol_name in {"visibility_intersection", "visible_intersection"}
            else protocol_name
        )
        return raw[inclusion_col].astype(str).str.lower().eq(expected.lower())
    if dataset_config.dataset_type == "infanface" and protocol_name == "non_contour":
        return landmark_index >= 17
    return mask


def make_source_protocol(name: str) -> EvaluationProtocolConfig:
    """Create an internal source-protocol config used for column resolution."""
    display_names = {
        "standard": "Standard valid-landmark metrics",
        "visibility_intersection": "Visibility intersection",
        "gt_valid": "GT-valid landmarks",
    }
    return EvaluationProtocolConfig(
        name=name,
        display_name=display_names.get(name, name.replace("_", " ").title()),
    )


def resolve_babyland_metric_source(
    analysis_protocol: EvaluationProtocolConfig,
    model_config: ModelBenchmarkConfig,
) -> str | None:
    """Resolve the BabyLand metric source for one model under one analysis."""
    if analysis_protocol.name == "main_comparison":
        return "gt_valid" if model_config.predicts_visibility else "standard"
    if analysis_protocol.name == "visibility_protocol_analysis":
        return None
    return analysis_protocol.name


def expand_models_for_analysis(
    config: BenchmarkAnalysisConfig,
    analysis_protocol: EvaluationProtocolConfig,
) -> tuple[list[tuple[ModelBenchmarkConfig, EvaluationProtocolConfig, str]], list[str]]:
    """Return model/source-protocol jobs for one analysis protocol."""
    import dataclasses

    jobs: list[tuple[ModelBenchmarkConfig, EvaluationProtocolConfig, str]] = []
    warnings: list[str] = []
    if config.dataset.dataset_type != "babyland72":
        return [
            (model_config, analysis_protocol, analysis_protocol.name)
            for model_config in config.models
        ], warnings

    if analysis_protocol.name == "visibility_protocol_analysis":
        for model_config in config.models:
            if not model_config.predicts_visibility:
                warnings.append(
                    f"{model_config.name} excluded from visibility_protocol_analysis "
                    "because predicts_visibility is false."
                )
                continue
            for source_name in ("visibility_intersection", "gt_valid"):
                source_protocol = make_source_protocol(source_name)
                variant_display = f"{model_config.display_name or model_config.name} ({source_protocol.display_name})"
                variant = dataclasses.replace(
                    model_config,
                    name=f"{model_config.name} [{source_name}]",
                    display_name=variant_display,
                )
                jobs.append((variant, source_protocol, source_name))
        return jobs, warnings

    for model_config in config.models:
        source_name = resolve_babyland_metric_source(analysis_protocol, model_config)
        if source_name is None:
            continue
        jobs.append((model_config, make_source_protocol(source_name), source_name))
    return jobs, warnings


def normalize_orientation_value(value: Any, mapping: dict[str, str]) -> str:
    """Normalize orientation labels, treating yaw_plus_*deg values as class labels."""
    if pd.isna(value):
        return ""
    normalized = normalize_orientation_label(value)
    if normalized is not None:
        return normalized
    raw = str(value).strip()
    try:
        numeric = str(int(float(raw)))
    except ValueError:
        numeric = raw
    return mapping.get(raw, mapping.get(numeric, ""))


def map_orientation_labels(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map numeric or string orientation labels to canonical class labels."""
    if series.empty:
        return series
    return series.map(lambda value: normalize_orientation_value(value, mapping))


def collect_orientation_warnings(
    raw: pd.DataFrame,
    orientation_col: str | None,
    mapped: pd.Series,
    model_name: str,
) -> list[dict[str, Any]]:
    """Collect unknown orientation labels for reporting."""
    if orientation_col is None:
        return []
    raw_values = raw[orientation_col]
    unknown_mask = (
        raw_values.notna() & raw_values.astype(str).str.strip().ne("") & mapped.eq("")
    )
    if not unknown_mask.any():
        return []
    rows = []
    counts = raw_values[unknown_mask].astype(str).value_counts()
    for label, count in counts.items():
        rows.append(
            {"model": model_name, "raw_orientation": label, "count": int(count)}
        )
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with a helpful missing-file error."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _truthy_detection_series(series: pd.Series) -> pd.Series:
    """Parse common truthy detection values from a CSV column."""
    return (
        series.astype(str)
        .str.lower()
        .isin({"1", "true", "yes", "y", "detected", "success", "valid"})
    )


def read_detection_rate_from_summary(summary_path: Path) -> DetectionRateInfo | None:
    """Read detection-rate metadata from a sibling metrics summary CSV if present."""
    if not summary_path.exists():
        return None
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return None
    if not {"metric", "value"}.issubset(summary.columns):
        return None
    values = {
        str(row["metric"]): row["value"]
        for _, row in summary.iterrows()
        if str(row.get("metric", "")).strip()
    }
    rate = pd.to_numeric(
        pd.Series([values.get("detection_rate")]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(rate):
        return None
    n_detected = pd.to_numeric(
        pd.Series([values.get("images_with_prediction")]),
        errors="coerce",
    ).iloc[0]
    n_total = pd.to_numeric(
        pd.Series([values.get("total_images") or values.get("num_samples")]),
        errors="coerce",
    ).iloc[0]
    return DetectionRateInfo(
        detection_rate=float(rate),
        detection_rate_source="summary_csv",
        n_detected=None if not np.isfinite(n_detected) else int(n_detected),
        n_total=None if not np.isfinite(n_total) else int(n_total),
    )


def infer_detection_rate_info(
    raw: pd.DataFrame | None,
    image_col: str | None,
    nme_col: str | None,
    detected_col: str | None,
    model_config: ModelBenchmarkConfig,
    dataset_config: DatasetBenchmarkConfig,
) -> DetectionRateInfo:
    """Infer detection rate from CSV columns, dataset totals, summaries, or fallback."""
    fallback = (
        model_config.detection_rate_fallback
        if model_config.detection_rate_fallback is not None
        else model_config.detection_rate
    )
    if raw is not None and detected_col is not None:
        detected = _truthy_detection_series(raw[detected_col])
        return DetectionRateInfo(
            detection_rate=float(detected.mean()) if len(detected) else None,
            detection_rate_source="detected_column",
            n_detected=int(detected.sum()),
            n_total=int(len(detected)),
            detected_column_name=detected_col,
        )

    if (
        raw is not None
        and image_col is not None
        and dataset_config.total_images is not None
        and dataset_config.total_images > 0
    ):
        if nme_col is not None:
            valid_rows = raw[pd.to_numeric(raw[nme_col], errors="coerce").notna()]
        else:
            valid_rows = raw
        n_detected = int(
            valid_rows[image_col]
            .map(
                lambda value: normalize_image_id(
                    value, dataset_config.image_id_strip_regexes
                )
            )
            .nunique()
        )
        return DetectionRateInfo(
            detection_rate=float(n_detected / dataset_config.total_images),
            detection_rate_source="n_detected_over_dataset_total",
            n_detected=n_detected,
            n_total=int(dataset_config.total_images),
            warning_message=(
                "Inferred from unique evaluated image IDs divided by dataset.total_images."
            ),
        )

    if model_config.per_image_csv is not None:
        summary_info = read_detection_rate_from_summary(
            model_config.per_image_csv.parent / "metrics_summary.csv"
        )
        if summary_info is not None:
            return summary_info

    if fallback is not None:
        return DetectionRateInfo(
            detection_rate=float(fallback),
            detection_rate_source="fallback_config",
            warning_message=(
                "Used manual detection_rate_fallback."
                if model_config.detection_rate_fallback is not None
                else "Used legacy detection_rate as fallback; prefer detection_rate_fallback."
            ),
        )

    return DetectionRateInfo(
        detection_rate=None,
        detection_rate_source="unavailable",
        warning_message=(
            "Could not infer detection rate. Provide dataset.total_images, a detected "
            "column, metrics_summary.csv, or detection_rate_fallback."
        ),
    )


def load_model_results(
    model_config: ModelBenchmarkConfig,
    config: BenchmarkAnalysisConfig,
    protocol_config: EvaluationProtocolConfig,
    metric_config: BenchmarkMetricConfig,
    drop_invalid_nme: bool = False,
    analysis_name: str | None = None,
    selected_metric_source: str | None = None,
) -> LoadedModelResults:
    """Load, normalize, and validate result CSVs for one model."""
    dataset_config = config.dataset
    analysis_name = analysis_name or protocol_config.name
    selected_metric_source = selected_metric_source or protocol_config.name
    warnings: list[str] = []
    orientation_warnings: list[dict[str, Any]] = []
    column_audit: list[dict[str, Any]] = []
    per_image = None
    per_landmark = None
    detection_rate_info = DetectionRateInfo()

    if model_config.per_image_csv is not None:
        raw = _read_csv(model_config.per_image_csv)
        mapping = model_config.columns.get("per_image", {})
        per_image_candidates = get_protocol_metric_candidates(
            config,
            "per_image",
            protocol_config,
            metric_config,
        )
        image_col = resolve_column_with_warning(
            raw,
            "image_id",
            IMAGE_ID_ALIASES,
            mapping,
            model_config.name,
            "per-image",
            warnings,
        )
        nme_col = resolve_column_with_warning(
            raw,
            "nme",
            per_image_candidates,
            mapping,
            model_config.name,
            "per-image",
            warnings,
        )
        orientation_col = resolve_column_with_warning(
            raw,
            "orientation",
            ORIENTATION_ALIASES,
            mapping,
            model_config.name,
            "per-image",
            warnings,
            required=False,
        )
        detected_col = resolve_column_with_warning(
            raw,
            "detected",
            DETECTED_ALIASES,
            mapping,
            model_config.name,
            "per-image",
            warnings,
            required=False,
        )
        missing = []
        if image_col is None:
            missing.append("image_id")
        if nme_col is None:
            missing.append("nme")
        detection_rate_info = infer_detection_rate_info(
            raw=raw,
            image_col=image_col,
            nme_col=nme_col,
            detected_col=detected_col,
            model_config=model_config,
            dataset_config=dataset_config,
        )
        if detection_rate_info.warning_message:
            warnings.append(
                f"{model_config.name} detection rate: {detection_rate_info.warning_message}"
            )
        column_audit.append(
            build_column_audit_row(
                model_config,
                "per_image",
                model_config.per_image_csv,
                raw,
                dataset_config,
                protocol_config,
                metric_config,
                nme_col,
                missing,
                analysis_name,
                selected_metric_source,
            )
        )
        if image_col is not None and nme_col is not None:
            per_image = pd.DataFrame(
                {
                    "model": model_config.name,
                    "image_id": raw[image_col].astype(str),
                    "image_key": raw[image_col].map(
                        lambda value: normalize_image_id(
                            value, dataset_config.image_id_strip_regexes
                        )
                    ),
                    "nme": pd.to_numeric(raw[nme_col], errors="coerce"),
                    "source_csv_path": str(model_config.per_image_csv),
                }
            )
            if orientation_col is not None:
                per_image["orientation"] = map_orientation_labels(
                    raw[orientation_col], dataset_config.orientation_mapping
                )
                orientation_warnings.extend(
                    collect_orientation_warnings(
                        raw,
                        orientation_col,
                        per_image["orientation"],
                        model_config.name,
                    )
                )
            else:
                per_image["orientation"] = ""
            if detected_col is not None:
                per_image["detected"] = _truthy_detection_series(raw[detected_col])
            else:
                per_image["detected"] = per_image["nme"].notna()
            per_image["analysis_name"] = analysis_name
            per_image["evaluation_protocol"] = protocol_config.name
            per_image["selected_metric_source"] = selected_metric_source
            per_image["metric_name"] = metric_config.name
            per_image["predicts_visibility"] = bool(model_config.predicts_visibility)
            per_image["selected_metric_column"] = nme_col

            warnings.extend(
                validate_nme_table(
                    per_image, model_config.name, "per-image", dataset_config
                )
            )
            if per_image["nme"].notna().sum() == 0:
                warnings.append(
                    f"{model_config.name} per-image: selected column {nme_col!r} has no usable values "
                    f"for {protocol_config.name}/{metric_config.name}."
                )
            if drop_invalid_nme:
                per_image = per_image[np.isfinite(per_image["nme"])].copy()
    else:
        detection_rate_info = infer_detection_rate_info(
            raw=None,
            image_col=None,
            nme_col=None,
            detected_col=None,
            model_config=model_config,
            dataset_config=dataset_config,
        )
        if detection_rate_info.warning_message:
            warnings.append(
                f"{model_config.name} detection rate: {detection_rate_info.warning_message}"
            )

    if metric_config.name == "hausdorff":
        warnings.append(
            f"{model_config.name}: Hausdorff is an image-level set distance; "
            "per-landmark analysis is skipped."
        )
    elif model_config.per_landmark_csv is not None:
        raw = _read_csv(model_config.per_landmark_csv)
        mapping = model_config.columns.get("per_landmark", {})
        per_landmark_candidates = get_protocol_metric_candidates(
            config,
            "per_landmark",
            protocol_config,
            metric_config,
        )
        image_col = resolve_column_with_warning(
            raw,
            "image_id",
            IMAGE_ID_ALIASES,
            mapping,
            model_config.name,
            "per-landmark",
            warnings,
        )
        nme_col = resolve_column_with_warning(
            raw,
            "nme",
            per_landmark_candidates,
            mapping,
            model_config.name,
            "per-landmark",
            warnings,
        )
        landmark_col = resolve_column_with_warning(
            raw,
            "landmark_index",
            LANDMARK_INDEX_ALIASES,
            mapping,
            model_config.name,
            "per-landmark",
            warnings,
        )
        orientation_col = resolve_column_with_warning(
            raw,
            "orientation",
            ORIENTATION_ALIASES,
            mapping,
            model_config.name,
            "per-landmark",
            warnings,
            required=False,
        )
        valid_col = resolve_column_with_warning(
            raw,
            "valid",
            ("valid", "is_valid"),
            mapping,
            model_config.name,
            "per-landmark",
            warnings,
            required=False,
        )
        missing = []
        if image_col is None:
            missing.append("image_id")
        if nme_col is None:
            missing.append("nme")
        if landmark_col is None:
            missing.append("landmark_index")
        column_audit.append(
            build_column_audit_row(
                model_config,
                "per_landmark",
                model_config.per_landmark_csv,
                raw,
                dataset_config,
                protocol_config,
                metric_config,
                nme_col,
                missing,
                analysis_name,
                selected_metric_source,
            )
        )
        if image_col is None or nme_col is None or landmark_col is None:
            return LoadedModelResults(
                model_config,
                per_image,
                per_landmark,
                detection_rate_info,
                warnings,
                orientation_warnings,
                column_audit,
            )

        landmark_index = pd.to_numeric(raw[landmark_col], errors="coerce")
        landmark_index = normalize_landmark_index(
            landmark_index,
            dataset_config.landmark_index_base,
            dataset_config.landmark_count,
        )
        protocol_mask = filter_per_landmark_by_protocol(
            raw,
            dataset_config,
            protocol_config,
            landmark_index,
        )
        raw = raw[protocol_mask].copy()
        landmark_index = landmark_index[protocol_mask]
        per_landmark = pd.DataFrame(
            {
                "model": model_config.name,
                "image_id": raw[image_col].astype(str),
                "image_key": raw[image_col].map(
                    lambda value: normalize_image_id(
                        value, dataset_config.image_id_strip_regexes
                    )
                ),
                "landmark_index": landmark_index,
                "nme": pd.to_numeric(raw[nme_col], errors="coerce"),
                "source_csv_path": str(model_config.per_landmark_csv),
            }
        )
        if orientation_col is not None:
            per_landmark["orientation"] = map_orientation_labels(
                raw[orientation_col], dataset_config.orientation_mapping
            )
            orientation_warnings.extend(
                collect_orientation_warnings(
                    raw, orientation_col, per_landmark["orientation"], model_config.name
                )
            )
        else:
            per_landmark["orientation"] = ""
        if valid_col is not None:
            per_landmark["valid"] = (
                raw[valid_col]
                .astype(str)
                .str.lower()
                .isin({"1", "true", "yes", "y", "valid"})
            )
        else:
            per_landmark["valid"] = per_landmark["nme"].notna()
        per_landmark["analysis_name"] = analysis_name
        per_landmark["evaluation_protocol"] = protocol_config.name
        per_landmark["selected_metric_source"] = selected_metric_source
        per_landmark["metric_name"] = metric_config.name
        per_landmark["predicts_visibility"] = bool(model_config.predicts_visibility)
        per_landmark["selected_metric_column"] = nme_col

        warnings.extend(
            validate_nme_table(
                per_landmark, model_config.name, "per-landmark", dataset_config
            )
        )
        if per_landmark["nme"].notna().sum() == 0:
            warnings.append(
                f"{model_config.name} per-landmark: selected column {nme_col!r} has no usable values "
                f"for {protocol_config.name}/{metric_config.name}."
            )
        if drop_invalid_nme:
            per_landmark = per_landmark[
                np.isfinite(per_landmark["nme"])
                & per_landmark["landmark_index"].notna()
            ].copy()

    return LoadedModelResults(
        model_config,
        per_image,
        per_landmark,
        detection_rate_info,
        warnings,
        orientation_warnings,
        column_audit,
    )


def normalize_landmark_index(
    values: pd.Series,
    index_base: str,
    landmark_count: int | None,
) -> pd.Series:
    """Normalize landmark indices to 0-based indexing."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return numeric
    if index_base == "0":
        return numeric
    if index_base == "1":
        return numeric - 1
    if index_base != "auto":
        raise ValueError(f"Unsupported landmark_index_base: {index_base}")
    max_expected = landmark_count or int(finite.max())
    if finite.min() >= 1 and finite.max() <= max_expected:
        return numeric - 1
    return numeric


def validate_nme_table(
    df: pd.DataFrame,
    model_name: str,
    table_name: str,
    dataset_config: DatasetBenchmarkConfig,
) -> list[str]:
    """Return non-fatal validation warnings for a normalized NME table."""
    warnings = []
    duplicate_count = (
        int(df.duplicated(["image_key"]).sum()) if table_name == "per-image" else 0
    )
    if duplicate_count:
        warnings.append(
            f"{model_name} {table_name}: {duplicate_count} duplicate normalized image IDs."
        )
    invalid_count = int((~np.isfinite(df["nme"])).sum())
    if invalid_count:
        warnings.append(
            f"{model_name} {table_name}: {invalid_count} rows have NaN or infinite NME."
        )
    suspicious_count = int((df["nme"] > dataset_config.suspicious_nme_threshold).sum())
    if suspicious_count:
        warnings.append(
            f"{model_name} {table_name}: {suspicious_count} rows have NME > "
            f"{dataset_config.suspicious_nme_threshold:g}."
        )
    if "landmark_index" in df.columns and dataset_config.landmark_count is not None:
        outside = df[
            df["landmark_index"].notna()
            & (
                (df["landmark_index"] < 0)
                | (df["landmark_index"] >= dataset_config.landmark_count)
            )
        ]
        if len(outside):
            warnings.append(
                f"{model_name} {table_name}: {len(outside)} rows have landmarks outside "
                f"0..{dataset_config.landmark_count - 1}."
            )
    return warnings


def finite_nme(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with finite NME values."""
    return df[np.isfinite(df["nme"])].copy()


def summarize_nme(values: pd.Series, prefix: str) -> dict[str, float | int]:
    """Compute standard summary metrics for an NME series."""
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return {
            f"n_{prefix}": 0,
            f"mean_{prefix}_nme": np.nan,
            f"median_{prefix}_nme": np.nan,
            f"std_{prefix}_nme": np.nan,
            f"p90_{prefix}_nme": np.nan,
            f"p95_{prefix}_nme": np.nan,
            f"p99_{prefix}_nme": np.nan,
            "FRI_0.05" if prefix == "image" else "FRL_0.05": np.nan,
            "FRI_0.10" if prefix == "image" else "FRL_0.10": np.nan,
            "FRI_0.20" if prefix == "image" else "FRL_0.20": np.nan,
        }
    failure_prefix = "FRI" if prefix == "image" else "FRL"
    return {
        f"n_{prefix}": int(clean.size),
        f"mean_{prefix}_nme": float(clean.mean()),
        f"median_{prefix}_nme": float(clean.median()),
        f"std_{prefix}_nme": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        f"p90_{prefix}_nme": float(clean.quantile(0.90)),
        f"p95_{prefix}_nme": float(clean.quantile(0.95)),
        f"p99_{prefix}_nme": float(clean.quantile(0.99)),
        f"{failure_prefix}_0.05": float((clean > 0.05).mean()),
        f"{failure_prefix}_0.10": float((clean > 0.10).mean()),
        f"{failure_prefix}_0.20": float((clean > 0.20).mean()),
    }


def compute_global_image_metrics(results: list[LoadedModelResults]) -> pd.DataFrame:
    """Compute one-row-per-model image-level benchmark metrics."""
    rows = []
    for result in results:
        if result.per_image is None:
            continue
        data = finite_nme(result.per_image)
        row = {"model": result.config.name}
        if not result.per_image.empty:
            first = result.per_image.iloc[0]
            row.update(
                {
                    "analysis_name": first.get("analysis_name", ""),
                    "selected_metric_source": first.get("selected_metric_source", ""),
                    "selected_metric_column": first.get("selected_metric_column", ""),
                    "predicts_visibility": bool(
                        first.get(
                            "predicts_visibility", result.config.predicts_visibility
                        )
                    ),
                }
            )
        else:
            row.update(
                {
                    "analysis_name": "",
                    "selected_metric_source": "",
                    "selected_metric_column": "",
                    "predicts_visibility": bool(result.config.predicts_visibility),
                }
            )
        row["detection_rate"] = result.detection_rate_info.detection_rate
        row["detection_rate_source"] = result.detection_rate_info.detection_rate_source
        row["n_detected"] = (
            result.detection_rate_info.n_detected
            if result.detection_rate_info.n_detected is not None
            else int(data.shape[0])
        )
        row["n_total"] = result.detection_rate_info.n_total
        row["detected_column_name"] = result.detection_rate_info.detected_column_name
        row["detection_rate_warning"] = result.detection_rate_info.warning_message
        row.update(
            {
                key: value
                for key, value in summarize_nme(data["nme"], "image").items()
                if key != "n_image"
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mean_image_nme", na_position="last")


def compute_best_worst_cases(
    results: list[LoadedModelResults],
    dataset_config: DatasetBenchmarkConfig,
    metric_config: BenchmarkMetricConfig,
    top_k: int = 10,
) -> pd.DataFrame:
    """Extract the best and worst image-level cases per model."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.per_image is None:
            continue
        data = finite_nme(result.per_image).sort_values("nme", ascending=True)
        for rank_type, subset in (
            ("best", data.head(top_k)),
            ("worst", data.tail(top_k).sort_values("nme", ascending=False)),
        ):
            for rank, (_, row) in enumerate(subset.iterrows(), start=1):
                rows.append(
                    {
                        "model_name": result.config.name,
                        "dataset_name": dataset_config.name,
                        "analysis_name": row.get("analysis_name", ""),
                        "evaluation_protocol": row.get("evaluation_protocol", ""),
                        "selected_metric_source": row.get("selected_metric_source", ""),
                        "selected_metric_column": row.get("selected_metric_column", ""),
                        "predicts_visibility": row.get("predicts_visibility", ""),
                        "metric_name": metric_config.name,
                        "metric_display_name": metric_config.display_name,
                        "image_id": row.get("image_id"),
                        "sample_id": row.get("image_id"),
                        "stem": row.get("image_key"),
                        "orientation": row.get("orientation", ""),
                        "image_level_nme": row.get("nme"),
                        "selected_image_nme": row.get("nme"),
                        "rank_type": rank_type,
                        "rank": rank,
                        "detection_status": row.get("detected", True),
                        "source_csv_path": row.get("source_csv_path", ""),
                        "original_image_path": row.get("original_image_path", ""),
                        "prediction_overlay_path": row.get(
                            "prediction_overlay_path", ""
                        ),
                        "gt_overlay_path": row.get("gt_overlay_path", ""),
                        "comparison_overlay_path": row.get(
                            "comparison_overlay_path", ""
                        ),
                    }
                )
    return pd.DataFrame(rows)


def split_best_worst_case_tables(
    best_worst_cases: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split per-model best/worst rows into two consolidated CSV-ready tables."""
    if best_worst_cases.empty:
        return {
            "best_cases": pd.DataFrame(),
            "worst_cases": pd.DataFrame(),
        }
    sort_columns = ["model_name", "rank"]
    return {
        "best_cases": best_worst_cases[
            best_worst_cases["rank_type"] == "best"
        ].sort_values(sort_columns),
        "worst_cases": best_worst_cases[
            best_worst_cases["rank_type"] == "worst"
        ].sort_values(sort_columns),
    }


def write_best_worst_case_tables(
    output_dir: Path, best_worst_cases: pd.DataFrame
) -> None:
    """Write combined best/worst CSV tables for one protocol/metric."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if best_worst_cases.empty:
        return
    best_worst_cases.to_csv(tables_dir / "best_worst_cases_per_model.csv", index=False)


def compute_orientation_metrics(
    results: list[LoadedModelResults],
    dataset_config: DatasetBenchmarkConfig,
) -> pd.DataFrame:
    """Compute image-level metrics grouped by orientation."""
    rows = []
    for result in results:
        if result.per_image is None or "orientation" not in result.per_image:
            continue
        data = finite_nme(result.per_image)
        data = data[data["orientation"].astype(str).str.len() > 0]
        if not dataset_config.include_unknown_orientations:
            data = data[data["orientation"].isin(dataset_config.orientation_order)]
        for orientation, group in data.groupby("orientation", dropna=True):
            first = group.iloc[0]
            row = {
                "model": result.config.name,
                "orientation": orientation,
                "analysis_name": first.get("analysis_name", ""),
                "selected_metric_source": first.get("selected_metric_source", ""),
                "selected_metric_column": first.get("selected_metric_column", ""),
                "predicts_visibility": bool(
                    first.get("predicts_visibility", result.config.predicts_visibility)
                ),
            }
            row.update(summarize_nme(group["nme"], "image"))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["orientation"] = pd.Categorical(
        out["orientation"],
        categories=dataset_config.orientation_order,
        ordered=True,
    )
    return out.sort_values(["orientation", "model"]).reset_index(drop=True)


def compute_landmark_pooled_metrics(results: list[LoadedModelResults]) -> pd.DataFrame:
    """Compute secondary landmark-pooled metrics by model."""
    rows = []
    for result in results:
        if result.per_landmark is None:
            continue
        data = finite_nme(result.per_landmark)
        row = {"model": result.config.name}
        if not data.empty:
            first = data.iloc[0]
            row.update(
                {
                    "analysis_name": first.get("analysis_name", ""),
                    "selected_metric_source": first.get("selected_metric_source", ""),
                    "selected_metric_column": first.get("selected_metric_column", ""),
                    "predicts_visibility": bool(
                        first.get(
                            "predicts_visibility", result.config.predicts_visibility
                        )
                    ),
                }
            )
        row.update(summarize_nme(data["nme"], "landmark"))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_per_landmark_metrics(results: list[LoadedModelResults]) -> pd.DataFrame:
    """Compute per-model, per-landmark metrics."""
    rows = []
    for result in results:
        if result.per_landmark is None:
            continue
        data = finite_nme(result.per_landmark)
        for landmark_index, group in data.groupby("landmark_index"):
            if pd.isna(landmark_index):
                continue
            first = group.iloc[0]
            row = {
                "model": result.config.name,
                "landmark_index": int(landmark_index),
                "analysis_name": first.get("analysis_name", ""),
                "selected_metric_source": first.get("selected_metric_source", ""),
                "selected_metric_column": first.get("selected_metric_column", ""),
                "predicts_visibility": bool(
                    first.get("predicts_visibility", result.config.predicts_visibility)
                ),
            }
            row.update(summarize_nme(group["nme"], "landmark"))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_anatomical_region_metrics(
    results: list[LoadedModelResults],
    regions: dict[str, list[int]],
) -> pd.DataFrame:
    """Compute per-model landmark metrics grouped by anatomical region."""
    if not regions:
        return pd.DataFrame()
    index_to_region = {
        int(index): region for region, indices in regions.items() for index in indices
    }
    rows = []
    for result in results:
        if result.per_landmark is None:
            continue
        data = finite_nme(result.per_landmark)
        data["region"] = data["landmark_index"].map(index_to_region)
        data = data[data["region"].notna()]
        for region, group in data.groupby("region"):
            first = group.iloc[0]
            summary = summarize_nme(group["nme"], "landmark")
            rows.append(
                {
                    "model": result.config.name,
                    "region": region,
                    "analysis_name": first.get("analysis_name", ""),
                    "selected_metric_source": first.get("selected_metric_source", ""),
                    "selected_metric_column": first.get("selected_metric_column", ""),
                    "predicts_visibility": bool(
                        first.get(
                            "predicts_visibility", result.config.predicts_visibility
                        )
                    ),
                    "n_observations": summary["n_landmark"],
                    "mean_nme": summary["mean_landmark_nme"],
                    "median_nme": summary["median_landmark_nme"],
                    "p90_nme": summary["p90_landmark_nme"],
                    "p95_nme": summary["p95_landmark_nme"],
                    "FRL_0.10": summary["FRL_0.10"],
                    "FRL_0.20": summary["FRL_0.20"],
                }
            )
    return pd.DataFrame(rows)


def compute_pairwise_image_comparisons(
    results: list[LoadedModelResults],
    dataset_config: DatasetBenchmarkConfig,
    bootstrap_iterations: int = 2000,
    random_seed: int = 12345,
    by_orientation: bool = False,
) -> pd.DataFrame:
    """Compute paired image-level comparisons on common images for all model pairs."""
    image_tables = {
        result.config.name: finite_nme(result.per_image)
        for result in results
        if result.per_image is not None
    }
    rng = np.random.default_rng(random_seed)
    rows = []
    names = list(image_tables)
    for i, model_a in enumerate(names):
        for model_b in names[i + 1 :]:
            groups = [("", None)]
            if by_orientation:
                common_orientations = set(
                    image_tables[model_a]["orientation"].dropna().astype(str)
                ) & set(image_tables[model_b]["orientation"].dropna().astype(str))
                orientations = [
                    orientation
                    for orientation in dataset_config.orientation_order
                    if orientation in common_orientations
                ]
                if dataset_config.include_unknown_orientations:
                    orientations.extend(
                        sorted(
                            orientation
                            for orientation in common_orientations
                            if orientation
                            and orientation not in dataset_config.orientation_order
                        )
                    )
                groups = [
                    (orientation, orientation)
                    for orientation in orientations
                    if orientation
                ]
            for orientation_label, orientation_value in groups:
                left = image_tables[model_a]
                right = image_tables[model_b]
                if orientation_value is not None:
                    left = left[left["orientation"] == orientation_value]
                    right = right[right["orientation"] == orientation_value]
                merged = left[["image_key", "nme"]].merge(
                    right[["image_key", "nme"]],
                    on="image_key",
                    suffixes=("_a", "_b"),
                )
                rows.append(
                    summarize_pairwise(
                        merged,
                        model_a,
                        model_b,
                        orientation_label,
                        bootstrap_iterations,
                        rng,
                    )
                )
    return pd.DataFrame(rows)


def summarize_pairwise(
    merged: pd.DataFrame,
    model_a: str,
    model_b: str,
    orientation: str,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Summarize a paired model comparison table."""
    row: dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        "orientation": orientation,
        "n_common_images": int(len(merged)),
    }
    if merged.empty:
        return row
    diff = merged["nme_a"].to_numpy() - merged["nme_b"].to_numpy()
    row.update(
        {
            "mean_nme_a": float(merged["nme_a"].mean()),
            "mean_nme_b": float(merged["nme_b"].mean()),
            "median_nme_a": float(merged["nme_a"].median()),
            "median_nme_b": float(merged["nme_b"].median()),
            "win_rate_a": float((merged["nme_a"] < merged["nme_b"]).mean()),
            "mean_signed_diff_a_minus_b": float(np.mean(diff)),
            "median_signed_diff_a_minus_b": float(np.median(diff)),
        }
    )
    scipy_stats = import_scipy_stats()
    if scipy_stats is not None and len(diff) > 0 and np.any(diff != 0):
        try:
            row["wilcoxon_pvalue"] = float(scipy_stats.wilcoxon(diff).pvalue)
        except ValueError:
            row["wilcoxon_pvalue"] = np.nan
    else:
        row["wilcoxon_pvalue"] = np.nan
    ci_low, ci_high = bootstrap_mean_ci(diff, bootstrap_iterations, rng)
    row["mean_diff_ci95_low"] = ci_low
    row["mean_diff_ci95_high"] = ci_high
    return row


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute a percentile bootstrap CI for the mean."""
    if len(values) == 0 or iterations <= 0:
        return (np.nan, np.nan)
    sample_means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        sample_means[index] = np.mean(sample)
    return (
        float(np.quantile(sample_means, 0.025)),
        float(np.quantile(sample_means, 0.975)),
    )


def compute_primary_vs_baselines(
    pairwise: pd.DataFrame,
    primary_model: str | None,
    reference_models: list[str],
) -> pd.DataFrame:
    """Extract primary-model pairwise comparisons and orient all diffs as primary minus baseline."""
    if not primary_model or pairwise.empty:
        return pd.DataFrame()
    refs = reference_models or sorted(
        set(pairwise["model_a"]).union(set(pairwise["model_b"])) - {primary_model}
    )
    rows = []
    base = pairwise[pairwise.get("orientation", "") == ""].copy()
    for ref in refs:
        direct = base[(base["model_a"] == primary_model) & (base["model_b"] == ref)]
        reverse = base[(base["model_a"] == ref) & (base["model_b"] == primary_model)]
        if not direct.empty:
            row = direct.iloc[0].to_dict()
            rows.append(
                {
                    "primary_model": primary_model,
                    "baseline_model": ref,
                    "n_common_images": row.get("n_common_images"),
                    "primary_mean_nme": row.get("mean_nme_a"),
                    "baseline_mean_nme": row.get("mean_nme_b"),
                    "primary_median_nme": row.get("median_nme_a"),
                    "baseline_median_nme": row.get("median_nme_b"),
                    "primary_win_rate": row.get("win_rate_a"),
                    "mean_signed_diff_primary_minus_baseline": row.get(
                        "mean_signed_diff_a_minus_b"
                    ),
                    "median_signed_diff_primary_minus_baseline": row.get(
                        "median_signed_diff_a_minus_b"
                    ),
                    "wilcoxon_pvalue": row.get("wilcoxon_pvalue"),
                    "mean_diff_ci95_low": row.get("mean_diff_ci95_low"),
                    "mean_diff_ci95_high": row.get("mean_diff_ci95_high"),
                }
            )
        elif not reverse.empty:
            row = reverse.iloc[0].to_dict()
            rows.append(
                {
                    "primary_model": primary_model,
                    "baseline_model": ref,
                    "n_common_images": row.get("n_common_images"),
                    "primary_mean_nme": row.get("mean_nme_b"),
                    "baseline_mean_nme": row.get("mean_nme_a"),
                    "primary_median_nme": row.get("median_nme_b"),
                    "baseline_median_nme": row.get("median_nme_a"),
                    "primary_win_rate": 1.0 - row.get("win_rate_a", np.nan),
                    "mean_signed_diff_primary_minus_baseline": -row.get(
                        "mean_signed_diff_a_minus_b", np.nan
                    ),
                    "median_signed_diff_primary_minus_baseline": -row.get(
                        "median_signed_diff_a_minus_b", np.nan
                    ),
                    "wilcoxon_pvalue": row.get("wilcoxon_pvalue"),
                    "mean_diff_ci95_low": -row.get("mean_diff_ci95_high", np.nan),
                    "mean_diff_ci95_high": -row.get("mean_diff_ci95_low", np.nan),
                }
            )
    return pd.DataFrame(rows)


def compute_model_ranking_summary(
    global_metrics: pd.DataFrame,
    orientation_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Identify best models for key benchmark metrics."""
    rows = []
    metric_specs = [
        ("mean_image_nme", True, "global_mean_image_nme"),
        ("median_image_nme", True, "global_median_image_nme"),
        ("p95_image_nme", True, "global_p95_image_nme"),
        ("FRI_0.20", True, "global_FRI_0.20"),
        ("detection_rate", False, "global_detection_rate"),
    ]
    for column, lower_is_better, label in metric_specs:
        if column in global_metrics and global_metrics[column].notna().any():
            idx = (
                global_metrics[column].idxmin()
                if lower_is_better
                else global_metrics[column].idxmax()
            )
            rows.append(
                {
                    "ranking_scope": label,
                    "group": "",
                    "metric": column,
                    "best_model": global_metrics.loc[idx, "model"],
                    "best_value": global_metrics.loc[idx, column],
                }
            )
    if not orientation_metrics.empty:
        for orientation, group in orientation_metrics.groupby("orientation"):
            idx = group["mean_image_nme"].idxmin()
            rows.append(
                {
                    "ranking_scope": "orientation",
                    "group": orientation,
                    "metric": "mean_image_nme",
                    "best_model": group.loc[idx, "model"],
                    "best_value": group.loc[idx, "mean_image_nme"],
                }
            )
    if not region_metrics.empty:
        for region, group in region_metrics.groupby("region"):
            idx = group["mean_nme"].idxmin()
            rows.append(
                {
                    "ranking_scope": "anatomical_region",
                    "group": region,
                    "metric": "mean_nme",
                    "best_model": group.loc[idx, "model"],
                    "best_value": group.loc[idx, "mean_nme"],
                }
            )
    return pd.DataFrame(rows)


def compute_dataset_input_summary(results: list[LoadedModelResults]) -> pd.DataFrame:
    """Summarize input files and available rows per model."""
    rows = []
    for result in results:
        if result.column_audit:
            for row in result.column_audit:
                row = dict(row)
                row["detection_rate"] = result.detection_rate_info.detection_rate
                row[
                    "detection_rate_source"
                ] = result.detection_rate_info.detection_rate_source
                row["n_detected"] = result.detection_rate_info.n_detected
                row["n_total"] = result.detection_rate_info.n_total
                row[
                    "detected_column_name"
                ] = result.detection_rate_info.detected_column_name
                row["warning_message"] = result.detection_rate_info.warning_message
                row[
                    "detection_rate_fallback_config"
                ] = result.config.detection_rate_fallback
                row["legacy_detection_rate_config"] = result.config.detection_rate
                row["landmark_format"] = result.config.landmark_format
                row["normalized_rows_used"] = (
                    0
                    if row["csv_type"] == "per_image" and result.per_image is None
                    else (
                        int(len(result.per_image))
                        if row["csv_type"] == "per_image"
                        else 0
                        if result.per_landmark is None
                        else int(len(result.per_landmark))
                    )
                )
                row["warnings"] = " | ".join(result.warnings)
                rows.append(row)
            continue
        rows.append(
            {
                "model_name": result.config.name,
                "predicts_visibility": bool(result.config.predicts_visibility),
                "analysis_name": (
                    ""
                    if result.per_image is None or result.per_image.empty
                    else result.per_image.iloc[0].get("analysis_name", "")
                ),
                "selected_metric_source": (
                    ""
                    if result.per_image is None or result.per_image.empty
                    else result.per_image.iloc[0].get("selected_metric_source", "")
                ),
                "selected_metric_column": (
                    ""
                    if result.per_image is None or result.per_image.empty
                    else result.per_image.iloc[0].get("selected_metric_column", "")
                ),
                "detection_rate": result.detection_rate_info.detection_rate,
                "detection_rate_source": result.detection_rate_info.detection_rate_source,
                "n_detected": result.detection_rate_info.n_detected,
                "n_total": result.detection_rate_info.n_total,
                "detected_column_name": result.detection_rate_info.detected_column_name,
                "warning_message": result.detection_rate_info.warning_message,
                "detection_rate_fallback_config": result.config.detection_rate_fallback,
                "legacy_detection_rate_config": result.config.detection_rate,
                "landmark_format": result.config.landmark_format,
                "per_image_csv": str(result.config.per_image_csv)
                if result.config.per_image_csv
                else "",
                "per_image_rows": 0
                if result.per_image is None
                else int(len(result.per_image)),
                "per_landmark_csv": str(result.config.per_landmark_csv)
                if result.config.per_landmark_csv
                else "",
                "per_landmark_rows": 0
                if result.per_landmark is None
                else int(len(result.per_landmark)),
                "warnings": " | ".join(result.warnings),
            }
        )
    return pd.DataFrame(rows)


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Write all non-empty result tables to CSV."""
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table is None:
            continue
        table.to_csv(table_dir / f"{name}.csv", index=False)


def add_analysis_context_to_tables(
    tables: dict[str, pd.DataFrame],
    dataset_config: DatasetBenchmarkConfig,
    protocol_config: EvaluationProtocolConfig,
    metric_config: BenchmarkMetricConfig,
) -> dict[str, pd.DataFrame]:
    """Add dataset/protocol/metric context columns to all output tables."""
    updated: dict[str, pd.DataFrame] = {}
    for name, table in tables.items():
        if table is None or table.empty:
            updated[name] = table
            continue
        table = table.copy()
        if "dataset_name" not in table.columns:
            table.insert(0, "dataset_name", dataset_config.name)
        if "evaluation_protocol" not in table.columns:
            table.insert(1, "evaluation_protocol", protocol_config.name)
        if "metric_name" not in table.columns:
            table.insert(2, "metric_name", metric_config.name)
        if "model" in table.columns and "model_name" not in table.columns:
            table.insert(3, "model_name", table["model"])
        updated[name] = table
    return updated


def import_matplotlib_pyplot() -> Any:
    """Import matplotlib lazily so table/report generation can run without it."""
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - depends on local binary deps.
        raise ImportError(
            f"Could not import matplotlib for plot generation: {error}"
        ) from error
    return plt


def import_scipy_stats() -> Any | None:
    """Import scipy.stats lazily; return None if scipy is absent or broken."""
    global _SCIPY_STATS, _SCIPY_IMPORT_ATTEMPTED
    if _SCIPY_IMPORT_ATTEMPTED:
        return _SCIPY_STATS
    _SCIPY_IMPORT_ATTEMPTED = True
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from scipy import stats as scipy_stats
    except Exception:
        _SCIPY_STATS = None
    else:
        _SCIPY_STATS = scipy_stats
    return _SCIPY_STATS


def save_figure(fig: Any, path: Path, save_pdf: bool = False, dpi: int = 300) -> None:
    """Save every matplotlib figure as matching PNG and PDF files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"), dpi=dpi)
    try:
        import_matplotlib_pyplot().close(fig)
    except ImportError:
        pass


def apply_axis_style(ax: Any, grid_axis: str = "y") -> None:
    """Apply a simple presentation-friendly axis style."""
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def maybe_percent(
    values: pd.Series | np.ndarray, use_percent_axis: bool
) -> pd.Series | np.ndarray:
    """Scale values to percent when configured."""
    return values * 100.0 if use_percent_axis else values


def format_metric_label(value: float, use_percent_axis: bool, digits: int = 1) -> str:
    """Format plot annotations for NME or percentage axes."""
    if not np.isfinite(value):
        return ""
    return f"{value * 100:.{digits}f}%" if use_percent_axis else f"{value:.3f}"


def annotate_bars(ax: Any, bars: Any, labels: list[str], padding: float = 3.0) -> None:
    """Annotate a bar container with preformatted labels."""
    for bar, label in zip(bars, labels):
        if not label:
            continue
        height = bar.get_height()
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_ced_reference_lines(ax: Any) -> None:
    """Add common NME reference lines to a CED plot."""
    for threshold, label in [(0.05, "5%"), (0.10, "10%"), (0.20, "20%")]:
        ax.axvline(threshold, color="0.45", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(
            threshold,
            0.03,
            label,
            rotation=90,
            va="bottom",
            ha="right",
            color="0.35",
            fontsize=8,
        )


def generate_benchmark_plots(
    output_dir: Path,
    results: list[LoadedModelResults],
    global_metrics: pd.DataFrame,
    orientation_metrics: pd.DataFrame,
    per_landmark_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    primary_vs_baselines: pd.DataFrame,
    config: BenchmarkAnalysisConfig,
) -> list[str]:
    """Generate benchmark plots and return relative plot paths for the report."""
    try:
        plt = import_matplotlib_pyplot()
    except ImportError as error:
        print(f"[WARNING] Skipping plot generation. {error}")
        return []

    plt.rcParams.update(
        {
            "figure.dpi": config.dataset.plot_dpi,
            "savefig.dpi": config.dataset.plot_dpi,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    plot_dir = output_dir / "plots"
    plot_paths: list[str] = []
    if global_metrics.empty:
        return plot_paths

    model_order = get_ordered_model_names(
        global_metrics["model"].tolist(), config.dataset
    )
    global_metrics = (
        global_metrics.set_index("model")
        .reindex(model_order)
        .dropna(how="all")
        .reset_index()
    )
    model_order = global_metrics["model"].tolist()
    display_names, color_map, _ = build_model_style_maps(model_order, config.dataset)
    labels = [label_for_model(model, display_names) for model in model_order]
    colors = [color_map[model] for model in model_order]
    x = np.arange(len(model_order))
    use_percent = config.dataset.use_percent_axis
    protocol_label = (
        config.dataset.evaluation_protocol_display_name
        or config.dataset.evaluation_protocol
    )
    metric_label = config.metrics[0].display_name if len(config.metrics) == 1 else "NME"
    title_prefix = " | ".join(
        part for part in [config.dataset.name, protocol_label, metric_label] if part
    )

    def contextual_title(title: str) -> str:
        """Return a plot title prefixed with dataset/protocol/metric context."""
        return f"{title_prefix}\n{title}" if title_prefix else title

    fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 4.5))
    detection_rates = pd.to_numeric(global_metrics["detection_rate"], errors="coerce")
    values = detection_rates.to_numpy() * 100.0
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Detection rate (%)")
    max_detection = np.nanmax(values) if np.isfinite(values).any() else 100.0
    ax.set_ylim(0, min(105, max(100, max_detection * 1.1)))
    ax.set_title(contextual_title("Detection rate by model"))
    apply_axis_style(ax)
    if config.dataset.annotate_bars:
        annotate_bars(
            ax,
            bars,
            [f"{value:.1f}%" if np.isfinite(value) else "n/a" for value in values],
        )
    path = plot_dir / "detection_rate_bar.png"
    save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
    plot_paths.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 4.5))
    values = maybe_percent(global_metrics["mean_image_nme"].to_numpy(), use_percent)
    bars = ax.bar(x, values, color=colors)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Mean image-level NME (%)" if use_percent else "Mean image-level NME")
    ax.set_title(contextual_title("Mean image-level NME by model"))
    apply_axis_style(ax)
    if config.dataset.annotate_bars:
        ann = [
            format_metric_label(value, use_percent)
            for value in global_metrics["mean_image_nme"]
        ]
        annotate_bars(ax, bars, ann)
        for idx, det in enumerate(detection_rates):
            det_label = f"det {det * 100:.1f}%" if np.isfinite(det) else "det n/a"
            ax.text(
                idx,
                0,
                det_label,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.35",
            )
    path = plot_dir / "mean_image_nme_bar.png"
    save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
    plot_paths.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    y_values = maybe_percent(global_metrics["mean_image_nme"].to_numpy(), use_percent)
    ax.scatter(detection_rates * 100.0, y_values, color=colors, s=70)
    for idx, row in global_metrics.iterrows():
        det = detection_rates.iloc[idx]
        if not np.isfinite(det):
            continue
        ax.annotate(
            labels[idx],
            (det * 100.0, y_values[idx]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Detection rate (%)")
    ax.set_ylabel("Mean image-level NME (%)" if use_percent else "Mean image-level NME")
    ax.set_title(contextual_title("Mean NME vs detection rate"))
    apply_axis_style(ax)
    path = plot_dir / "mean_nme_vs_detection_rate.png"
    save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
    plot_paths.append(str(path.relative_to(output_dir)))

    image_frames = [
        finite_nme(result.per_image)
        for result in results
        if result.per_image is not None
    ]
    if image_frames:
        image_all = pd.concat(image_frames, ignore_index=True)
        ordered_data = [
            image_all.loc[image_all["model"] == model, "nme"].to_numpy()
            for model in model_order
        ]
        plot_data = [maybe_percent(values, use_percent) for values in ordered_data]

        fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 5.0))
        box = ax.boxplot(
            plot_data,
            tick_labels=labels,
            showfliers=False,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 6,
            },
            medianprops={"color": "black", "linewidth": 1.8},
            patch_artist=True,
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Image-level NME (%)" if use_percent else "Image-level NME")
        ax.set_title(contextual_title("Image-level NME distribution by model"))
        ax.plot([], [], color="black", linewidth=1.8, label="Median")
        ax.plot(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="None",
            label="Mean",
        )
        ax.legend(loc="upper right")
        apply_axis_style(ax)
        if config.dataset.annotate_boxplot_means:
            for idx, values in enumerate(ordered_data, start=1):
                if len(values):
                    mean = float(np.mean(values))
                    display_value = mean * 100.0 if use_percent else mean
                    ax.annotate(
                        format_metric_label(mean, use_percent),
                        xy=(idx, display_value),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                    )
        path = plot_dir / "image_nme_boxplot_by_model.png"
        save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
        plot_paths.append(str(path.relative_to(output_dir)))

        if config.dataset.generate_log_scale_variants and any(
            np.nanmax(values)
            / max(np.nanmedian(values), config.dataset.log_scale_epsilon)
            > 4
            for values in ordered_data
            if len(values)
        ):
            fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 5.0))
            box = ax.boxplot(
                plot_data,
                tick_labels=labels,
                showfliers=False,
                showmeans=True,
                patch_artist=True,
            )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            ax.set_yscale("log")
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Image-level NME (%)" if use_percent else "Image-level NME")
            ax.set_title(
                contextual_title("Image-level NME distribution by model (log scale)")
            )
            apply_axis_style(ax)
            path = plot_dir / "image_nme_boxplot_by_model_log.png"
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
            plot_paths.append(str(path.relative_to(output_dir)))

        if config.dataset.show_violin_plots:
            fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 5.0))
            violin = ax.violinplot(
                plot_data, showmeans=False, showmedians=False, showextrema=False
            )
            for body, color in zip(violin["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.45)
            for idx, values in enumerate(plot_data, start=1):
                if len(values) == 0:
                    continue
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                mean = np.mean(values)
                ax.vlines(idx, q1, q3, color="black", linewidth=3)
                ax.scatter(
                    idx,
                    median,
                    marker="_",
                    color="black",
                    s=120,
                    zorder=3,
                    label="Median" if idx == 1 else None,
                )
                ax.scatter(
                    idx,
                    mean,
                    marker="D",
                    facecolor="white",
                    edgecolor="black",
                    s=36,
                    zorder=3,
                    label="Mean" if idx == 1 else None,
                )
            ax.set_xticks(
                np.arange(1, len(model_order) + 1), labels, rotation=30, ha="right"
            )
            ax.set_ylabel("Image-level NME (%)" if use_percent else "Image-level NME")
            ax.set_title(contextual_title("Image-level NME distribution by model"))
            ax.legend(loc="upper right")
            apply_axis_style(ax)
            path = plot_dir / "image_nme_violin_by_model.png"
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
            plot_paths.append(str(path.relative_to(output_dir)))

            if config.dataset.generate_log_scale_variants and any(
                np.nanmax(values)
                / max(np.nanmedian(values), config.dataset.log_scale_epsilon)
                > 4
                for values in ordered_data
                if len(values)
            ):
                epsilon = config.dataset.log_scale_epsilon * (
                    100.0 if use_percent else 1.0
                )
                log_plot_data = [np.maximum(values, epsilon) for values in plot_data]
                fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 5.0))
                violin = ax.violinplot(
                    log_plot_data, showmeans=False, showmedians=False, showextrema=False
                )
                for body, color in zip(violin["bodies"], colors):
                    body.set_facecolor(color)
                    body.set_edgecolor("black")
                    body.set_alpha(0.45)
                for idx, values in enumerate(log_plot_data, start=1):
                    if len(values) == 0:
                        continue
                    q1, median, q3 = np.percentile(values, [25, 50, 75])
                    mean = np.mean(values)
                    ax.vlines(idx, q1, q3, color="black", linewidth=3)
                    ax.scatter(
                        idx,
                        median,
                        marker="_",
                        color="black",
                        s=120,
                        zorder=3,
                        label="Median" if idx == 1 else None,
                    )
                    ax.scatter(
                        idx,
                        mean,
                        marker="D",
                        facecolor="white",
                        edgecolor="black",
                        s=36,
                        zorder=3,
                        label="Mean" if idx == 1 else None,
                    )
                ax.set_yscale("log")
                ax.set_xticks(
                    np.arange(1, len(model_order) + 1), labels, rotation=30, ha="right"
                )
                ax.set_ylabel(
                    "Image-level NME (%)" if use_percent else "Image-level NME"
                )
                ax.set_title(
                    contextual_title(
                        "Image-level NME distribution by model (log scale)"
                    )
                )
                ax.legend(loc="upper right")
                apply_axis_style(ax)
                path = plot_dir / "image_nme_violin_by_model_log.png"
                save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
                plot_paths.append(str(path.relative_to(output_dir)))

        def plot_ced(path: Path, zoom: bool = False) -> None:
            fig, ax = plt.subplots(figsize=(7.2, 5.0))
            for model, label, color in zip(model_order, labels, colors):
                values = np.sort(
                    image_all.loc[image_all["model"] == model, "nme"].to_numpy()
                )
                if len(values) == 0:
                    continue
                y = np.arange(1, len(values) + 1) / len(values)
                ax.plot(values, y, label=label, color=color, linewidth=2.2)
            add_ced_reference_lines(ax)
            if zoom:
                ax.set_xlim(0, config.dataset.ced_zoom_max_nme)
            ax.set_ylim(0, 1.01)
            ax.set_xlabel("Image-level NME threshold")
            ax.set_ylabel("Fraction of images with NME <= threshold")
            ax.set_title(
                contextual_title("Cumulative Error Distribution (Image-level NME)")
            )
            ax.legend(fontsize=9)
            apply_axis_style(ax)
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)

        path = plot_dir / "ced_curves_image_nme.png"
        plot_ced(path, zoom=False)
        plot_paths.append(str(path.relative_to(output_dir)))
        path = plot_dir / "ced_curves_image_nme_zoomed.png"
        plot_ced(path, zoom=True)
        plot_paths.append(str(path.relative_to(output_dir)))

    if not orientation_metrics.empty:
        orient = orientation_metrics.copy()
        orient = orient[orient["orientation"].isin(config.dataset.orientation_order)]
        pivot = orient.pivot(
            index="model", columns="orientation", values="mean_image_nme"
        )
        pivot = pivot.reindex(
            index=model_order, columns=config.dataset.orientation_order
        ).dropna(how="all")
        if not pivot.empty:
            fig, ax = plt.subplots(
                figsize=(max(7, pivot.shape[1] * 1.15), max(4, pivot.shape[0] * 0.55))
            )
            values = pivot.to_numpy() * (100.0 if use_percent else 1.0)
            image = ax.imshow(values, aspect="auto", cmap="viridis")
            ax.set_xticks(
                np.arange(pivot.shape[1]), pivot.columns, rotation=30, ha="right"
            )
            ax.set_yticks(
                np.arange(pivot.shape[0]),
                [label_for_model(model, display_names) for model in pivot.index],
            )
            ax.set_title(contextual_title("Orientation-wise mean image NME"))
            fig.colorbar(
                image,
                ax=ax,
                label="Mean image NME (%)" if use_percent else "Mean image NME",
            )
            path = plot_dir / "orientation_mean_nme_heatmap.png"
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
            plot_paths.append(str(path.relative_to(output_dir)))

            fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.6), 5))
            width = 0.8 / max(len(pivot.index), 1)
            orientation_x = np.arange(len(pivot.columns))
            for idx, model in enumerate(pivot.index):
                offset = (idx - (len(pivot.index) - 1) / 2) * width
                ax.bar(
                    orientation_x + offset,
                    pivot.loc[model].to_numpy() * (100.0 if use_percent else 1.0),
                    width,
                    label=label_for_model(model, display_names),
                    color=color_map[model],
                )
            ax.set_xticks(orientation_x, pivot.columns, rotation=30, ha="right")
            ax.set_ylabel("Mean image NME (%)" if use_percent else "Mean image NME")
            ax.set_title(contextual_title("Orientation-wise mean image NME"))
            ax.legend(title="Model", fontsize=8)
            apply_axis_style(ax)
            path = plot_dir / "orientation_mean_nme_grouped_bar.png"
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
            plot_paths.append(str(path.relative_to(output_dir)))

    primary = config.dataset.primary_model
    if primary and not per_landmark_metrics.empty:
        primary_raw = next(
            (
                finite_nme(r.per_landmark)
                for r in results
                if r.config.name == primary and r.per_landmark is not None
            ),
            pd.DataFrame(),
        )
        if not primary_raw.empty:
            landmark_indices = sorted(primary_raw["landmark_index"].dropna().unique())
            data = [
                primary_raw.loc[primary_raw["landmark_index"] == idx, "nme"].to_numpy()
                for idx in landmark_indices
            ]
            fig, ax = plt.subplots(figsize=(max(10, len(landmark_indices) * 0.22), 4.8))
            ax.boxplot(
                data,
                tick_labels=[str(int(idx)) for idx in landmark_indices],
                showfliers=False,
            )
            ax.set_xlabel("Landmark index (0-based)")
            ax.set_ylabel("Per-landmark NME")
            ax.set_title(
                contextual_title(
                    f"Per-landmark NME for {label_for_model(primary, display_names)}"
                )
            )
            apply_axis_style(ax)
            path = plot_dir / "primary_per_landmark_nme_boxplot.png"
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
            plot_paths.append(str(path.relative_to(output_dir)))

        fig, ax = plt.subplots(figsize=(10, 5))
        common_limit = min(
            [
                int(pd.to_numeric(m.landmark_format, errors="coerce"))
                for m in (r.config for r in results)
                if m.landmark_format and str(m.landmark_format).isdigit()
            ]
            or [int(per_landmark_metrics["landmark_index"].max()) + 1]
        )
        for model, color in zip(model_order, colors):
            model_rows = per_landmark_metrics[
                (per_landmark_metrics["model"] == model)
                & (per_landmark_metrics["landmark_index"] < common_limit)
            ].sort_values("landmark_index")
            if not model_rows.empty:
                ax.plot(
                    model_rows["landmark_index"],
                    model_rows["mean_landmark_nme"],
                    label=label_for_model(model, display_names),
                    color=color,
                    linewidth=1.8,
                )
        ax.set_xlabel("Landmark index (0-based)")
        ax.set_ylabel("Mean landmark NME")
        ax.set_title(contextual_title("Per-landmark mean NME across models"))
        ax.legend(fontsize=8)
        apply_axis_style(ax)
        path = plot_dir / "per_landmark_mean_nme_lines_common_landmarks.png"
        save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
        plot_paths.append(str(path.relative_to(output_dir)))

    if not region_metrics.empty:
        region_order = list(config.dataset.anatomical_regions.keys()) or sorted(
            region_metrics["region"].unique()
        )
        pivot = region_metrics.pivot(
            index="region", columns="model", values="mean_nme"
        ).reindex(index=region_order, columns=model_order)
        pivot = pivot.dropna(how="all")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(max(11, len(pivot) * 0.75), 5.5))
            pivot.rename(columns=display_names).plot(
                kind="bar", ax=ax, color=[color_map[m] for m in pivot.columns]
            )
            ax.set_ylabel("Mean NME")
            ax.set_title(contextual_title("Anatomical-region mean NME"))
            ax.legend(title="Model", fontsize=8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            apply_axis_style(ax)
            path = plot_dir / "anatomical_region_mean_nme_bar.png"
            save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
            plot_paths.append(str(path.relative_to(output_dir)))

            common68_regions = [
                region
                for region in pivot.index
                if max(config.dataset.anatomical_regions.get(region, [-1])) < 68
            ]
            pivot68 = pivot.loc[common68_regions]
            if not pivot68.empty:
                fig, ax = plt.subplots(figsize=(max(10, len(pivot68) * 0.85), 5.2))
                pivot68.rename(columns=display_names).plot(
                    kind="bar", ax=ax, color=[color_map[m] for m in pivot68.columns]
                )
                ax.set_ylabel("Mean NME")
                ax.set_title(
                    contextual_title("Anatomical-region mean NME (common 68 landmarks)")
                )
                ax.legend(title="Model", fontsize=8)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                apply_axis_style(ax)
                path = plot_dir / "anatomical_region_mean_nme_bar_common68.png"
                save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
                plot_paths.append(str(path.relative_to(output_dir)))

    if not primary_vs_baselines.empty:
        primary_color = color_map.get(primary, "#333333")
        baselines = primary_vs_baselines["baseline_model"].tolist()
        x_win = np.arange(len(baselines))
        fig, ax = plt.subplots(figsize=(max(6.5, len(baselines) * 1.35), 4.8))
        values = primary_vs_baselines["primary_win_rate"].to_numpy() * 100.0
        bars = ax.bar(x_win, values, color=primary_color, alpha=0.85)
        ax.axhline(50.0, color="black", linestyle="--", linewidth=1.1, label="50%")
        ax.set_ylim(0, 100)
        ax.set_xticks(
            x_win,
            [label_for_model(model, display_names) for model in baselines],
            rotation=30,
            ha="right",
        )
        ax.set_ylabel("Images where primary model has lower NME (%)")
        ax.set_title(
            contextual_title(
                f"{label_for_model(primary, display_names)} pairwise win rate on common images"
            )
        )
        if config.dataset.annotate_bars:
            ann = [
                f"{rate:.1f}%\n(n={int(n)})"
                for rate, n in zip(values, primary_vs_baselines["n_common_images"])
            ]
            annotate_bars(ax, bars, ann)
        ax.legend(loc="upper right")
        apply_axis_style(ax)
        path = plot_dir / "primary_pairwise_win_rate.png"
        save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
        plot_paths.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(max(7, len(model_order) * 1.25), 4.8))
    width = 0.35
    fri10 = global_metrics["FRI_0.10"].to_numpy() * 100.0
    fri20 = global_metrics["FRI_0.20"].to_numpy() * 100.0
    bars10 = ax.bar(x - width / 2, fri10, width, label="NME > 10%", color="#6B8EC1")
    bars20 = ax.bar(x + width / 2, fri20, width, label="NME > 20%", color="#D65F5F")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Image failure rate (%)")
    ax.set_title(contextual_title("Image-level failure rates"))
    if config.dataset.annotate_bars:
        annotate_bars(ax, bars10, [f"{value:.1f}%" for value in fri10])
        annotate_bars(ax, bars20, [f"{value:.1f}%" for value in fri20])
    ax.legend()
    apply_axis_style(ax)
    path = plot_dir / "image_failure_rate_bar.png"
    save_figure(fig, path, config.save_pdf, config.dataset.plot_dpi)
    plot_paths.append(str(path.relative_to(output_dir)))

    return plot_paths


def format_float(value: Any, digits: int = 4) -> str:
    """Format floats for Markdown tables."""
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a compact Markdown table without requiring tabulate."""
    if df.empty:
        return "_Unavailable._"
    shown = df.head(max_rows).copy()
    columns = list(shown.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append(
            "| " + " | ".join(format_float(row[col]) for col in columns) + " |"
        )
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def get_babyland_visibility_report_note(
    dataset: DatasetBenchmarkConfig,
    protocol_config: EvaluationProtocolConfig | None,
) -> list[str]:
    """Return BabyLand-72 visibility-protocol interpretation notes."""
    if dataset.dataset_type != "babyland72" or protocol_config is None:
        return []
    if protocol_config.name == "main_comparison":
        return [
            "**BabyLand-72 visibility note:** Only visibility-aware models produce predicted "
            "visibility flags. Therefore, the main SOTA comparison uses GT-valid metrics for "
            "visibility-aware models and standard valid-landmark metrics for models without "
            "predicted visibility. The visibility-intersection protocol is reported separately "
            "for visibility-aware models only.",
            "",
        ]
    if protocol_config.name == "visibility_protocol_analysis":
        return [
            "**BabyLand-72 visibility note:** This analysis includes only models with "
            "`predicts_visibility: true`. External SOTA models without predicted visibility "
            "cannot be evaluated under `gt_visibility == 1 AND pred_visibility == 1`, so they "
            "are excluded from this protocol comparison.",
            "",
        ]
    return []


def write_markdown_report(
    output_dir: Path,
    config: BenchmarkAnalysisConfig,
    tables: dict[str, pd.DataFrame],
    plot_paths: list[str],
    warnings: list[str],
    metric_config: BenchmarkMetricConfig | None = None,
    protocol_config: EvaluationProtocolConfig | None = None,
) -> None:
    """Write the benchmark Markdown report."""
    dataset = config.dataset
    global_metrics = tables["global_image_metrics"]
    orientation_metrics = tables["orientation_image_metrics"]
    primary_vs = tables["primary_model_vs_baselines"]
    rankings = tables["model_ranking_summary"]
    report_path = output_dir / "report.md"

    lines = [
        f"# Benchmark Report: {dataset.name}",
        "",
        "## Inputs",
        f"- Dataset name: `{dataset.name}`",
        f"- Evaluation protocol: `{protocol_config.display_name if protocol_config else dataset.evaluation_protocol}`",
        f"- Geometric metric: `{metric_config.display_name if metric_config else 'NME'}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Output directory: `{output_dir}`",
        f"- Primary model: `{dataset.primary_model or ''}`",
        f"- Models included: {', '.join(model.name for model in config.models)}",
        "",
        dataframe_to_markdown(tables["dataset_input_summary"], max_rows=50),
        "",
        "## Main results: image-level metrics",
        "",
        "The main ranking uses image-level NME, so each evaluated image contributes once. "
        "This is preferred for visibility-aware datasets because landmark-pooled metrics "
        "can overweight images with more visible landmarks.",
        "",
        *get_babyland_visibility_report_note(dataset, protocol_config),
        *(
            [
                "Hausdorff distance measures the worst-case mismatch between the "
                "predicted and ground-truth landmark point sets. Lower Hausdorff is "
                "better. It is sensitive to outlier landmarks, so it complements "
                "mean and median NME and should be interpreted together with "
                "failure rates and CED curves.",
                "",
            ]
            if metric_config is not None and metric_config.name == "hausdorff"
            else []
        ),
        dataframe_to_markdown(global_metrics, max_rows=50),
        "",
        "- `mean_image_nme`: average image-level normalized mean error.",
        "- `median_image_nme`: median image-level error, less sensitive to outliers.",
        "- `p90/p95/p99`: tail-error percentiles.",
        "- `FRI_0.10`: fraction of images with image-level NME greater than 0.10.",
        "- Detection rate is reported separately from accuracy on detected/evaluated images.",
        "",
        "Boxplot interpretation: the box shows the interquartile range, the line inside "
        "the box is the median, and the diamond marker is the mean. A large separation "
        "between mean and median indicates skew or outlier-heavy behavior.",
        "",
        "Failure-rate interpretation: lower is better. `FRI > 0.10` is the percentage "
        "of images with NME greater than 10%, which captures moderate/severe failures; "
        "`FRI > 0.20` captures severe failures.",
        "",
        "CED interpretation: a Cumulative Error Distribution plots the image-level NME "
        "threshold on the x-axis and the fraction of images with NME less than or equal "
        "to that threshold on the y-axis. Curves farther left and higher are better; "
        "long right tails indicate outliers.",
        "",
        "Violin plot interpretation: violin width indicates density. Wider regions mean "
        "more images in that error range, while long upper tails indicate difficult cases "
        "or outliers.",
        "",
        "## Orientation analysis",
        "",
    ]
    if orientation_metrics.empty:
        lines.append("_Orientation labels were unavailable or not resolved._")
    else:
        lines.append(dataframe_to_markdown(orientation_metrics, max_rows=80))
        best_by_orientation = rankings[rankings["ranking_scope"] == "orientation"]
        if not best_by_orientation.empty:
            lines.extend(["", "Best model by orientation:"])
            for _, row in best_by_orientation.iterrows():
                lines.append(
                    f"- `{row['group']}`: `{row['best_model']}` ({row['metric']}={format_float(row['best_value'])})"
                )

    lines.extend(["", "## Pairwise analysis", ""])
    if primary_vs.empty:
        lines.append("_Primary-model pairwise comparisons were unavailable._")
    else:
        lines.append(dataframe_to_markdown(primary_vs, max_rows=50))
        lines.append("")
        lines.append(
            "Negative signed differences mean the primary model has lower NME than the baseline."
        )
        lines.append(
            "Primary win rate is computed only on common images: it is the percentage "
            "of shared images where the primary model has lower image-level NME than "
            "the baseline. Values above 50% mean the primary model wins on most shared "
            "images, but this should be interpreted together with mean NME, P95/P99 and "
            "CED curves because a model can win often and still lose badly on a smaller "
            "set of outlier cases."
        )

    lines.extend(["", "## Best and worst cases per model", ""])
    best_cases = tables.get("best_cases", pd.DataFrame())
    worst_cases = tables.get("worst_cases", pd.DataFrame())
    if best_cases.empty and worst_cases.empty:
        lines.append("_Best/worst case extraction was unavailable._")
    else:
        model_names = pd.concat(
            [
                best_cases.get("model_name", pd.Series(dtype=object)),
                worst_cases.get("model_name", pd.Series(dtype=object)),
            ],
            ignore_index=True,
        ).drop_duplicates()
        for model_name in model_names:
            lines.extend(["", f"### {model_name}", "", "10 best cases:"])
            best = best_cases[best_cases["model_name"] == model_name][
                [
                    "rank",
                    "image_id",
                    "orientation",
                    "image_level_nme",
                    "prediction_overlay_path",
                    "comparison_overlay_path",
                ]
            ]
            lines.append(dataframe_to_markdown(best, max_rows=10))
            lines.extend(["", "10 worst cases:"])
            worst = worst_cases[worst_cases["model_name"] == model_name][
                [
                    "rank",
                    "image_id",
                    "orientation",
                    "image_level_nme",
                    "prediction_overlay_path",
                    "comparison_overlay_path",
                ]
            ]
            lines.append(dataframe_to_markdown(worst, max_rows=10))

    lines.extend(["", "## Per-landmark analysis", ""])
    if metric_config is not None and metric_config.name == "hausdorff":
        lines.append(
            "_Skipped: Hausdorff is an image-level set distance and does not have "
            "meaningful per-landmark values._"
        )
    elif tables["per_landmark_metrics"].empty:
        lines.append("_Per-landmark CSVs were unavailable._")
    else:
        lines.append(
            "Landmark-pooled metrics are secondary and should not replace the image-level ranking."
        )
        lines.append("")
        lines.append(
            dataframe_to_markdown(tables["global_landmark_pooled_metrics"], max_rows=50)
        )
        primary = dataset.primary_model
        if primary:
            primary_landmarks = tables["per_landmark_metrics"][
                tables["per_landmark_metrics"]["model"] == primary
            ].sort_values("mean_landmark_nme", ascending=False)
            if not primary_landmarks.empty:
                lines.extend(["", f"Hardest landmarks for `{primary}` by mean NME:"])
                for _, row in primary_landmarks.head(10).iterrows():
                    lines.append(
                        f"- Landmark {int(row['landmark_index'])}: mean NME {format_float(row['mean_landmark_nme'])}"
                    )

    lines.extend(["", "## Anatomical-region analysis", ""])
    if tables["anatomical_region_metrics"].empty:
        lines.append(
            "_Anatomical-region mapping was unavailable or no per-landmark data were provided._"
        )
    else:
        if "babyland" in dataset.name.lower():
            lines.append(
                "BabyLand-72 uses 72 anatomical landmarks with visibility-aware metrics; "
                "it does not simply trace the visible face contour. Models with 68-landmark "
                "formats are not directly comparable on landmarks 68-71."
            )
        lines.append("")
        lines.append(
            dataframe_to_markdown(tables["anatomical_region_metrics"], max_rows=100)
        )

    lines.extend(["", "## Strengths of the primary model", ""])
    lines.extend(
        generate_primary_strengths(global_metrics, primary_vs, dataset.primary_model)
    )
    lines.extend(["", "## Weaknesses / limitations", ""])
    lines.extend(
        generate_primary_limitations(global_metrics, primary_vs, dataset.primary_model)
    )
    lines.extend(["", "## Recommended claims", ""])
    lines.extend(
        generate_recommended_claims(global_metrics, primary_vs, dataset.primary_model)
    )

    lines.extend(["", "## Figures", ""])
    for plot_path in plot_paths:
        lines.append(f"- [{plot_path}]({plot_path})")

    lines.extend(["", "## Warnings and sanity checks", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("_No warnings._")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_primary_strengths(
    global_metrics: pd.DataFrame,
    primary_vs: pd.DataFrame,
    primary_model: str | None,
) -> list[str]:
    """Generate conservative primary-model strength bullets."""
    if not primary_model or global_metrics.empty:
        return ["- Primary model not configured."]
    lines = []
    row = global_metrics[global_metrics["model"] == primary_model]
    if row.empty:
        return [f"- `{primary_model}` was not found in global metrics."]
    row = row.iloc[0]
    detection_rate = pd.to_numeric(
        pd.Series([row.get("detection_rate")]), errors="coerce"
    ).iloc[0]
    if np.isfinite(detection_rate) and detection_rate >= 0.999:
        lines.append(
            f"- `{primary_model}` achieves full or near-full detection coverage."
        )
    if not primary_vs.empty:
        wins = primary_vs[primary_vs["primary_win_rate"] > 0.5]
        for _, item in wins.iterrows():
            lines.append(
                f"- `{primary_model}` has lower NME than `{item['baseline_model']}` on "
                f"{format_float(item['primary_win_rate'] * 100, 1)}% of common images."
            )
    best_mean = global_metrics.sort_values("mean_image_nme").iloc[0]
    if best_mean["model"] == primary_model:
        lines.append(
            f"- `{primary_model}` has the lowest mean image-level NME among evaluated models."
        )
    return lines or [
        "- No automatic strength was identified; inspect the tables for nuanced behavior."
    ]


def generate_primary_limitations(
    global_metrics: pd.DataFrame,
    primary_vs: pd.DataFrame,
    primary_model: str | None,
) -> list[str]:
    """Generate conservative primary-model limitation bullets."""
    if not primary_model or global_metrics.empty:
        return ["- Primary model not configured."]
    lines = []
    row = global_metrics[global_metrics["model"] == primary_model]
    if row.empty:
        return [f"- `{primary_model}` was not found in global metrics."]
    best_mean = global_metrics.sort_values("mean_image_nme").iloc[0]
    if best_mean["model"] != primary_model:
        lines.append(
            f"- `{primary_model}` does not have the lowest global mean image NME; "
            f"`{best_mean['model']}` is lower on this metric."
        )
    best_p95 = global_metrics.sort_values("p95_image_nme").iloc[0]
    if best_p95["model"] != primary_model:
        lines.append(
            f"- `{primary_model}` does not have the best P95 tail error; "
            f"`{best_p95['model']}` is lower."
        )
    if not primary_vs.empty:
        losses = primary_vs[primary_vs["primary_win_rate"] < 0.5]
        for _, item in losses.iterrows():
            lines.append(
                f"- `{primary_model}` has lower NME than `{item['baseline_model']}` on fewer than half "
                "of common images."
            )
    return lines or [
        "- No automatic limitation was identified; still avoid claims beyond the configured datasets."
    ]


def generate_recommended_claims(
    global_metrics: pd.DataFrame,
    primary_vs: pd.DataFrame,
    primary_model: str | None,
) -> list[str]:
    """Generate cautious benchmark claims."""
    if not primary_model:
        return ["- Configure `primary_model` to generate primary-model claims."]
    claims = [
        "- Report image-level NME as the primary accuracy metric and detection rate as a separate coverage metric.",
        "- Use pairwise common-image comparisons when claiming improvement over a baseline.",
    ]
    if not primary_vs.empty:
        for _, row in primary_vs.iterrows():
            if row.get("primary_win_rate", 0) > 0.5:
                claims.append(
                    f"- `{primary_model}` is better than `{row['baseline_model']}` on most common images "
                    f"({format_float(row['primary_win_rate'] * 100, 1)}% win rate)."
                )
    if not global_metrics.empty:
        primary_row = global_metrics[global_metrics["model"] == primary_model]
        if primary_row.empty:
            return claims[:5]
        detection_rate = pd.to_numeric(
            pd.Series([primary_row.iloc[0].get("detection_rate")]),
            errors="coerce",
        ).iloc[0]
        if np.isfinite(detection_rate) and detection_rate >= 0.999:
            claims.append(
                f"- `{primary_model}` provides full detection coverage on this benchmark configuration."
            )
    return claims[:5]


def write_top_level_benchmark_summary(
    output_dir: Path,
    config: BenchmarkAnalysisConfig,
    summary: pd.DataFrame,
) -> None:
    """Write a top-level summary across all protocol/metric benchmark runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Benchmark Summary: {config.dataset.name}",
        "",
        f"- Dataset type: `{config.dataset.dataset_type or 'unspecified'}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Protocols",
        "",
    ]
    for protocol in config.evaluation_protocols:
        description = f": {protocol.description}" if protocol.description else ""
        lines.append(f"- `{protocol.name}` ({protocol.display_name}){description}")
    if config.dataset.dataset_type == "babyland72":
        lines.extend(
            [
                "",
                "For BabyLand-72, `main_comparison` includes all models with compatible metrics. "
                "Visibility-aware models use GT-valid columns, while non-visibility SOTA models "
                "use their standard valid-landmark columns.",
                "",
                "`visibility_protocol_analysis` includes only models configured with "
                "`predicts_visibility: true` and compares their `visibility_intersection` and "
                "`gt_valid` metric sources.",
            ]
        )
    lines.extend(["", "## Metrics", ""])
    for metric in config.metrics:
        lines.append(f"- `{metric.name}` ({metric.display_name})")
    lines.extend(["", "## Main Results", ""])
    if summary.empty:
        lines.append("_No protocol/metric combination produced image-level results._")
    else:
        lines.append(dataframe_to_markdown(summary, max_rows=100))
        summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    lines.extend(
        [
            "",
            "## Column Audit",
            "",
            "See `dataset_input_summary.csv` for the selected CSV columns, available columns, "
            "missing mappings, and row counts for each model/protocol/metric combination.",
            "",
        ]
    )
    (output_dir / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_benchmark_analysis(config: BenchmarkAnalysisConfig) -> dict[str, pd.DataFrame]:
    """Run the complete benchmark analysis pipeline."""
    all_tables: dict[str, pd.DataFrame] = {}
    input_summary_parts = []
    benchmark_rows = []
    for protocol_config in config.evaluation_protocols:
        for metric_config in config.metrics:
            metric_output_dir = (
                config.dataset.output_dir / protocol_config.name / metric_config.name
            )
            metric_config_dataset = dataclasses_replace_dataset_output(
                config,
                metric_output_dir,
                protocol_config,
                metric_config,
            )
            tables = run_benchmark_analysis_for_metric(
                metric_config_dataset,
                protocol_config,
                metric_config,
            )
            all_tables.update(
                {
                    f"{protocol_config.name}/{metric_config.name}/{key}": value
                    for key, value in tables.items()
                }
            )
            dataset_input = tables.get("dataset_input_summary")
            if dataset_input is not None and not dataset_input.empty:
                input_summary_parts.append(dataset_input)
            global_metrics = tables.get("global_image_metrics", pd.DataFrame())
            if not global_metrics.empty and "mean_image_nme" in global_metrics:
                ranked = global_metrics.sort_values("mean_image_nme").reset_index(
                    drop=True
                )
                best = ranked.iloc[0]
                primary_rank = ""
                if config.dataset.primary_model:
                    matches = ranked.index[
                        ranked["model"].eq(config.dataset.primary_model)
                    ]
                    primary_rank = int(matches[0] + 1) if len(matches) else ""
                benchmark_rows.append(
                    {
                        "evaluation_protocol": protocol_config.name,
                        "metric_name": metric_config.name,
                        "best_model_by_mean_image_nme": best.get("model", ""),
                        "best_mean_image_nme": best.get("mean_image_nme", np.nan),
                        "primary_model_rank": primary_rank,
                        "output_dir": str(metric_output_dir),
                    }
                )
    config.dataset.output_dir.mkdir(parents=True, exist_ok=True)
    if input_summary_parts:
        pd.concat(input_summary_parts, ignore_index=True).to_csv(
            config.dataset.output_dir / "dataset_input_summary.csv",
            index=False,
        )
    write_top_level_benchmark_summary(
        config.dataset.output_dir, config, pd.DataFrame(benchmark_rows)
    )
    return all_tables


def dataclasses_replace_dataset_output(
    config: BenchmarkAnalysisConfig,
    output_dir: Path,
    protocol_config: EvaluationProtocolConfig | None = None,
    metric_config: BenchmarkMetricConfig | None = None,
) -> BenchmarkAnalysisConfig:
    """Return a shallow config copy with a metric-specific output directory."""
    import dataclasses

    dataset = dataclasses.replace(
        config.dataset,
        output_dir=output_dir,
        evaluation_protocol="" if protocol_config is None else protocol_config.name,
        evaluation_protocol_display_name=""
        if protocol_config is None
        else protocol_config.display_name,
    )
    metrics = config.metrics if metric_config is None else [metric_config]
    return dataclasses.replace(config, dataset=dataset, metrics=metrics)


def run_benchmark_analysis_for_metric(
    config: BenchmarkAnalysisConfig,
    protocol_config: EvaluationProtocolConfig,
    metric_config: BenchmarkMetricConfig,
) -> dict[str, pd.DataFrame]:
    """Run the complete benchmark analysis pipeline for one geometric metric."""
    output_dir = config.dataset.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    model_jobs, analysis_warnings = expand_models_for_analysis(config, protocol_config)
    results = [
        load_model_results(
            model,
            config,
            source_protocol,
            metric_config,
            drop_invalid_nme=config.drop_invalid_nme,
            analysis_name=protocol_config.name,
            selected_metric_source=selected_source_name,
        )
        for model, source_protocol, selected_source_name in model_jobs
    ]
    warnings = list(analysis_warnings)
    if not results:
        warnings.append(
            f"No models were available for analysis `{protocol_config.name}`."
        )
    warnings.extend([warning for result in results for warning in result.warnings])
    _, _, style_warnings = build_model_style_maps(
        [result.config.name for result in results],
        config.dataset,
    )
    warnings.extend(style_warnings)
    orientation_warning_table = pd.DataFrame(
        [row for result in results for row in result.orientation_warnings]
    )

    global_metrics = compute_global_image_metrics(results)
    orientation_metrics = compute_orientation_metrics(results, config.dataset)
    landmark_pooled = compute_landmark_pooled_metrics(results)
    per_landmark = compute_per_landmark_metrics(results)
    region_metrics = compute_anatomical_region_metrics(
        results, config.dataset.anatomical_regions
    )
    pairwise = compute_pairwise_image_comparisons(
        results,
        config.dataset,
        bootstrap_iterations=config.bootstrap_iterations,
        random_seed=config.random_seed,
        by_orientation=False,
    )
    pairwise_orientation = compute_pairwise_image_comparisons(
        results,
        config.dataset,
        bootstrap_iterations=config.bootstrap_iterations,
        random_seed=config.random_seed,
        by_orientation=True,
    )
    primary_vs = compute_primary_vs_baselines(
        pairwise,
        config.dataset.primary_model,
        config.dataset.reference_models,
    )
    rankings = compute_model_ranking_summary(
        global_metrics, orientation_metrics, region_metrics
    )
    input_summary = compute_dataset_input_summary(results)
    best_worst_cases = compute_best_worst_cases(results, config.dataset, metric_config)
    best_worst_tables = split_best_worst_case_tables(best_worst_cases)

    if not pairwise.empty:
        missing_common = pairwise[pairwise["n_common_images"].fillna(0) == 0]
        for _, row in missing_common.iterrows():
            warnings.append(
                f"No common images for {row['model_a']} vs {row['model_b']}."
            )

    tables = {
        "global_image_metrics": global_metrics,
        "orientation_image_metrics": orientation_metrics,
        "global_landmark_pooled_metrics": landmark_pooled,
        "per_landmark_metrics": per_landmark,
        "anatomical_region_metrics": region_metrics,
        "pairwise_image_comparisons": pairwise,
        "pairwise_orientation_comparisons": pairwise_orientation,
        "primary_model_vs_baselines": primary_vs,
        "model_ranking_summary": rankings,
        "dataset_input_summary": input_summary,
        "orientation_label_warnings": orientation_warning_table,
        "best_worst_cases_per_model": best_worst_cases,
        "best_cases": best_worst_tables["best_cases"],
        "worst_cases": best_worst_tables["worst_cases"],
    }
    tables = add_analysis_context_to_tables(
        tables, config.dataset, protocol_config, metric_config
    )
    write_tables(output_dir, tables)
    write_best_worst_case_tables(output_dir, best_worst_cases)
    plot_paths = generate_benchmark_plots(
        output_dir,
        results,
        global_metrics,
        orientation_metrics,
        per_landmark,
        region_metrics,
        primary_vs,
        config,
    )
    write_markdown_report(
        output_dir,
        config,
        tables,
        plot_paths,
        warnings,
        metric_config,
        protocol_config,
    )
    print(f"Benchmark analysis completed: {output_dir}")
    print(f"Tables: {output_dir / 'tables'}")
    print(f"Plots: {output_dir / 'plots'}")
    print(f"Report: {output_dir / 'report.md'}")
    return tables


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze landmark detection benchmark CSVs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML or JSON benchmark config.",
    )
    parser.add_argument(
        "--drop-invalid-nme",
        action="store_true",
        help="Drop rows with NaN or infinite NME after reporting them.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    config = load_config(args.config, drop_invalid_nme=args.drop_invalid_nme)
    run_benchmark_analysis(config)


if __name__ == "__main__":
    main()
