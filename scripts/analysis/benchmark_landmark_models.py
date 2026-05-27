from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent.
    yaml = None

_SCIPY_STATS: Any | None = None
_SCIPY_IMPORT_ATTEMPTED = False


IMAGE_ID_ALIASES = ("image_id", "sample_id", "stem", "filename", "file", "image")
IMAGE_NME_ALIASES = (
    "nme",
    "image_nme",
    "mean_nme",
    "mean_nme_box",
    "box_nme",
    "nme_box",
)
LANDMARK_NME_ALIASES = ("nme", "point_to_point_nme_box", "landmark_nme", "nme_box")
LANDMARK_INDEX_ALIASES = ("landmark_index", "landmark_idx", "landmark_id", "landmark", "point_index")
ORIENTATION_ALIASES = ("orientation", "yaw_group", "yaw_angle", "class_idx", "pose")
DETECTED_ALIASES = ("detected", "is_detected", "success", "valid_detection")

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


@dataclass
class ModelBenchmarkConfig:
    """Configuration for one model in the landmark benchmark."""

    name: str
    per_image_csv: Path | None = None
    per_landmark_csv: Path | None = None
    detection_rate: float | None = None
    landmark_format: str | None = None
    columns: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class DatasetBenchmarkConfig:
    """Dataset-level benchmark configuration."""

    name: str
    output_dir: Path
    primary_model: str | None = None
    reference_models: list[str] = field(default_factory=list)
    orientation_mapping: dict[str, str] = field(default_factory=dict)
    anatomical_regions: dict[str, list[int]] = field(default_factory=dict)
    landmark_count: int | None = None
    landmark_index_base: str = "auto"
    image_id_strip_regexes: list[str] = field(default_factory=lambda: [r"__det_\\d+$"])
    suspicious_nme_threshold: float = 1.0


@dataclass
class BenchmarkAnalysisConfig:
    """Complete benchmark-analysis configuration."""

    dataset: DatasetBenchmarkConfig
    models: list[ModelBenchmarkConfig]
    drop_invalid_nme: bool = False
    bootstrap_iterations: int = 2000
    random_seed: int = 12345
    save_pdf: bool = False


@dataclass
class LoadedModelResults:
    """Loaded and normalized result tables for one model."""

    config: ModelBenchmarkConfig
    per_image: pd.DataFrame | None
    per_landmark: pd.DataFrame | None
    warnings: list[str] = field(default_factory=list)


def load_config(config_path: str | Path, drop_invalid_nme: bool | None = None) -> BenchmarkAnalysisConfig:
    """Load a benchmark config from YAML or JSON."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs. Install pyyaml or use JSON.")
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    else:
        with config_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

    dataset_raw = raw.get("dataset", {})
    dataset_name = str(dataset_raw.get("name", "benchmark"))
    regions = dataset_raw.get("anatomical_regions") or {}
    if not regions and "babyland" in dataset_name.lower():
        regions = DEFAULT_BABYLAND72_REGIONS

    orientation_mapping = dataset_raw.get("orientation_mapping") or {}
    if not orientation_mapping and "babyland" in dataset_name.lower():
        orientation_mapping = DEFAULT_BABYLAND_ORIENTATION_MAPPING

    dataset = DatasetBenchmarkConfig(
        name=dataset_name,
        output_dir=Path(dataset_raw["output_dir"]),
        primary_model=dataset_raw.get("primary_model"),
        reference_models=list(dataset_raw.get("reference_models", [])),
        orientation_mapping={str(key): str(value) for key, value in orientation_mapping.items()},
        anatomical_regions={str(key): list(value) for key, value in regions.items()},
        landmark_count=dataset_raw.get("landmark_count"),
        landmark_index_base=str(dataset_raw.get("landmark_index_base", "auto")),
        image_id_strip_regexes=list(dataset_raw.get("image_id_strip_regexes", [r"__det_\\d+$"])),
        suspicious_nme_threshold=float(dataset_raw.get("suspicious_nme_threshold", 1.0)),
    )

    models = []
    for item in raw.get("models", []):
        models.append(
            ModelBenchmarkConfig(
                name=str(item["name"]),
                detection_rate=(
                    None if item.get("detection_rate") is None else float(item["detection_rate"])
                ),
                per_image_csv=Path(item["per_image_csv"]) if item.get("per_image_csv") else None,
                per_landmark_csv=Path(item["per_landmark_csv"]) if item.get("per_landmark_csv") else None,
                landmark_format=None if item.get("landmark_format") is None else str(item["landmark_format"]),
                columns=dict(item.get("columns", {})),
            )
        )

    if not models:
        raise ValueError("Config must contain at least one model.")

    return BenchmarkAnalysisConfig(
        dataset=dataset,
        models=models,
        drop_invalid_nme=bool(raw.get("drop_invalid_nme", False) if drop_invalid_nme is None else drop_invalid_nme),
        bootstrap_iterations=int(raw.get("bootstrap_iterations", 2000)),
        random_seed=int(raw.get("random_seed", 12345)),
        save_pdf=bool(raw.get("save_pdf", False)),
    )


def normalize_image_id(value: Any, strip_regexes: list[str] | None = None) -> str:
    """Normalize image identifiers so rows from different model CSVs can be paired."""
    text = "" if pd.isna(value) else str(value)
    text = Path(text).name
    text = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", "", text, flags=re.IGNORECASE)
    for pattern in strip_regexes or []:
        text = re.sub(pattern, "", text)
    return text.strip()


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
            raise ValueError(f"Configured column {column!r} for {canonical_name!r} is not present.")
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


def map_orientation_labels(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map numeric or string orientation labels to display labels."""
    if series.empty:
        return series

    def _map_one(value: Any) -> str:
        if pd.isna(value):
            return ""
        raw = str(value).strip()
        numeric = raw
        try:
            numeric = str(int(float(raw)))
        except ValueError:
            pass
        return mapping.get(raw, mapping.get(numeric, raw))

    return series.map(_map_one)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with a helpful missing-file error."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def load_model_results(
    model_config: ModelBenchmarkConfig,
    dataset_config: DatasetBenchmarkConfig,
    drop_invalid_nme: bool = False,
) -> LoadedModelResults:
    """Load, normalize, and validate result CSVs for one model."""
    warnings: list[str] = []
    per_image = None
    per_landmark = None

    if model_config.per_image_csv is not None:
        raw = _read_csv(model_config.per_image_csv)
        mapping = model_config.columns.get("per_image", {})
        image_col = resolve_column(raw, "image_id", IMAGE_ID_ALIASES, mapping)
        nme_col = resolve_column(raw, "nme", IMAGE_NME_ALIASES, mapping)
        orientation_col = resolve_column(raw, "orientation", ORIENTATION_ALIASES, mapping, required=False)
        detected_col = resolve_column(raw, "detected", DETECTED_ALIASES, mapping, required=False)

        per_image = pd.DataFrame(
            {
                "model": model_config.name,
                "image_id": raw[image_col].astype(str),
                "image_key": raw[image_col].map(
                    lambda value: normalize_image_id(value, dataset_config.image_id_strip_regexes)
                ),
                "nme": pd.to_numeric(raw[nme_col], errors="coerce"),
            }
        )
        if orientation_col is not None:
            per_image["orientation"] = map_orientation_labels(
                raw[orientation_col], dataset_config.orientation_mapping
            )
        else:
            per_image["orientation"] = ""
        if detected_col is not None:
            per_image["detected"] = raw[detected_col].astype(str).str.lower().isin(
                {"1", "true", "yes", "y", "detected", "success"}
            )
        else:
            per_image["detected"] = per_image["nme"].notna()

        warnings.extend(validate_nme_table(per_image, model_config.name, "per-image", dataset_config))
        if drop_invalid_nme:
            per_image = per_image[np.isfinite(per_image["nme"])].copy()

    if model_config.per_landmark_csv is not None:
        raw = _read_csv(model_config.per_landmark_csv)
        mapping = model_config.columns.get("per_landmark", {})
        image_col = resolve_column(raw, "image_id", IMAGE_ID_ALIASES, mapping)
        nme_col = resolve_column(raw, "nme", LANDMARK_NME_ALIASES, mapping)
        landmark_col = resolve_column(raw, "landmark_index", LANDMARK_INDEX_ALIASES, mapping)
        orientation_col = resolve_column(raw, "orientation", ORIENTATION_ALIASES, mapping, required=False)
        valid_col = resolve_column(raw, "valid", ("valid", "is_valid"), mapping, required=False)

        landmark_index = pd.to_numeric(raw[landmark_col], errors="coerce")
        landmark_index = normalize_landmark_index(
            landmark_index,
            dataset_config.landmark_index_base,
            dataset_config.landmark_count,
        )
        per_landmark = pd.DataFrame(
            {
                "model": model_config.name,
                "image_id": raw[image_col].astype(str),
                "image_key": raw[image_col].map(
                    lambda value: normalize_image_id(value, dataset_config.image_id_strip_regexes)
                ),
                "landmark_index": landmark_index,
                "nme": pd.to_numeric(raw[nme_col], errors="coerce"),
            }
        )
        if orientation_col is not None:
            per_landmark["orientation"] = map_orientation_labels(
                raw[orientation_col], dataset_config.orientation_mapping
            )
        else:
            per_landmark["orientation"] = ""
        if valid_col is not None:
            per_landmark["valid"] = raw[valid_col].astype(str).str.lower().isin(
                {"1", "true", "yes", "y", "valid"}
            )
        else:
            per_landmark["valid"] = per_landmark["nme"].notna()

        warnings.extend(validate_nme_table(per_landmark, model_config.name, "per-landmark", dataset_config))
        if drop_invalid_nme:
            per_landmark = per_landmark[
                np.isfinite(per_landmark["nme"]) & per_landmark["landmark_index"].notna()
            ].copy()

    return LoadedModelResults(model_config, per_image, per_landmark, warnings)


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
    duplicate_count = int(df.duplicated(["image_key"]).sum()) if table_name == "per-image" else 0
    if duplicate_count:
        warnings.append(f"{model_name} {table_name}: {duplicate_count} duplicate normalized image IDs.")
    invalid_count = int((~np.isfinite(df["nme"])).sum())
    if invalid_count:
        warnings.append(f"{model_name} {table_name}: {invalid_count} rows have NaN or infinite NME.")
    suspicious_count = int((df["nme"] > dataset_config.suspicious_nme_threshold).sum())
    if suspicious_count:
        warnings.append(
            f"{model_name} {table_name}: {suspicious_count} rows have NME > "
            f"{dataset_config.suspicious_nme_threshold:g}."
        )
    if "landmark_index" in df.columns and dataset_config.landmark_count is not None:
        outside = df[
            df["landmark_index"].notna()
            & ((df["landmark_index"] < 0) | (df["landmark_index"] >= dataset_config.landmark_count))
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
        row["detection_rate"] = (
            result.config.detection_rate
            if result.config.detection_rate is not None
            else float(result.per_image["detected"].mean())
        )
        row["n_detected"] = int(data.shape[0])
        row.update({key: value for key, value in summarize_nme(data["nme"], "image").items() if key != "n_image"})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_image_nme", na_position="last")


def compute_orientation_metrics(results: list[LoadedModelResults]) -> pd.DataFrame:
    """Compute image-level metrics grouped by orientation."""
    rows = []
    for result in results:
        if result.per_image is None or "orientation" not in result.per_image:
            continue
        data = finite_nme(result.per_image)
        data = data[data["orientation"].astype(str).str.len() > 0]
        for orientation, group in data.groupby("orientation", dropna=True):
            row = {"model": result.config.name, "orientation": orientation}
            row.update(summarize_nme(group["nme"], "image"))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_landmark_pooled_metrics(results: list[LoadedModelResults]) -> pd.DataFrame:
    """Compute secondary landmark-pooled metrics by model."""
    rows = []
    for result in results:
        if result.per_landmark is None:
            continue
        data = finite_nme(result.per_landmark)
        row = {"model": result.config.name}
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
            row = {"model": result.config.name, "landmark_index": int(landmark_index)}
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
            summary = summarize_nme(group["nme"], "landmark")
            rows.append(
                {
                    "model": result.config.name,
                    "region": region,
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
                orientations = sorted(
                    set(image_tables[model_a]["orientation"].dropna().astype(str))
                    & set(image_tables[model_b]["orientation"].dropna().astype(str))
                )
                groups = [(orientation, orientation) for orientation in orientations if orientation]
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
    return (float(np.quantile(sample_means, 0.025)), float(np.quantile(sample_means, 0.975)))


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
                    "mean_signed_diff_primary_minus_baseline": row.get("mean_signed_diff_a_minus_b"),
                    "median_signed_diff_primary_minus_baseline": row.get("median_signed_diff_a_minus_b"),
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
                    "mean_signed_diff_primary_minus_baseline": -row.get("mean_signed_diff_a_minus_b", np.nan),
                    "median_signed_diff_primary_minus_baseline": -row.get("median_signed_diff_a_minus_b", np.nan),
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
            idx = global_metrics[column].idxmin() if lower_is_better else global_metrics[column].idxmax()
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
        rows.append(
            {
                "model": result.config.name,
                "detection_rate_config": result.config.detection_rate,
                "landmark_format": result.config.landmark_format,
                "per_image_csv": str(result.config.per_image_csv) if result.config.per_image_csv else "",
                "per_image_rows": 0 if result.per_image is None else int(len(result.per_image)),
                "per_landmark_csv": str(result.config.per_landmark_csv) if result.config.per_landmark_csv else "",
                "per_landmark_rows": 0 if result.per_landmark is None else int(len(result.per_landmark)),
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


def import_matplotlib_pyplot() -> Any:
    """Import matplotlib lazily so table/report generation can run without it."""
    try:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - depends on local binary deps.
        raise ImportError(f"Could not import matplotlib for plot generation: {error}") from error
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


def save_figure(fig: Any, path: Path, save_pdf: bool = False) -> None:
    """Save a matplotlib figure as PNG and optionally PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    if save_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    try:
        import_matplotlib_pyplot().close(fig)
    except ImportError:
        pass


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

    plot_dir = output_dir / "plots"
    plot_paths: list[str] = []
    if global_metrics.empty:
        return plot_paths

    models = global_metrics["model"].tolist()
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4))
    ax.bar(x, global_metrics["detection_rate"])
    ax.set_xticks(x, models, rotation=30, ha="right")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, min(1.05, max(1.0, global_metrics["detection_rate"].max() * 1.1)))
    ax.set_title("Detection rate by model")
    path = plot_dir / "detection_rate_bar.png"
    save_figure(fig, path, config.save_pdf)
    plot_paths.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4))
    ax.bar(x, global_metrics["mean_image_nme"])
    ax.set_xticks(x, models, rotation=30, ha="right")
    ax.set_ylabel("Mean image NME")
    ax.set_title("Mean image-level NME by model")
    path = plot_dir / "mean_image_nme_bar.png"
    save_figure(fig, path, config.save_pdf)
    plot_paths.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(global_metrics["detection_rate"], global_metrics["mean_image_nme"])
    for _, row in global_metrics.iterrows():
        ax.annotate(row["model"], (row["detection_rate"], row["mean_image_nme"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Detection rate")
    ax.set_ylabel("Mean image NME")
    ax.set_title("Accuracy vs coverage")
    path = plot_dir / "mean_nme_vs_detection_rate.png"
    save_figure(fig, path, config.save_pdf)
    plot_paths.append(str(path.relative_to(output_dir)))

    image_frames = [finite_nme(result.per_image) for result in results if result.per_image is not None]
    if image_frames:
        image_all = pd.concat(image_frames, ignore_index=True)
        ordered_data = [image_all.loc[image_all["model"] == model, "nme"].to_numpy() for model in models]
        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4.5))
        ax.boxplot(ordered_data, labels=models, showfliers=False)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Image-level NME")
        ax.set_title("Image-level NME distribution")
        path = plot_dir / "image_nme_boxplot_by_model.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

        fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4.5))
        ax.violinplot(ordered_data, showmeans=True, showextrema=True)
        ax.set_xticks(np.arange(1, len(models) + 1), models, rotation=30, ha="right")
        ax.set_ylabel("Image-level NME")
        ax.set_title("Image-level NME violin plot")
        path = plot_dir / "image_nme_violin_by_model.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

        fig, ax = plt.subplots(figsize=(7, 5))
        for model in models:
            values = np.sort(image_all.loc[image_all["model"] == model, "nme"].to_numpy())
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, y, label=model)
        ax.set_xlabel("Image-level NME")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title("CED curves")
        ax.legend(fontsize=8)
        path = plot_dir / "ced_curves_image_nme.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

    if not orientation_metrics.empty:
        pivot = orientation_metrics.pivot(index="model", columns="orientation", values="mean_image_nme")
        fig, ax = plt.subplots(figsize=(max(7, pivot.shape[1] * 1.1), max(4, pivot.shape[0] * 0.5)))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
        ax.set_title("Orientation-wise mean image NME")
        fig.colorbar(image, ax=ax, label="Mean image NME")
        path = plot_dir / "orientation_mean_nme_heatmap.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

        fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.5), 5))
        pivot.T.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean image NME")
        ax.set_title("Orientation-wise mean image NME")
        ax.legend(title="Model", fontsize=8)
        path = plot_dir / "orientation_mean_nme_grouped_bar.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

    primary = config.dataset.primary_model
    if primary and not per_landmark_metrics.empty:
        primary_landmarks = per_landmark_metrics[per_landmark_metrics["model"] == primary]
        primary_raw = next((finite_nme(r.per_landmark) for r in results if r.config.name == primary and r.per_landmark is not None), pd.DataFrame())
        if not primary_raw.empty:
            landmark_indices = sorted(primary_raw["landmark_index"].dropna().unique())
            data = [primary_raw.loc[primary_raw["landmark_index"] == idx, "nme"].to_numpy() for idx in landmark_indices]
            fig, ax = plt.subplots(figsize=(max(10, len(landmark_indices) * 0.22), 4.5))
            ax.boxplot(data, labels=[str(int(idx)) for idx in landmark_indices], showfliers=False)
            ax.set_xlabel("Landmark index (0-based)")
            ax.set_ylabel("Per-landmark NME")
            ax.set_title(f"Per-landmark NME for {primary}")
            path = plot_dir / "primary_per_landmark_nme_boxplot.png"
            save_figure(fig, path, config.save_pdf)
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
        for model in models:
            model_rows = per_landmark_metrics[
                (per_landmark_metrics["model"] == model)
                & (per_landmark_metrics["landmark_index"] < common_limit)
            ].sort_values("landmark_index")
            if not model_rows.empty:
                ax.plot(model_rows["landmark_index"], model_rows["mean_landmark_nme"], label=model)
        ax.set_xlabel("Landmark index (0-based)")
        ax.set_ylabel("Mean landmark NME")
        ax.set_title("Per-landmark mean NME across models")
        ax.legend(fontsize=8)
        path = plot_dir / "per_landmark_mean_nme_lines_common_landmarks.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

    if not region_metrics.empty:
        pivot = region_metrics.pivot(index="region", columns="model", values="mean_nme")
        fig, ax = plt.subplots(figsize=(max(9, len(pivot) * 0.8), 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Mean NME")
        ax.set_title("Anatomical-region mean NME")
        ax.legend(title="Model", fontsize=8)
        path = plot_dir / "anatomical_region_mean_nme_bar.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

    if not primary_vs_baselines.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(primary_vs_baselines) * 1.2), 4))
        ax.bar(primary_vs_baselines["baseline_model"], primary_vs_baselines["primary_win_rate"])
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Primary win rate")
        ax.set_title(f"{primary} pairwise win rate")
        ax.set_xticklabels(primary_vs_baselines["baseline_model"], rotation=30, ha="right")
        path = plot_dir / "primary_pairwise_win_rate.png"
        save_figure(fig, path, config.save_pdf)
        plot_paths.append(str(path.relative_to(output_dir)))

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.2), 4))
    width = 0.35
    ax.bar(x - width / 2, global_metrics["FRI_0.10"], width, label="FRI > 0.10")
    ax.bar(x + width / 2, global_metrics["FRI_0.20"], width, label="FRI > 0.20")
    ax.set_xticks(x, models, rotation=30, ha="right")
    ax.set_ylabel("Failure rate")
    ax.set_title("Image-level failure rates")
    ax.legend()
    path = plot_dir / "image_failure_rate_bar.png"
    save_figure(fig, path, config.save_pdf)
    plot_paths.append(str(path.relative_to(output_dir)))

    return plot_paths


def format_float(value: Any, digits: int = 4) -> str:
    """Format floats for Markdown tables."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
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
        lines.append("| " + " | ".join(format_float(row[col]) for col in columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing first {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def write_markdown_report(
    output_dir: Path,
    config: BenchmarkAnalysisConfig,
    tables: dict[str, pd.DataFrame],
    plot_paths: list[str],
    warnings: list[str],
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
        dataframe_to_markdown(global_metrics, max_rows=50),
        "",
        "- `mean_image_nme`: average image-level normalized mean error.",
        "- `median_image_nme`: median image-level error, less sensitive to outliers.",
        "- `p90/p95/p99`: tail-error percentiles.",
        "- `FRI_0.10`: fraction of images with image-level NME greater than 0.10.",
        "- Detection rate is reported separately from accuracy on detected/evaluated images.",
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
                lines.append(f"- `{row['group']}`: `{row['best_model']}` ({row['metric']}={format_float(row['best_value'])})")

    lines.extend(["", "## Pairwise analysis", ""])
    if primary_vs.empty:
        lines.append("_Primary-model pairwise comparisons were unavailable._")
    else:
        lines.append(dataframe_to_markdown(primary_vs, max_rows=50))
        lines.append("")
        lines.append("Negative signed differences mean the primary model has lower NME than the baseline.")

    lines.extend(["", "## Per-landmark analysis", ""])
    if tables["per_landmark_metrics"].empty:
        lines.append("_Per-landmark CSVs were unavailable._")
    else:
        lines.append("Landmark-pooled metrics are secondary and should not replace the image-level ranking.")
        lines.append("")
        lines.append(dataframe_to_markdown(tables["global_landmark_pooled_metrics"], max_rows=50))
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
        lines.append("_Anatomical-region mapping was unavailable or no per-landmark data were provided._")
    else:
        if "babyland" in dataset.name.lower():
            lines.append(
                "BabyLand-72 uses 72 anatomical landmarks with visibility-aware metrics; "
                "it does not simply trace the visible face contour. Models with 68-landmark "
                "formats are not directly comparable on landmarks 68-71."
            )
        lines.append("")
        lines.append(dataframe_to_markdown(tables["anatomical_region_metrics"], max_rows=100))

    lines.extend(["", "## Strengths of the primary model", ""])
    lines.extend(generate_primary_strengths(global_metrics, primary_vs, dataset.primary_model))
    lines.extend(["", "## Weaknesses / limitations", ""])
    lines.extend(generate_primary_limitations(global_metrics, primary_vs, dataset.primary_model))
    lines.extend(["", "## Recommended claims", ""])
    lines.extend(generate_recommended_claims(global_metrics, primary_vs, dataset.primary_model))

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
    if row.get("detection_rate", np.nan) >= 0.999:
        lines.append(f"- `{primary_model}` achieves full or near-full detection coverage.")
    if not primary_vs.empty:
        wins = primary_vs[primary_vs["primary_win_rate"] > 0.5]
        for _, item in wins.iterrows():
            lines.append(
                f"- `{primary_model}` has lower NME than `{item['baseline_model']}` on "
                f"{format_float(item['primary_win_rate'] * 100, 1)}% of common images."
            )
    best_mean = global_metrics.sort_values("mean_image_nme").iloc[0]
    if best_mean["model"] == primary_model:
        lines.append(f"- `{primary_model}` has the lowest mean image-level NME among evaluated models.")
    return lines or ["- No automatic strength was identified; inspect the tables for nuanced behavior."]


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
    return lines or ["- No automatic limitation was identified; still avoid claims beyond the configured datasets."]


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
        if not primary_row.empty and primary_row.iloc[0].get("detection_rate", np.nan) >= 0.999:
            claims.append(f"- `{primary_model}` provides full detection coverage on this benchmark configuration.")
    return claims[:5]


def run_benchmark_analysis(config: BenchmarkAnalysisConfig) -> dict[str, pd.DataFrame]:
    """Run the complete benchmark analysis pipeline."""
    output_dir = config.dataset.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    results = [
        load_model_results(model, config.dataset, drop_invalid_nme=config.drop_invalid_nme)
        for model in config.models
    ]
    warnings = [warning for result in results for warning in result.warnings]

    global_metrics = compute_global_image_metrics(results)
    orientation_metrics = compute_orientation_metrics(results)
    landmark_pooled = compute_landmark_pooled_metrics(results)
    per_landmark = compute_per_landmark_metrics(results)
    region_metrics = compute_anatomical_region_metrics(results, config.dataset.anatomical_regions)
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
    rankings = compute_model_ranking_summary(global_metrics, orientation_metrics, region_metrics)
    input_summary = compute_dataset_input_summary(results)

    if not pairwise.empty:
        missing_common = pairwise[pairwise["n_common_images"].fillna(0) == 0]
        for _, row in missing_common.iterrows():
            warnings.append(f"No common images for {row['model_a']} vs {row['model_b']}.")

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
    }
    write_tables(output_dir, tables)
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
    write_markdown_report(output_dir, config, tables, plot_paths, warnings)
    print(f"Benchmark analysis completed: {output_dir}")
    print(f"Tables: {output_dir / 'tables'}")
    print(f"Plots: {output_dir / 'plots'}")
    print(f"Report: {output_dir / 'report.md'}")
    return tables


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze landmark detection benchmark CSVs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML or JSON benchmark config.")
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
