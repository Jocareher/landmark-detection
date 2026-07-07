from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

from .confidence_metrics import compute_heatmap_confidence_metrics
from .confidence_error_plots import (
    figure_markdown_lines,
    save_confidence_error_figure_set,
)
from .metrics import decode_heatmaps_to_image_coords
from .metrics import compute_box_normalization_factor
from .pca_shape_prior import compute_pca_projection_loss
from .postprocessing import (
    apply_homogeneous_transform,
    extract_batched_size,
    project_landmarks_between_sizes,
    project_landmarks_to_original_size,
)
from .tta_consistency import compute_tta_consistency
from ..utils.natural_labels import compute_natural_valid_landmark_mask

FINE_REGION_NAMES = [
    "face_contour",
    "right_eyebrow",
    "left_eyebrow",
    "nose_bridge",
    "nose_base",
    "right_eye",
    "left_eye",
    "outer_lip",
    "inner_lip",
    "under_lip",
    "upper_chin",
    "left_chin",
    "right_chin",
]
GROUPED_REGION_MAP = {
    "face_contour": "contour",
    "right_eyebrow": "eyebrows",
    "left_eyebrow": "eyebrows",
    "right_eye": "eyes",
    "left_eye": "eyes",
    "nose_bridge": "nose",
    "nose_base": "nose",
    "outer_lip": "mouth",
    "inner_lip": "mouth",
    "under_lip": "mouth",
    "upper_chin": "contour",
    "left_chin": "contour",
    "right_chin": "contour",
}
ERROR_HIGH_SIGNALS = {"heatmap_entropy", "heatmap_variance", "tta_variance"}
CONFIDENCE_SIGNALS = [
    "heatmap_max",
    "heatmap_entropy",
    "heatmap_variance",
    "peak_sharpness",
    "tta_variance",
    "pca_reconstruction_error",
]


def run_confidence_error_analysis(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    dataset_description: str,
    eval_mode: str,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
    tta_samples: int = 0,
    pca_prior: dict[str, Any] | None = None,
    failure_thresholds: Iterable[float] = (0.05, 0.08, 0.10),
    retention_fractions: Iterable[float] = (0.10, 0.25, 0.50, 1.0),
    max_visual_examples: int = 12,
    max_batches: int | None = None,
    visibility_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run offline confidence-error analysis and save CSV, plots, and a report."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    per_landmark_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    example_candidates: list[dict[str, Any]] = []
    total_landmarks_seen = 0
    valid_gt_landmarks_seen = 0
    invalid_gt_landmarks_seen = 0
    nan_target_coordinate_rows_seen = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(dataloader, desc="Analyzing confidence", dynamic_ncols=True)
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            heatmaps = outputs["heatmaps"]
            predicted_input = decode_heatmaps_to_image_coords(
                heatmaps=heatmaps,
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=True,
                decoder=coordinate_decoder,
                softmax_temperature=wasserstein_softmax_temperature,
            ).cpu()
            predicted_visibility = (
                torch.sigmoid(outputs["visibility_logits"].detach().cpu())
                >= visibility_threshold
            ).to(torch.int64)
            heatmap_metrics = compute_heatmap_confidence_metrics(
                heatmaps=heatmaps.detach(),
                temperature=wasserstein_softmax_temperature,
            )
            tta_result = compute_tta_consistency(
                model=model,
                images=images,
                num_samples=tta_samples,
                coordinate_decoder=coordinate_decoder,
                wasserstein_softmax_temperature=wasserstein_softmax_temperature,
            )
            tta_variance_input = (
                tta_result.variance.detach().cpu() if tta_result is not None else None
            )
            tta_predictions_input = (
                tta_result.predictions.detach().cpu()
                if tta_result is not None
                else None
            )

            pca_errors = None
            if pca_prior is not None:
                pca_errors = _compute_sample_pca_errors(
                    predicted_landmarks=predicted_input.to(device),
                    pca_prior=pca_prior,
                )

            metadata_batch = batch["metadata"]
            target_landmarks_batch = batch["landmarks"].cpu()
            target_visibility_batch = batch.get("visibility")
            if target_visibility_batch is not None:
                target_visibility_batch = target_visibility_batch.cpu()

            for sample_index in range(images.shape[0]):
                sample = _prepare_sample_coordinates(
                    eval_mode=eval_mode,
                    metadata_batch=metadata_batch,
                    sample_index=sample_index,
                    predicted_input=predicted_input[sample_index],
                    target_landmarks=target_landmarks_batch[sample_index],
                    tta_variance_input=(
                        None
                        if tta_variance_input is None
                        else tta_variance_input[sample_index]
                    ),
                    tta_predictions_input=(
                        None
                        if tta_predictions_input is None
                        else tta_predictions_input[:, sample_index]
                    ),
                )
                target_visibility = (
                    None
                    if target_visibility_batch is None
                    else _tensor_to_numpy(
                        target_visibility_batch[sample_index],
                        dtype=np.int64,
                    )
                )
                predicted_visibility_sample = _tensor_to_numpy(
                    predicted_visibility[sample_index],
                    dtype=np.int64,
                )
                sample_rows, image_row = _build_sample_rows(
                    sample=sample,
                    target_visibility=target_visibility,
                    predicted_visibility=predicted_visibility_sample,
                    heatmap_metrics=heatmap_metrics,
                    sample_index=sample_index,
                    pca_error=(
                        None if pca_errors is None else float(pca_errors[sample_index])
                    ),
                )
                total_landmarks_seen += int(image_row["total_landmarks"])
                valid_gt_landmarks_seen += int(image_row["number_of_valid_landmarks"])
                invalid_gt_landmarks_seen += int(
                    image_row["number_of_invalid_landmarks"]
                )
                nan_target_coordinate_rows_seen += int(
                    image_row["number_of_nan_target_landmarks"]
                )
                per_landmark_rows.extend(sample_rows)
                per_image_rows.append(image_row)
                example_candidates.append(
                    {
                        "sample": sample,
                        "image_row": image_row,
                        "landmark_rows": sample_rows,
                        "predicted_visibility": predicted_visibility_sample,
                        "target_visibility": target_visibility,
                    }
                )

    if not per_landmark_rows:
        raise RuntimeError("No confidence-error rows were produced.")

    evaluable_rows = _evaluable_rows(per_landmark_rows)
    summary_by_region = summarize_by_region(per_landmark_rows)
    summary_by_pose = summarize_by_pose(per_image_rows, per_landmark_rows)
    correlations = compute_correlations(per_landmark_rows)
    quantile_rows = compute_quantile_error_rows(per_landmark_rows)
    retention_rows = compute_retention_curve_rows(
        rows=per_landmark_rows,
        failure_threshold=float(next(iter(failure_thresholds))),
        retention_fractions=list(retention_fractions),
    )
    failure_rows, roc_curves = compute_failure_detection_rows(
        rows=per_landmark_rows,
        failure_thresholds=list(failure_thresholds),
    )
    viability_rows = compute_region_viability_rows(retention_rows)

    _write_csv(output_dir / "per_landmark_confidence_error.csv", per_landmark_rows)
    _write_csv(output_dir / "per_image_confidence_error.csv", per_image_rows)
    _write_csv(output_dir / "summary_by_region.csv", summary_by_region)
    _write_csv(output_dir / "summary_by_pose.csv", summary_by_pose)
    _write_csv(output_dir / "confidence_error_correlations.csv", correlations)
    _write_csv(output_dir / "confidence_quantile_errors.csv", quantile_rows)
    _write_csv(output_dir / "retention_curves.csv", retention_rows)
    _write_csv(output_dir / "failure_detection.csv", failure_rows)
    _write_csv(output_dir / "region_pseudo_label_viability.csv", viability_rows)
    sanity_checks = build_sanity_checks(
        per_landmark_rows=per_landmark_rows,
        per_image_rows=per_image_rows,
        total_landmarks_seen=total_landmarks_seen,
        valid_gt_landmarks_seen=valid_gt_landmarks_seen,
        invalid_gt_landmarks_seen=invalid_gt_landmarks_seen,
        nan_target_coordinate_rows_seen=nan_target_coordinate_rows_seen,
    )
    sanity_checks.update(
        build_official_metric_warning(
            eval_mode=eval_mode,
            mean_nme_percent=sanity_checks.get("mean_nme_percent"),
        )
    )
    (output_dir / "sanity_checks.json").write_text(
        json.dumps(sanity_checks, indent=2),
        encoding="utf-8",
    )

    plot_outputs = save_confidence_error_figure_set(
        per_landmark_rows=per_landmark_rows,
        per_image_rows=per_image_rows,
        summary_by_region=summary_by_region,
        pose_summary_rows=summary_by_pose,
        correlations=correlations,
        retention_rows=retention_rows,
        failure_rows=failure_rows,
        viability_rows=viability_rows,
        figures_dir=figures_dir,
    )
    example_outputs = save_example_visualizations(
        candidates=example_candidates,
        examples_dir=examples_dir,
        max_visual_examples=max_visual_examples,
    )
    report_summary = write_markdown_report(
        output_path=output_dir / "confidence_error_report.md",
        dataset_description=dataset_description,
        checkpoint_path=checkpoint_path,
        per_image_rows=per_image_rows,
        correlations=correlations,
        summary_by_region=summary_by_region,
        summary_by_pose=summary_by_pose,
        retention_rows=retention_rows,
        viability_rows=viability_rows,
        sanity_checks=sanity_checks,
        plot_outputs=plot_outputs,
        example_outputs=example_outputs,
        pose_available=any(_has_value(row.get("pose")) for row in per_landmark_rows),
        visibility_available=any(
            _has_value(row.get("visibility")) for row in per_landmark_rows
        ),
        pca_available=pca_prior is not None,
        tta_samples=tta_samples,
    )

    summary = {
        "num_images": len(per_image_rows),
        "num_landmark_rows": len(per_landmark_rows),
        "num_evaluable_landmark_rows": len(evaluable_rows),
        "num_invalid_gt_landmark_rows": invalid_gt_landmarks_seen,
        "num_nan_target_coordinate_rows": nan_target_coordinate_rows_seen,
        "mean_nme_fraction": _safe_mean([row["mean_nme"] for row in per_image_rows]),
        "mean_nme_percent": _scale_fraction_to_percent(
            _safe_mean([row["mean_nme"] for row in per_image_rows])
        ),
        "median_image_nme_fraction": _safe_median(
            [row["mean_nme"] for row in per_image_rows]
        ),
        "global_landmark_mean_nme_fraction": _safe_mean(
            [row["normalized_error"] for row in evaluable_rows]
        ),
        "global_landmark_median_nme_fraction": _safe_median(
            [row["normalized_error"] for row in evaluable_rows]
        ),
        "evaluation_protocol": sanity_checks["evaluation_protocol"],
        "official_metric_warning": sanity_checks.get("official_metric_warning"),
        "outputs": {
            "report": str(report_summary),
            "figures": [str(path) for path in plot_outputs],
            "examples": [str(path) for path in example_outputs],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def summarize_by_region(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize NME and confidence metrics by fine and grouped facial regions."""
    output_rows: list[dict[str, Any]] = []
    for region in _ordered_regions(rows):
        region_rows = [row for row in rows if row["region"] == region]
        output_rows.append(_summarize_region(region, region_rows))
    for region in ["contour", "eyebrows", "eyes", "nose", "mouth"]:
        region_rows = [row for row in rows if row["grouped_region"] == region]
        output_rows.append(_summarize_region(region, region_rows))
    return output_rows


def summarize_by_pose(
    per_image_rows: list[dict[str, Any]],
    per_landmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize official NME and failure rates by pose when pose metadata exists."""
    poses = sorted(
        {
            str(row["pose"])
            for row in per_image_rows
            if _has_value(row.get("pose"))
        }
    )
    output_rows: list[dict[str, Any]] = []
    for pose in poses:
        image_rows = [
            row
            for row in per_image_rows
            if str(row.get("pose")) == pose and _is_finite_number(row.get("mean_nme"))
        ]
        landmark_rows = [
            row
            for row in _evaluable_rows(per_landmark_rows)
            if str(row.get("pose")) == pose
        ]
        errors = [row["normalized_error"] for row in landmark_rows]
        output_rows.append(
            {
                "pose": pose,
                "image_count": len(image_rows),
                "evaluable_landmark_count": len(landmark_rows),
                "mean_nme": _safe_mean([row["mean_nme"] for row in image_rows]),
                "mean_nme_percent": _scale_fraction_to_percent(
                    _safe_mean([row["mean_nme"] for row in image_rows])
                ),
                "median_nme": _safe_median([row["median_nme"] for row in image_rows]),
                "median_nme_percent": _scale_fraction_to_percent(
                    _safe_median([row["median_nme"] for row in image_rows])
                ),
                "landmark_mean_nme": _safe_mean(errors),
                "landmark_mean_nme_percent": _scale_fraction_to_percent(
                    _safe_mean(errors)
                ),
                "failure_rate_nme_gt_0_05": _failure_rate(errors, 0.05),
                "valid_gt_landmark_count": sum(
                    int(row.get("number_of_valid_landmarks", 0) or 0)
                    for row in image_rows
                ),
                "invalid_gt_landmark_count": sum(
                    int(row.get("number_of_invalid_landmarks", 0) or 0)
                    for row in image_rows
                ),
            }
        )
    return output_rows


def compute_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute Pearson and Spearman confidence-error correlations."""
    output_rows: list[dict[str, Any]] = []
    rows = _evaluable_rows(rows)
    scopes = (
        [("global", rows)]
        + [
            (region, [row for row in rows if row["region"] == region])
            for region in _ordered_regions(rows)
        ]
        + [
            (
                region,
                [row for row in rows if row.get("grouped_region") == region],
            )
            for region in ["contour", "eyebrows", "eyes", "nose", "mouth"]
        ]
    )
    for scope_name, scope_rows in scopes:
        for signal in CONFIDENCE_SIGNALS:
            pairs = _valid_pairs(scope_rows, signal, "normalized_error")
            if len(pairs) < 2:
                continue
            x_values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
            y_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            output_rows.append(
                {
                    "scope": scope_name,
                    "confidence_signal": signal,
                    "pearson": _pearsonr(x_values, y_values),
                    "spearman": _spearmanr(x_values, y_values),
                    "n": int(len(pairs)),
                }
            )
    return output_rows


def compute_quantile_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize error retained by top/bottom confidence quantiles."""
    output_rows: list[dict[str, Any]] = []
    rows = _evaluable_rows(rows)
    for signal in CONFIDENCE_SIGNALS:
        scored = _sorted_by_confidence(rows, signal)
        if not scored:
            continue
        for label, fraction, from_top in (
            ("top_10", 0.10, True),
            ("top_25", 0.25, True),
            ("top_50", 0.50, True),
            ("bottom_50", 0.50, False),
        ):
            selected = _select_fraction(scored, fraction=fraction, from_top=from_top)
            errors = [row["normalized_error"] for row in selected]
            output_rows.append(
                {
                    "confidence_signal": signal,
                    "quantile": label,
                    "retained_landmarks": len(selected),
                    "mean_nme": _safe_mean(errors),
                    "median_nme": _safe_median(errors),
                }
            )
    return output_rows


def compute_retention_curve_rows(
    rows: list[dict[str, Any]],
    failure_threshold: float,
    retention_fractions: list[float],
) -> list[dict[str, Any]]:
    """Build retention curves ordered by each confidence signal."""
    output_rows: list[dict[str, Any]] = []
    rows = _evaluable_rows(rows)
    scopes = (
        ["global"]
        + _ordered_regions(rows)
        + ["contour", "eyebrows", "eyes", "nose", "mouth"]
    )
    for signal in CONFIDENCE_SIGNALS:
        for region in scopes:
            scope_rows = (
                rows
                if region == "global"
                else [
                    row
                    for row in rows
                    if row["region"] == region or row["grouped_region"] == region
                ]
            )
            scored = _sorted_by_confidence(scope_rows, signal)
            if not scored:
                continue
            for fraction in sorted(set(float(value) for value in retention_fractions)):
                selected = _select_fraction(scored, fraction=fraction, from_top=True)
                errors = [row["normalized_error"] for row in selected]
                output_rows.append(
                    {
                        "confidence_signal": signal,
                        "region": region,
                        "retained_fraction": float(len(selected) / len(scored)),
                        "retained_landmarks": int(len(selected)),
                        "mean_nme": _safe_mean(errors),
                        "median_nme": _safe_median(errors),
                        "failure_rate": _failure_rate(errors, failure_threshold),
                    }
                )
    return output_rows


def compute_failure_detection_rows(
    rows: list[dict[str, Any]],
    failure_thresholds: list[float],
) -> tuple[list[dict[str, Any]], dict[tuple[str, float], list[tuple[float, float]]]]:
    """Evaluate whether confidence scores detect high-error failures."""
    output_rows: list[dict[str, Any]] = []
    roc_curves: dict[tuple[str, float], list[tuple[float, float]]] = {}
    rows = _evaluable_rows(rows)
    for signal in CONFIDENCE_SIGNALS:
        pairs = _valid_pairs(rows, signal, "normalized_error")
        if len(pairs) < 2:
            continue
        scores = np.asarray([_failure_score(signal, pair[0]) for pair in pairs])
        errors = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        for threshold in failure_thresholds:
            labels = errors > float(threshold)
            if labels.min() == labels.max():
                continue
            auroc, roc_points = _binary_auroc(labels, scores)
            auprc = _binary_auprc(labels, scores)
            roc_curves[(signal, float(threshold))] = roc_points
            for retained_fraction in (0.10, 0.25, 0.50):
                retained = _select_fraction(
                    _sorted_by_confidence(rows, signal), retained_fraction, True
                )
                selected_ids = {
                    (row["image_id"], row["landmark_index"]) for row in retained
                }
                predicted_good = np.asarray(
                    [
                        (row["image_id"], row["landmark_index"]) in selected_ids
                        for row in rows
                        if _is_finite_number(row.get(signal))
                    ],
                    dtype=bool,
                )
                actual_good = ~labels
                precision, recall = _precision_recall(actual_good, predicted_good)
                output_rows.append(
                    {
                        "confidence_signal": signal,
                        "failure_threshold": float(threshold),
                        "auroc": auroc,
                        "auprc": auprc,
                        "confidence_selection": f"top_{int(retained_fraction * 100)}",
                        "precision_good": precision,
                        "recall_good": recall,
                        "n": int(len(labels)),
                    }
                )
    return output_rows, roc_curves


def compute_region_viability_rows(
    retention_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create region-level pseudo-label viability recommendations."""
    output_rows: list[dict[str, Any]] = []
    grouped_regions = ["contour", "eyebrows", "eyes", "nose", "mouth"]
    for region in grouped_regions:
        candidates = [
            row
            for row in retention_rows
            if row["region"] == region
            and math.isclose(float(row["retained_fraction"]), 0.25, rel_tol=0.05)
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: float(row["mean_nme"]))
        mean_nme = float(best["mean_nme"])
        failure_rate = float(best["failure_rate"])
        if mean_nme <= 0.05 and failure_rate <= 0.10:
            recommendation = "use_early"
            suitable = True
        elif mean_nme <= 0.08 and failure_rate <= 0.25:
            recommendation = "use_with_strict_filtering"
            suitable = True
        elif mean_nme <= 0.10 and failure_rate <= 0.40:
            recommendation = "use_late"
            suitable = True
        else:
            recommendation = "exclude_or_delay"
            suitable = False
        output_rows.append(
            {
                "region": region,
                "best_signal_at_25pct": best["confidence_signal"],
                "retained_fraction": best["retained_fraction"],
                "retained_landmarks": best["retained_landmarks"],
                "mean_nme": best["mean_nme"],
                "median_nme": best["median_nme"],
                "failure_rate": best["failure_rate"],
                "suitable_for_pseudo_labeling": suitable,
                "recommendation": recommendation,
            }
        )
    return output_rows


def save_confidence_error_plots(
    rows: list[dict[str, Any]],
    summary_by_region: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    roc_curves: dict[tuple[str, float], list[tuple[float, float]]],
    figures_dir: Path,
) -> list[Path]:
    """Save diagnostic plots for confidence-error analysis."""
    plt = _import_pyplot()
    if plt is None:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    evaluable_rows = _evaluable_rows(rows)
    best_signal = _best_signal_from_rows(evaluable_rows) or "heatmap_max"
    signal_rows = [
        row for row in evaluable_rows if _is_finite_number(row.get(best_signal))
    ]
    if signal_rows:
        outputs.append(_plot_scatter(signal_rows, best_signal, figures_dir))
        outputs.append(_plot_quantile_boxplot(signal_rows, best_signal, figures_dir))
    outputs.append(_plot_region_bars(summary_by_region, figures_dir))
    outputs.append(_plot_retention_curves(retention_rows, figures_dir))
    if roc_curves:
        outputs.append(_plot_roc_curves(roc_curves, figures_dir))
    return [path for path in outputs if path is not None]


def save_example_visualizations(
    candidates: list[dict[str, Any]],
    examples_dir: Path,
    max_visual_examples: int,
) -> list[Path]:
    """Save qualitative examples for confidence/error quadrants."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    if max_visual_examples <= 0:
        return []

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["image_row"].get("mean_heatmap_max") is not None
        and candidate["image_row"].get("mean_nme") is not None
    ]
    if not valid_candidates:
        return []
    confidence_values = np.asarray(
        [candidate["image_row"]["mean_heatmap_max"] for candidate in valid_candidates],
        dtype=np.float64,
    )
    error_values = np.asarray(
        [candidate["image_row"]["mean_nme"] for candidate in valid_candidates],
        dtype=np.float64,
    )
    high_conf = float(np.quantile(confidence_values, 0.75))
    low_conf = float(np.quantile(confidence_values, 0.25))
    high_error = float(np.quantile(error_values, 0.75))
    low_error = float(np.quantile(error_values, 0.25))
    buckets = {
        "high_confidence_low_error": lambda c: c["image_row"]["mean_heatmap_max"]
        >= high_conf
        and c["image_row"]["mean_nme"] <= low_error,
        "high_confidence_high_error": lambda c: c["image_row"]["mean_heatmap_max"]
        >= high_conf
        and c["image_row"]["mean_nme"] >= high_error,
        "low_confidence_high_error": lambda c: c["image_row"]["mean_heatmap_max"]
        <= low_conf
        and c["image_row"]["mean_nme"] >= high_error,
        "low_confidence_low_error": lambda c: c["image_row"]["mean_heatmap_max"]
        <= low_conf
        and c["image_row"]["mean_nme"] <= low_error,
    }
    outputs: list[Path] = []
    per_bucket = max(1, max_visual_examples // len(buckets))
    for bucket_name, predicate in buckets.items():
        selected = [candidate for candidate in valid_candidates if predicate(candidate)]
        selected = sorted(selected, key=lambda c: c["image_row"]["mean_nme"])
        if "high_error" in bucket_name:
            selected = list(reversed(selected))
        for candidate in selected[:per_bucket]:
            sample = candidate["sample"]
            output_path = examples_dir / f"{bucket_name}_{sample['image_id']}.jpg"
            try:
                from ..utils.visualization import save_landmark_comparison_overlay_image

                save_landmark_comparison_overlay_image(
                    image_path=Path(sample["overlay_image_path"]),
                    output_path=output_path,
                    predicted_landmarks=sample["predicted_landmarks"],
                    predicted_visibility=candidate["predicted_visibility"],
                    target_landmarks=sample["target_landmarks"],
                    target_visibility=(
                        candidate["target_visibility"]
                        if candidate["target_visibility"] is not None
                        else np.ones(
                            sample["target_landmarks"].shape[0], dtype=np.int64
                        )
                    ),
                    show_indices=False,
                    point_radius=5,
                    line_width=2,
                )
                outputs.append(output_path)
            except Exception:
                continue
    return outputs


def write_markdown_report(
    output_path: Path,
    dataset_description: str,
    checkpoint_path: str | Path,
    per_image_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    summary_by_region: list[dict[str, Any]],
    summary_by_pose: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    viability_rows: list[dict[str, Any]],
    sanity_checks: dict[str, Any],
    plot_outputs: list[Path],
    example_outputs: list[Path],
    pose_available: bool,
    visibility_available: bool,
    pca_available: bool,
    tta_samples: int,
) -> Path:
    """Write a concise Markdown report for the offline diagnostic analysis."""
    global_nme = _safe_mean([row["mean_nme"] for row in per_image_rows])
    global_median = _safe_median([row["median_nme"] for row in per_image_rows])
    global_correlations = [row for row in correlations if row["scope"] == "global"]
    ranked = sorted(
        global_correlations,
        key=lambda row: abs(float(row.get("spearman", 0.0))),
        reverse=True,
    )
    best_lines = [
        f"- `{row['confidence_signal']}`: Spearman {float(row['spearman']):.4f}, Pearson {float(row['pearson']):.4f}"
        for row in ranked[:5]
    ]
    region_lines = [
        f"- {row['region']}: mean NME {float(row['mean_nme']):.4f}, median NME {float(row['median_nme']):.4f}"
        for row in summary_by_region
        if row["region"] in ["contour", "eyebrows", "eyes", "nose", "mouth"]
        and _is_finite_number(row.get("mean_nme"))
    ]
    viability_lines = [
        f"- {row['region']}: {row['recommendation']} using `{row['best_signal_at_25pct']}` at {float(row['retained_fraction']):.2f} retained, mean NME {float(row['mean_nme']) * 100.0:.2f}%, failure rate {float(row['failure_rate']) * 100.0:.1f}%"
        for row in viability_rows
    ]
    plot_lines = figure_markdown_lines(plot_outputs)
    example_lines = [f"- `{path.name}`" for path in example_outputs[:10]]
    pose_lines = [
        f"- {row['pose']}: mean image NME {_format_optional_float(row.get('mean_nme_percent'), 2)}%, images={int(row.get('image_count', 0))}"
        for row in summary_by_pose
        if _has_value(row.get("pose"))
    ]
    metric_warning = sanity_checks.get("official_metric_warning")
    text = "\n".join(
        [
            "# Confidence-Error Analysis Report",
            "",
            "## Scope",
            "",
            f"- Dataset: {dataset_description}",
            f"- Checkpoint: `{checkpoint_path}`",
            f"- Images analyzed: {len(per_image_rows)}",
            f"- Mean image NME fraction: {_format_optional_float(global_nme, 4)}",
            f"- Mean image NME percent: {_format_optional_float(_scale_fraction_to_percent(global_nme), 2)}",
            f"- Median image NME fraction: {_format_optional_float(global_median, 4)}",
            f"- Median image NME percent: {_format_optional_float(_scale_fraction_to_percent(global_median), 2)}",
            f"- Total landmark rows: {sanity_checks['total_landmark_rows']}",
            f"- Valid GT landmark rows: {sanity_checks['valid_gt_landmark_rows']}",
            f"- Invalid GT landmark rows: {sanity_checks['invalid_gt_landmark_rows']}",
            f"- NaN target coordinate rows: {sanity_checks['nan_target_coordinate_rows']}",
            f"- TTA samples: {tta_samples}",
            f"- PCA shape plausibility: {'enabled' if pca_available else 'not computed'}",
            f"- Evaluation protocol: {sanity_checks['evaluation_protocol']}",
            f"- BabyLand reference NME: {sanity_checks.get('paper_reference_nme_percent', 10.41):.2f}% when checkpoint, data, decoder, crops, normalization, and postprocessing match the paper setup.",
            f"- Reference metric warning: {metric_warning if metric_warning else 'none'}",
            "",
            "This is an offline diagnostic analysis. BabyLand labels are used only to measure confidence-error behavior and must not be used for training, adaptation, or final model tuning without a proper validation split.",
            "",
            "## Confidence Signal Semantics",
            "",
            "- `heatmap_max`: higher means more confident.",
            "- `heatmap_variance`: lower means more confident.",
            "- `heatmap_entropy`: lower means more confident.",
            "- `peak_sharpness`: higher should mean more confident, but it may be weak.",
            "- `tta_variance`: lower means more stable and more confident.",
            "- `pca_reconstruction_error`: image-level shape plausibility, not landmark-level confidence.",
            "",
            "Spearman correlation is the main ranking diagnostic for pseudo-label selection because candidate selection is order-based. Pearson correlation is useful for assessing whether a signal has a roughly linear relationship with error magnitude.",
            "",
            "## Best Confidence Signals",
            "",
            *(
                best_lines
                or ["- No valid confidence-error correlations were available."]
            ),
            "",
            "## Pose Reliability",
            "",
            *(pose_lines or ["- Pose grouping was unavailable in metadata."]),
            "",
            "## Region Reliability",
            "",
            *(region_lines or ["- No valid region-level rows were available."]),
            "",
            "## Pseudo-Label Viability",
            "",
            *(
                viability_lines
                or ["- No region met the retention-summary requirements."]
            ),
            "",
            "Do not pseudo-label all 72 landmarks initially. Start with internal regions, prioritize the safest retained subsets, and delay or exclude contour at the beginning.",
            "",
            "Recommended UDA sequencing: first run a consistency-only baseline, then add conservative region-specific pseudo-labeling if the consistency-only result is stable. Broad all-landmark pseudo-labeling is not supported by this diagnostic.",
            "",
            "## Grouping Availability",
            "",
            f"- Pose grouping: {'available' if pose_available else 'not available in metadata; skipped'}",
            f"- Visibility grouping: {'available' if visibility_available else 'not available; skipped'}",
            "",
            "## Outputs",
            "",
            "- `per_landmark_confidence_error.csv`",
            "- `per_image_confidence_error.csv`",
            "- `summary_by_region.csv`",
            "- `summary_by_pose.csv`",
            "- `retention_curves.csv`",
            "- `failure_detection.csv`",
            "- `confidence_error_correlations.csv`",
            "",
            "## Plots",
            "",
            *(
                plot_lines
                or ["- No figures were generated."]
            ),
            "",
            "## Examples",
            "",
            *(example_lines or ["- No qualitative examples were saved."]),
            "",
            "## Caveats",
            "",
            "- Error-based summaries exclude invalid GT landmarks; invalid rows remain in the per-landmark CSV with NaN errors for diagnostics.",
            "- Predicted visibility is retained as an analysis variable only and does not define the official GT error mask.",
            "- Heatmap entropy and spatial variance are computed from a spatial softmax over predicted heatmaps.",
            "- Peak sharpness is the gap between the two strongest local peaks, with a flat top-2 fallback.",
            "- TTA consistency avoids horizontal flips unless a future analysis wires in verified left-right remapping.",
            "- Region recommendations are diagnostic heuristics, not an adaptation policy.",
        ]
    )
    output_path.write_text(text + "\n", encoding="utf-8")
    return output_path


def _prepare_sample_coordinates(
    eval_mode: str,
    metadata_batch: dict[str, Any],
    sample_index: int,
    predicted_input: torch.Tensor,
    target_landmarks: torch.Tensor,
    tta_variance_input: torch.Tensor | None,
    tta_predictions_input: torch.Tensor | None,
) -> dict[str, Any]:
    sample_id = str(metadata_batch["sample_id"][sample_index])
    if eval_mode == "natural":
        network_input_size = extract_batched_size(
            metadata_batch["transformed_size"], sample_index
        )
        crop_size = extract_batched_size(metadata_batch["crop_size"], sample_index)
        transform_crop_to_orig = metadata_batch["transform_crop_to_orig"][sample_index]
        predicted_crop = project_landmarks_between_sizes(
            landmarks=predicted_input,
            source_size=network_input_size,
            target_size=crop_size,
        )
        predicted_output = apply_homogeneous_transform(
            predicted_crop,
            transform_crop_to_orig,
        ).astype(np.float32)
        tta_variance_output = None
        if tta_predictions_input is not None:
            projected_tta = []
            for prediction in tta_predictions_input:
                prediction_crop = project_landmarks_between_sizes(
                    landmarks=prediction,
                    source_size=network_input_size,
                    target_size=crop_size,
                )
                projected_tta.append(
                    apply_homogeneous_transform(
                        prediction_crop,
                        transform_crop_to_orig,
                    ).astype(np.float32)
                )
            tta_variance_output = (
                np.stack(projected_tta, axis=0).var(axis=0).sum(axis=1)
            )
        elif tta_variance_input is not None:
            scale = max(
                crop_size[0] / network_input_size[0],
                crop_size[1] / network_input_size[1],
            )
            tta_variance_output = _tensor_to_numpy(
                tta_variance_input,
                dtype=np.float32,
            ) * float(scale * scale)
        return {
            "image_id": str(metadata_batch["source_image_name"][sample_index]),
            "prediction_id": sample_id,
            "image_path": str(metadata_batch["source_image_path"][sample_index]),
            "overlay_image_path": str(
                metadata_batch["source_image_path"][sample_index]
            ),
            "predicted_landmarks": predicted_output,
            "target_landmarks": _tensor_to_numpy(target_landmarks, dtype=np.float32),
            "tta_variance": tta_variance_output,
            "pose": str(metadata_batch.get("orientation", [""])[sample_index]),
        }

    transformed_size = extract_batched_size(
        metadata_batch["transformed_size"], sample_index
    )
    original_size = extract_batched_size(metadata_batch["original_size"], sample_index)
    predicted_output = project_landmarks_to_original_size(
        landmarks=predicted_input,
        transformed_size=transformed_size,
        original_size=original_size,
    )
    predicted_output = _tensor_to_numpy(predicted_output, dtype=np.float32)
    tta_variance_output = None
    if tta_predictions_input is not None:
        projected_tta = [
            _tensor_to_numpy(
                project_landmarks_to_original_size(
                    landmarks=prediction,
                    transformed_size=transformed_size,
                    original_size=original_size,
                ),
                dtype=np.float32,
            )
            for prediction in tta_predictions_input
        ]
        tta_variance_output = np.stack(projected_tta, axis=0).var(axis=0).sum(axis=1)
    elif tta_variance_input is not None:
        scale = max(
            original_size[0] / transformed_size[0],
            original_size[1] / transformed_size[1],
        )
        tta_variance_output = _tensor_to_numpy(
            tta_variance_input,
            dtype=np.float32,
        ) * float(scale * scale)
    pose = None
    if "yaw_group" in metadata_batch:
        pose = str(metadata_batch["yaw_group"][sample_index])
    elif "yaw_angle" in metadata_batch:
        pose = str(metadata_batch["yaw_angle"][sample_index])
    return {
        "image_id": sample_id,
        "prediction_id": sample_id,
        "image_path": str(metadata_batch["image_path"][sample_index]),
        "overlay_image_path": str(metadata_batch["image_path"][sample_index]),
        "predicted_landmarks": predicted_output,
        "target_landmarks": _tensor_to_numpy(target_landmarks, dtype=np.float32),
        "tta_variance": tta_variance_output,
        "pose": pose,
    }


def _build_sample_rows(
    sample: dict[str, Any],
    target_visibility: np.ndarray | None,
    predicted_visibility: np.ndarray,
    heatmap_metrics: Any,
    sample_index: int,
    pca_error: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted = sample["predicted_landmarks"]
    target = sample["target_landmarks"]
    target_visibility_for_mask = (
        np.ones(target.shape[0], dtype=np.int64)
        if target_visibility is None
        else np.asarray(target_visibility, dtype=np.int64)
    )
    finite_target_mask = np.isfinite(target[:, 0]) & np.isfinite(target[:, 1])
    finite_prediction_mask = np.isfinite(predicted[:, 0]) & np.isfinite(predicted[:, 1])
    valid_gt_mask = compute_natural_valid_landmark_mask(
        landmarks=target,
        visibility=target_visibility_for_mask,
    )
    evaluable_mask = valid_gt_mask & finite_prediction_mask
    if valid_gt_mask.any():
        normalization = compute_box_normalization_factor(target[valid_gt_mask])
    else:
        normalization = float("nan")
    rows: list[dict[str, Any]] = []
    for landmark_index in range(predicted.shape[0]):
        region = _get_landmark_anatomical_group(landmark_index)
        is_valid_gt = bool(valid_gt_mask[landmark_index])
        is_evaluable = bool(evaluable_mask[landmark_index])
        pixel_error = (
            float(np.linalg.norm(predicted[landmark_index] - target[landmark_index]))
            if is_evaluable
            else float("nan")
        )
        normalized_error = (
            float(pixel_error / normalization)
            if is_evaluable and math.isfinite(normalization)
            else float("nan")
        )
        row = {
            "image_id": sample["image_id"],
            "image_path": sample["image_path"],
            "prediction_id": sample["prediction_id"],
            "landmark_index": int(landmark_index),
            "region": region,
            "grouped_region": GROUPED_REGION_MAP.get(region, "other"),
            "predicted_x": float(predicted[landmark_index, 0]),
            "predicted_y": float(predicted[landmark_index, 1]),
            "target_x": float(target[landmark_index, 0]),
            "target_y": float(target[landmark_index, 1]),
            "target_is_finite": bool(finite_target_mask[landmark_index]),
            "prediction_is_finite": bool(finite_prediction_mask[landmark_index]),
            "gt_valid_for_error": is_valid_gt,
            "evaluable_for_error": is_evaluable,
            "pixel_error": pixel_error,
            "normalized_error": normalized_error,
            "normalized_error_percent": _scale_fraction_to_percent(normalized_error),
            "heatmap_max": float(
                heatmap_metrics.heatmap_max[sample_index, landmark_index].detach().cpu()
            ),
            "heatmap_entropy": float(
                heatmap_metrics.heatmap_entropy[sample_index, landmark_index]
                .detach()
                .cpu()
            ),
            "heatmap_variance": float(
                heatmap_metrics.heatmap_variance[sample_index, landmark_index]
                .detach()
                .cpu()
            ),
            "peak_sharpness": float(
                heatmap_metrics.peak_sharpness[sample_index, landmark_index]
                .detach()
                .cpu()
            ),
            "tta_variance": (
                None
                if sample["tta_variance"] is None
                else float(sample["tta_variance"][landmark_index])
            ),
            "pca_reconstruction_error": pca_error,
            "visibility": (
                None
                if target_visibility is None
                else int(target_visibility[landmark_index])
            ),
            "predicted_visibility": int(predicted_visibility[landmark_index]),
            "pose": sample.get("pose"),
        }
        rows.append(row)
    evaluable_rows = _evaluable_rows(rows)
    image_errors = [row["normalized_error"] for row in evaluable_rows]
    valid_gt_count = int(valid_gt_mask.sum())
    invalid_gt_count = int(len(valid_gt_mask) - valid_gt_count)
    nan_target_count = int((~finite_target_mask).sum())
    image_row = {
        "image_id": sample["image_id"],
        "image_path": sample["image_path"],
        "mean_nme": _safe_mean(image_errors),
        "mean_nme_percent": _scale_fraction_to_percent(_safe_mean(image_errors)),
        "median_nme": _safe_median(image_errors),
        "median_nme_percent": _scale_fraction_to_percent(_safe_median(image_errors)),
        "max_nme": _safe_max(image_errors),
        "max_nme_percent": _scale_fraction_to_percent(_safe_max(image_errors)),
        "mean_heatmap_max": _safe_mean([row["heatmap_max"] for row in evaluable_rows]),
        "mean_heatmap_entropy": _safe_mean(
            [row["heatmap_entropy"] for row in evaluable_rows]
        ),
        "mean_tta_variance": _safe_mean(
            [row["tta_variance"] for row in evaluable_rows]
        ),
        "mean_pca_reconstruction_error": pca_error,
        "pose": sample.get("pose"),
        "total_landmarks": int(predicted.shape[0]),
        "number_of_valid_landmarks": valid_gt_count,
        "number_of_evaluable_landmarks": len(evaluable_rows),
        "number_of_invalid_landmarks": invalid_gt_count,
        "number_of_nan_target_landmarks": nan_target_count,
        "box_normalization_factor": normalization,
    }
    return rows, image_row


def _compute_sample_pca_errors(
    predicted_landmarks: torch.Tensor,
    pca_prior: dict[str, Any],
) -> list[float]:
    errors = []
    for sample_index in range(predicted_landmarks.shape[0]):
        try:
            loss = compute_pca_projection_loss(
                predicted_landmarks=predicted_landmarks[
                    sample_index : sample_index + 1
                ],
                pca_prior=pca_prior,
            )
            errors.append(float(loss.detach().cpu().item()))
        except Exception:
            errors.append(float("nan"))
    return errors


def _summarize_region(region: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable_rows = _evaluable_rows(rows)
    return {
        "region": region,
        "landmark_count": len(evaluable_rows),
        "total_rows": len(rows),
        "valid_gt_landmark_count": sum(
            1 for row in rows if bool(row.get("gt_valid_for_error"))
        ),
        "invalid_gt_landmark_count": sum(
            1 for row in rows if not bool(row.get("gt_valid_for_error"))
        ),
        "mean_nme": _safe_mean([row["normalized_error"] for row in evaluable_rows]),
        "mean_nme_percent": _scale_fraction_to_percent(
            _safe_mean([row["normalized_error"] for row in evaluable_rows])
        ),
        "median_nme": _safe_median([row["normalized_error"] for row in evaluable_rows]),
        "median_nme_percent": _scale_fraction_to_percent(
            _safe_median([row["normalized_error"] for row in evaluable_rows])
        ),
        "mean_heatmap_max": _safe_mean([row["heatmap_max"] for row in evaluable_rows]),
        "mean_heatmap_entropy": _safe_mean(
            [row["heatmap_entropy"] for row in evaluable_rows]
        ),
        "mean_tta_variance": _safe_mean(
            [row["tta_variance"] for row in evaluable_rows]
        ),
        "spearman_heatmap_max_vs_error": _correlation_for_rows(
            evaluable_rows, "heatmap_max", "spearman"
        ),
        "spearman_entropy_vs_error": _correlation_for_rows(
            evaluable_rows, "heatmap_entropy", "spearman"
        ),
        "spearman_tta_variance_vs_error": _correlation_for_rows(
            evaluable_rows, "tta_variance", "spearman"
        ),
    }


def _write_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sanity_checks(
    per_landmark_rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    total_landmarks_seen: int,
    valid_gt_landmarks_seen: int,
    invalid_gt_landmarks_seen: int,
    nan_target_coordinate_rows_seen: int,
) -> dict[str, Any]:
    """Build explicit protocol checks for valid-GT-only evaluation."""
    evaluable_rows = _evaluable_rows(per_landmark_rows)
    image_valid_counts = [
        int(row["number_of_valid_landmarks"]) for row in per_image_rows
    ]
    all_images_have_72_valid = bool(image_valid_counts) and all(
        count == 72 for count in image_valid_counts
    )
    mean_image_nme = _safe_mean([row["mean_nme"] for row in per_image_rows])
    return {
        "evaluation_protocol": (
            "valid_gt_mask = visibility == 1 AND finite(target_x, target_y); "
            "predicted_visibility is not used to define the error mask"
        ),
        "normalization": (
            "box normalization from valid GT landmarks for each image, matching "
            "compute_box_normalization_factor"
        ),
        "nme_scale": "fraction; percent fields multiply by 100",
        "aggregation": (
            "per-image NME is the mean over valid GT landmarks; mean_nme_fraction "
            "is the mean of per-image NME values"
        ),
        "total_landmark_rows": int(total_landmarks_seen),
        "valid_gt_landmark_rows": int(valid_gt_landmarks_seen),
        "invalid_gt_landmark_rows": int(invalid_gt_landmarks_seen),
        "evaluable_landmark_rows": int(len(evaluable_rows)),
        "nan_target_coordinate_rows": int(nan_target_coordinate_rows_seen),
        "invalid_gt_rows_have_nan_error": all(
            not _is_finite_number(row.get("normalized_error"))
            for row in per_landmark_rows
            if not bool(row.get("gt_valid_for_error"))
        ),
        "valid_landmark_count_min": min(image_valid_counts)
        if image_valid_counts
        else None,
        "valid_landmark_count_max": max(image_valid_counts)
        if image_valid_counts
        else None,
        "valid_landmark_count_mean": _safe_mean(image_valid_counts),
        "all_images_have_72_valid_landmarks": all_images_have_72_valid,
        "mean_nme_fraction": mean_image_nme,
        "mean_nme_percent": _scale_fraction_to_percent(mean_image_nme),
    }


def build_official_metric_warning(
    eval_mode: str,
    mean_nme_percent: Any,
    reference_percent: float = 10.41,
    tolerance_percent: float = 0.50,
) -> dict[str, Any]:
    """Build a non-blocking warning when BabyLand natural NME differs materially."""
    warning: str | None = None
    if eval_mode == "natural" and _is_finite_number(mean_nme_percent):
        observed = float(mean_nme_percent)
        if abs(observed - reference_percent) > tolerance_percent:
            warning = (
                f"Observed natural BabyLand mean NME is {observed:.2f}%, which is "
                f"outside +/-{tolerance_percent:.2f}% of the reference "
                f"{reference_percent:.2f}%. Check checkpoint, crops, decoder, "
                "normalization, and postprocessing before drawing conclusions."
            )
    return {
        "paper_reference_nme_percent": reference_percent,
        "paper_reference_tolerance_percent": tolerance_percent,
        "official_metric_warning": warning,
    }


def _ordered_regions(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["region"] for row in rows}
    return [region for region in FINE_REGION_NAMES if region in present]


def _valid_pairs(
    rows: list[dict[str, Any]], x_key: str, y_key: str
) -> list[tuple[float, float]]:
    return [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if _is_finite_number(row.get(x_key)) and _is_finite_number(row.get(y_key))
    ]


def _evaluable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that participate in official error-based summaries."""
    return [
        row
        for row in rows
        if bool(row.get("evaluable_for_error"))
        and _is_finite_number(row.get("normalized_error"))
    ]


def _correlation_for_rows(
    rows: list[dict[str, Any]], signal: str, kind: str
) -> float | None:
    pairs = _valid_pairs(rows, signal, "normalized_error")
    if len(pairs) < 2:
        return None
    x_values = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    return (
        _spearmanr(x_values, y_values)
        if kind == "spearman"
        else _pearsonr(x_values, y_values)
    )


def _pearsonr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    if x_values.size < 2 or np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _spearmanr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    return _pearsonr(_rankdata(x_values), _rankdata(y_values))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _sorted_by_confidence(
    rows: list[dict[str, Any]], signal: str
) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if _is_finite_number(row.get(signal))]
    reverse = signal not in ERROR_HIGH_SIGNALS
    return sorted(valid_rows, key=lambda row: float(row[signal]), reverse=reverse)


def _select_fraction(
    rows: list[dict[str, Any]], fraction: float, from_top: bool
) -> list[dict[str, Any]]:
    if not rows:
        return []
    count = max(1, int(math.ceil(len(rows) * min(max(float(fraction), 0.0), 1.0))))
    return rows[:count] if from_top else rows[-count:]


def _failure_score(signal: str, confidence_value: float) -> float:
    return (
        float(confidence_value)
        if signal in ERROR_HIGH_SIGNALS
        else -float(confidence_value)
    )


def _binary_auroc(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, list[tuple[float, float]]]:
    order = np.argsort(scores)[::-1]
    labels = labels[order].astype(bool)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    tp = 0
    fp = 0
    points = [(0.0, 0.0)]
    for label in labels:
        tp += int(label)
        fp += int(not label)
        points.append((fp / max(negatives, 1), tp / max(positives, 1)))
    points.append((1.0, 1.0))
    points_np = np.asarray(points, dtype=np.float64)
    auroc = _trapezoid_area(points_np[:, 1], points_np[:, 0])
    return auroc, points


def _binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    labels = labels[order].astype(bool)
    positives = max(int(labels.sum()), 1)
    tp = 0
    fp = 0
    precision = [1.0]
    recall = [0.0]
    for label in labels:
        tp += int(label)
        fp += int(not label)
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / positives)
    return _trapezoid_area(np.asarray(precision), np.asarray(recall))


def _precision_recall(
    actual_good: np.ndarray, predicted_good: np.ndarray
) -> tuple[float, float]:
    tp = float((actual_good & predicted_good).sum())
    fp = float((~actual_good & predicted_good).sum())
    fn = float((actual_good & ~predicted_good).sum())
    precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    return float(precision), float(recall)


def _trapezoid_area(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """Compute trapezoidal area with NumPy-version-compatible fallbacks."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    if hasattr(np, "trapz"):
        return float(np.trapz(y_values, x_values))
    if len(y_values) < 2:
        return 0.0
    widths = x_values[1:] - x_values[:-1]
    heights = (y_values[1:] + y_values[:-1]) * 0.5
    return float((widths * heights).sum())


def _failure_rate(errors: list[Any], threshold: float) -> float | None:
    values = [float(value) for value in errors if _is_finite_number(value)]
    if not values:
        return None
    return float(np.mean(np.asarray(values) > float(threshold)))


def _best_signal_from_rows(rows: list[dict[str, Any]]) -> str | None:
    correlations = compute_correlations(rows)
    global_rows = [row for row in correlations if row["scope"] == "global"]
    if not global_rows:
        return None
    return max(global_rows, key=lambda row: abs(float(row["spearman"])))[
        "confidence_signal"
    ]


def _plot_scatter(rows: list[dict[str, Any]], signal: str, figures_dir: Path) -> Path:
    plt = _import_pyplot()
    sample_rows = rows
    if len(sample_rows) > 10000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(sample_rows), size=10000, replace=False)
        sample_rows = [sample_rows[int(index)] for index in indices]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        [row[signal] for row in sample_rows],
        [row["normalized_error"] for row in sample_rows],
        s=8,
        alpha=0.25,
    )
    axis.set_xlabel(signal)
    axis.set_ylabel("normalized error")
    axis.set_title(f"{signal} vs normalized error")
    output_path = figures_dir / f"scatter_{signal}_vs_normalized_error.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_quantile_boxplot(
    rows: list[dict[str, Any]], signal: str, figures_dir: Path
) -> Path:
    plt = _import_pyplot()
    scored = _sorted_by_confidence(rows, signal)
    groups = [
        ("top 10%", _select_fraction(scored, 0.10, True)),
        ("top 25%", _select_fraction(scored, 0.25, True)),
        ("top 50%", _select_fraction(scored, 0.50, True)),
        ("bottom 50%", _select_fraction(scored, 0.50, False)),
    ]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot(
        [[row["normalized_error"] for row in group] for _, group in groups],
        labels=[label for label, _ in groups],
        showfliers=False,
    )
    axis.set_ylabel("normalized error")
    axis.set_title(f"NME by {signal} quantile")
    output_path = figures_dir / f"boxplot_nme_by_{signal}_quantile.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_region_bars(
    summary_by_region: list[dict[str, Any]], figures_dir: Path
) -> Path:
    plt = _import_pyplot()
    rows = [
        row
        for row in summary_by_region
        if row["region"] in ["contour", "eyebrows", "eyes", "nose", "mouth"]
    ]
    labels = [row["region"] for row in rows]
    x_values = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x_values, [row["mean_nme"] for row in rows], color="#4C78A8")
    axes[0].set_xticks(x_values, labels, rotation=25, ha="right")
    axes[0].set_ylabel("mean NME")
    axes[0].set_title("Mean NME by region")
    axes[1].bar(x_values, [row["mean_heatmap_max"] for row in rows], color="#F58518")
    axes[1].set_xticks(x_values, labels, rotation=25, ha="right")
    axes[1].set_ylabel("mean heatmap max")
    axes[1].set_title("Mean confidence by region")
    output_path = figures_dir / "region_mean_nme_and_confidence.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_retention_curves(
    retention_rows: list[dict[str, Any]], figures_dir: Path
) -> Path:
    plt = _import_pyplot()
    rows = [row for row in retention_rows if row["region"] == "global"]
    fig, axis = plt.subplots(figsize=(7, 5))
    for signal in sorted({row["confidence_signal"] for row in rows}):
        signal_rows = sorted(
            [row for row in rows if row["confidence_signal"] == signal],
            key=lambda row: float(row["retained_fraction"]),
        )
        axis.plot(
            [row["retained_fraction"] for row in signal_rows],
            [row["mean_nme"] for row in signal_rows],
            marker="o",
            label=signal,
        )
    axis.set_xlabel("retained fraction")
    axis.set_ylabel("mean NME")
    axis.set_title("Retention curve ordered by confidence")
    axis.legend(fontsize=8)
    output_path = figures_dir / "retention_curve_global.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_roc_curves(
    roc_curves: dict[tuple[str, float], list[tuple[float, float]]],
    figures_dir: Path,
) -> Path:
    plt = _import_pyplot()
    fig, axis = plt.subplots(figsize=(7, 5))
    for (signal, threshold), points in sorted(roc_curves.items()):
        if not math.isclose(threshold, 0.05):
            continue
        points_np = np.asarray(points, dtype=np.float64)
        axis.plot(points_np[:, 0], points_np[:, 1], label=f"{signal}, >{threshold}")
    axis.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title("Failure-detection ROC")
    axis.legend(fontsize=8)
    output_path = figures_dir / "failure_detection_roc.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _safe_mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    return float(np.mean(finite)) if finite else None


def _safe_median(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    return float(np.median(finite)) if finite else None


def _safe_max(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    return float(np.max(finite)) if finite else None


def _scale_fraction_to_percent(value: Any) -> float | None:
    """Convert a finite NME fraction to percent while preserving missing values."""
    return float(value) * 100.0 if _is_finite_number(value) else None


def _format_optional_float(value: Any, decimals: int = 4) -> str:
    """Format optional float values for reports."""
    return f"{float(value):.{decimals}f}" if _is_finite_number(value) else "n/a"


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _tensor_to_numpy(
    value: torch.Tensor | np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Convert a tensor or array to NumPy, falling back when torch lacks NumPy."""
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        try:
            return detached.numpy().astype(dtype, copy=False)
        except RuntimeError:
            return np.asarray(detached.tolist(), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if _is_finite_number(value):
        return True
    return isinstance(value, str) and value.lower() != "nan"


def _get_landmark_anatomical_group(landmark_index: int) -> str:
    if 0 <= landmark_index <= 16:
        return "face_contour"
    if 17 <= landmark_index <= 21:
        return "right_eyebrow"
    if 22 <= landmark_index <= 26:
        return "left_eyebrow"
    if 27 <= landmark_index <= 30:
        return "nose_bridge"
    if 31 <= landmark_index <= 35:
        return "nose_base"
    if 36 <= landmark_index <= 41:
        return "right_eye"
    if 42 <= landmark_index <= 47:
        return "left_eye"
    if 48 <= landmark_index <= 59:
        return "outer_lip"
    if 60 <= landmark_index <= 67:
        return "inner_lip"
    if landmark_index == 68:
        return "under_lip"
    if landmark_index == 69:
        return "upper_chin"
    if landmark_index == 70:
        return "left_chin"
    if landmark_index == 71:
        return "right_chin"
    return "unknown"


def _import_pyplot() -> Any | None:
    try:
        import contextlib
        import io
        import os
        import tempfile

        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "landmarks_detection_matplotlib"),
        )
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as pyplot

        return pyplot
    except Exception:
        return None
