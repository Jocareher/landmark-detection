from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .evaluation_reporting import (
    REPORT_CATEGORIES,
    collect_official_metric_rows,
    format_report_value,
    write_official_metric_exports,
)
from .metrics import decode_heatmaps_to_image_coords
from ..models import NormalizedLandmarker


def compute_residual_total_variation(residual: torch.Tensor) -> torch.Tensor:
    """Return mean anisotropic total variation for a batched residual tensor."""
    vertical = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().mean()
    horizontal = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def compute_image_regularization(
    input_images: torch.Tensor,
    normalized_images: torch.Tensor,
    lambda_l1: float,
    lambda_tv: float,
) -> dict[str, torch.Tensor]:
    """Compute optional L1 and residual-TV regularization components."""
    residual = normalized_images - input_images
    l1 = residual.abs().mean()
    tv = compute_residual_total_variation(residual)
    total = float(lambda_l1) * l1 + float(lambda_tv) * tv
    return {
        "image_regularization_loss": total,
        "image_l1_loss": l1,
        "image_tv_loss": tv,
    }


def run_normalizer_diagnostics(
    model: NormalizedLandmarker,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    dataset_name: str,
    coordinate_decoder: str,
    softmax_temperature: float,
    visibility_threshold: float,
    mean: Sequence[float],
    std: Sequence[float],
    changed_pixel_thresholds: Sequence[float] = (1e-4, 1e-3, 1e-2),
    num_visual_examples: int = 32,
    save_visual_examples: bool = True,
) -> dict[str, Any]:
    """Measure image changes and prediction drift introduced by the normalizer."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    visual_root = output_dir / "visualizations" / dataset_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()

    aggregate = {
        "absolute_sum": 0.0,
        "squared_sum": 0.0,
        "element_count": 0,
        "residual_sum": 0.0,
        "residual_squared_sum": 0.0,
        "residual_min": float("inf"),
        "residual_max": float("-inf"),
        "tv_sum": 0.0,
        "sample_count": 0,
        "heatmap_absolute_sum": 0.0,
        "heatmap_squared_sum": 0.0,
        "heatmap_element_count": 0,
        "landmark_displacement_sum": 0.0,
        "landmark_displacement_count": 0,
        "visibility_logit_absolute_sum": 0.0,
        "visibility_logit_count": 0,
        "visibility_agreement_sum": 0.0,
    }
    changed_counts = {float(threshold): 0 for threshold in changed_pixel_thresholds}
    drift_rows: list[dict[str, Any]] = []
    visuals_saved = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            normalized_images = model.normalize_images(images)
            baseline_outputs = model.landmarker(images)
            normalized_outputs = model.forward_normalized(normalized_images)
            residual = normalized_images - images
            absolute_residual = residual.abs()

            aggregate["absolute_sum"] += float(absolute_residual.sum())
            aggregate["squared_sum"] += float(residual.square().sum())
            aggregate["element_count"] += residual.numel()
            aggregate["residual_sum"] += float(residual.sum())
            aggregate["residual_squared_sum"] += float(residual.square().sum())
            aggregate["residual_min"] = min(
                aggregate["residual_min"], float(residual.min())
            )
            aggregate["residual_max"] = max(
                aggregate["residual_max"], float(residual.max())
            )
            aggregate["tv_sum"] += (
                float(compute_residual_total_variation(residual)) * images.shape[0]
            )
            aggregate["sample_count"] += images.shape[0]
            for threshold in changed_counts:
                changed_counts[threshold] += int((absolute_residual > threshold).sum())

            baseline_heatmaps = baseline_outputs["heatmaps"]
            normalized_heatmaps = normalized_outputs["heatmaps"]
            heatmap_difference = normalized_heatmaps - baseline_heatmaps
            aggregate["heatmap_absolute_sum"] += float(heatmap_difference.abs().sum())
            aggregate["heatmap_squared_sum"] += float(heatmap_difference.square().sum())
            aggregate["heatmap_element_count"] += heatmap_difference.numel()

            baseline_landmarks = decode_heatmaps_to_image_coords(
                baseline_heatmaps,
                image_height=images.shape[2],
                image_width=images.shape[3],
                decoder=coordinate_decoder,
                softmax_temperature=softmax_temperature,
            )
            normalized_landmarks = decode_heatmaps_to_image_coords(
                normalized_heatmaps,
                image_height=images.shape[2],
                image_width=images.shape[3],
                decoder=coordinate_decoder,
                softmax_temperature=softmax_temperature,
            )
            displacement = torch.linalg.vector_norm(
                normalized_landmarks - baseline_landmarks, dim=-1
            )
            aggregate["landmark_displacement_sum"] += float(displacement.sum())
            aggregate["landmark_displacement_count"] += displacement.numel()

            baseline_logits = baseline_outputs["visibility_logits"]
            normalized_logits = normalized_outputs["visibility_logits"]
            aggregate["visibility_logit_absolute_sum"] += float(
                (normalized_logits - baseline_logits).abs().sum()
            )
            aggregate["visibility_logit_count"] += baseline_logits.numel()
            baseline_visibility = torch.sigmoid(baseline_logits) >= visibility_threshold
            normalized_visibility = (
                torch.sigmoid(normalized_logits) >= visibility_threshold
            )
            agreement = (
                (baseline_visibility == normalized_visibility).float().mean(dim=1)
            )
            aggregate["visibility_agreement_sum"] += float(agreement.sum())

            metadata = batch.get("metadata", {})
            for sample_index in range(images.shape[0]):
                sample_id = _extract_sample_id(
                    metadata, sample_index, batch_index, images.shape[0]
                )
                current_displacement = displacement[sample_index]
                drift_rows.append(
                    {
                        "dataset": dataset_name,
                        "sample_id": sample_id,
                        "mean_landmark_displacement_px": float(
                            current_displacement.mean()
                        ),
                        "max_landmark_displacement_px": float(
                            current_displacement.max()
                        ),
                        "normalized_landmark_displacement": float(
                            current_displacement.mean()
                            / max(images.shape[2], images.shape[3])
                        ),
                        "visibility_agreement": float(agreement[sample_index]),
                    }
                )
                if save_visual_examples and visuals_saved < num_visual_examples:
                    _save_visual_example(
                        input_image=images[sample_index].cpu(),
                        normalized_image=normalized_images[sample_index].cpu(),
                        output_root=visual_root,
                        sample_id=sample_id,
                        mean=mean,
                        std=std,
                    )
                    visuals_saved += 1

    element_count = max(int(aggregate["element_count"]), 1)
    residual_mean = aggregate["residual_sum"] / element_count
    residual_variance = max(
        aggregate["residual_squared_sum"] / element_count - residual_mean**2,
        0.0,
    )
    heatmap_count = max(int(aggregate["heatmap_element_count"]), 1)
    landmark_count = max(int(aggregate["landmark_displacement_count"]), 1)
    visibility_count = max(int(aggregate["visibility_logit_count"]), 1)
    sample_count = max(int(aggregate["sample_count"]), 1)
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "num_samples": int(aggregate["sample_count"]),
        "num_image_elements": int(aggregate["element_count"]),
        "mean_l1_difference": aggregate["absolute_sum"] / element_count,
        "mean_l2_difference": (aggregate["squared_sum"] / element_count) ** 0.5,
        "max_absolute_difference": max(
            abs(float(aggregate["residual_min"])),
            abs(float(aggregate["residual_max"])),
        ),
        "residual_mean": residual_mean,
        "residual_std": residual_variance**0.5,
        "residual_min": aggregate["residual_min"],
        "residual_max": aggregate["residual_max"],
        "residual_total_variation": aggregate["tv_sum"] / sample_count,
        "heatmap_mean_absolute_difference": aggregate["heatmap_absolute_sum"]
        / heatmap_count,
        "heatmap_mse": aggregate["heatmap_squared_sum"] / heatmap_count,
        "mean_landmark_displacement_px": aggregate["landmark_displacement_sum"]
        / landmark_count,
        "visibility_logit_mean_absolute_difference": aggregate[
            "visibility_logit_absolute_sum"
        ]
        / visibility_count,
        "visibility_prediction_agreement": aggregate["visibility_agreement_sum"]
        / sample_count,
        "visual_examples_saved": visuals_saved,
    }
    for threshold, count in changed_counts.items():
        summary[f"pixels_changed_above_{threshold:g}"] = count / element_count

    _write_dict_csv(metrics_dir / f"{dataset_name}_normalizer_diagnostics.csv", summary)
    _write_rows_csv(metrics_dir / f"{dataset_name}_prediction_drift.csv", drift_rows)
    (metrics_dir / f"{dataset_name}_normalizer_diagnostics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def write_combined_normalizer_diagnostics(
    output_dir: str | Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    """Save a cross-dataset table and combined prediction-drift table."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = list(summaries.values())
    if rows:
        total_samples = max(sum(int(row["num_samples"]) for row in rows), 1)
        total_elements = max(sum(int(row["num_image_elements"]) for row in rows), 1)
        global_row: dict[str, Any] = {
            "dataset": "global",
            "num_samples": total_samples,
            "num_image_elements": total_elements,
        }
        for key in rows[0]:
            if key in {"dataset", "num_samples", "num_image_elements"}:
                continue
            if key in {"max_absolute_difference", "residual_max"}:
                global_row[key] = max(float(row[key]) for row in rows)
            elif key == "residual_min":
                global_row[key] = min(float(row[key]) for row in rows)
            elif key == "visual_examples_saved":
                global_row[key] = sum(int(row[key]) for row in rows)
            elif key == "mean_l2_difference":
                global_row[key] = (
                    sum(
                        float(row[key]) ** 2 * int(row["num_image_elements"])
                        for row in rows
                    )
                    / total_elements
                ) ** 0.5
            elif key == "residual_std":
                global_mean = (
                    sum(
                        float(row["residual_mean"]) * int(row["num_image_elements"])
                        for row in rows
                    )
                    / total_elements
                )
                global_row[key] = (
                    sum(
                        (
                            float(row[key]) ** 2
                            + (float(row["residual_mean"]) - global_mean) ** 2
                        )
                        * int(row["num_image_elements"])
                        for row in rows
                    )
                    / total_elements
                ) ** 0.5
            elif key.startswith("pixels_changed_") or key in {
                "mean_l1_difference",
                "residual_mean",
            }:
                global_row[key] = (
                    sum(
                        float(row[key]) * int(row["num_image_elements"]) for row in rows
                    )
                    / total_elements
                )
            else:
                global_row[key] = (
                    sum(float(row[key]) * int(row["num_samples"]) for row in rows)
                    / total_samples
                )
        rows = [global_row, *rows]
    _write_rows_csv(metrics_dir / "normalizer_diagnostics.csv", rows)
    combined_payload = {row["dataset"]: row for row in rows}
    (metrics_dir / "normalizer_diagnostics.json").write_text(
        json.dumps(combined_payload, indent=2), encoding="utf-8"
    )

    drift_rows: list[dict[str, Any]] = []
    for dataset_name in summaries:
        drift_path = metrics_dir / f"{dataset_name}_prediction_drift.csv"
        if not drift_path.exists():
            continue
        with drift_path.open("r", newline="", encoding="utf-8") as file:
            drift_rows.extend(csv.DictReader(file))
    _write_rows_csv(metrics_dir / "prediction_drift.csv", drift_rows)


def save_modular_checkpoints(
    model: NormalizedLandmarker,
    output_dir: str | Path,
    base_checkpoint_path: str | Path,
    experiment_mode: str,
    resolved_config_path: str | Path,
    landmarker_updated: bool,
    normalizer_updated: bool,
    decoder_name: str,
    loss_pipeline_name: str,
    evaluation_protocol: str,
) -> dict[str, str]:
    """Save modular checkpoints and a manifest for one completed experiment."""
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    best_source = output_dir / "best_model.pth"
    last_source = output_dir / "last_model.pth"
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for source, name in (
        (best_source, "full_model_best.pth"),
        (last_source, "full_model_last.pth"),
    ):
        if source.exists():
            payload = torch.load(source, map_location="cpu", weights_only=False)
        else:
            payload = {
                "epoch": -1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": None,
                "metrics": {},
            }
        checkpoint_payloads[name] = payload
        target = checkpoints_dir / name
        torch.save(payload, target)
        saved[name] = str(target)

    if model.normalizer is not None:
        for full_name, name in (
            ("full_model_best.pth", "normalizer_best.pth"),
            ("full_model_last.pth", "normalizer_last.pth"),
        ):
            full_state = checkpoint_payloads[full_name]["model_state_dict"]
            normalizer_state = {
                key.removeprefix("normalizer."): value
                for key, value in full_state.items()
                if key.startswith("normalizer.")
            }
            target = checkpoints_dir / name
            torch.save(
                {
                    "experiment_mode": experiment_mode,
                    "normalizer_state_dict": normalizer_state
                    or model.normalizer.state_dict(),
                    "architecture": model.normalizer.architecture_config(),
                },
                target,
            )
            saved[name] = str(target)
    if landmarker_updated:
        for full_name, name in (
            ("full_model_best.pth", "landmarker_best.pth"),
            ("full_model_last.pth", "landmarker_last.pth"),
        ):
            full_state = checkpoint_payloads[full_name]["model_state_dict"]
            landmarker_state = {
                key.removeprefix("landmarker."): value
                for key, value in full_state.items()
                if key.startswith("landmarker.")
            }
            target = checkpoints_dir / name
            torch.save(
                {
                    "landmarker_state_dict": landmarker_state
                    or model.landmarker.state_dict()
                },
                target,
            )
            saved[name] = str(target)

    manifest = {
        "experiment_mode": experiment_mode,
        "base_landmarker_checkpoint": str(base_checkpoint_path),
        "landmarker_updated": bool(landmarker_updated),
        "normalizer_updated": bool(normalizer_updated),
        "checkpoint_paths": saved,
        "decoder_name": decoder_name,
        "loss_pipeline_name": loss_pipeline_name,
        "evaluation_protocol": evaluation_protocol,
        "git_commit": _git_commit_hash(),
        "resolved_config_path": str(resolved_config_path),
    }
    manifest_path = checkpoints_dir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    saved["checkpoint_manifest.json"] = str(manifest_path)
    return saved


def write_experiment_report(
    output_dir: str | Path,
    experiment_mode: str,
    objective: str,
    checkpoint_path: str | Path,
    parameter_counts: dict[str, dict[str, int]],
    evaluation_summaries: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
    checkpoint_paths: dict[str, str],
    normalizer_architecture: dict[str, Any] | None = None,
    training_protocol: str = "",
    warnings: Sequence[str] = (),
) -> Path:
    """Write a concise Markdown report for one normalizer experiment."""
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "report.md"
    official_metric_rows = collect_official_metric_rows(evaluation_summaries)
    write_official_metric_exports(reports_dir, official_metric_rows)
    lines = [
        f"# {experiment_mode}",
        "",
        f"**Objective:** {objective}",
        "",
        f"**Base checkpoint:** `{checkpoint_path}`",
        "",
        f"**Training protocol:** {training_protocol or 'Not provided'}",
        "",
        "## Normalizer architecture",
        "",
        "```json",
        json.dumps(normalizer_architecture or {}, indent=2),
        "```",
        "",
        "## Parameter counts",
        "",
        "| Module | Total | Trainable | Frozen |",
        "|---|---:|---:|---:|",
    ]
    for module_name, counts in parameter_counts.items():
        lines.append(
            f"| {module_name} | {counts['total']} | {counts['trainable']} | {counts['frozen']} |"
        )
    lines.extend(["", "## Official evaluation summaries", ""])
    for dataset_name, evaluation_summary in evaluation_summaries.items():
        dataset_rows = [
            row for row in official_metric_rows if row["dataset"] == dataset_name
        ]
        lines.extend([f"### {dataset_name}", ""])
        if not dataset_rows:
            lines.extend(["See the existing evaluation output directory.", ""])
            continue
        for category in REPORT_CATEGORIES:
            category_rows = [row for row in dataset_rows if row["category"] == category]
            if not category_rows:
                continue
            lines.extend([f"#### {category}", "", "| Metric | Value |", "|---|---:|"])
            lines.extend(
                f"| {row['metric']} | {format_report_value(row['value'])} |"
                for row in category_rows
            )
            lines.append("")
    lines.extend(
        [
            "Excel-ready exports:",
            "",
            "- `official_metrics_long.csv`: one row per dataset and metric.",
            "- `official_metrics_wide.csv`: one dataset per row.",
            "- `official_metrics_copy_paste.tsv`: tab-separated wide table for direct copy/paste.",
            "",
        ]
    )
    lines.extend(["", "## Normalizer and prediction-drift diagnostics", ""])
    for dataset_name, summary in diagnostics.items():
        lines.extend(
            [
                f"### {dataset_name}",
                "",
                f"- Mean L1 image difference: `{summary['mean_l1_difference']:.8f}`",
                f"- Maximum absolute image difference: `{summary['max_absolute_difference']:.8f}`",
                f"- Heatmap mean absolute difference: `{summary['heatmap_mean_absolute_difference']:.8f}`",
                f"- Mean decoded landmark displacement: `{summary['mean_landmark_displacement_px']:.6f}` px",
                f"- Visibility prediction agreement: `{summary['visibility_prediction_agreement']:.6f}`",
                "",
            ]
        )
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    if experiment_mode == "normalizer_sanity":
        lines.extend(
            [
                "## Sanity conclusion",
                "",
                (
                    "The identity wrapper preserved images and predictions within the configured tolerance; the pipeline is safe to advance to Experiment 2."
                    if not warnings
                    else "Unexpected identity drift was detected. Experiment 2 should not start until the warnings are resolved."
                ),
                "",
            ]
        )
    lines.extend(["## Checkpoints", ""])
    lines.extend(f"- `{name}`: `{path}`" for name, path in checkpoint_paths.items())
    lines.extend(
        [
            "",
            "## Visual examples",
            "",
            f"Inverse-normalized RGB examples are stored under `{output_dir / 'visualizations'}`.",
            "",
            "## Interpretation",
            "",
            "Official landmark metrics remain those produced by the existing Wasserstein/barycenter evaluation pipeline. Image-change and prediction-drift values above are diagnostics only.",
            "",
            "## Recommended next step",
            "",
            (
                "Proceed to frozen-landmarker normalizer training only if all identity drift is within tolerance."
                if experiment_mode == "normalizer_sanity"
                else "Compare official natural-domain gains against SynBaby preservation and inspect residual artifacts before increasing model flexibility."
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _save_visual_example(
    input_image: torch.Tensor,
    normalized_image: torch.Tensor,
    output_root: Path,
    sample_id: str,
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    """Save input, normalized, residual, and side-by-side RGB previews."""
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    input_rgb = _to_display_image(input_image, mean, std)
    normalized_rgb = _to_display_image(normalized_image, mean, std)
    residual = np.abs(normalized_rgb.astype(np.float32) - input_rgb.astype(np.float32))
    residual_max = float(residual.max())
    residual_rgb = (
        np.zeros_like(input_rgb)
        if residual_max <= 0
        else np.clip(residual / residual_max * 255.0, 0, 255).astype(np.uint8)
    )
    for subdirectory, array in (
        ("input", input_rgb),
        ("normalized", normalized_rgb),
        ("residual_abs", residual_rgb),
    ):
        directory = output_root / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(directory / f"{safe_id}.png")
    side_by_side = np.concatenate([input_rgb, normalized_rgb, residual_rgb], axis=1)
    side_directory = output_root / "side_by_side"
    side_directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(side_by_side).save(side_directory / f"{safe_id}.png")


def _to_display_image(
    image: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    """Convert a channel-normalized tensor into a displayable RGB array."""
    mean_tensor = torch.as_tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype).view(-1, 1, 1)
    denormalized = image * std_tensor + mean_tensor
    display_tensor = (
        denormalized.permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu()
    )
    # ``tolist`` keeps visualization available in environments where the
    # optional PyTorch-to-NumPy binary bridge is version-incompatible.
    return np.asarray(display_tensor.tolist(), dtype=np.uint8)


def _extract_sample_id(
    metadata: Any, sample_index: int, batch_index: int, batch_size: int
) -> str:
    """Resolve a stable sample identifier from collated metadata."""
    if isinstance(metadata, dict) and "sample_id" in metadata:
        values = metadata["sample_id"]
        if isinstance(values, (list, tuple)):
            return str(values[sample_index])
        try:
            return str(values[sample_index])
        except Exception:
            return str(values)
    return f"batch_{batch_index:05d}_sample_{sample_index:03d}_{batch_size}"


def _write_dict_csv(path: Path, row: dict[str, Any]) -> None:
    """Write one dictionary row to CSV."""
    _write_rows_csv(path, [row])


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionary rows to CSV when at least one row is available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit_hash() -> str:
    """Return the current short Git commit hash when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "N/A"
