from __future__ import annotations

import csv
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

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
    residual_display_scale: float = 0.02,
    residual_amplification: float = 25.0,
) -> dict[str, Any]:
    """Measure image changes and prediction drift introduced by the normalizer."""
    if residual_display_scale <= 0:
        raise ValueError("residual_display_scale must be positive.")
    if residual_amplification <= 0:
        raise ValueError("residual_amplification must be positive.")
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "diagnostic_tables"
    visual_root = output_dir / "image_comparisons" / dataset_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if save_visual_examples and num_visual_examples > 0:
        _write_visualization_readme(
            output_dir / "image_comparisons",
            residual_display_scale=residual_display_scale,
            residual_amplification=residual_amplification,
        )
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
                        residual_display_scale=residual_display_scale,
                        residual_amplification=residual_amplification,
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
    metrics_dir = output_dir / "diagnostic_tables"
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
    base_checkpoint_path: str | Path | None,
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
        payload["model_type"] = type(model).__name__
        if model.normalizer is not None:
            payload["normalizer_architecture"] = model.normalizer.architecture_config()
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
                    "model_type": type(model.normalizer).__name__,
                    "experiment_mode": experiment_mode,
                    "normalizer_state_dict": normalizer_state
                    or model.normalizer.state_dict(),
                    "architecture": model.normalizer.architecture_config(),
                },
                target,
            )
            saved[name] = str(target)
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
        landmarker_state = landmarker_state or model.landmarker.state_dict()
        target = checkpoints_dir / name
        torch.save(
            {
                "model_type": type(model.landmarker).__name__,
                "model_state_dict": landmarker_state,
                "landmarker_state_dict": landmarker_state,
                "source_full_checkpoint": str(checkpoints_dir / full_name),
            },
            target,
        )
        saved[name] = str(target)

    manifest = {
        "experiment_mode": experiment_mode,
        "base_landmarker_checkpoint": (
            str(base_checkpoint_path) if base_checkpoint_path is not None else None
        ),
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
    checkpoint_path: str | Path | None,
    parameter_counts: dict[str, dict[str, int]],
    diagnostics: dict[str, dict[str, Any]],
    checkpoint_paths: dict[str, str],
    normalizer_architecture: dict[str, Any] | None = None,
    training_protocol: str = "",
    warnings: Sequence[str] = (),
) -> Path:
    """Write a concise Markdown report for one normalizer experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "experiment_report.md"
    lines = [
        f"# {experiment_mode}",
        "",
        f"**Objective:** {objective}",
        "",
        f"**Base checkpoint:** `{checkpoint_path if checkpoint_path is not None else 'none'}`",
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
    lines.extend(
        [
            "## Official evaluation results",
            "",
            "Dataset metrics and spreadsheet-ready consolidated reports are stored only under `evaluation/` and `evaluation/reports/`.",
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
            f"Inverse-normalized RGB examples are stored under `{output_dir / 'image_comparisons'}`.",
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
    residual_display_scale: float = 0.02,
    residual_amplification: float = 25.0,
) -> None:
    """Save input, normalized, and consistently scaled residual previews.

    The legacy ``residual_abs`` artifact is auto-scaled independently for each
    image and is retained for backward compatibility. The fixed-scale and
    signed artifacts use the same scale for every image, so their intensity is
    directly comparable across samples and datasets.
    """
    if residual_display_scale <= 0:
        raise ValueError("residual_display_scale must be positive.")
    if residual_amplification <= 0:
        raise ValueError("residual_amplification must be positive.")
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    input_float = _to_display_float(input_image, mean, std)
    normalized_float = _to_display_float(normalized_image, mean, std)
    signed_residual = normalized_float - input_float
    absolute_residual = np.abs(signed_residual)
    residual_max = float(absolute_residual.max())

    input_rgb = _float_rgb_to_uint8(input_float)
    normalized_rgb = _float_rgb_to_uint8(normalized_float)
    residual_abs_auto = _float_rgb_to_uint8(
        np.zeros_like(absolute_residual)
        if residual_max <= 0
        else absolute_residual / residual_max
    )
    residual_abs_fixed = _float_rgb_to_uint8(
        np.clip(absolute_residual / residual_display_scale, 0.0, 1.0)
    )
    residual_signed = _float_rgb_to_uint8(
        np.clip(
            0.5 + signed_residual / (2.0 * residual_display_scale),
            0.0,
            1.0,
        )
    )
    normalized_amplified = _float_rgb_to_uint8(
        np.clip(
            input_float + residual_amplification * signed_residual,
            0.0,
            1.0,
        )
    )
    for subdirectory, array in (
        ("input", input_rgb),
        ("normalized", normalized_rgb),
        ("residual_abs", residual_abs_auto),
        ("residual_abs_fixed", residual_abs_fixed),
        ("residual_signed", residual_signed),
        ("normalized_change_amplified", normalized_amplified),
    ):
        directory = output_root / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(directory / f"{safe_id}.png")

    mean_absolute = float(absolute_residual.mean())
    mean_signed_channels = signed_residual.mean(axis=(0, 1))
    side_by_side = _build_residual_comparison_panel(
        tiles=(
            input_rgb,
            normalized_rgb,
            residual_signed,
            residual_abs_fixed,
            normalized_amplified,
        ),
        labels=(
            "Input",
            "Normalized",
            f"Signed residual (gray=0, scale=+/-{residual_display_scale:g})",
            f"Absolute residual (fixed scale={residual_display_scale:g})",
            f"Input + {residual_amplification:g}x residual",
        ),
        statistics=(
            f"mean |delta|={mean_absolute:.6f}   max |delta|={residual_max:.6f}   "
            f"mean delta RGB=({mean_signed_channels[0]:+.6f}, "
            f"{mean_signed_channels[1]:+.6f}, {mean_signed_channels[2]:+.6f})"
        ),
    )
    side_directory = output_root / "side_by_side"
    side_directory.mkdir(parents=True, exist_ok=True)
    side_by_side.save(side_directory / f"{safe_id}.png")


def _build_residual_comparison_panel(
    tiles: Sequence[np.ndarray],
    labels: Sequence[str],
    statistics: str,
) -> Image.Image:
    """Build a labeled comparison panel with one shared statistics header."""
    if not tiles or len(tiles) != len(labels):
        raise ValueError("tiles and labels must be non-empty and have equal length.")
    tile_images = [Image.fromarray(tile) for tile in tiles]
    width, height = tile_images[0].size
    if any(image.size != (width, height) for image in tile_images):
        raise ValueError("All comparison tiles must have the same dimensions.")
    statistics_height = 28
    label_height = 50
    canvas = Image.new(
        "RGB",
        (width * len(tile_images), statistics_height + label_height + height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), statistics, fill="black")
    for index, (image, label) in enumerate(zip(tile_images, labels)):
        x = index * width
        wrapped_label = textwrap.wrap(label, width=max(12, width // 7))
        for line_index, line in enumerate(wrapped_label[:3]):
            draw.text(
                (x + 5, statistics_height + 3 + 13 * line_index),
                line,
                fill="black",
            )
        canvas.paste(image, (x, statistics_height + label_height))
    return canvas


def _write_visualization_readme(
    visualizations_root: Path,
    residual_display_scale: float,
    residual_amplification: float,
) -> None:
    """Write an interpretation guide next to normalizer visual artifacts."""
    visualizations_root.mkdir(parents=True, exist_ok=True)
    content = f"""# Normalizer visualization guide

All RGB differences are computed after reversing the model's channel
normalization and clipping both images to the display range `[0, 1]`.

- `input`: original image passed to the normalizer.
- `normalized`: actual normalizer output passed to the landmarker.
- `residual_abs`: legacy absolute residual, independently auto-scaled by each
  image's maximum. It shows spatial support but cannot compare magnitudes.
- `residual_abs_fixed`: absolute residual using a shared scale. A channel value
  of `{residual_display_scale:g}` maps to full intensity in every image.
- `residual_signed`: signed residual using the same shared scale. Middle gray
  means zero; values above/below gray mean positive/negative channel changes.
- `normalized_change_amplified`: input plus `{residual_amplification:g}` times
  the signed residual. This is diagnostic only and is not passed to the model.
- `side_by_side`: labeled panel containing input, actual normalized output,
  signed residual, fixed-scale absolute residual, and amplified preview.

The panel header reports mean absolute RGB residual, maximum absolute RGB
residual, and mean signed change for red, green, and blue. These values use the
`[0, 1]` RGB range. The signed and absolute fixed-scale images are comparable
across samples only when generated with the same residual display scale.
"""
    (visualizations_root / "README.md").write_text(content, encoding="utf-8")


def _to_display_float(
    image: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    """Convert a channel-normalized tensor to uncluttered RGB floats in [0, 1]."""
    mean_tensor = torch.as_tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype).view(-1, 1, 1)
    rgb = (image * std_tensor + mean_tensor).permute(1, 2, 0).clamp(0, 1)
    return np.asarray(rgb.tolist(), dtype=np.float32)


def _float_rgb_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert RGB floats in [0, 1] into a displayable uint8 image."""
    return np.clip(array * 255.0, 0, 255).round().astype(np.uint8)


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
