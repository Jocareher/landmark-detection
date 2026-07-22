from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .metrics import decode_heatmaps_to_image_coords
from ..models import NormalizedLandmarker


class NormalizerProbeMonitor:
    """Track a fixed image set while an image normalizer is optimized.

    Probe tensors are copied once at construction time. Captures therefore use
    identical inputs, display clipping, coordinate decoding, and difference-map
    scaling at every checkpoint. Ground truth is optional and is never consumed
    by the model or by an adaptation loss.
    """

    def __init__(
        self,
        model: NormalizedLandmarker,
        probe_batch: dict[str, Any],
        device: torch.device,
        output_dir: str | Path,
        mean: Sequence[float],
        std: Sequence[float],
        coordinate_decoder: str,
        softmax_temperature: float,
        max_images: int = 4,
        difference_display_max: float = 0.15,
        registration_warning_px: float = 1.0,
        edge_correlation_warning: float = 0.90,
        wandb_module: Any | None = None,
        wandb_prefix: str = "normalizer_monitor",
    ) -> None:
        """Snapshot probes and initialize local and optional W&B logging."""
        if max_images <= 0:
            raise ValueError("max_images must be positive.")
        if difference_display_max <= 0:
            raise ValueError("difference_display_max must be positive.")
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.panels_dir = self.output_dir / "panels"
        self.grids_dir = self.output_dir / "checkpoint_grids"
        self.animations_dir = self.output_dir / "animations"
        self.plots_dir = self.output_dir / "plots"
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.grids_dir.mkdir(parents=True, exist_ok=True)
        self.animations_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.coordinate_decoder = coordinate_decoder
        self.softmax_temperature = float(softmax_temperature)
        self.difference_display_max = float(difference_display_max)
        self.registration_warning_px = float(registration_warning_px)
        self.edge_correlation_warning = float(edge_correlation_warning)
        self.wandb = wandb_module
        self.wandb_prefix = wandb_prefix
        self.images = probe_batch["image"][:max_images].detach().cpu().clone()
        self.landmarks = _optional_probe_tensor(probe_batch, "landmarks", max_images)
        self.visibility = _optional_probe_tensor(probe_batch, "visibility", max_images)
        self.sample_ids = _extract_sample_ids(
            probe_batch.get("metadata", {}), self.images.shape[0]
        )
        self.rows: list[dict[str, Any]] = []
        self.loss_rows: list[dict[str, float | int | str]] = []
        self.panel_history: dict[str, list[Path]] = {
            sample_id: [] for sample_id in self.sample_ids
        }

    @classmethod
    def from_dataloader(
        cls,
        model: NormalizedLandmarker,
        dataloader: torch.utils.data.DataLoader,
        **kwargs: Any,
    ) -> NormalizerProbeMonitor:
        """Build a monitor from one deterministic loader batch."""
        return cls(model=model, probe_batch=next(iter(dataloader)), **kwargs)

    def capture(
        self,
        stage: str,
        step: int,
        adaptation_loss: float | None = None,
        structural_prior_loss: float | None = None,
        is_final: bool = False,
    ) -> dict[str, float]:
        """Capture panels and metrics for the unchanged probe images.

        For future test-time adaptation, call this method at steps 0, 1, 5,
        10, 20, and the final step and pass both loss values. Omit ground truth
        from ``probe_batch`` for real target images.
        """
        if not isinstance(self.model, NormalizedLandmarker):
            raise TypeError("Normalizer monitoring requires NormalizedLandmarker.")
        step_name = "final" if is_final else f"step_{step:06d}"
        capture_dir = self.panels_dir / f"{_safe_name(stage)}_{step_name}"
        capture_dir.mkdir(parents=True, exist_ok=True)
        was_training = self.model.training
        self.model.eval()
        images = self.images.to(self.device)
        with torch.inference_mode():
            normalized = self.model.normalize_images(images)
            before_outputs = self.model.landmarker(images)
            after_outputs = self.model.forward_normalized(normalized)
            before_landmarks = decode_heatmaps_to_image_coords(
                before_outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                decoder=self.coordinate_decoder,
                softmax_temperature=self.softmax_temperature,
            )
            after_landmarks = decode_heatmaps_to_image_coords(
                after_outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                decoder=self.coordinate_decoder,
                softmax_temperature=self.softmax_temperature,
            )
            before_confidence = _heatmap_peak_confidence(before_outputs["heatmaps"])
            after_confidence = _heatmap_peak_confidence(after_outputs["heatmaps"])
        if was_training:
            self.model.train()

        capture_rows: list[dict[str, Any]] = []
        wandb_images: list[Any] = []
        for index, sample_id in enumerate(self.sample_ids):
            original_rgb = _tensor_to_rgb_float(
                images[index].cpu(), self.mean, self.std
            )
            normalized_rgb = _tensor_to_rgb_float(
                normalized[index].cpu(), self.mean, self.std
            )
            difference = np.abs(normalized_rgb - original_rgb)
            difference_rgb = np.clip(difference / self.difference_display_max, 0.0, 1.0)
            gt_landmarks = self.landmarks[index] if self.landmarks is not None else None
            gt_visibility = (
                self.visibility[index] if self.visibility is not None else None
            )
            panel = _build_six_part_panel(
                original_rgb=original_rgb,
                normalized_rgb=normalized_rgb,
                difference_rgb=difference_rgb,
                before_landmarks=before_landmarks[index].cpu(),
                after_landmarks=after_landmarks[index].cpu(),
                ground_truth_landmarks=gt_landmarks,
                ground_truth_visibility=gt_visibility,
                title=f"{stage} | {step_name} | {sample_id}",
            )
            safe_id = _safe_name(sample_id)
            panel_path = capture_dir / f"{safe_id}.png"
            panel.save(panel_path)
            self.panel_history[sample_id].append(panel_path)
            row = _compute_probe_metrics(
                original_rgb=original_rgb,
                normalized_rgb=normalized_rgb,
                before_landmarks=before_landmarks[index].cpu(),
                after_landmarks=after_landmarks[index].cpu(),
                before_confidence=before_confidence[index].cpu(),
                after_confidence=after_confidence[index].cpu(),
                ground_truth_landmarks=gt_landmarks,
                ground_truth_visibility=gt_visibility,
            )
            row.update(
                {
                    "stage": stage,
                    "step": int(step),
                    "checkpoint": step_name,
                    "sample_id": sample_id,
                    "geometry_warning": bool(
                        abs(float(row["registration_shift_x_px"]))
                        > self.registration_warning_px
                        or abs(float(row["registration_shift_y_px"]))
                        > self.registration_warning_px
                        or float(row["edge_correlation"])
                        < self.edge_correlation_warning
                    ),
                }
            )
            capture_rows.append(row)
            self.rows.append(row)
            if self.wandb is not None:
                wandb_images.append(
                    self.wandb.Image(panel, caption=f"{sample_id} {step_name}")
                )

        summary = _summarize_numeric_rows(capture_rows)
        summary["geometry_warning_count"] = float(
            sum(bool(row["geometry_warning"]) for row in capture_rows)
        )
        self._write_outputs()
        self._write_history_visuals()
        if adaptation_loss is not None or structural_prior_loss is not None:
            self.loss_rows.append(
                {
                    "stage": stage,
                    "step": int(step),
                    "adaptation_loss": float(adaptation_loss)
                    if adaptation_loss is not None
                    else math.nan,
                    "structural_prior_loss": float(structural_prior_loss)
                    if structural_prior_loss is not None
                    else math.nan,
                }
            )
            _write_rows_csv(self.output_dir / "adaptation_losses.csv", self.loss_rows)
            _save_loss_plot(self.loss_rows, self.output_dir / "adaptation_losses.png")
        if self.wandb is not None:
            payload = {
                f"{self.wandb_prefix}/{stage}/{key}": value
                for key, value in summary.items()
            }
            payload[f"{self.wandb_prefix}/{stage}/capture_step"] = int(step)
            payload[f"{self.wandb_prefix}/{stage}/panels"] = wandb_images
            if adaptation_loss is not None:
                payload[f"{self.wandb_prefix}/{stage}/adaptation_loss"] = float(
                    adaptation_loss
                )
            if structural_prior_loss is not None:
                payload[f"{self.wandb_prefix}/{stage}/structural_prior_loss"] = float(
                    structural_prior_loss
                )
            self.wandb.log(payload)
        return summary

    def _write_outputs(self) -> None:
        """Persist raw metrics, per-checkpoint summaries, and readable plots."""
        _write_rows_csv(self.output_dir / "probe_metrics.csv", self.rows)
        checkpoint_rows = _summarize_probe_checkpoints(self.rows)
        _write_rows_csv(
            self.output_dir / "probe_metrics_by_checkpoint.csv", checkpoint_rows
        )
        if checkpoint_rows:
            _write_rows_csv(
                self.output_dir / "probe_metrics_final_summary.csv",
                [checkpoint_rows[-1]],
            )
        _write_probe_metric_plots(self.rows, self.plots_dir)
        _write_interpretation_report(
            self.output_dir / "monitoring_report.md",
            self.rows,
            difference_display_max=self.difference_display_max,
        )

    def _write_history_visuals(self) -> None:
        """Refresh checkpoint grids and GIFs from all captures so far."""
        for sample_id, paths in self.panel_history.items():
            if not paths:
                continue
            frames = [Image.open(path).convert("RGB") for path in paths]
            safe_id = _safe_name(sample_id)
            grid = _make_checkpoint_grid(frames)
            grid.save(self.grids_dir / f"{safe_id}.png")
            frames[0].save(
                self.animations_dir / f"{safe_id}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=900,
                loop=0,
            )
            for frame in frames:
                frame.close()


def should_capture_source_step(
    epoch_number: int,
    configured_steps: Sequence[int],
    is_final: bool = False,
) -> bool:
    """Return whether an epoch checkpoint belongs to the monitoring schedule."""
    return is_final or int(epoch_number) in {int(step) for step in configured_steps}


def _optional_probe_tensor(
    batch: dict[str, Any], key: str, max_images: int
) -> torch.Tensor | None:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    return value[:max_images].detach().cpu().clone()


def _extract_sample_ids(metadata: Any, count: int) -> list[str]:
    if isinstance(metadata, dict) and "sample_id" in metadata:
        values = metadata["sample_id"]
        if isinstance(values, (list, tuple)):
            return [str(value) for value in values[:count]]
        try:
            return [str(values[index]) for index in range(count)]
        except Exception:
            pass
    return [f"probe_{index:03d}" for index in range(count)]


def _tensor_to_rgb_float(
    image: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    mean_tensor = torch.as_tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=image.dtype).view(-1, 1, 1)
    rgb = (image * std_tensor + mean_tensor).permute(1, 2, 0).clamp(0, 1)
    return np.asarray(rgb.mul(255).round().byte().tolist(), dtype=np.uint8) / 255.0


def _heatmap_peak_confidence(heatmaps: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(heatmaps.flatten(start_dim=-2), dim=-1)
    return probabilities.amax(dim=-1)


def _build_six_part_panel(
    original_rgb: np.ndarray,
    normalized_rgb: np.ndarray,
    difference_rgb: np.ndarray,
    before_landmarks: torch.Tensor,
    after_landmarks: torch.Tensor,
    ground_truth_landmarks: torch.Tensor | None,
    ground_truth_visibility: torch.Tensor | None,
    title: str,
) -> Image.Image:
    original = _array_to_image(original_rgb)
    normalized = _array_to_image(normalized_rgb)
    difference = _array_to_image(difference_rgb)
    before = _draw_landmarks(original.copy(), before_landmarks, "#00FFFF")
    after = _draw_landmarks(normalized.copy(), after_landmarks, "#FF00FF")
    ground_truth = original.copy()
    if ground_truth_landmarks is not None:
        ground_truth = _draw_landmarks(
            ground_truth,
            ground_truth_landmarks,
            "#00FF00",
            ground_truth_visibility,
        )
    names = (
        "Original",
        "Normalized",
        "Absolute difference",
        "Before normalization",
        "After normalization",
        "Ground truth (source only)",
    )
    tiles = [original, normalized, difference, before, after, ground_truth]
    width, height = original.size
    header = 22
    title_height = 26
    canvas = Image.new(
        "RGB", (3 * width, 2 * (height + header) + title_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill="black")
    for index, (name, tile) in enumerate(zip(names, tiles)):
        column = index % 3
        row = index // 3
        x = column * width
        y = title_height + row * (height + header)
        draw.text((x + 4, y + 3), name, fill="black")
        canvas.paste(tile, (x, y + header))
    return canvas


def _array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0, 255).round().astype(np.uint8))


def _draw_landmarks(
    image: Image.Image,
    landmarks: torch.Tensor,
    color: str,
    visibility: torch.Tensor | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for index, point in enumerate(landmarks):
        if visibility is not None and float(visibility[index]) <= 0:
            continue
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        radius = 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return image


def _compute_probe_metrics(
    original_rgb: np.ndarray,
    normalized_rgb: np.ndarray,
    before_landmarks: torch.Tensor,
    after_landmarks: torch.Tensor,
    before_confidence: torch.Tensor,
    after_confidence: torch.Tensor,
    ground_truth_landmarks: torch.Tensor | None,
    ground_truth_visibility: torch.Tensor | None,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for name, array in (("input", original_rgb), ("normalized", normalized_rgb)):
        for channel_index, channel_name in enumerate(("r", "g", "b")):
            channel = array[..., channel_index]
            row[f"{name}_{channel_name}_mean"] = float(channel.mean())
            row[f"{name}_{channel_name}_std"] = float(channel.std())
        gray = _to_gray(array)
        row[f"{name}_contrast"] = float(gray.std())
        row[f"{name}_dynamic_range"] = float(
            np.percentile(gray, 99) - np.percentile(gray, 1)
        )
        row[f"{name}_high_frequency_ratio"] = _high_frequency_ratio(gray)
    difference = normalized_rgb - original_rgb
    row["mean_absolute_pixel_difference"] = float(np.abs(difference).mean())
    row["max_absolute_pixel_difference"] = float(np.abs(difference).max())
    before_gray = _to_gray(original_rgb)
    after_gray = _to_gray(normalized_rgb)
    shift_y, shift_x = _phase_correlation_shift(before_gray, after_gray)
    row["registration_shift_x_px"] = float(shift_x)
    row["registration_shift_y_px"] = float(shift_y)
    before_edges = _sobel_edges(before_gray)
    after_edges = _sobel_edges(after_gray)
    row["edge_mean_absolute_difference"] = float(
        np.abs(after_edges - before_edges).mean()
    )
    row["edge_correlation"] = _safe_correlation(before_edges, after_edges)
    displacement = torch.linalg.vector_norm(after_landmarks - before_landmarks, dim=-1)
    row["mean_landmark_displacement_px"] = float(displacement.mean())
    row["max_landmark_displacement_px"] = float(displacement.max())
    row["heatmap_confidence_before"] = float(before_confidence.mean())
    row["heatmap_confidence_after"] = float(after_confidence.mean())
    row["heatmap_confidence_change"] = float(
        after_confidence.mean() - before_confidence.mean()
    )
    row["localization_error_before_px"] = math.nan
    row["localization_error_after_px"] = math.nan
    row["localization_error_change_px"] = math.nan
    if ground_truth_landmarks is not None:
        valid = torch.isfinite(ground_truth_landmarks).all(dim=-1)
        if ground_truth_visibility is not None:
            valid &= ground_truth_visibility > 0
        if bool(valid.any()):
            before_error = torch.linalg.vector_norm(
                before_landmarks[valid] - ground_truth_landmarks[valid], dim=-1
            ).mean()
            after_error = torch.linalg.vector_norm(
                after_landmarks[valid] - ground_truth_landmarks[valid], dim=-1
            ).mean()
            row["localization_error_before_px"] = float(before_error)
            row["localization_error_after_px"] = float(after_error)
            row["localization_error_change_px"] = float(after_error - before_error)
    return row


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32
    )


def _high_frequency_ratio(gray: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2)
    high_frequency = radius >= 0.25 * min(height, width)
    total = float(spectrum.sum())
    return float(spectrum[high_frequency].sum() / max(total, 1e-12))


def _phase_correlation_shift(
    reference: np.ndarray, moving: np.ndarray
) -> tuple[int, int]:
    reference_fft = np.fft.fft2(reference - reference.mean())
    moving_fft = np.fft.fft2(moving - moving.mean())
    cross_power = reference_fft * np.conj(moving_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.abs(np.fft.ifft2(cross_power))
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    height, width = reference.shape
    if peak_y > height // 2:
        peak_y -= height
    if peak_x > width // 2:
        peak_x -= width
    return int(peak_y), int(peak_x)


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    # ``tolist`` keeps this diagnostic usable when the optional PyTorch/NumPy
    # binary bridge is unavailable or ABI-incompatible.
    tensor = torch.tensor(gray.tolist(), dtype=torch.float32).view(1, 1, *gray.shape)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gradient_x = F.conv2d(tensor, kernel_x, padding=1)
    gradient_y = F.conv2d(tensor, kernel_y, padding=1)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)
    return np.asarray(magnitude[0, 0].tolist(), dtype=np.float32)


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_centered = first_flat - first_flat.mean()
    second_centered = second_flat - second_flat.mean()
    denominator = float(
        np.sqrt(np.square(first_centered).sum() * np.square(second_centered).sum())
    )
    if denominator <= 1e-12:
        return 1.0 if np.allclose(first_flat, second_flat) else 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def _summarize_numeric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not rows:
        return summary
    for key in rows[0]:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
        ]
        if values:
            summary[f"mean_{key}"] = float(sum(values) / len(values))
    return summary


def _summarize_probe_checkpoints(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate fixed-probe metrics independently for every capture."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["stage"]), str(row["checkpoint"]), int(row["step"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (stage, checkpoint, step), checkpoint_rows in grouped.items():
        metric_summary = _summarize_numeric_rows(checkpoint_rows)
        metric_summary.pop("mean_step", None)
        output.append(
            {
                "stage": stage,
                "checkpoint": checkpoint,
                "step": step,
                "num_probes": len(checkpoint_rows),
                "geometry_warning_count": sum(
                    bool(row.get("geometry_warning")) for row in checkpoint_rows
                ),
                **metric_summary,
            }
        )
    return output


def _write_probe_metric_plots(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    """Create trajectory and final-probe charts from ``probe_metrics.csv``."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    captures = _ordered_capture_groups(rows)
    trajectory_specs = (
        ("mean_absolute_pixel_difference", "Appearance change", "mean |RGB delta|"),
        ("mean_landmark_displacement_px", "Prediction displacement", "pixels"),
        ("localization_error_change_px", "Localization error change", "pixels"),
        ("edge_correlation", "Edge preservation", "correlation"),
    )
    charts = [
        _draw_metric_chart(
            labels=[label for label, _ in captures],
            groups=[group for _, group in captures],
            metric=metric,
            title=title,
            y_label=y_label,
        )
        for metric, title, y_label in trajectory_specs
    ]
    dashboard = Image.new("RGB", (1440, 790), "#F4F7FB")
    draw = ImageDraw.Draw(dashboard)
    draw.text(
        (32, 18),
        "Normalizer monitoring - fixed-probe evolution",
        fill="#17233C",
    )
    draw.text(
        (32, 42),
        "Line = probe mean; shaded band = min-max across the unchanged probe set",
        fill="#5A667D",
    )
    for index, chart in enumerate(charts):
        dashboard.paste(chart, (20 + (index % 2) * 710, 75 + (index // 2) * 350))
    dashboard.save(output_dir / "probe_metric_trajectories.png")

    latest_label, latest_rows = captures[-1]
    profile_specs = (
        ("mean_absolute_pixel_difference", "Appearance change"),
        ("mean_landmark_displacement_px", "Landmark displacement (px)"),
        ("localization_error_change_px", "Localization error change (px)"),
        ("edge_correlation", "Edge correlation"),
    )
    profile = _draw_final_probe_profile(latest_label, latest_rows, profile_specs)
    profile.save(output_dir / "final_probe_profile.png")


def _ordered_capture_groups(
    rows: Sequence[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["stage"]), str(row["checkpoint"]))
        groups.setdefault(key, []).append(row)
    return [
        (checkpoint.replace("step_", ""), group)
        for (_, checkpoint), group in groups.items()
    ]


def _finite_metric_values(rows: Sequence[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if isinstance(row.get(metric), (int, float))
        and math.isfinite(float(row[metric]))
    ]


def _draw_metric_chart(
    labels: Sequence[str],
    groups: Sequence[Sequence[dict[str, Any]]],
    metric: str,
    title: str,
    y_label: str,
) -> Image.Image:
    width, height = 690, 330
    left, right, top, bottom = 74, 24, 48, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), 12, outline="#D8DFEA")
    draw.text((18, 14), title, fill="#17233C")
    draw.text((18, 31), y_label, fill="#647089")
    series = [_finite_metric_values(group, metric) for group in groups]
    valid_series = [(index, values) for index, values in enumerate(series) if values]
    draw.line((left, top, left, height - bottom), fill="#9AA6B8", width=1)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="#9AA6B8",
        width=1,
    )
    if not valid_series:
        draw.text((left + 20, top + 80), "Metric unavailable", fill="#8B96A8")
        return image
    all_values = [value for _, values in valid_series for value in values]
    value_min, value_max = min(all_values), max(all_values)
    if metric in {"mean_absolute_pixel_difference", "mean_landmark_displacement_px"}:
        value_min = min(0.0, value_min)
    span = max(value_max - value_min, 1e-9)
    padding = 0.08 * span
    value_min -= padding
    value_max += padding
    span = value_max - value_min
    x_span = max(len(labels) - 1, 1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + index / x_span * (width - left - right)
        y = height - bottom - (value - value_min) / span * (height - top - bottom)
        return x, y

    upper = [point(index, max(values)) for index, values in valid_series]
    lower = [point(index, min(values)) for index, values in reversed(valid_series)]
    if len(upper) > 1:
        draw.polygon([*upper, *lower], fill="#DCEBFA")
    means = [point(index, sum(values) / len(values)) for index, values in valid_series]
    if len(means) > 1:
        draw.line(means, fill="#1769AA", width=3)
    for x, y in means:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#1769AA")
    for tick in range(5):
        value = value_min + tick / 4 * span
        _, y = point(0, value)
        draw.line((left - 4, y, left, y), fill="#9AA6B8")
        draw.text((6, y - 7), f"{value:.3g}", fill="#5A667D")
    visible_label_indices = sorted({0, len(labels) // 2, len(labels) - 1})
    for index in visible_label_indices:
        x, _ = point(index, value_min)
        draw.text((x - 14, height - bottom + 10), labels[index][:10], fill="#5A667D")
    return image


def _draw_final_probe_profile(
    checkpoint: str,
    rows: Sequence[dict[str, Any]],
    specs: Sequence[tuple[str, str]],
) -> Image.Image:
    width = 1180
    row_height = 34
    section_height = 74 + row_height * len(rows)
    height = 70 + section_height * len(specs)
    image = Image.new("RGB", (width, height), "#F4F7FB")
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), f"Final fixed-probe profile - {checkpoint}", fill="#17233C")
    draw.text(
        (28, 40),
        "Bars compare individual probes; exact values remain available in probe_metrics.csv",
        fill="#5A667D",
    )
    y = 68
    colors = ("#1769AA", "#37A67E", "#D8842F", "#8A62B3", "#D24D57")
    for metric, title in specs:
        values = _finite_metric_values(rows, metric)
        draw.rounded_rectangle(
            (18, y, width - 18, y + section_height - 8),
            10,
            fill="white",
            outline="#D8DFEA",
        )
        draw.text((34, y + 14), title, fill="#17233C")
        if not values:
            draw.text((34, y + 42), "Metric unavailable", fill="#8B96A8")
            y += section_height
            continue
        scale_min = min(0.0, min(values))
        scale_max = max(0.0, max(values))
        span = max(scale_max - scale_min, 1e-9)
        chart_left, chart_right = 270, width - 92
        zero_x = chart_left + (0.0 - scale_min) / span * (chart_right - chart_left)
        draw.line((zero_x, y + 43, zero_x, y + section_height - 20), fill="#AAB4C4")
        for index, row in enumerate(rows):
            value = float(row.get(metric, math.nan))
            if not math.isfinite(value):
                continue
            row_y = y + 48 + index * row_height
            label = str(row.get("sample_id", f"probe_{index}"))[:28]
            draw.text((34, row_y), label, fill="#465168")
            value_x = chart_left + (value - scale_min) / span * (
                chart_right - chart_left
            )
            draw.rectangle(
                (min(zero_x, value_x), row_y + 2, max(zero_x, value_x), row_y + 17),
                fill=colors[index % len(colors)],
            )
            draw.text((chart_right + 10, row_y), f"{value:.4g}", fill="#465168")
        y += section_height
    return image


def _make_checkpoint_grid(frames: Sequence[Image.Image]) -> Image.Image:
    columns = min(2, len(frames))
    rows = math.ceil(len(frames) / columns)
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    for index, frame in enumerate(frames):
        grid.paste(frame, ((index % columns) * width, (index // columns) * height))
    return grid


def _save_loss_plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    width, height = 720, 420
    margin = 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.line(
        (margin, height - margin, width - 15, height - margin), fill="black", width=2
    )
    draw.line((margin, 20, margin, height - margin), fill="black", width=2)
    series = (
        ("adaptation_loss", "#0066CC"),
        ("structural_prior_loss", "#CC3300"),
    )
    finite_values = [
        float(row[key])
        for row in rows
        for key, _ in series
        if math.isfinite(float(row.get(key, math.nan)))
    ]
    if finite_values:
        maximum = max(finite_values)
        minimum = min(finite_values)
        span = max(maximum - minimum, 1e-12)
        steps = [int(row["step"]) for row in rows]
        step_span = max(max(steps) - min(steps), 1)
        for key, color in series:
            points = []
            for row in rows:
                value = float(row.get(key, math.nan))
                if not math.isfinite(value):
                    continue
                x = margin + (int(row["step"]) - min(steps)) / step_span * (
                    width - margin - 15
                )
                y = height - margin - (value - minimum) / span * (height - margin - 20)
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse(
                    (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                    fill=color,
                )
        draw.text((margin + 8, 25), "adaptation loss", fill=series[0][1])
        draw.text((margin + 150, 25), "structural-prior loss", fill=series[1][1])
    draw.text((width // 2 - 20, height - 25), "step", fill="black")
    canvas.save(path)


def _write_interpretation_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    difference_display_max: float,
) -> None:
    """Write conservative artifact flags for the most recent checkpoint."""
    if not rows:
        return
    latest_checkpoint = str(rows[-1]["checkpoint"])
    latest_rows = [row for row in rows if str(row["checkpoint"]) == latest_checkpoint]
    geometry_warnings = sum(bool(row["geometry_warning"]) for row in latest_rows)
    smoothing_warnings = sum(
        float(row["normalized_high_frequency_ratio"])
        < 0.7 * max(float(row["input_high_frequency_ratio"]), 1e-12)
        for row in latest_rows
    )
    color_collapse_warnings = 0
    for row in latest_rows:
        input_std = sum(float(row[f"input_{channel}_std"]) for channel in "rgb") / 3
        normalized_std = (
            sum(float(row[f"normalized_{channel}_std"]) for channel in "rgb") / 3
        )
        color_collapse_warnings += normalized_std < 0.5 * max(input_std, 1e-12)
    excessive_change_warnings = sum(
        float(row["mean_absolute_pixel_difference"]) > 0.5 * difference_display_max
        for row in latest_rows
    )
    residuals = [float(row["mean_absolute_pixel_difference"]) for row in latest_rows]
    residual_mean = sum(residuals) / max(len(residuals), 1)
    residual_variance = sum((value - residual_mean) ** 2 for value in residuals) / max(
        len(residuals), 1
    )
    residual_cv = math.sqrt(residual_variance) / max(residual_mean, 1e-12)
    identity_dependence_warning = (
        len(residuals) > 1 and residual_mean > 1e-6 and residual_cv > 1.0
    )
    summary = _summarize_numeric_rows(latest_rows)
    error_change = summary.get("mean_localization_error_change_px", math.nan)
    confidence_change = summary.get("mean_heatmap_confidence_change", math.nan)
    lines = [
        "# Normalizer monitoring report",
        "",
        f"Latest checkpoint: `{latest_checkpoint}` ({len(latest_rows)} fixed probes).",
        "",
        "## Automatic diagnostic flags",
        "",
        f"- Possible geometry change: {geometry_warnings}/{len(latest_rows)} probes.",
        f"- Possible excessive smoothing: {smoothing_warnings}/{len(latest_rows)} probes.",
        f"- Possible color collapse: {color_collapse_warnings}/{len(latest_rows)} probes.",
        f"- Large appearance residual: {excessive_change_warnings}/{len(latest_rows)} probes.",
        (
            "- Possible identity-dependent change: yes "
            f"(residual coefficient of variation `{residual_cv:.4f}`)."
            if identity_dependence_warning
            else "- Possible identity-dependent change: not flagged by residual variability."
        ),
        "",
        "## Relationship to landmark predictions",
        "",
        f"- Mean heatmap-confidence change: `{confidence_change:.8f}`.",
        (
            f"- Mean source localization-error change: `{error_change:.6f}` px."
            if math.isfinite(error_change)
            else "- Source localization-error change: unavailable because no ground truth was logged."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "These flags identify suspicious appearance or structural changes; they do not prove that the output is closer to an unseen target domain. Appearance correction must be supported by improved landmark metrics/confidence without geometry warnings and by visual inspection of the checkpoint grid or GIF."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")
