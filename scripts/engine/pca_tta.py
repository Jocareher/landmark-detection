from __future__ import annotations

import csv
import json
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .pca_shape_prior import (
    compute_pca_projection_loss,
    softargmax_heatmaps_to_image_coords,
)
from ..models import NormalizedLandmarker


@dataclass(frozen=True)
class PCATTAConfig:
    """Configuration for episodic PCA-guided test-time adaptation."""

    steps: int = 20
    learning_rate: float = 1e-4
    monitor_steps: tuple[int, ...] = (0, 1, 5, 10, 20)
    probe_count: int = 4
    difference_display_max: float = 0.15
    normalization_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, ...] = (0.229, 0.224, 0.225)

    def validate(self) -> None:
        """Validate values before any target image is adapted."""
        if self.steps < 1:
            raise ValueError("PCA TTA requires at least one adaptation step.")
        if self.learning_rate <= 0:
            raise ValueError("PCA TTA learning_rate must be positive.")
        if self.probe_count < 0:
            raise ValueError("PCA TTA probe_count cannot be negative.")
        if self.difference_display_max <= 0:
            raise ValueError("difference_display_max must be positive.")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("PCA TTA visualization expects three-channel normalization.")


class PCAGuidedTTA:
    """Adapt only an image normalizer using PCA reconstruction loss.

    Adaptation is episodic: the source-trained normalizer and a fresh Adam
    optimizer are restored for every input image. The landmarker and PCA prior
    remain frozen. No target ground truth is consumed by this class.
    """

    def __init__(
        self,
        model: NormalizedLandmarker,
        pca_prior: dict[str, Any],
        device: torch.device,
        output_dir: str | Path,
        config: PCATTAConfig,
        wandb_module: Any | None = None,
    ) -> None:
        """Snapshot source weights and initialize trajectory reporting."""
        config.validate()
        if not isinstance(model, NormalizedLandmarker) or model.normalizer is None:
            raise TypeError(
                "PCA-guided TTA requires a NormalizedLandmarker with an active "
                "external image normalizer."
            )
        self.model = model
        self.pca_prior = pca_prior
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.wandb = wandb_module
        self.source_normalizer_state = {
            key: value.detach().cpu().clone()
            for key, value in model.normalizer.state_dict().items()
        }
        self.trajectory_rows: list[dict[str, Any]] = []
        self.summary_rows: list[dict[str, Any]] = []
        self.processed_samples = 0
        self.failed_samples = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "probes").mkdir(parents=True, exist_ok=True)
        self.model.to(device)
        self.model.freeze_landmarker()
        self.model.unfreeze_normalizer()
        self._restore_source_normalizer()
        self._validate_parameter_partition()
        landmarker_total = _count_parameters(self.model.landmarker.parameters())
        normalizer_total = _count_parameters(self.model.normalizer.parameters())
        print(
            "[PCA-TTA] Parameter audit | "
            f"landmarker_total={landmarker_total:,} "
            "landmarker_trainable=0 "
            f"normalizer_total={normalizer_total:,} "
            f"normalizer_trainable={normalizer_total:,}"
        )

    def adapt_batch(
        self,
        images: torch.Tensor,
        sample_ids: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Adapt independently to every image and concatenate final outputs."""
        if images.ndim != 4:
            raise ValueError(
                f"Expected image batch with shape (B, C, H, W), got {images.shape}."
            )
        ids = (
            [str(value) for value in sample_ids]
            if sample_ids is not None
            else [
                f"sample_{self.processed_samples + index:06d}"
                for index in range(images.shape[0])
            ]
        )
        if len(ids) != images.shape[0]:
            raise ValueError("sample_ids length must match the image batch size.")

        batched_outputs: dict[str, list[torch.Tensor]] = {}
        for sample_index, sample_id in enumerate(ids):
            outputs = self._adapt_sample(
                image=images[sample_index : sample_index + 1],
                sample_id=sample_id,
            )
            for key, value in outputs.items():
                batched_outputs.setdefault(key, []).append(value)
        return {
            key: torch.cat(values, dim=0) for key, values in batched_outputs.items()
        }

    def _adapt_sample(
        self,
        image: torch.Tensor,
        sample_id: str,
    ) -> dict[str, torch.Tensor]:
        """Run one independent adaptation episode and restore source weights."""
        self._restore_source_normalizer()
        self._validate_parameter_partition()
        self.model.landmarker.eval()
        assert self.model.normalizer is not None
        self.model.normalizer.train()
        optimizer = torch.optim.Adam(
            self.model.normalizer.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.0,
        )
        episode_number = self.processed_samples + 1
        print(
            f"[PCA-TTA][episode={episode_number:06d}] sample={sample_id} | "
            "source_normalizer_restored=yes optimizer=fresh "
            "landmarker=frozen normalizer=trainable"
        )
        image = image.detach().to(self.device)
        sample_rows: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        baseline_landmarks: torch.Tensor | None = None
        final_outputs: dict[str, torch.Tensor] | None = None
        failed = False
        failure_message = ""

        try:
            for step in range(self.config.steps + 1):
                outputs = self.model(image)
                landmarks = softargmax_heatmaps_to_image_coords(
                    heatmaps=outputs["heatmaps"],
                    image_height=image.shape[2],
                    image_width=image.shape[3],
                )
                reconstruction_loss = compute_pca_projection_loss(
                    predicted_landmarks=landmarks,
                    pca_prior=self.pca_prior,
                )
                if not bool(torch.isfinite(reconstruction_loss).item()):
                    raise FloatingPointError(
                        f"Non-finite PCA reconstruction loss at step {step}."
                    )
                if baseline_landmarks is None:
                    baseline_landmarks = landmarks.detach().clone()
                drift = torch.linalg.norm(
                    landmarks.detach() - baseline_landmarks, dim=-1
                )
                with torch.no_grad():
                    normalized = self.model.normalize_images(image)
                    image_change = (normalized - image).abs().mean()
                row = {
                    "sample_id": sample_id,
                    "step": int(step),
                    "pca_reconstruction_loss": float(
                        reconstruction_loss.detach().item()
                    ),
                    "total_tta_loss": float(reconstruction_loss.detach().item()),
                    "gradient_norm": math.nan,
                    "mean_landmark_drift_px": float(drift.mean().item()),
                    "max_landmark_drift_px": float(drift.max().item()),
                    "mean_absolute_normalizer_change": float(image_change.item()),
                    "failed": False,
                    "failure_message": "",
                }
                sample_rows.append(row)

                if step in self._capture_steps():
                    snapshots.append(
                        {
                            "step": step,
                            "normalized": normalized.detach().cpu().clone(),
                            "landmarks": landmarks.detach().cpu().clone(),
                            "loss": float(reconstruction_loss.detach().item()),
                        }
                    )
                final_outputs = {
                    key: value.detach().clone() for key, value in outputs.items()
                }
                if step == self.config.steps:
                    break

                optimizer.zero_grad(set_to_none=True)
                reconstruction_loss.backward()
                gradient_norm = _gradient_norm(self.model.normalizer.parameters())
                sample_rows[-1]["gradient_norm"] = gradient_norm
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite normalizer gradient at step {step}."
                    )
                optimizer.step()
        except Exception as error:
            failed = True
            failure_message = str(error)
            self.failed_samples += 1
            warnings.warn(
                f"PCA TTA failed for '{sample_id}'; using the unadapted source "
                f"prediction. Reason: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._restore_source_normalizer()
            self.model.eval()
            with torch.inference_mode():
                direct_outputs = self.model(image)
            final_outputs = {
                key: value.detach().clone() for key, value in direct_outputs.items()
            }
            if not sample_rows:
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "step": 0,
                        "pca_reconstruction_loss": math.nan,
                        "total_tta_loss": math.nan,
                        "gradient_norm": math.nan,
                        "mean_landmark_drift_px": 0.0,
                        "max_landmark_drift_px": 0.0,
                        "mean_absolute_normalizer_change": math.nan,
                        "failed": True,
                        "failure_message": failure_message,
                    }
                )
        finally:
            for row in sample_rows:
                row["failed"] = failed
                row["failure_message"] = failure_message
            self.trajectory_rows.extend(sample_rows)
            sample_summary = _summarize_sample_rows(sample_rows)
            self.summary_rows.append(sample_summary)
            if self.processed_samples < self.config.probe_count and snapshots:
                self._save_probe(
                    sample_id=sample_id,
                    image=image.detach().cpu(),
                    baseline_landmarks=baseline_landmarks,
                    snapshots=snapshots,
                )
            self.processed_samples += 1
            self._restore_source_normalizer()
            self.model.eval()
            print(
                f"[PCA-TTA][episode={episode_number:06d}] sample={sample_id} | "
                f"status={'failed' if failed else 'ok'} "
                f"initial_loss={sample_summary['initial_pca_reconstruction_loss']:.6g} "
                f"final_loss={sample_summary['final_pca_reconstruction_loss']:.6g} "
                "source_normalizer_restored_after=yes"
            )

        assert final_outputs is not None
        return final_outputs

    def _capture_steps(self) -> set[int]:
        """Return requested monitoring steps clipped to the episode length."""
        steps = {
            int(step)
            for step in self.config.monitor_steps
            if 0 <= int(step) <= self.config.steps
        }
        steps.update({0, self.config.steps})
        return steps

    def _restore_source_normalizer(self) -> None:
        """Restore the immutable source state before or after an episode."""
        assert self.model.normalizer is not None
        self.model.normalizer.load_state_dict(
            deepcopy(self.source_normalizer_state), strict=True
        )
        for parameter in self.model.normalizer.parameters():
            parameter.requires_grad = True
        current_state = self.model.normalizer.state_dict()
        if any(
            not torch.equal(current_state[key].detach().cpu(), source_value)
            for key, source_value in self.source_normalizer_state.items()
        ):
            raise RuntimeError("Failed to restore the source normalizer state exactly.")

    def _validate_parameter_partition(self) -> None:
        """Fail fast unless only the complete normalizer remains trainable."""
        trainable_landmarker = sum(
            parameter.numel()
            for parameter in self.model.landmarker.parameters()
            if parameter.requires_grad
        )
        frozen_normalizer = sum(
            parameter.numel()
            for parameter in self.model.normalizer.parameters()
            if not parameter.requires_grad
        )
        if trainable_landmarker:
            raise RuntimeError(
                "PCA TTA requires the complete landmarker to be frozen, but "
                f"{trainable_landmarker:,} parameters remain trainable."
            )
        if frozen_normalizer:
            raise RuntimeError(
                "PCA TTA requires the complete normalizer to be trainable, but "
                f"{frozen_normalizer:,} parameters remain frozen."
            )

    def _save_probe(
        self,
        sample_id: str,
        image: torch.Tensor,
        baseline_landmarks: torch.Tensor | None,
        snapshots: list[dict[str, Any]],
    ) -> None:
        """Save fixed-scale crop-space panels for one adapted target image."""
        if baseline_landmarks is None:
            return
        probe_dir = self.output_dir / "probes" / _safe_name(sample_id)
        probe_dir.mkdir(parents=True, exist_ok=True)
        original_rgb = _tensor_to_rgb(
            image[0],
            mean=self.config.normalization_mean,
            std=self.config.normalization_std,
        )
        panel_paths: list[Path] = []
        for snapshot in snapshots:
            normalized_rgb = _tensor_to_rgb(
                snapshot["normalized"][0],
                mean=self.config.normalization_mean,
                std=self.config.normalization_std,
            )
            difference = np.abs(normalized_rgb - original_rgb)
            difference_magnitude = difference.mean(axis=2)
            fixed_difference_rgb = _colorize_difference(
                difference_magnitude,
                display_max=self.config.difference_display_max,
            )
            enhanced_display_max = _robust_difference_display_max(
                difference_magnitude
            )
            enhanced_difference_rgb = _colorize_difference(
                difference_magnitude,
                display_max=enhanced_display_max,
            )
            panel = _build_probe_panel(
                original_rgb=original_rgb,
                normalized_rgb=normalized_rgb,
                fixed_difference_rgb=fixed_difference_rgb,
                enhanced_difference_rgb=enhanced_difference_rgb,
                fixed_display_max=self.config.difference_display_max,
                enhanced_display_max=enhanced_display_max,
                baseline_landmarks=baseline_landmarks[0].cpu(),
                adapted_landmarks=snapshot["landmarks"][0],
                title=(
                    f"{sample_id} | step {snapshot['step']} | "
                    f"PCA loss {snapshot['loss']:.6g} | "
                    f"RGB MAE {difference.mean():.5f} | "
                    f"RGB max {difference.max():.5f}"
                ),
            )
            panel_path = probe_dir / f"step_{int(snapshot['step']):04d}.png"
            panel.save(panel_path)
            panel_paths.append(panel_path)
        if panel_paths:
            frames = [Image.open(path).convert("RGB") for path in panel_paths]
            _stack_vertically(frames).save(probe_dir / "adaptation_grid.png")
            frames[0].save(
                probe_dir / "adaptation.gif",
                save_all=True,
                append_images=frames[1:],
                duration=900,
                loop=0,
            )
            for frame in frames:
                frame.close()

    def finalize(self) -> dict[str, Any]:
        """Write raw, per-image, aggregate, plot, and interpretation artifacts."""
        trajectory_path = self.output_dir / "trajectories.csv"
        summary_path = self.output_dir / "image_summary.csv"
        aggregate_path = self.output_dir / "aggregate_curves.csv"
        _write_rows_csv(trajectory_path, self.trajectory_rows)
        _write_rows_csv(summary_path, self.summary_rows)
        aggregate_rows = _aggregate_trajectories(self.trajectory_rows)
        _write_rows_csv(aggregate_path, aggregate_rows)
        _save_aggregate_plot(
            aggregate_rows,
            self.output_dir / "figures" / "pca_reconstruction_loss_curve.png",
        )
        summary = {
            "method": "episodic_pca_reconstruction_tta",
            "adapted_module": "external_image_normalizer",
            "loss": "pca_reconstruction_loss_only",
            "steps": self.config.steps,
            "learning_rate": self.config.learning_rate,
            "processed_samples": self.processed_samples,
            "failed_samples": self.failed_samples,
            "trajectory_csv": str(trajectory_path),
            "image_summary_csv": str(summary_path),
            "aggregate_curves_csv": str(aggregate_path),
            "probe_dir": str(self.output_dir / "probes"),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (self.output_dir / "README.md").write_text(
            _tta_readme(self.config), encoding="utf-8"
        )
        if self.wandb is not None:
            for row in aggregate_rows:
                self.wandb.log(
                    {
                        "tta/step": int(row["step"]),
                        "tta/pca_loss_mean": float(row["mean"]),
                        "tta/pca_loss_median": float(row["median"]),
                        "tta/pca_loss_p25": float(row["p25"]),
                        "tta/pca_loss_p75": float(row["p75"]),
                    }
                )
            if self.summary_rows:
                columns = list(self.summary_rows[0].keys())
                data = [[row[column] for column in columns] for row in self.summary_rows]
                self.wandb.log(
                    {"tta/image_summary": self.wandb.Table(columns=columns, data=data)}
                )
            probe_grids = sorted((self.output_dir / "probes").glob("*/adaptation_grid.png"))
            if probe_grids:
                self.wandb.log(
                    {
                        "tta/probe_grids": [
                            self.wandb.Image(str(path), caption=path.parent.name)
                            for path in probe_grids
                        ]
                    }
                )
        return summary


def _count_parameters(parameters: Any) -> int:
    """Count scalar parameters from an iterable without modifying it."""
    return sum(parameter.numel() for parameter in parameters)


def _gradient_norm(parameters: Any) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared_norm)


def _summarize_sample_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial = rows[0]
    final = rows[-1]
    finite_losses = [
        float(row["pca_reconstruction_loss"])
        for row in rows
        if math.isfinite(float(row["pca_reconstruction_loss"]))
    ]
    initial_loss = float(initial["pca_reconstruction_loss"])
    final_loss = float(final["pca_reconstruction_loss"])
    relative_reduction = math.nan
    if math.isfinite(initial_loss) and abs(initial_loss) > 1e-12:
        relative_reduction = (initial_loss - final_loss) / initial_loss
    return {
        "sample_id": initial["sample_id"],
        "initial_pca_reconstruction_loss": initial_loss,
        "final_pca_reconstruction_loss": final_loss,
        "minimum_pca_reconstruction_loss": min(finite_losses)
        if finite_losses
        else math.nan,
        "relative_loss_reduction": relative_reduction,
        "final_mean_landmark_drift_px": final["mean_landmark_drift_px"],
        "final_max_landmark_drift_px": final["max_landmark_drift_px"],
        "failed": bool(final["failed"]),
        "failure_message": final["failure_message"],
    }


def _aggregate_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, list[float]] = {}
    for row in rows:
        value = float(row["pca_reconstruction_loss"])
        if not bool(row["failed"]) and math.isfinite(value):
            by_step.setdefault(int(row["step"]), []).append(value)
    aggregate: list[dict[str, Any]] = []
    for step in sorted(by_step):
        values = np.asarray(by_step[step], dtype=np.float64)
        aggregate.append(
            {
                "step": step,
                "count": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
            }
        )
    return aggregate


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_aggregate_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    width, height = 1400, 820
    left, right, top, bottom = 150, 45, 80, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    steps = [int(row["step"]) for row in rows]
    y_values = [
        float(row[key])
        for row in rows
        for key in ("p10", "p25", "median", "p75", "p90")
    ]
    y_min = min(y_values)
    y_max = max(y_values)
    if math.isclose(y_min, y_max):
        y_max = y_min + max(abs(y_min) * 0.05, 1e-12)
    x_min, x_max = min(steps), max(steps)
    if x_min == x_max:
        x_max = x_min + 1

    def point(step: int, value: float) -> tuple[int, int]:
        x_coord = left + int((step - x_min) / (x_max - x_min) * plot_width)
        y_coord = top + int((y_max - value) / (y_max - y_min) * plot_height)
        return x_coord, y_coord

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "PCA-guided test-time adaptation", fill="#102A43")
    for fraction in np.linspace(0.0, 1.0, 6):
        y_coord = top + int(fraction * plot_height)
        value = y_max - fraction * (y_max - y_min)
        draw.line((left, y_coord, left + plot_width, y_coord), fill="#D9E2EC")
        draw.text((8, y_coord - 7), f"{value:.3e}", fill="#486581")
    draw.line((left, top, left, top + plot_height), fill="#243B53", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="#243B53",
        width=2,
    )
    outer = [point(step, float(row["p90"])) for step, row in zip(steps, rows)]
    outer += [
        point(step, float(row["p10"]))
        for step, row in reversed(list(zip(steps, rows)))
    ]
    inner = [point(step, float(row["p75"])) for step, row in zip(steps, rows)]
    inner += [
        point(step, float(row["p25"]))
        for step, row in reversed(list(zip(steps, rows)))
    ]
    if len(outer) >= 3:
        draw.polygon(outer, fill="#D9EAF7")
        draw.polygon(inner, fill="#9CCAE5")
    median_points = [
        point(step, float(row["median"])) for step, row in zip(steps, rows)
    ]
    if len(median_points) >= 2:
        draw.line(median_points, fill="#0B6E99", width=5, joint="curve")
    for step, current_point in zip(steps, median_points):
        draw.ellipse(
            (
                current_point[0] - 4,
                current_point[1] - 4,
                current_point[0] + 4,
                current_point[1] + 4,
            ),
            fill="#0B6E99",
        )
        if step in {x_min, x_max} or step in {1, 5, 10, 20}:
            draw.text(
                (current_point[0] - 6, top + plot_height + 16),
                str(step),
                fill="#486581",
            )
    draw.text(
        (left + plot_width // 2 - 65, height - 52),
        "Episodic TTA step",
        fill="#243B53",
    )
    draw.text((left + 18, top + 16), "Median", fill="#0B6E99")
    draw.rectangle((left + 115, top + 16, left + 145, top + 28), fill="#9CCAE5")
    draw.text((left + 152, top + 15), "P25–P75", fill="#486581")
    draw.rectangle((left + 265, top + 16, left + 295, top + 28), fill="#D9EAF7")
    draw.text((left + 302, top + 15), "P10–P90", fill="#486581")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _tensor_to_rgb(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
    rgb = (tensor.cpu() * std_tensor + mean_tensor).permute(1, 2, 0)
    return np.asarray(rgb.clamp(0, 1).tolist(), dtype=np.float32)


def _draw_landmarks(image: Image.Image, landmarks: torch.Tensor, color: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for x_coord, y_coord in landmarks.tolist():
        radius = 2
        draw.ellipse(
            (
                float(x_coord) - radius,
                float(y_coord) - radius,
                float(x_coord) + radius,
                float(y_coord) + radius,
            ),
            fill=color,
        )
    return image


def _as_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def _robust_difference_display_max(difference: np.ndarray) -> float:
    """Return a per-panel P99 scale that exposes subtle non-zero changes."""
    positive = np.asarray(difference, dtype=np.float32)
    if not np.any(positive > 0.0):
        return 1.0
    return max(float(np.percentile(positive, 99.0)), 1e-8)


def _colorize_difference(
    difference: np.ndarray,
    display_max: float,
) -> np.ndarray:
    """Map scalar absolute RGB differences to a perceptual black-to-red heatmap."""
    if display_max <= 0:
        raise ValueError("display_max must be positive.")
    values = np.clip(
        np.asarray(difference, dtype=np.float32) / float(display_max),
        0.0,
        1.0,
    )
    stops = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.15, 0.55],
            [0.00, 0.75, 0.95],
            [1.00, 0.90, 0.10],
            [0.90, 0.05, 0.02],
        ],
        dtype=np.float32,
    )
    scaled = values * float(len(stops) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    return stops[lower] * (1.0 - fraction) + stops[upper] * fraction


def _build_probe_panel(
    original_rgb: np.ndarray,
    normalized_rgb: np.ndarray,
    fixed_difference_rgb: np.ndarray,
    enhanced_difference_rgb: np.ndarray,
    fixed_display_max: float,
    enhanced_display_max: float,
    baseline_landmarks: torch.Tensor,
    adapted_landmarks: torch.Tensor,
    title: str,
) -> Image.Image:
    original = _as_image(original_rgb)
    normalized = _as_image(normalized_rgb)
    fixed_difference = _as_image(fixed_difference_rgb)
    enhanced_difference = _as_image(enhanced_difference_rgb)
    baseline = _draw_landmarks(original.copy(), baseline_landmarks, "#00FFFF")
    adapted = _draw_landmarks(normalized.copy(), adapted_landmarks, "#FF00FF")
    names = (
        "Input crop",
        "Normalizer output",
        f"Abs. RGB diff (fixed 0–{fixed_display_max:.3g})",
        f"Abs. RGB diff (enhanced P99={enhanced_display_max:.3g})",
        "Step-0 landmarks",
        "Current landmarks",
    )
    tiles = (
        original,
        normalized,
        fixed_difference,
        enhanced_difference,
        baseline,
        adapted,
    )
    width, height = original.size
    header_height = 22
    title_height = 28
    canvas = Image.new(
        "RGB", (len(tiles) * width, height + header_height + title_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill="black")
    for index, (name, tile) in enumerate(zip(names, tiles)):
        x_offset = index * width
        draw.text((x_offset + 4, title_height + 4), name, fill="black")
        canvas.paste(tile, (x_offset, title_height + header_height))
    return canvas


def _stack_vertically(images: list[Image.Image]) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    y_offset = 0
    for image in images:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height
    return canvas


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _tta_readme(config: PCATTAConfig) -> str:
    return f"""# PCA-guided episodic TTA

Only the external image normalizer is updated. The landmarker and the PCA prior
remain frozen. The sole optimization objective is PCA reconstruction loss; no
target ground truth, image regularizer, parameter regularizer, or consistency
loss is used.

- Adaptation steps per image: `{config.steps}`
- Adam learning rate: `{config.learning_rate}`
- Source normalizer and optimizer state are reset before every image.
- `trajectories.csv` contains one row per image and adaptation step.
- `image_summary.csv` contains one row per image.
- `aggregate_curves.csv` and `figures/` summarize the loss distribution.
- `probes/` shows crop-space normalizer and landmark evolution. Final evaluator
  predictions are independently reprojected from crop coordinates to the
  original image by the existing BabyLand/InfAnFace pipeline.
- Every probe contains both a fixed-scale absolute RGB-difference heatmap for
  comparisons across images/steps and a contrast-enhanced P99 heatmap for
  exposing subtle spatial changes. The enhanced panel must not be used to
  compare change magnitude; its numeric P99 scale is printed in the header.

Ground-truth metrics are computed only after adaptation by the standard dataset
evaluator and never feed the optimizer.
"""
