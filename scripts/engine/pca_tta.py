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
from ..utils.visualization import plt as plotting


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
        if self.steps < 0:
            raise ValueError("PCA TTA adaptation steps cannot be negative.")
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
        self.summary_by_sample_id: dict[str, dict[str, Any]] = {}
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
        baseline_outputs: dict[str, torch.Tensor] | None = None
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
                    baseline_outputs = {
                        key: value.detach().clone() for key, value in outputs.items()
                    }
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
            if baseline_outputs is None:
                baseline_outputs = {
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
            self.summary_by_sample_id[sample_id] = sample_summary
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
        assert baseline_outputs is not None
        final_outputs["tta_baseline_heatmaps"] = baseline_outputs["heatmaps"]
        final_outputs["tta_baseline_visibility_logits"] = baseline_outputs[
            "visibility_logits"
        ]
        return final_outputs

    def record_evaluation_metrics(
        self,
        *,
        sample_id: str,
        orientation: str,
        initial_nme_box_gt_valid: float | None,
        final_nme_box_gt_valid: float | None,
        initial_hausdorff_box_gt_valid: float | None,
        final_hausdorff_box_gt_valid: float | None,
        number_of_valid_landmarks: int,
    ) -> None:
        """Attach post-adaptation GT diagnostics without affecting optimization."""
        row = self.summary_by_sample_id.get(str(sample_id))
        if row is None:
            raise KeyError(f"No completed PCA-TTA episode exists for '{sample_id}'.")
        initial_nme = _finite_or_nan(initial_nme_box_gt_valid)
        final_nme = _finite_or_nan(final_nme_box_gt_valid)
        nme_delta = final_nme - initial_nme
        relative_nme_change = (
            nme_delta / initial_nme
            if math.isfinite(initial_nme) and abs(initial_nme) > 1e-12
            else math.nan
        )
        row.update(
            {
                "orientation": str(orientation),
                "number_of_valid_landmarks": int(number_of_valid_landmarks),
                "initial_nme_box_gt_valid": initial_nme,
                "final_nme_box_gt_valid": final_nme,
                "delta_nme_box_gt_valid": nme_delta,
                "relative_nme_change": relative_nme_change,
                "nme_improved": bool(nme_delta < 0.0),
                "initial_hausdorff_box_gt_valid": _finite_or_nan(
                    initial_hausdorff_box_gt_valid
                ),
                "final_hausdorff_box_gt_valid": _finite_or_nan(
                    final_hausdorff_box_gt_valid
                ),
            }
        )

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
        _save_drift_plot(
            aggregate_rows,
            self.output_dir / "figures" / "landmark_drift_curve.png",
        )
        _save_final_adaptation_distribution(
            self.summary_rows,
            self.output_dir / "figures" / "final_adaptation_distribution.png",
        )
        evaluation_analysis = _save_evaluation_analysis(
            self.summary_rows,
            output_dir=self.output_dir,
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
            "evaluation_analysis": evaluation_analysis,
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


def _finite_or_nan(value: float | None) -> float:
    """Convert an optional numeric diagnostic to a finite float or NaN."""
    if value is None:
        return math.nan
    numeric = float(value)
    return numeric if math.isfinite(numeric) else math.nan


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
    initial_by_sample = {
        str(row["sample_id"]): float(row["pca_reconstruction_loss"])
        for row in rows
        if int(row["step"]) == 0
        and not bool(row["failed"])
        and math.isfinite(float(row["pca_reconstruction_loss"]))
    }
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        value = float(row["pca_reconstruction_loss"])
        if not bool(row["failed"]) and math.isfinite(value):
            by_step.setdefault(int(row["step"]), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for step in sorted(by_step):
        step_rows = by_step[step]
        values = np.asarray(
            [float(row["pca_reconstruction_loss"]) for row in step_rows],
            dtype=np.float64,
        )
        relative_reductions = np.asarray(
            [
                (
                    initial_by_sample[str(row["sample_id"])]
                    - float(row["pca_reconstruction_loss"])
                )
                / initial_by_sample[str(row["sample_id"])]
                for row in step_rows
                if str(row["sample_id"]) in initial_by_sample
                and abs(initial_by_sample[str(row["sample_id"])]) > 1e-12
            ],
            dtype=np.float64,
        )
        mean_drifts = np.asarray(
            [float(row["mean_landmark_drift_px"]) for row in step_rows],
            dtype=np.float64,
        )
        max_drifts = np.asarray(
            [float(row["max_landmark_drift_px"]) for row in step_rows],
            dtype=np.float64,
        )
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
                "relative_reduction_mean": float(relative_reductions.mean()),
                "relative_reduction_median": float(np.median(relative_reductions)),
                "relative_reduction_p25": float(
                    np.percentile(relative_reductions, 25)
                ),
                "relative_reduction_p75": float(
                    np.percentile(relative_reductions, 75)
                ),
                "mean_drift_mean_px": float(mean_drifts.mean()),
                "mean_drift_median_px": float(np.median(mean_drifts)),
                "mean_drift_p25_px": float(np.percentile(mean_drifts, 25)),
                "mean_drift_p75_px": float(np.percentile(mean_drifts, 75)),
                "max_drift_median_px": float(np.median(max_drifts)),
                "max_drift_p25_px": float(np.percentile(max_drifts, 25)),
                "max_drift_p75_px": float(np.percentile(max_drifts, 75)),
                "max_drift_p90_px": float(np.percentile(max_drifts, 90)),
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
    """Save a slide-ready absolute-loss and relative-improvement figure."""
    if not rows:
        return
    if plotting is None:
        _save_plotting_unavailable(path, "PCA reconstruction loss")
        return
    plt = plotting
    from matplotlib.ticker import PercentFormatter

    _configure_plot_style(plt)
    steps = np.asarray([int(row["step"]) for row in rows])
    values = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "mean",
            "median",
            "p10",
            "p25",
            "p75",
            "p90",
            "relative_reduction_median",
            "relative_reduction_p25",
            "relative_reduction_p75",
        )
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    loss_axis, reduction_axis = axes
    loss_axis.fill_between(
        steps, values["p10"], values["p90"], color="#CFE8F3", label="P10–P90"
    )
    loss_axis.fill_between(
        steps, values["p25"], values["p75"], color="#79B9D1", label="P25–P75"
    )
    loss_axis.plot(
        steps, values["median"], color="#005F73", linewidth=3, label="Median"
    )
    loss_axis.plot(
        steps,
        values["mean"],
        color="#BB3E03",
        linewidth=2.2,
        linestyle="--",
        label="Mean",
    )
    loss_axis.set_yscale("log")
    loss_axis.set_title("Absolute PCA reconstruction loss")
    loss_axis.set_xlabel("TTA optimization step")
    loss_axis.set_ylabel("PCA reconstruction loss (log scale)")
    loss_axis.legend(loc="best")

    reduction_axis.fill_between(
        steps,
        values["relative_reduction_p25"],
        values["relative_reduction_p75"],
        color="#94D2BD",
        alpha=0.8,
        label="P25–P75",
    )
    reduction_axis.plot(
        steps,
        values["relative_reduction_median"],
        color="#0A9396",
        linewidth=3,
        label="Median reduction",
    )
    reduction_axis.axhline(0.0, color="#52606D", linewidth=1)
    reduction_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    reduction_axis.set_title("Reduction relative to each image at step 0")
    reduction_axis.set_xlabel("TTA optimization step")
    reduction_axis.set_ylabel("Relative PCA-loss reduction")
    reduction_axis.legend(loc="best")

    figure.suptitle("PCA-guided episodic test-time adaptation", fontsize=18, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _configure_plot_style(plt: Any) -> None:
    """Apply a consistent high-legibility style for meeting figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def _save_plotting_unavailable(path: Path, title: str) -> None:
    """Preserve report generation when optional Matplotlib binaries are absent."""
    image = Image.new("RGB", (1600, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 55), title, fill="#102A43")
    draw.text(
        (60, 130),
        "Plot unavailable: install a Matplotlib build compatible with the active NumPy.",
        fill="#486581",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _save_drift_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Plot typical and worst-landmark displacement throughout adaptation."""
    if not rows:
        return
    if plotting is None:
        _save_plotting_unavailable(path, "Landmark drift introduced by TTA")
        return
    plt = plotting

    _configure_plot_style(plt)
    steps = np.asarray([int(row["step"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharex=True)
    definitions = (
        (
            axes[0],
            "mean_drift_median_px",
            "mean_drift_p25_px",
            "mean_drift_p75_px",
            "Mean displacement across 72 landmarks",
        ),
        (
            axes[1],
            "max_drift_median_px",
            "max_drift_p25_px",
            "max_drift_p75_px",
            "Largest single-landmark displacement",
        ),
    )
    for axis, median_key, low_key, high_key, title in definitions:
        low = np.asarray([float(row[low_key]) for row in rows])
        high = np.asarray([float(row[high_key]) for row in rows])
        median = np.asarray([float(row[median_key]) for row in rows])
        axis.fill_between(steps, low, high, color="#A9D6C9", label="P25–P75")
        axis.plot(steps, median, color="#006D77", linewidth=3, label="Median")
        axis.set_title(title)
        axis.set_xlabel("TTA optimization step")
        axis.set_ylabel("Landmark displacement from step 0 (pixels)")
        axis.legend(loc="upper left")
    figure.suptitle("Landmark drift introduced by TTA", fontsize=18, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_final_adaptation_distribution(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Show final loss-reduction and landmark-drift distributions."""
    valid = [row for row in rows if not bool(row.get("failed", False))]
    if not valid:
        return
    if plotting is None:
        _save_plotting_unavailable(path, "Distribution of TTA effects")
        return
    plt = plotting
    from matplotlib.ticker import PercentFormatter

    _configure_plot_style(plt)
    reductions = np.asarray(
        [float(row["relative_loss_reduction"]) for row in valid], dtype=np.float64
    )
    mean_drifts = np.asarray(
        [float(row["final_mean_landmark_drift_px"]) for row in valid],
        dtype=np.float64,
    )
    max_drifts = np.asarray(
        [float(row["final_max_landmark_drift_px"]) for row in valid],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    axes[0].hist(reductions, bins=30, color="#0A9396", alpha=0.85)
    axes[0].axvline(np.median(reductions), color="#9B2226", linewidth=2.5, label="Median")
    axes[0].xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_title("PCA-loss reduction per image")
    axes[0].set_xlabel("Relative reduction from step 0")
    axes[0].set_ylabel("Number of images")
    axes[0].legend()
    axes[1].hist(
        mean_drifts,
        bins=30,
        color="#005F73",
        alpha=0.82,
        label="Mean across landmarks",
    )
    axes[1].hist(
        max_drifts,
        bins=30,
        color="#EE9B00",
        alpha=0.58,
        label="Maximum landmark",
    )
    axes[1].set_title("Final landmark displacement")
    axes[1].set_xlabel("Displacement from step 0 (pixels)")
    axes[1].set_ylabel("Number of images")
    axes[1].legend()
    figure.suptitle("Distribution of TTA effects", fontsize=18, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_evaluation_analysis(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Save GT-only post-hoc analyses after adaptation has fully completed."""
    required = {
        "orientation",
        "initial_nme_box_gt_valid",
        "final_nme_box_gt_valid",
        "delta_nme_box_gt_valid",
    }
    valid = [
        row
        for row in rows
        if required.issubset(row)
        and math.isfinite(float(row["initial_nme_box_gt_valid"]))
        and math.isfinite(float(row["final_nme_box_gt_valid"]))
    ]
    if not valid:
        return None

    figures_dir = output_dir / "figures"
    orientation_rows = _summarize_evaluation_by_orientation(valid)
    orientation_csv = output_dir / "orientation_tta_summary.csv"
    _write_rows_csv(orientation_csv, orientation_rows)
    initial = np.asarray(
        [float(row["initial_nme_box_gt_valid"]) for row in valid], dtype=np.float64
    )
    final = np.asarray(
        [float(row["final_nme_box_gt_valid"]) for row in valid], dtype=np.float64
    )
    delta = final - initial
    pca_reduction = np.asarray(
        [float(row["relative_loss_reduction"]) for row in valid], dtype=np.float64
    )
    correlation = _spearman_correlation(pca_reduction, delta)
    _save_nme_before_after_scatter(
        valid, figures_dir / "nme_before_after_scatter.png"
    )
    _save_pca_vs_nme_scatter(
        valid,
        correlation=correlation,
        path=figures_dir / "pca_loss_reduction_vs_nme_change.png",
    )
    _save_orientation_nme_plot(
        orientation_rows,
        path=figures_dir / "nme_before_after_by_orientation.png",
    )
    _save_orientation_delta_boxplot(
        valid,
        path=figures_dir / "nme_change_by_orientation.png",
    )
    analysis = {
        "num_images": len(valid),
        "num_improved": int((delta < 0.0).sum()),
        "num_worsened": int((delta > 0.0).sum()),
        "num_unchanged": int((delta == 0.0).sum()),
        "mean_initial_nme_box_gt_valid": float(initial.mean()),
        "mean_final_nme_box_gt_valid": float(final.mean()),
        "mean_delta_nme_box_gt_valid": float(delta.mean()),
        "median_delta_nme_box_gt_valid": float(np.median(delta)),
        "spearman_pca_reduction_vs_nme_delta": correlation,
        "orientation_summary_csv": str(orientation_csv),
        "ground_truth_usage": "post_hoc_analysis_only_not_adaptation",
    }
    analysis_path = output_dir / "evaluation_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    analysis["analysis_json"] = str(analysis_path)
    return analysis


def _summarize_evaluation_by_orientation(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate before/after NME and improvement rate by face orientation."""
    orientation_order = ["left", "quarter_left", "frontal", "quarter_right", "right"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["orientation"]), []).append(row)
    result: list[dict[str, Any]] = []
    ordered = [name for name in orientation_order if name in grouped]
    ordered.extend(sorted(set(grouped).difference(ordered)))
    for orientation in ordered:
        current = grouped[orientation]
        initial = np.asarray(
            [float(row["initial_nme_box_gt_valid"]) for row in current]
        )
        final = np.asarray([float(row["final_nme_box_gt_valid"]) for row in current])
        delta = final - initial
        result.append(
            {
                "orientation": orientation,
                "count": int(len(current)),
                "mean_nme_before": float(initial.mean()),
                "mean_nme_after": float(final.mean()),
                "mean_nme_delta": float(delta.mean()),
                "median_nme_before": float(np.median(initial)),
                "median_nme_after": float(np.median(final)),
                "median_nme_delta": float(np.median(delta)),
                "improved_images": int((delta < 0.0).sum()),
                "worsened_images": int((delta > 0.0).sum()),
                "improvement_rate": float((delta < 0.0).mean()),
            }
        )
    return result


def _orientation_colors() -> dict[str, str]:
    return {
        "left": "#5E3C99",
        "quarter_left": "#3288BD",
        "frontal": "#1A9850",
        "quarter_right": "#F6A01A",
        "right": "#D73027",
        "unknown": "#6B7280",
    }


def _save_nme_before_after_scatter(rows: list[dict[str, Any]], path: Path) -> None:
    if plotting is None:
        _save_plotting_unavailable(path, "Per-image localization before and after TTA")
        return
    plt = plotting

    _configure_plot_style(plt)
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    colors = _orientation_colors()
    for orientation in dict.fromkeys(str(row["orientation"]) for row in rows):
        current = [row for row in rows if str(row["orientation"]) == orientation]
        x_values = np.asarray(
            [float(row["initial_nme_box_gt_valid"]) * 100.0 for row in current]
        )
        y_values = np.asarray(
            [float(row["final_nme_box_gt_valid"]) * 100.0 for row in current]
        )
        axis.scatter(
            x_values,
            y_values,
            s=34,
            alpha=0.68,
            color=colors.get(orientation, colors["unknown"]),
            label=orientation.replace("_", " ").title(),
        )
    all_values = np.asarray(
        [
            float(row[key]) * 100.0
            for row in rows
            for key in ("initial_nme_box_gt_valid", "final_nme_box_gt_valid")
        ]
    )
    positive = all_values[all_values > 0]
    lower = max(float(positive.min()) * 0.85, 1e-3)
    upper = float(all_values.max()) * 1.1
    axis.plot([lower, upper], [lower, upper], color="#263238", linestyle="--", linewidth=2)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("NME before TTA (%; GT-valid)")
    axis.set_ylabel("NME after TTA (%; GT-valid)")
    axis.set_title("Per-image localization before and after TTA")
    axis.legend(title="Orientation", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.text(
        0.03,
        0.97,
        "Below diagonal: improved\nAbove diagonal: worsened",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#CBD5E1"},
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_pca_vs_nme_scatter(
    rows: list[dict[str, Any]],
    *,
    correlation: float,
    path: Path,
) -> None:
    if plotting is None:
        _save_plotting_unavailable(path, "PCA loss reduction versus NME change")
        return
    plt = plotting
    from matplotlib.ticker import PercentFormatter

    _configure_plot_style(plt)
    figure, axis = plt.subplots(figsize=(9.2, 6.6))
    colors = _orientation_colors()
    for orientation in dict.fromkeys(str(row["orientation"]) for row in rows):
        current = [row for row in rows if str(row["orientation"]) == orientation]
        axis.scatter(
            [float(row["relative_loss_reduction"]) for row in current],
            [float(row["delta_nme_box_gt_valid"]) * 100.0 for row in current],
            s=34,
            alpha=0.68,
            color=colors.get(orientation, colors["unknown"]),
            label=orientation.replace("_", " ").title(),
        )
    axis.axhline(0.0, color="#263238", linewidth=2, linestyle="--")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_xlabel("Relative PCA reconstruction-loss reduction")
    axis.set_ylabel("NME change after TTA (percentage points)")
    axis.set_title("Does improved PCA plausibility predict improved localization?")
    axis.legend(title="Orientation", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.text(
        0.03,
        0.97,
        f"Spearman ρ = {correlation:.3f}\nNegative ΔNME indicates improvement",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#CBD5E1"},
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_orientation_nme_plot(
    rows: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    if plotting is None:
        _save_plotting_unavailable(path, "TTA performance by orientation")
        return
    plt = plotting

    _configure_plot_style(plt)
    labels = [str(row["orientation"]).replace("_", " ").title() for row in rows]
    positions = np.arange(len(rows))
    width = 0.36
    before = np.asarray([float(row["mean_nme_before"]) * 100.0 for row in rows])
    after = np.asarray([float(row["mean_nme_after"]) * 100.0 for row in rows])
    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    axis.bar(positions - width / 2, before, width, color="#94A3B8", label="Before TTA")
    axis.bar(positions + width / 2, after, width, color="#0A9396", label="After TTA")
    for index, row in enumerate(rows):
        axis.text(
            positions[index],
            max(before[index], after[index]) + max(after.max(), before.max()) * 0.025,
            f"n={int(row['count'])}\n{float(row['improvement_rate']):.0%} improved",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Mean NME (%; GT-valid)")
    axis.set_title("TTA localization performance by face orientation")
    axis.legend()
    axis.set_ylim(0, max(before.max(), after.max()) * 1.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _save_orientation_delta_boxplot(
    rows: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    if plotting is None:
        _save_plotting_unavailable(path, "NME change by orientation")
        return
    plt = plotting

    _configure_plot_style(plt)
    order = ["left", "quarter_left", "frontal", "quarter_right", "right"]
    present = [name for name in order if any(str(row["orientation"]) == name for row in rows)]
    present.extend(
        sorted(
            {str(row["orientation"]) for row in rows}.difference(present)
        )
    )
    values = [
        [
            float(row["delta_nme_box_gt_valid"]) * 100.0
            for row in rows
            if str(row["orientation"]) == orientation
        ]
        for orientation in present
    ]
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    box = axis.boxplot(values, patch_artist=True, showfliers=False)
    colors = _orientation_colors()
    for patch, orientation in zip(box["boxes"], present):
        patch.set_facecolor(colors.get(orientation, colors["unknown"]))
        patch.set_alpha(0.72)
    axis.axhline(0.0, color="#263238", linewidth=2, linestyle="--")
    axis.set_xticks(
        np.arange(1, len(present) + 1),
        [name.replace("_", " ").title() for name in present],
    )
    axis.set_ylabel("NME change after TTA (percentage points)")
    axis.set_title("Distribution of per-image TTA effect by orientation")
    axis.text(
        0.02,
        0.97,
        "Negative values indicate improvement",
        transform=axis.transAxes,
        va="top",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _spearman_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute Spearman correlation with average ranks for tied values."""
    if len(x_values) < 2 or len(y_values) != len(x_values):
        return math.nan
    x_rank = _average_ranks(np.asarray(x_values, dtype=np.float64))
    y_rank = _average_ranks(np.asarray(y_values, dtype=np.float64))
    if np.std(x_rank) <= 0 or np.std(y_rank) <= 0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


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
- `image_summary.csv` additionally contains before/after GT-valid NME and
  Hausdorff diagnostics when the dataset evaluator provides ground truth after
  adaptation. These values never participate in optimization.
- `aggregate_curves.csv` and `figures/` summarize absolute loss, relative loss
  reduction, landmark drift, and the distribution of final adaptation effects.
- `orientation_tta_summary.csv` and `evaluation_analysis.json` summarize NME
  before/after, improvement rates, and the Spearman association between PCA-loss
  reduction and NME change.
- `figures/nme_before_after_scatter.png`,
  `figures/pca_loss_reduction_vs_nme_change.png`, and the orientation figures
  are generated only when post-hoc ground truth metrics are available.
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
