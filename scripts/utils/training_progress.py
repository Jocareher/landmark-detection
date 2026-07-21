from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, TextIO


def format_metric(value: Any, precision: int = 6) -> str:
    """Format an optional scalar metric for terminal presentation.

    Args:
        value: Scalar value to format. Missing values produce ``N/A``.
        precision: Number of digits after the decimal point.

    Returns:
        A stable plain-text representation suitable for TTY and log files.
    """
    if precision < 0:
        raise ValueError("precision must be non-negative.")
    if value is None:
        return "N/A"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(numeric_value):
        return "non-finite"
    return f"{numeric_value:.{precision}f}"


def format_duration(seconds: float | None) -> str:
    """Format an optional duration as ``HH:MM:SS`` or ``MM:SS``."""
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds < 0:
        return "N/A"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


class TrainingProgressReporter:
    """Render training progress with Rich and a non-TTY plain-text fallback.

    The reporter owns presentation only. It never changes model state,
    optimization, checkpoints, metrics, or W&B logging. Interactive terminals
    use Rich when it is installed; redirected SLURM logs receive one stable
    plain-text line per completed phase and no ANSI control sequences.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        force_interactive: bool | None = None,
        enable_rich: bool = True,
    ) -> None:
        """Initialize terminal capability detection and rendering state.

        Args:
            stream: Output stream. Defaults to ``sys.stdout``.
            force_interactive: Test or application override for TTY detection.
            enable_rich: Allow Rich rendering when available and interactive.
        """
        self.stream = stream or sys.stdout
        stream_is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interactive = (
            stream_is_tty if force_interactive is None else bool(force_interactive)
        )
        self.rich_enabled = False
        self.console: Any | None = None
        self.progress: Any | None = None
        self.tqdm_progress: Any | None = None
        self.task_id: Any | None = None
        self.phase: str | None = None
        self.epoch = 0
        self.total_epochs = 0
        self.run_started_at: float | None = None
        self._current_lr = 0.0
        if self.interactive and enable_rich:
            try:
                from rich.console import Console

                self.console = Console(
                    file=self.stream,
                    force_terminal=True,
                    color_system="auto",
                    highlight=False,
                )
                self.rich_enabled = True
            except ImportError:
                self.rich_enabled = False

    def start_run(
        self,
        *,
        run_name: str | None,
        device: str,
        device_name: str,
        train_samples: int,
        validation_samples: int,
        batch_size: int | None,
        epochs: int,
        optimizer_name: str,
        learning_rate: float,
        wandb_project: str | None,
        wandb_url: str | None,
        checkpoint_dir: str | Path,
    ) -> None:
        """Print the run header and start wall-clock timing."""
        if train_samples < 0 or validation_samples < 0:
            raise ValueError("Dataset sample counts cannot be negative.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        self.run_started_at = time.monotonic()
        metadata = [
            ("Run", run_name or "unnamed"),
            ("Device", f"{device} ({device_name})"),
            ("Samples", f"train={train_samples}, validation={validation_samples}"),
            ("Batch size", str(batch_size) if batch_size is not None else "N/A"),
            ("Epochs", str(epochs)),
            ("Optimizer", optimizer_name),
            ("Initial learning rate", f"{learning_rate:.6g}"),
            ("W&B project", wandb_project or "disabled"),
            ("W&B URL", wandb_url or "unavailable"),
            ("Checkpoints", str(checkpoint_dir)),
        ]
        if self.rich_enabled:
            from rich import box
            from rich.table import Table

            table = Table(title="Training run", box=box.SIMPLE, show_header=False)
            table.add_column("Field", style="dim")
            table.add_column("Value")
            for key, value in metadata:
                table.add_row(key, value)
            self.console.print(table)
            return
        self._plain("Training run")
        for key, value in metadata:
            self._plain(f"  {key:<22} {value}")

    def start_epoch(
        self,
        epoch: int,
        total_epochs: int,
        learning_rate: float,
    ) -> None:
        """Record epoch context and print a non-TTY epoch marker."""
        if epoch <= 0 or total_epochs <= 0 or epoch > total_epochs:
            raise ValueError("Epoch must be within [1, total_epochs].")
        self.epoch = int(epoch)
        self.total_epochs = int(total_epochs)
        self._current_lr = float(learning_rate)
        if not self.interactive:
            self._plain(f"Epoch {epoch}/{total_epochs}")

    def start_train(self, total_batches: int) -> None:
        """Start the TRAIN progress display for the current epoch."""
        self._start_phase("TRAIN", total_batches, "cyan")

    def update_train_batch(
        self,
        *,
        total_loss: float,
        nme: float | None,
        visible_nme: float | None,
        learning_rate: float,
    ) -> None:
        """Update TRAIN with running epoch averages."""
        self._update_phase(total_loss, nme, visible_nme, learning_rate)

    def finish_train(self, metrics: dict[str, float]) -> None:
        """Finish TRAIN and emit a stable line in non-TTY mode."""
        self._finish_phase("TRAIN", metrics)

    def start_validation(self, total_batches: int) -> None:
        """Start the VALIDATION progress display for the current epoch."""
        self._start_phase("VALIDATION", total_batches, "magenta")

    def update_validation_batch(
        self,
        *,
        total_loss: float,
        nme: float | None,
        visible_nme: float | None,
        learning_rate: float,
    ) -> None:
        """Update VALIDATION with running epoch averages."""
        self._update_phase(total_loss, nme, visible_nme, learning_rate)

    def finish_validation(self, metrics: dict[str, float]) -> None:
        """Finish VALIDATION and emit a stable line in non-TTY mode."""
        self._finish_phase("VALIDATION", metrics)

    def report_checkpoint(self, checkpoint_path: str | Path, is_best: bool) -> None:
        """Report a successful checkpoint save."""
        label = "Best checkpoint saved" if is_best else "Checkpoint saved"
        self._message(
            f"{label}: {checkpoint_path}", style="green" if is_best else "dim"
        )

    def report_learning_rate_change(self, previous: float, current: float) -> None:
        """Report a scheduler-induced learning-rate change."""
        if math.isclose(float(previous), float(current), rel_tol=0.0, abs_tol=0.0):
            return
        self._message(
            f"Learning rate changed: {previous:.6g} -> {current:.6g}",
            style="yellow",
        )

    def warning(self, message: str) -> None:
        """Print a warning without suppressing or redirecting other output."""
        if not message:
            raise ValueError("Warning message cannot be empty.")
        self._message(f"Warning: {message}", style="yellow")

    def error(self, message: str) -> None:
        """Print an error-level presentation message."""
        if not message:
            raise ValueError("Error message cannot be empty.")
        self._message(f"Error: {message}", style="red")

    def info(self, message: str) -> None:
        """Print secondary run information."""
        if message:
            self._message(message, style="dim")

    def finish_epoch(
        self,
        *,
        train_metrics: dict[str, float],
        validation_metrics: dict[str, float],
        learning_rate: float,
        best_validation_nme: float,
        best_nme_epoch: int,
        checkpoint_improved: bool,
        early_stopping_counter: int,
        patience: int,
    ) -> None:
        """Print an aligned epoch summary and checkpoint/NME status."""
        metric_rows = [
            ("Total loss", "total_loss"),
            ("Full landmark loss", "full_landmark_loss"),
            ("Image L1 loss", "image_l1_loss"),
            ("Image TV loss", "image_tv_loss"),
            ("NME", "nme"),
            ("PCA loss", "pca_loss"),
            ("Visibility loss", "visibility_loss"),
            ("Visible landmark loss", "visible_landmark_loss"),
            ("Visible-landmark NME", "visible_nme"),
        ]
        epoch_seconds = float(train_metrics.get("epoch_time", 0.0)) + float(
            validation_metrics.get("epoch_time", 0.0)
        )
        if self.rich_enabled:
            from rich import box
            from rich.table import Table

            table = Table(
                title=f"Epoch {self.epoch}/{self.total_epochs}",
                box=box.SIMPLE_HEAVY,
            )
            table.add_column("Metric")
            table.add_column("Train", justify="right", style="cyan")
            table.add_column("Validation", justify="right", style="magenta")
            for label, key in metric_rows:
                table.add_row(
                    label,
                    format_metric(train_metrics.get(key)),
                    format_metric(validation_metrics.get(key)),
                )
            table.add_row(
                "Learning rate", f"{learning_rate:.6g}", f"{learning_rate:.6g}"
            )
            table.add_row(
                "Epoch time",
                format_duration(epoch_seconds),
                format_duration(epoch_seconds),
            )
            self.console.print(table)
        else:
            self._plain(f"{'Metric':<26} {'Train':>14} {'Validation':>14}")
            self._plain("-" * 56)
            for label, key in metric_rows:
                self._plain(
                    f"{label:<26} "
                    f"{format_metric(train_metrics.get(key)):>14} "
                    f"{format_metric(validation_metrics.get(key)):>14}"
                )
            self._plain(f"{'Learning rate':<26} {learning_rate:>14.6g}")
            self._plain(f"{'Epoch time':<26} {format_duration(epoch_seconds):>14}")
        improvement_text = (
            "current epoch improved the best checkpoint"
            if checkpoint_improved
            else "current epoch did not improve the best checkpoint"
        )
        self._message(
            f"Best validation NME: {format_metric(best_validation_nme)} "
            f"(epoch {best_nme_epoch}); {improvement_text}.",
            style="green" if checkpoint_improved else "dim",
        )
        counter_style = "yellow" if early_stopping_counter > 0 else "dim"
        self._message(
            f"Early stopping: {early_stopping_counter}/{patience}",
            style=counter_style,
        )

    def finish_run(
        self,
        *,
        best_epoch: int,
        best_validation_nme: float,
        final_train_nme: float | None,
        final_validation_nme: float | None,
        final_train_pca_loss: float | None,
        final_validation_pca_loss: float | None,
        best_checkpoint_path: str | Path,
        wandb_url: str | None,
    ) -> None:
        """Print the compact end-of-training summary."""
        total_time = (
            time.monotonic() - self.run_started_at
            if self.run_started_at is not None
            else None
        )
        rows = [
            ("Best checkpoint epoch", str(best_epoch)),
            ("Best validation NME", format_metric(best_validation_nme)),
            ("Final training NME", format_metric(final_train_nme)),
            ("Final validation NME", format_metric(final_validation_nme)),
            ("Final training PCA loss", format_metric(final_train_pca_loss)),
            (
                "Final validation PCA loss",
                format_metric(final_validation_pca_loss),
            ),
            ("Total training time", format_duration(total_time)),
            ("Best checkpoint", str(best_checkpoint_path)),
            ("W&B URL", wandb_url or "unavailable"),
        ]
        if self.rich_enabled:
            from rich import box
            from rich.table import Table

            table = Table(title="Training complete", box=box.SIMPLE, show_header=False)
            table.add_column("Field", style="dim")
            table.add_column("Value")
            for label, value in rows:
                table.add_row(label, value)
            self.console.print(table)
            return
        self._plain("Training complete")
        for label, value in rows:
            self._plain(f"  {label:<24} {value}")

    def _start_phase(self, phase: str, total_batches: int, color: str) -> None:
        """Create one interactive Rich progress task."""
        if total_batches < 0:
            raise ValueError("total_batches cannot be negative.")
        self.phase = phase
        if not self.interactive:
            return
        if not self.rich_enabled:
            from tqdm.auto import tqdm

            self.tqdm_progress = tqdm(
                total=total_batches,
                desc=f"{phase} {self.epoch}/{self.total_epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
                file=self.stream,
            )
            return
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        self.progress = Progress(
            TextColumn(
                "[{task.fields[color]}]{task.fields[phase]}[/] "
                "{task.fields[epoch]}/{task.fields[total_epochs]} "
                "lr={task.fields[lr]:.3g} "
                "loss={task.fields[loss]} "
                "NME={task.fields[nme]} "
                "vis-NME={task.fields[visible_nme]}",
                justify="right",
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            console=self.console,
            transient=False,
            auto_refresh=True,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            phase,
            total=total_batches,
            phase=phase,
            color=color,
            epoch=self.epoch,
            total_epochs=self.total_epochs,
            lr=self._current_lr,
            loss="N/A",
            nme="N/A",
            visible_nme="N/A",
        )

    def _update_phase(
        self,
        total_loss: float,
        nme: float | None,
        visible_nme: float | None,
        learning_rate: float,
    ) -> None:
        """Advance an active interactive task by one batch."""
        if not self.interactive:
            return
        if not self.rich_enabled:
            if self.tqdm_progress is None:
                raise RuntimeError(
                    "A progress phase must be started before updating it."
                )
            self.tqdm_progress.update(1)
            self.tqdm_progress.set_postfix(
                loss=format_metric(total_loss, 4),
                nme=format_metric(nme, 4),
                visible_nme=format_metric(visible_nme, 4),
                lr=f"{learning_rate:.3g}",
                refresh=True,
            )
            return
        if self.progress is None or self.task_id is None:
            raise RuntimeError("A progress phase must be started before updating it.")
        self.progress.update(
            self.task_id,
            advance=1,
            lr=float(learning_rate),
            loss=format_metric(total_loss, precision=4),
            nme=format_metric(nme, precision=4),
            visible_nme=format_metric(visible_nme, precision=4),
        )

    def _finish_phase(self, expected_phase: str, metrics: dict[str, float]) -> None:
        """Close an interactive progress task or print a plain phase summary."""
        if self.phase != expected_phase:
            raise RuntimeError(
                f"Cannot finish {expected_phase}; active phase is {self.phase!r}."
            )
        if self.rich_enabled:
            if self.progress is not None:
                self.progress.stop()
            self.progress = None
            self.task_id = None
        elif self.interactive:
            if self.tqdm_progress is not None:
                self.tqdm_progress.close()
            self.tqdm_progress = None
        else:
            self._plain(
                f"{expected_phase:<10} "
                f"loss={format_metric(metrics.get('total_loss'), 4)} "
                f"NME={format_metric(metrics.get('nme'), 4)} "
                f"visible-NME={format_metric(metrics.get('visible_nme'), 4)} "
                f"time={format_duration(metrics.get('epoch_time'))}"
            )
        self.phase = None

    def _message(self, message: str, style: str) -> None:
        """Render one styled or plain status message."""
        if self.rich_enabled:
            self.console.print(message, style=style)
        else:
            self._plain(message)

    def _plain(self, message: str) -> None:
        """Write one flushed plain-text line without ANSI control sequences."""
        print(message, file=self.stream, flush=True)
