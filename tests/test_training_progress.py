from __future__ import annotations

import io

import torch

from scripts.engine.train import _compute_visible_box_normalized_nme
from scripts.utils.training_progress import (
    TrainingProgressReporter,
    format_duration,
    format_metric,
)


def _metrics(**overrides: float) -> dict[str, float]:
    """Return a complete minimal metric dictionary for reporter tests."""
    metrics = {
        "total_loss": 0.02,
        "full_landmark_loss": 0.001,
        "image_l1_loss": 0.0,
        "image_tv_loss": 0.0,
        "nme": 0.1,
        "pca_loss": 0.0,
        "visibility_loss": 0.01,
        "visible_landmark_loss": 0.002,
        "visible_nme": 0.08,
        "epoch_time": 2.0,
    }
    metrics.update(overrides)
    return metrics


def test_metric_and_duration_formatting() -> None:
    assert format_metric(0.1234567, precision=4) == "0.1235"
    assert format_metric(None) == "N/A"
    assert format_metric(float("nan")) == "non-finite"
    assert format_metric(float("inf")) == "non-finite"
    assert format_duration(272.0) == "04:32"


def test_visible_nme_uses_only_visible_landmark_errors() -> None:
    targets = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]])
    predictions = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [20.0, 20.0]]])
    visibility = torch.tensor([[1.0, 1.0, 0.0]])

    visible_nme = _compute_visible_box_normalized_nme(
        predictions,
        targets,
        visibility,
    )

    torch.testing.assert_close(visible_nme, torch.tensor([0.5]))


def test_non_tty_mode_is_plain_and_handles_missing_metrics() -> None:
    stream = io.StringIO()
    reporter = TrainingProgressReporter(
        stream=stream,
        force_interactive=False,
    )
    reporter.start_run(
        run_name="plain-test",
        device="cpu",
        device_name="CPU",
        train_samples=8,
        validation_samples=4,
        batch_size=2,
        epochs=1,
        optimizer_name="Adam",
        learning_rate=1e-4,
        wandb_project=None,
        wandb_url=None,
        checkpoint_dir="runs/plain-test",
    )
    reporter.start_epoch(1, 1, 1e-4)
    reporter.start_train(1)
    reporter.update_train_batch(
        total_loss=0.02,
        nme=None,
        visible_nme=None,
        learning_rate=1e-4,
    )
    reporter.finish_train({"total_loss": 0.02, "epoch_time": 1.0})
    reporter.start_validation(1)
    reporter.update_validation_batch(
        total_loss=0.03,
        nme=0.1,
        visible_nme=0.08,
        learning_rate=1e-4,
    )
    reporter.finish_validation(_metrics(total_loss=0.03))
    reporter.finish_epoch(
        train_metrics={"total_loss": 0.02, "epoch_time": 1.0},
        validation_metrics=_metrics(total_loss=0.03),
        learning_rate=1e-4,
        best_validation_nme=0.1,
        best_nme_epoch=1,
        checkpoint_improved=False,
        early_stopping_counter=1,
        patience=15,
    )

    output = stream.getvalue()
    assert "\x1b[" not in output
    assert "TRAIN" in output
    assert "VALIDATION" in output
    assert "N/A" in output
    assert "Early stopping: 1/15" in output


def test_tty_mode_and_best_checkpoint_highlighting() -> None:
    stream = io.StringIO()
    reporter = TrainingProgressReporter(
        stream=stream,
        force_interactive=True,
    )
    reporter.start_epoch(1, 1, 1e-4)
    reporter.start_train(1)
    reporter.update_train_batch(
        total_loss=0.02,
        nme=0.1,
        visible_nme=0.08,
        learning_rate=1e-4,
    )
    reporter.finish_train(_metrics())
    reporter.report_checkpoint("runs/test/best_model.pth", is_best=True)
    reporter.finish_epoch(
        train_metrics=_metrics(),
        validation_metrics=_metrics(nme=0.09),
        learning_rate=1e-4,
        best_validation_nme=0.09,
        best_nme_epoch=1,
        checkpoint_improved=True,
        early_stopping_counter=0,
        patience=15,
    )

    output = stream.getvalue()
    assert "TRAIN" in output
    assert "Best checkpoint saved" in output
    assert "current epoch improved" in output
    assert "checkpoint." in output
