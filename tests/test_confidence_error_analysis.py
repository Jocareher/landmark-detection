from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from scripts.engine.confidence_error_analysis import _build_sample_rows


def test_invalid_gt_landmarks_are_kept_but_excluded_from_error_metrics() -> None:
    """BabyLand invalid GT landmarks should not contribute to NME statistics."""
    target_landmarks = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    predicted_landmarks = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [2.0, 4.0],
            [9.0, 9.0],
        ],
        dtype=np.float32,
    )
    heatmap_metrics = SimpleNamespace(
        heatmap_max=torch.ones(1, 4),
        heatmap_entropy=torch.ones(1, 4),
        heatmap_variance=torch.ones(1, 4),
        peak_sharpness=torch.ones(1, 4),
    )
    rows, image_row = _build_sample_rows(
        sample={
            "image_id": "sample",
            "image_path": "sample.png",
            "prediction_id": "sample",
            "predicted_landmarks": predicted_landmarks,
            "target_landmarks": target_landmarks,
            "tta_variance": None,
            "pose": "frontal",
        },
        target_visibility=np.array([1, 1, 1, 0], dtype=np.int64),
        predicted_visibility=np.array([1, 1, 1, 1], dtype=np.int64),
        heatmap_metrics=heatmap_metrics,
        sample_index=0,
        pca_error=None,
    )

    assert len(rows) == 4
    assert image_row["total_landmarks"] == 4
    assert image_row["number_of_valid_landmarks"] == 3
    assert image_row["number_of_invalid_landmarks"] == 1
    assert image_row["number_of_nan_target_landmarks"] == 1
    assert rows[3]["gt_valid_for_error"] is False
    assert np.isnan(rows[3]["pixel_error"])
    assert np.isnan(rows[3]["normalized_error"])

    normalization = np.sqrt(4.0)
    expected_errors = np.array([0.0, 1.0 / normalization, 2.0 / normalization])
    assert np.isclose(image_row["mean_nme"], expected_errors.mean())
    assert np.isclose(image_row["mean_nme_percent"], expected_errors.mean() * 100.0)
