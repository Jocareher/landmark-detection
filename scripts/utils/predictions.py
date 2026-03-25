from __future__ import annotations

from pathlib import Path

import numpy as np


def save_prediction_file(
    output_path: Path,
    landmarks: np.ndarray,
    visibility: np.ndarray,
) -> None:
    """Save predicted landmarks and visibility to a plain text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for landmark_index in range(landmarks.shape[0]):
            x_coord = float(landmarks[landmark_index, 0])
            y_coord = float(landmarks[landmark_index, 1])
            visibility_value = int(visibility[landmark_index])
            file.write(f"{x_coord:.6f} {y_coord:.6f} {visibility_value}\n")
