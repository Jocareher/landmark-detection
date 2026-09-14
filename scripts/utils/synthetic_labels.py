from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SYNTHETIC_YAW_ANGLES = (-75, -65, -55, -45, -30, -15, 0, 15, 30, 45, 55, 65, 75)
UNKNOWN_SYNTHETIC_YAW_GROUP = "unknown_yaw"


@dataclass(frozen=True)
class SyntheticLandmarkLabel:
    """Parsed synthetic landmark label file."""

    landmarks: np.ndarray
    visibility: np.ndarray
    yaw_angle: float | None
    yaw_group: str


def format_synthetic_yaw_group(yaw_angle: float | int | None) -> str:
    """Format a synthetic yaw angle as a signed degree label."""
    if yaw_angle is None:
        return UNKNOWN_SYNTHETIC_YAW_GROUP
    yaw_value = float(yaw_angle)
    if yaw_value.is_integer():
        yaw_text = f"{int(yaw_value):+d}" if yaw_value > 0 else f"{int(yaw_value):d}"
    else:
        yaw_text = f"{yaw_value:+g}" if yaw_value > 0 else f"{yaw_value:g}"
    return f"{yaw_text}\N{DEGREE SIGN}"


def _parse_yaw_angle(raw_line: str, label_path: Path) -> float:
    tokens = raw_line.split()
    if len(tokens) != 1:
        raise ValueError(
            f"Expected one yaw_angle token in first line of {label_path}, got: {raw_line!r}."
        )
    try:
        yaw_angle = float(tokens[0])
    except ValueError as error:
        raise ValueError(
            f"Expected numeric yaw_angle in first line of {label_path}, got {tokens[0]!r}."
        ) from error
    return yaw_angle


def parse_synthetic_landmark_label(
    label_path: str | Path,
    expected_num_landmarks: int,
) -> SyntheticLandmarkLabel:
    """Parse a synthetic landmark label file.

    Supported formats:

    New format:
        yaw_angle
        x1 y1 v1
        ...

    Legacy format:
        x1 y1 v1
        ...
    """
    label_path = Path(label_path)
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty synthetic label file: {label_path}")

    first_tokens = lines[0].split()
    if len(first_tokens) == 1:
        yaw_angle = _parse_yaw_angle(lines[0], label_path)
        landmark_lines = lines[1:]
    elif len(first_tokens) == 3:
        yaw_angle = None
        landmark_lines = lines
    else:
        raise ValueError(
            f"Could not parse first line of {label_path}. Expected either "
            "a one-value yaw_angle header or a three-value landmark row."
        )

    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(
        landmark_lines,
        start=(2 if yaw_angle is not None else 1),
    ):
        tokens = raw_line.split()
        if len(tokens) != 3:
            raise ValueError(
                f"Invalid landmark row in {label_path} at line {line_number}. "
                f"Expected 'x y visibility', got: {raw_line!r}."
            )
        rows.append([float(token) for token in tokens])

    data = np.asarray(rows, dtype=np.float32)
    expected_shape = (expected_num_landmarks, 3)
    if data.shape != expected_shape:
        raise ValueError(
            f"Invalid label shape in '{label_path}'. Expected {expected_shape}, got {data.shape}."
        )

    visibility = data[:, 2].astype(np.float32, copy=True)
    invalid_visibility = ~np.isin(visibility, [0.0, 1.0])
    if invalid_visibility.any():
        invalid_values = sorted(
            {float(value) for value in visibility[invalid_visibility]}
        )
        raise ValueError(
            f"Invalid visibility values in '{label_path}': {invalid_values}. "
            "Expected only 0 or 1."
        )

    return SyntheticLandmarkLabel(
        landmarks=data[:, :2].astype(np.float32, copy=True),
        visibility=visibility,
        yaw_angle=yaw_angle,
        yaw_group=format_synthetic_yaw_group(yaw_angle),
    )
