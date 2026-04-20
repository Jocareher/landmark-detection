from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


NATURAL_CLASS_ID_TO_ORIENTATION = {
    0: "left",
    1: "quarter_left",
    2: "frontal",
    3: "quarter_right",
    4: "right",
}
NATURAL_ORIENTATION_NAMES = [
    "left",
    "quarter_left",
    "frontal",
    "quarter_right",
    "right",
]
UNKNOWN_ORIENTATION = "unknown"


@dataclass(frozen=True)
class NaturalLandmarkLabel:
    """Parsed natural-image label file."""

    landmarks: np.ndarray
    visibility: np.ndarray
    class_idx: int | None
    orientation: str


def orientation_from_class_idx(class_idx: int | None) -> str:
    """Map a natural-image class id to the repository orientation name."""
    if class_idx is None:
        return UNKNOWN_ORIENTATION
    if class_idx not in NATURAL_CLASS_ID_TO_ORIENTATION:
        raise ValueError(
            f"Unsupported natural orientation class_idx={class_idx}. "
            f"Expected one of {sorted(NATURAL_CLASS_ID_TO_ORIENTATION)}."
        )
    return NATURAL_CLASS_ID_TO_ORIENTATION[class_idx]


def _parse_class_idx(raw_line: str, label_path: Path) -> int:
    tokens = raw_line.split()
    if len(tokens) != 1:
        raise ValueError(
            f"Expected one class_idx token in first line of {label_path}, got: {raw_line!r}."
        )
    class_value = float(tokens[0])
    class_idx = int(class_value)
    if class_value != float(class_idx):
        raise ValueError(
            f"Expected integer class_idx in first line of {label_path}, got {tokens[0]!r}."
        )
    orientation_from_class_idx(class_idx)
    return class_idx


def parse_natural_landmark_label(
    label_path: str | Path,
    expected_num_landmarks: int,
) -> NaturalLandmarkLabel:
    """Parse a natural-image landmark label file.

    Supported formats:

    New format:
        class_idx
        x1 y1 v1
        ...

    Legacy format:
        x1 y1 v1
        ...

    The legacy format is accepted only when the first non-empty line has three
    numeric columns, which avoids silently interpreting a class header as a landmark.
    """
    label_path = Path(label_path)
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Empty natural label file: {label_path}")

    first_tokens = lines[0].split()
    if len(first_tokens) == 1:
        class_idx = _parse_class_idx(lines[0], label_path)
        landmark_lines = lines[1:]
    elif len(first_tokens) == 3:
        class_idx = None
        landmark_lines = lines
    else:
        raise ValueError(
            f"Could not parse first line of {label_path}. Expected either "
            "a one-value class_idx header or a three-value landmark row."
        )

    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(
        landmark_lines,
        start=(2 if class_idx is not None else 1),
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

    return NaturalLandmarkLabel(
        landmarks=data[:, :2].astype(np.float32, copy=True),
        visibility=visibility,
        class_idx=class_idx,
        orientation=orientation_from_class_idx(class_idx),
    )
