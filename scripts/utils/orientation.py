from __future__ import annotations

import re
from typing import Any

ORIENTATION_CLASS_TO_NAME = {
    0: "left",
    1: "quarter_left",
    2: "frontal",
    3: "quarter_right",
    4: "right",
}

ORIENTATION_ORDER = [
    "left",
    "quarter_left",
    "frontal",
    "quarter_right",
    "right",
]

_CANONICAL_ORIENTATION_NAMES = {
    "left": "left",
    "quarter_left": "quarter_left",
    "quarter left": "quarter_left",
    "frontal": "frontal",
    "front": "frontal",
    "quarter_right": "quarter_right",
    "quarter right": "quarter_right",
    "right": "right",
}


def normalize_orientation_label(value: Any) -> str | None:
    """Normalize orientation class labels to repository pose names.

    The SOTA exports sometimes encode pose classes as strings that look like yaw
    angles, for example ``yaw_plus_0deg``. In those files the trailing value is a
    class index, not a degree value.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    lower = raw.lower()
    if lower in _CANONICAL_ORIENTATION_NAMES:
        return _CANONICAL_ORIENTATION_NAMES[lower]

    yaw_class_match = re.fullmatch(r"yaw_(?:plus|minus)_([0-4])deg", lower)
    if yaw_class_match:
        return ORIENTATION_CLASS_TO_NAME[int(yaw_class_match.group(1))]

    try:
        numeric = int(float(raw))
    except ValueError:
        return None
    return ORIENTATION_CLASS_TO_NAME.get(numeric)
