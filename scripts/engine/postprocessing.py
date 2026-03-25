from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def project_landmarks_to_original_size(
    landmarks: torch.Tensor,
    transformed_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Project landmark coordinates from transformed image space back to original size."""
    transformed_height, transformed_width = transformed_size
    original_height, original_width = original_size

    scale_x = original_width / float(transformed_width)
    scale_y = original_height / float(transformed_height)

    projected_landmarks = landmarks.clone()
    projected_landmarks[:, 0] *= scale_x
    projected_landmarks[:, 1] *= scale_y
    return projected_landmarks


def extract_batched_size(
    batched_size: Any,
    sample_index: int,
) -> tuple[int, int]:
    """Extract one `(height, width)` tuple from collated metadata."""
    if isinstance(batched_size, torch.Tensor):
        if batched_size.ndim == 2 and batched_size.shape[1] == 2:
            return int(batched_size[sample_index, 0].item()), int(
                batched_size[sample_index, 1].item()
            )
        raise ValueError(
            f"Unsupported tensor shape for batched size: {tuple(batched_size.shape)}."
        )

    if isinstance(batched_size, Sequence) and not isinstance(
        batched_size, (str, bytes)
    ):
        if len(batched_size) == 2:
            first, second = batched_size

            if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
                if first.ndim == 1 and second.ndim == 1:
                    return int(first[sample_index].item()), int(
                        second[sample_index].item()
                    )

            if isinstance(first, Sequence) and isinstance(second, Sequence):
                return int(first[sample_index]), int(second[sample_index])

        current_value = batched_size[sample_index]

        if isinstance(current_value, torch.Tensor) and current_value.numel() == 2:
            flat_value = current_value.view(-1)
            return int(flat_value[0].item()), int(flat_value[1].item())

        if isinstance(current_value, Sequence) and len(current_value) == 2:
            return int(current_value[0]), int(current_value[1])

    raise ValueError(
        f"Could not parse batched size field of type {type(batched_size)}."
    )
