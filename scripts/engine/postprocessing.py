from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def project_landmarks_between_sizes(
    landmarks: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Project landmark coordinates between two image spaces using size scaling."""
    source_height, source_width = source_size
    target_height, target_width = target_size

    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)

    projected_landmarks = landmarks.clone()
    projected_landmarks[:, 0] *= scale_x
    projected_landmarks[:, 1] *= scale_y
    return projected_landmarks


def project_landmarks_to_original_size(
    landmarks: torch.Tensor,
    transformed_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Project landmark coordinates from transformed image space back to original size."""
    return project_landmarks_between_sizes(
        landmarks=landmarks,
        source_size=transformed_size,
        target_size=original_size,
    )


def apply_homogeneous_transform(
    landmarks: np.ndarray | torch.Tensor,
    transform_matrix: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to a set of `(x, y)` landmark coordinates."""
    landmarks_np = _as_numpy_array(landmarks, dtype=np.float32)
    transform_np = _as_numpy_array(transform_matrix, dtype=np.float32)

    if landmarks_np.ndim != 2 or landmarks_np.shape[1] != 2:
        raise ValueError(
            f"Expected landmarks with shape (N, 2), got {landmarks_np.shape}."
        )
    if transform_np.shape != (3, 3):
        raise ValueError(
            f"Expected transform matrix with shape (3, 3), got {transform_np.shape}."
        )

    homogeneous_landmarks = np.concatenate(
        [
            landmarks_np.astype(np.float32, copy=False),
            np.ones((landmarks_np.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    transformed = homogeneous_landmarks @ transform_np.T
    scale = transformed[:, 2:3]
    scale[scale == 0.0] = 1.0
    return transformed[:, :2] / scale


def _as_numpy_array(
    value: np.ndarray | torch.Tensor,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Convert arrays or tensors to NumPy, with a PyTorch no-NumPy fallback."""
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        try:
            return detached.numpy().astype(dtype, copy=False)
        except RuntimeError:
            return np.asarray(detached.tolist(), dtype=dtype)
    return np.asarray(value, dtype=dtype)


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
