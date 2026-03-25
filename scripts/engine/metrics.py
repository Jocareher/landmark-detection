from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AverageMeter:
    """Track running averages for scalar metrics."""

    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        """Clear the meter state."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Accumulate a new value observed over `n` samples."""
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def get_preds_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """Extract argmax landmark coordinates from a batch of heatmaps."""
    if heatmaps.ndim != 4:
        raise ValueError(
            f"Expected heatmaps with shape (B, K, H, W), got {tuple(heatmaps.shape)}."
        )

    batch_size, num_landmarks, _, heatmap_width = heatmaps.shape
    flattened = heatmaps.view(batch_size, num_landmarks, -1)
    max_indices = flattened.argmax(dim=2)
    pred_x = (max_indices % heatmap_width).float()
    pred_y = torch.div(max_indices, heatmap_width, rounding_mode="floor").float()
    return torch.stack([pred_x, pred_y], dim=-1)


def decode_heatmaps_to_image_coords(
    heatmaps: torch.Tensor,
    image_height: int,
    image_width: int,
    use_subpixel: bool = True,
) -> torch.Tensor:
    """Map heatmap-space coordinates back into image-space pixel coordinates."""
    preds = get_preds_from_heatmaps(heatmaps)
    batch_size, num_landmarks, heatmap_height, heatmap_width = heatmaps.shape

    if use_subpixel:
        refined_preds = preds.clone()
        for batch_index in range(batch_size):
            for landmark_index in range(num_landmarks):
                px = int(preds[batch_index, landmark_index, 0].item())
                py = int(preds[batch_index, landmark_index, 1].item())
                if 1 <= px < heatmap_width - 1 and 1 <= py < heatmap_height - 1:
                    current_map = heatmaps[batch_index, landmark_index]
                    diff_x = current_map[py, px + 1] - current_map[py, px - 1]
                    diff_y = current_map[py + 1, px] - current_map[py - 1, px]
                    refined_preds[batch_index, landmark_index, 0] += (
                        diff_x.sign() * 0.25
                    )
                    refined_preds[batch_index, landmark_index, 1] += (
                        diff_y.sign() * 0.25
                    )
        preds = refined_preds

    scale_x = image_width / float(heatmap_width)
    scale_y = image_height / float(heatmap_height)
    preds_image = preds.clone()
    preds_image[..., 0] *= scale_x
    preds_image[..., 1] *= scale_y
    return preds_image


def compute_box_normalized_nme(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> np.ndarray:
    """Compute the box-normalized mean error for a batch of predicted landmarks."""
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    batch_size = preds_np.shape[0]
    nme_values = np.zeros(batch_size, dtype=np.float32)

    for batch_index in range(batch_size):
        gt_landmarks = targets_np[batch_index]
        pred_landmarks = preds_np[batch_index]
        min_xy = gt_landmarks.min(axis=0)
        max_xy = gt_landmarks.max(axis=0)
        box_width = max_xy[0] - min_xy[0]
        box_height = max_xy[1] - min_xy[1]
        normalization = np.sqrt(max(box_width * box_height, eps))
        point_errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)
        nme_values[batch_index] = point_errors.mean() / normalization
    return nme_values
