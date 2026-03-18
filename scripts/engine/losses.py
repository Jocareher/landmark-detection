from __future__ import annotations

import torch


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    heatmap_loss_fn: torch.nn.Module,
    visibility_loss_fn: torch.nn.Module,
    lambda_heatmap: float = 1.0,
    lambda_visibility: float = 1.0,
) -> dict[str, torch.Tensor]:
    predicted_heatmaps = outputs["heatmaps"]
    predicted_visibility_logits = outputs["visibility_logits"]
    target_heatmaps = batch["heatmaps"]
    target_visibility = batch["visibility"]

    heatmap_loss = heatmap_loss_fn(predicted_heatmaps, target_heatmaps)
    visibility_loss = visibility_loss_fn(predicted_visibility_logits, target_visibility)
    total_loss = lambda_heatmap * heatmap_loss + lambda_visibility * visibility_loss
    return {
        "total_loss": total_loss,
        "heatmap_loss": heatmap_loss,
        "visibility_loss": visibility_loss,
    }
