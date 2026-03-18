from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .metrics import compute_box_normalized_nme, decode_heatmaps_to_image_coords


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_nme: bool = True,
) -> dict[str, Any]:
    model.eval()
    all_predictions = []
    nme_values = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            pred_landmarks = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=False,
            )
            all_predictions.append(pred_landmarks.cpu())
            if compute_nme and "landmarks" in batch:
                nme_batch = compute_box_normalized_nme(
                    preds=pred_landmarks,
                    targets=batch["landmarks"].to(device),
                )
                nme_values.append(nme_batch)

    predictions = torch.cat(all_predictions, dim=0)
    results: dict[str, Any] = {"predictions": predictions}
    if nme_values:
        results["nme"] = float(np.concatenate(nme_values).mean())
    return results
