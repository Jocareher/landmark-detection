from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_box_normalized_nme, decode_heatmaps_to_image_coords
from .postprocessing import (
    apply_homogeneous_transform,
    extract_batched_size,
    project_landmarks_between_sizes,
    project_landmarks_to_original_size,
)
from ..utils.predictions import save_prediction_file


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    compute_nme: bool = True,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
) -> dict[str, Any]:
    """Run model inference on a dataloader and optionally compute mean NME."""
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
                use_subpixel=True,
                decoder=coordinate_decoder,
                softmax_temperature=wasserstein_softmax_temperature,
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


def export_inference_outputs(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    visibility_threshold: float = 0.5,
    save_overlays: bool = True,
    show_indices: bool = False,
    point_radius: int = 10,
    line_width: int = 4,
    line_color: str = "#FFD400",
    project_to_original: bool = False,
    save_crop_overlays: bool = False,
    landmark_loss: str | None = None,
    coordinate_decoder: str = "argmax_subpixel",
    wasserstein_softmax_temperature: float = 1.0,
) -> dict[str, Any]:
    """Run inference and persist predicted labels and optional overlays."""
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    prediction_labels_dir = predictions_dir / "labels"
    prediction_overlays_dir = predictions_dir / "images"
    prediction_crops_dir = predictions_dir / "crops"

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    prediction_labels_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        prediction_overlays_dir.mkdir(parents=True, exist_ok=True)
        if project_to_original and save_crop_overlays:
            prediction_crops_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)
    save_landmark_overlay_image = None
    if save_overlays or (project_to_original and save_crop_overlays):
        from ..utils.visualization import save_landmark_overlay_image

    all_predictions: list[torch.Tensor] = []
    saved_samples = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Inferring", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)

            predicted_landmarks_batch = decode_heatmaps_to_image_coords(
                heatmaps=outputs["heatmaps"],
                image_height=images.shape[2],
                image_width=images.shape[3],
                use_subpixel=True,
                decoder=coordinate_decoder,
                softmax_temperature=wasserstein_softmax_temperature,
            ).cpu()
            all_predictions.append(predicted_landmarks_batch)

            predicted_visibility_logits = outputs["visibility_logits"].cpu()
            predicted_visibility_scores = torch.sigmoid(predicted_visibility_logits)
            predicted_visibility_batch = (
                predicted_visibility_scores >= visibility_threshold
            ).to(torch.int64)

            metadata_batch = batch["metadata"]
            batch_size = images.shape[0]

            for sample_index in range(batch_size):
                sample_id = str(metadata_batch["sample_id"][sample_index])
                image_path = Path(metadata_batch["image_path"][sample_index])
                transformed_size = extract_batched_size(
                    batched_size=metadata_batch["transformed_size"],
                    sample_index=sample_index,
                )
                original_size = extract_batched_size(
                    batched_size=metadata_batch["original_size"],
                    sample_index=sample_index,
                )

                predicted_visibility = (
                    predicted_visibility_batch[sample_index].numpy().astype(np.int64)
                )

                if (
                    project_to_original
                    and "crop_size" in metadata_batch
                    and "transform_crop_to_orig" in metadata_batch
                    and "source_image_path" in metadata_batch
                ):
                    crop_image_path = Path(
                        metadata_batch["crop_image_path"][sample_index]
                    )
                    crop_size = extract_batched_size(
                        batched_size=metadata_batch["crop_size"],
                        sample_index=sample_index,
                    )
                    transform_crop_to_orig = metadata_batch["transform_crop_to_orig"][
                        sample_index
                    ]
                    predicted_landmarks_crop = project_landmarks_between_sizes(
                        landmarks=predicted_landmarks_batch[sample_index],
                        source_size=transformed_size,
                        target_size=crop_size,
                    )
                    predicted_landmarks_output = apply_homogeneous_transform(
                        landmarks=predicted_landmarks_crop,
                        transform_matrix=transform_crop_to_orig,
                    )
                    overlay_image_path = Path(
                        metadata_batch["source_image_path"][sample_index]
                    )
                    overlay_output_path = prediction_overlays_dir / f"{sample_id}.jpg"
                else:
                    predicted_landmarks_output = project_landmarks_to_original_size(
                        landmarks=predicted_landmarks_batch[sample_index],
                        transformed_size=transformed_size,
                        original_size=original_size,
                    ).numpy()
                    overlay_image_path = image_path
                    overlay_output_path = prediction_overlays_dir / f"{sample_id}.png"

                save_prediction_file(
                    output_path=prediction_labels_dir / f"{sample_id}.txt",
                    landmarks=predicted_landmarks_output,
                    visibility=predicted_visibility,
                )

                if save_overlays:
                    save_landmark_overlay_image(
                        image_path=overlay_image_path,
                        output_path=overlay_output_path,
                        predicted_landmarks=predicted_landmarks_output,
                        predicted_visibility=predicted_visibility,
                        show_indices=show_indices,
                        point_radius=point_radius,
                        line_width=line_width,
                        line_color=line_color,
                    )
                    if (
                        project_to_original
                        and "crop_size" in metadata_batch
                        and "transform_crop_to_orig" in metadata_batch
                        and save_crop_overlays
                        and prediction_crops_dir is not None
                    ):
                        save_landmark_overlay_image(
                            image_path=crop_image_path,
                            output_path=prediction_crops_dir / f"{sample_id}.png",
                            predicted_landmarks=predicted_landmarks_crop.numpy(),
                            predicted_visibility=predicted_visibility,
                            show_indices=show_indices,
                            point_radius=point_radius,
                            line_width=line_width,
                            line_color=line_color,
                        )

                saved_samples += 1

    predictions = (
        torch.cat(all_predictions, dim=0) if all_predictions else torch.empty(0)
    )
    return {
        "num_samples": saved_samples,
        "predictions": predictions,
        "predictions_dir": str(predictions_dir),
        "prediction_labels_dir": str(prediction_labels_dir),
        "prediction_overlays_dir": str(prediction_overlays_dir)
        if save_overlays
        else None,
        "prediction_crop_overlays_dir": str(prediction_crops_dir)
        if save_overlays and project_to_original and save_crop_overlays
        else None,
        "landmark_loss": landmark_loss,
        "coordinate_decoder": coordinate_decoder,
    }
