from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import (
    ExperimentConfig,
    build_config,
    config_to_serializable_dict,
    resolve_inference_output_dir,
)
from scripts.engine.inference import export_inference_outputs
from scripts.engine.metrics import decoder_from_landmark_loss
from scripts.models import build_model_from_checkpoints
from scripts.utils import get_default_device, set_seed


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class InferenceImageFolderDataset(Dataset):
    """Dataset that reads every image under one directory recursively."""

    def __init__(self, input_dir: str | Path, config: ExperimentConfig) -> None:
        self.input_dir = Path(input_dir)
        self.config = config
        self.image_paths = sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.input_dir}.")

        self.mean = torch.tensor(config.normalization_mean, dtype=torch.float32).view(
            3, 1, 1
        )
        self.std = torch.tensor(config.normalization_std, dtype=torch.float32).view(
            3, 1, 1
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        target_height, target_width = self.config.image_size
        resized_image = image.resize((target_width, target_height), Image.BILINEAR)
        image_np = np.asarray(resized_image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        image_tensor = (image_tensor - self.mean) / self.std

        relative_path = image_path.relative_to(self.input_dir)
        sample_id = str(relative_path.with_suffix("")).replace("/", "__")

        metadata = {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "original_size": (original_height, original_width),
            "transformed_size": (target_height, target_width),
        }
        return {
            "image": image_tensor,
            "metadata": metadata,
        }


class DetectorExportInferenceDataset(Dataset):
    """Inference dataset for detector-export crops with optional reprojection metadata."""

    def __init__(
        self,
        export_root: str | Path,
        config: ExperimentConfig,
        source_root: str | Path | None = None,
    ) -> None:
        self.export_root = Path(export_root)
        self.images_dir = self.export_root / "images"
        self.metadata_dir = self.export_root / "metadata"
        self.source_root = Path(source_root) if source_root is not None else None
        self.config = config

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Detector export images directory not found: {self.images_dir}"
            )
        if not self.metadata_dir.exists():
            raise FileNotFoundError(
                f"Detector export metadata directory not found: {self.metadata_dir}"
            )

        self.image_paths = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(
                f"No detector-export crop images found under {self.images_dir}."
            )

        self.mean = torch.tensor(config.normalization_mean, dtype=torch.float32).view(
            3, 1, 1
        )
        self.std = torch.tensor(config.normalization_std, dtype=torch.float32).view(
            3, 1, 1
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        metadata_path = self.metadata_dir / f"{image_path.stem}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found for crop '{image_path.name}'."
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata_json = json.load(file)
        if not isinstance(metadata_json, dict):
            raise ValueError(f"Expected a JSON object in {metadata_path}.")

        image = Image.open(image_path).convert("RGB")
        crop_width, crop_height = image.size
        target_height, target_width = self.config.image_size
        resized_image = image.resize((target_width, target_height), Image.BILINEAR)
        image_np = np.asarray(resized_image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        image_tensor = (image_tensor - self.mean) / self.std

        source_image_path = self._resolve_source_image_path(
            metadata=metadata_json,
            metadata_path=metadata_path,
        )
        original_width, original_height = self._load_image_size(source_image_path)
        crop_id = str(metadata_json.get("crop_id") or image_path.stem)

        metadata = {
            "sample_id": crop_id,
            "image_path": str(image_path),
            "crop_image_path": str(image_path),
            "source_image_path": str(source_image_path),
            "original_size": (original_height, original_width),
            "crop_size": (crop_height, crop_width),
            "transformed_size": (target_height, target_width),
            "transform_crop_to_orig": torch.tensor(
                np.asarray(
                    metadata_json.get("transform_crop_to_orig"), dtype=np.float32
                ),
                dtype=torch.float32,
            ),
        }
        return {
            "image": image_tensor,
            "metadata": metadata,
        }

    def _resolve_source_image_path(
        self,
        metadata: dict[str, Any],
        metadata_path: Path,
    ) -> Path:
        """Resolve a source image path stored in detector-export metadata."""
        raw_path = metadata.get("source_image_path")
        if not raw_path:
            raise KeyError(f"Missing 'source_image_path' in {metadata_path}.")

        source_path = Path(str(raw_path))
        candidate_paths = []
        if source_path.is_absolute():
            candidate_paths.append(source_path)
        else:
            if self.source_root is not None:
                candidate_paths.append(self.source_root / source_path)
            candidate_paths.append(self.export_root / source_path)
            candidate_paths.append(metadata_path.parent / source_path)

        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path.resolve()

        raise FileNotFoundError(
            f"Could not resolve source image path '{raw_path}' from {metadata_path}."
        )

    @staticmethod
    def _load_image_size(image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            return image.size


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for folder-based standalone inference."""
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Run inference on every image found under one input directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint used to generate predictions.",
    )
    parser.add_argument(
        "--normalizer-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional normalizer-only checkpoint to combine with a landmarker-only "
            "--checkpoint. Do not use it with a full-model checkpoint."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory scanned recursively for images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where predicted labels and overlays will be written. If omitted, it is derived from the checkpoint name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.eval_batch_size,
        help="Mini-batch size for inference. If omitted, the training batch size from config is reused.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.num_workers,
        help="Number of worker processes for the dataloader.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=defaults.device,
        help="Device used to run inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed used for deterministic inference.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=defaults.visibility_threshold,
        help="Threshold applied to visibility logits to obtain binary predictions.",
    )
    parser.add_argument(
        "--landmark-loss",
        choices=["mse", "adaptive_wing", "wasserstein"],
        default=defaults.landmark_loss,
        help="Loss regime used by the checkpoint; controls coordinate decoding.",
    )
    parser.add_argument(
        "--wasserstein-softmax-temperature",
        type=float,
        default=defaults.wasserstein_softmax_temperature,
        help="Spatial softmax temperature used by barycenter decoding.",
    )
    parser.add_argument(
        "--disable-overlays",
        action="store_true",
        default=not defaults.save_inference_overlays,
        help="Disable writing image overlays with predicted landmarks.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        default=defaults.show_landmark_indices,
        help="Draw landmark indices next to each predicted point.",
    )
    parser.add_argument(
        "--project-to-original",
        action="store_true",
        default=False,
        help="When detector-export metadata is available, reproject predictions to original-image coordinates and draw overlays on the original source image.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional root used to resolve relative source_image_path values from detector-export metadata.",
    )
    parser.add_argument(
        "--save-crop-overlays",
        action="store_true",
        default=defaults.save_natural_crop_overlays,
        help="When combined with --project-to-original, also save qualitative crop-space overlays under predictions/crops/.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save the resolved configuration JSON next to the inference outputs.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build the inference config from CLI values and config defaults."""
    config = build_config()
    config.eval_batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device
    config.seed = args.seed
    config.visibility_threshold = args.visibility_threshold
    config.landmark_loss = args.landmark_loss
    config.coordinate_decoder = decoder_from_landmark_loss(args.landmark_loss)
    config.wasserstein_softmax_temperature = args.wasserstein_softmax_temperature
    config.save_inference_overlays = not args.disable_overlays
    config.show_landmark_indices = args.show_indices
    config.project_to_original = args.project_to_original
    config.source_root = args.source_root
    config.save_natural_crop_overlays = args.save_crop_overlays

    if args.output_dir is None:
        resolve_inference_output_dir(config, args.checkpoint)
        config.output_dir = config.output_dir.parent / (
            f"{config.output_dir.name}_{args.input_dir.resolve().name}"
        )
    else:
        config.output_dir = args.output_dir

    return config


def maybe_save_config(config: ExperimentConfig, output_dir: Path) -> None:
    """Save the resolved config as JSON."""
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config_to_serializable_dict(config), indent=2),
        encoding="utf-8",
    )


def _normalizer_architecture_from_config(config: ExperimentConfig) -> dict[str, object]:
    """Return fallback normalizer metadata for legacy full checkpoints."""
    return {
        "type": "residual",
        "input_channels": config.normalizer_input_channels,
        "hidden_channels": config.normalizer_hidden_channels,
        "num_layers": config.normalizer_num_layers,
        "kernel_size": config.normalizer_kernel_size,
        "activation": config.normalizer_activation,
        "normalization": config.normalizer_internal_normalization,
        "residual_scale": config.normalizer_residual_scale,
        "initialize_identity": config.normalizer_initialize_identity,
        "clamp_output": config.normalizer_clamp_output,
        "clamp_min": config.normalizer_clamp_min,
        "clamp_max": config.normalizer_clamp_max,
    }


def build_inference_dataloader(
    input_dir: Path,
    config: ExperimentConfig,
) -> DataLoader:
    """Create a dataloader for arbitrary image folders."""
    metadata_dir = input_dir / "metadata"
    images_dir = input_dir / "images"

    if config.project_to_original and metadata_dir.exists() and images_dir.exists():
        dataset: Dataset = DetectorExportInferenceDataset(
            export_root=input_dir,
            config=config,
            source_root=config.source_root,
        )
    else:
        dataset = InferenceImageFolderDataset(input_dir=input_dir, config=config)

    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and config.device in {"cuda", "auto"},
    )


def main() -> None:
    """Run standalone inference over an arbitrary folder of images."""
    args = parse_args()
    config = build_config_from_args(args)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = get_default_device(config.device)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Input dir: {args.input_dir}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Output dir: {output_dir}")
    print(
        "[INFO] Loss/decoder: "
        f"landmark_loss={config.landmark_loss}, "
        f"coordinate_decoder={config.coordinate_decoder}"
    )

    dataloader = build_inference_dataloader(args.input_dir, config)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalizer_checkpoint = (
        torch.load(args.normalizer_checkpoint, map_location="cpu", weights_only=False)
        if args.normalizer_checkpoint is not None
        else None
    )
    model = build_model_from_checkpoints(
        checkpoint,
        num_landmarks=config.num_landmarks,
        normalizer_checkpoint=normalizer_checkpoint,
        fallback_normalizer_architecture=_normalizer_architecture_from_config(config),
    )
    model.to(device)

    if args.save_config:
        maybe_save_config(config, output_dir)

    summary = export_inference_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=output_dir,
        visibility_threshold=config.visibility_threshold,
        save_overlays=config.save_inference_overlays,
        show_indices=config.show_landmark_indices,
        point_radius=config.overlay_point_radius,
        line_width=config.overlay_line_width,
        line_color=config.overlay_connection_color,
        project_to_original=config.project_to_original,
        save_crop_overlays=config.save_natural_crop_overlays,
        landmark_loss=config.landmark_loss,
        coordinate_decoder=config.coordinate_decoder,
        wasserstein_softmax_temperature=config.wasserstein_softmax_temperature,
    )

    print("[INFO] Inference finished.")
    print(f"[INFO] Samples processed: {summary['num_samples']}")
    print(f"[INFO] Labels dir: {summary['prediction_labels_dir']}")
    if summary["prediction_overlays_dir"] is not None:
        print(f"[INFO] Overlays dir: {summary['prediction_overlays_dir']}")
    if summary.get("prediction_crop_overlays_dir") is not None:
        print(f"[INFO] Crop overlays dir: {summary['prediction_crop_overlays_dir']}")


if __name__ == "__main__":
    main()
