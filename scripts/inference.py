from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from scripts.models import HRNetLandmarkVisibility
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
    config.save_inference_overlays = not args.disable_overlays
    config.show_landmark_indices = args.show_indices

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


def build_model(config: ExperimentConfig) -> HRNetLandmarkVisibility:
    """Instantiate the model used for inference."""
    return HRNetLandmarkVisibility(num_landmarks=config.num_landmarks)


def build_inference_dataloader(
    input_dir: Path,
    config: ExperimentConfig,
) -> DataLoader:
    """Create a dataloader for arbitrary image folders."""
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

    dataloader = build_inference_dataloader(args.input_dir, config)

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    )

    print("[INFO] Inference finished.")
    print(f"[INFO] Samples processed: {summary['num_samples']}")
    print(f"[INFO] Labels dir: {summary['prediction_labels_dir']}")
    if summary["prediction_overlays_dir"] is not None:
        print(f"[INFO] Overlays dir: {summary['prediction_overlays_dir']}")


if __name__ == "__main__":
    main()
