from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.engine.postprocessing import apply_homogeneous_transform

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

MULTIPIE_68_CONNECTIONS = [
    list(range(0, 17)),
    list(range(17, 22)),
    list(range(22, 27)),
    list(range(27, 31)),
    list(range(31, 36)),
    [36, 37, 38, 39, 40, 41, 36],
    [42, 43, 44, 45, 46, 47, 42],
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
    [60, 61, 62, 63, 64, 65, 66, 67, 60],
]


@dataclass
class ExternalInfAnFaceInferenceConfig:
    """Configuration for running the external infant landmark model adapter."""

    external_repo_root: Path
    checkpoint_path: Path
    crop_images_dir: Path
    crop_metadata_path: Path
    original_images_dir: Path | None
    output_dir: Path
    external_config_path: Path | None = None
    device: str = "auto"
    batch_size: int = 8
    num_workers: int = 0
    crop_scale_multiplier: float = 1.0
    save_overlays: bool = True
    show_indices: bool = False
    point_radius: int = 4
    line_width: int = 2
    line_color: str = "#00FF00"


@dataclass
class CropSample:
    """One detector crop plus metadata needed for original-image reprojection."""

    image_stem: str
    crop_image_path: Path
    metadata_path: Path
    source_image_path: Path
    crop_size: tuple[int, int]
    transform_crop_to_orig: np.ndarray


class RecursiveConfig(dict):
    """Dictionary with attribute access for external HRNet config fields."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def to_recursive_config(value: Any) -> Any:
    """Convert nested dictionaries into attribute-accessible config objects."""
    if isinstance(value, dict):
        return RecursiveConfig(
            {key: to_recursive_config(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [to_recursive_config(item) for item in value]
    return value


def build_external_config(config_path: Path) -> RecursiveConfig:
    """Load the external HRNet YAML without requiring the external yacs dependency."""
    raw = load_yaml_or_json(config_path)
    raw.setdefault("MODEL", {})
    raw["MODEL"].setdefault("INIT_WEIGHTS", False)
    raw["MODEL"].setdefault("PRETRAINED", "")
    raw["MODEL"].setdefault("NUM_JOINTS", 68)
    raw["MODEL"].setdefault("IMAGE_SIZE", [256, 256])
    raw["MODEL"].setdefault("HEATMAP_SIZE", [64, 64])
    raw["MODEL"].setdefault("EXTRA", {})
    return to_recursive_config(raw)


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "PyYAML is required to read YAML config files. Install pyyaml or use JSON."
        ) from error
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return loaded


def load_config(config_path: Path) -> ExternalInfAnFaceInferenceConfig:
    """Build an inference config from a YAML or JSON file."""
    raw = load_yaml_or_json(config_path)
    external = raw.get("external_model", {})
    data = raw.get("data", {})
    output = raw.get("output", {})
    runtime = raw.get("runtime", {})
    visualization = raw.get("visualization", {})

    metadata_path = data.get("crop_metadata_path") or data.get("crop_metadata_dir")
    if metadata_path is None:
        raise KeyError("Config must define data.crop_metadata_path or data.crop_metadata_dir.")

    external_repo_root = Path(external["external_repo_root"])
    external_config_path = external.get("external_config_path")
    return ExternalInfAnFaceInferenceConfig(
        external_repo_root=external_repo_root,
        checkpoint_path=Path(external["checkpoint_path"]),
        external_config_path=(
            Path(external_config_path) if external_config_path else None
        ),
        crop_images_dir=Path(data["crop_images_dir"]),
        crop_metadata_path=Path(metadata_path),
        original_images_dir=(
            Path(data["original_images_dir"])
            if data.get("original_images_dir")
            else None
        ),
        output_dir=Path(output["output_dir"]),
        device=str(runtime.get("device", "auto")),
        batch_size=int(runtime.get("batch_size", 8)),
        num_workers=int(runtime.get("num_workers", 0)),
        crop_scale_multiplier=float(runtime.get("crop_scale_multiplier", 1.0)),
        save_overlays=bool(visualization.get("save_overlays", True)),
        show_indices=bool(visualization.get("show_indices", False)),
        point_radius=int(visualization.get("point_radius", 4)),
        line_width=int(visualization.get("line_width", 2)),
        line_color=str(visualization.get("line_color", "#00FF00")),
    )


def select_inference_device(device: str = "auto") -> torch.device:
    """Select a torch device for inference."""
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device!r}.")
    return torch.device(requested)


def resolve_path(path: Path, base_dir: Path) -> Path:
    """Resolve a path, treating relative paths as relative to a base directory."""
    return path if path.is_absolute() else base_dir / path


def get_external_transform(
    center: torch.Tensor | np.ndarray,
    scale: float,
    output_size: list[int] | tuple[int, int],
) -> np.ndarray:
    """Return the HRNet-style transform matrix used by the external repository."""
    center_np = (
        np.asarray(center.detach().cpu().tolist(), dtype=np.float32)
        if isinstance(center, torch.Tensor)
        else np.asarray(center, dtype=np.float32)
    )
    height = 200.0 * float(scale)
    transform = np.zeros((3, 3), dtype=np.float32)
    transform[0, 0] = float(output_size[1]) / height
    transform[1, 1] = float(output_size[0]) / height
    transform[0, 2] = output_size[1] * (-float(center_np[0]) / height + 0.5)
    transform[1, 2] = output_size[0] * (-float(center_np[1]) / height + 0.5)
    transform[2, 2] = 1.0
    return transform


def transform_pixel_external_hrnet(
    point: torch.Tensor | np.ndarray | list[float],
    center: torch.Tensor | np.ndarray,
    scale: float,
    output_size: list[int] | tuple[int, int],
    invert: bool = False,
) -> np.ndarray:
    """Transform one pixel using the external HRNet integer-coordinate convention."""
    transform = get_external_transform(center, scale, output_size)
    if invert:
        transform = np.linalg.inv(transform)
    point_np = (
        np.asarray(point.detach().cpu().tolist(), dtype=np.float32)
        if isinstance(point, torch.Tensor)
        else np.asarray(point, dtype=np.float32)
    )
    transformed = transform @ np.asarray([point_np[0] - 1.0, point_np[1] - 1.0, 1.0])
    return transformed[:2].astype(int) + 1


def crop_external_hrnet(
    image: np.ndarray,
    center: torch.Tensor,
    scale: float,
    output_size: list[int] | tuple[int, int],
    rot: float = 0,
) -> np.ndarray:
    """Crop one image with the external HRNet preprocessing convention.

    The adapter uses this only with ``rot=0`` for inference on detector crops.
    """
    if rot != 0:
        raise ValueError("Rotated external HRNet crops are not supported in this adapter.")
    center_new = center.clone()
    height, width = image.shape[:2]
    scale_factor = scale * 200.0 / output_size[0]
    if scale_factor < 2:
        scale_factor = 1.0
    else:
        new_height = int(math.floor(height / scale_factor))
        new_width = int(math.floor(width / scale_factor))
        if max(new_height, new_width) < 2:
            return np.zeros((output_size[0], output_size[1], image.shape[2]), dtype=np.float32)
        image = np.asarray(
            Image.fromarray(image.astype(np.uint8)).resize(
                (new_width, new_height),
                Image.BILINEAR,
            ),
            dtype=np.float32,
        )
        center_new[0] = center_new[0] / scale_factor
        center_new[1] = center_new[1] / scale_factor
        scale = scale / scale_factor

    upper_left = np.asarray(
        transform_pixel_external_hrnet([0, 0], center_new, scale, output_size, invert=True)
    )
    bottom_right = np.asarray(
        transform_pixel_external_hrnet(output_size, center_new, scale, output_size, invert=True)
    )
    new_shape = [int(bottom_right[1] - upper_left[1]), int(bottom_right[0] - upper_left[0])]
    if image.ndim > 2:
        new_shape.append(image.shape[2])
    new_image = np.zeros(new_shape, dtype=np.float32)

    new_x = max(0, -upper_left[0]), min(bottom_right[0], image.shape[1]) - upper_left[0]
    new_y = max(0, -upper_left[1]), min(bottom_right[1], image.shape[0]) - upper_left[1]
    old_x = max(0, upper_left[0]), min(image.shape[1], bottom_right[0])
    old_y = max(0, upper_left[1]), min(image.shape[0], bottom_right[1])
    new_image[new_y[0] : new_y[1], new_x[0] : new_x[1]] = image[
        old_y[0] : old_y[1], old_x[0] : old_x[1]
    ]
    return np.asarray(
        Image.fromarray(new_image.astype(np.uint8)).resize(
            (output_size[1], output_size[0]),
            Image.BILINEAR,
        ),
        dtype=np.float32,
    )


def get_preds_external_hrnet(scores: torch.Tensor) -> torch.Tensor:
    """Get maximum-response heatmap coordinates using the external HRNet convention."""
    if scores.dim() != 4:
        raise ValueError("Score maps should be 4D tensors.")
    maxval, index = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    index = index.view(scores.size(0), scores.size(1), 1) + 1
    preds = index.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1
    return preds * maxval.gt(0).repeat(1, 1, 2).float()


def decode_preds_external_hrnet(
    output: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    resolution: list[int] | tuple[int, int],
) -> torch.Tensor:
    """Decode heatmaps with the external HRNet argmax/subpixel convention."""
    coords = get_preds_external_hrnet(output).cpu()
    for batch_index in range(coords.size(0)):
        for point_index in range(coords.size(1)):
            heatmap = output[batch_index][point_index].cpu()
            px = int(math.floor(float(coords[batch_index][point_index][0])))
            py = int(math.floor(float(coords[batch_index][point_index][1])))
            if (px > 1) and (px < resolution[0]) and (py > 1) and (py < resolution[1]):
                diff = torch.tensor(
                    [
                        heatmap[py - 1][px] - heatmap[py - 1][px - 2],
                        heatmap[py][px - 1] - heatmap[py - 2][px - 1],
                    ]
                )
                coords[batch_index][point_index] += diff.sign() * 0.25
    coords += 0.5
    preds = coords.clone()
    for batch_index in range(coords.size(0)):
        for point_index in range(coords.size(1)):
            preds[batch_index, point_index, 0:2] = torch.tensor(
                transform_pixel_external_hrnet(
                    coords[batch_index, point_index, 0:2],
                    center[batch_index],
                    float(scale[batch_index].item()),
                    resolution,
                    invert=True,
                ).tolist(),
                dtype=torch.float32,
            )
    return preds


def import_external_modules(external_repo_root: Path) -> dict[str, Any]:
    """Import the external repository modules without copying them."""
    external_repo_root = external_repo_root.resolve()
    if not external_repo_root.exists():
        raise FileNotFoundError(f"External repo root not found: {external_repo_root}")
    sys.path.insert(0, str(external_repo_root))
    try:
        import lib.models as external_models
    except Exception:
        if sys.path and sys.path[0] == str(external_repo_root):
            sys.path.pop(0)
        raise
    return {
        "models": external_models,
        "decode_preds": decode_preds_external_hrnet,
        "crop": crop_external_hrnet,
    }


def load_external_infanface_model(
    config: ExternalInfAnFaceInferenceConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any, Any]:
    """Load the external HRNet infant landmark model and its preprocessing helpers."""
    modules = import_external_modules(config.external_repo_root)
    external_models = modules["models"]

    config_path = (
        config.external_config_path
        or config.external_repo_root / "experiments" / "300w" / "hrnet-r90jt.yaml"
    )
    config_path = resolve_path(config_path, config.external_repo_root)
    external_config = build_external_config(config_path)
    external_config.MODEL.INIT_WEIGHTS = False

    model = external_models.get_face_alignment_net(external_config)
    checkpoint_path = resolve_path(config.checkpoint_path, config.external_repo_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned_state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, external_config, modules["crop"], modules["decode_preds"]


def load_crop_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    """Load detector crop metadata from a JSON directory, JSON file, or CSV file."""
    metadata_path = Path(metadata_path)
    if metadata_path.is_dir():
        rows = {}
        for path in sorted(metadata_path.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in metadata file: {path}")
            rows[path.stem] = {**payload, "_metadata_path": str(path)}
        return rows

    if metadata_path.suffix.lower() == ".json":
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue
                key = str(
                    item.get("crop_id")
                    or item.get("sample_id")
                    or item.get("image_stem")
                    or Path(str(item.get("crop_image_path", ""))).stem
                )
                rows[key] = {**item, "_metadata_path": str(metadata_path)}
            return rows
        if isinstance(payload, dict):
            if all(isinstance(value, dict) for value in payload.values()):
                return {
                    str(key): {**value, "_metadata_path": str(metadata_path)}
                    for key, value in payload.items()
                }
            key = str(
                payload.get("crop_id")
                or payload.get("sample_id")
                or Path(str(payload.get("crop_image_path", metadata_path.stem))).stem
            )
            return {key: {**payload, "_metadata_path": str(metadata_path)}}
        raise ValueError(f"Unsupported JSON metadata structure: {metadata_path}")

    if metadata_path.suffix.lower() == ".csv":
        rows = {}
        with metadata_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for item in reader:
                key = str(
                    item.get("crop_id")
                    or item.get("sample_id")
                    or item.get("image_stem")
                    or Path(str(item.get("crop_image_path", ""))).stem
                )
                rows[key] = {**item, "_metadata_path": str(metadata_path)}
        return rows

    raise ValueError(f"Unsupported metadata path: {metadata_path}")


def parse_transform_matrix(metadata: dict[str, Any]) -> np.ndarray:
    """Parse the crop-to-original 3x3 transform from detector metadata."""
    raw = metadata.get("transform_crop_to_orig")
    if raw is None:
        raise KeyError("Missing transform_crop_to_orig in crop metadata.")
    if isinstance(raw, str):
        raw = json.loads(raw)
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected transform_crop_to_orig shape (3, 3), got {matrix.shape}.")
    return matrix


def resolve_original_image_path(
    metadata: dict[str, Any],
    original_images_dir: Path | None,
    crop_metadata_path: Path,
) -> Path:
    """Resolve the original full-image path referenced by crop metadata."""
    raw_path = (
        metadata.get("source_image_path")
        or metadata.get("original_image_path")
        or metadata.get("image_path_original")
    )
    if not raw_path:
        raise KeyError("Missing source_image_path/original_image_path in crop metadata.")
    path = Path(str(raw_path))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if original_images_dir is not None:
            candidates.append(original_images_dir / path)
            candidates.append(original_images_dir / path.name)
        candidates.append(crop_metadata_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve original image path from metadata: {raw_path}")


def index_crop_samples(config: ExternalInfAnFaceInferenceConfig) -> list[CropSample]:
    """Index crop images and pair them with reprojection metadata."""
    metadata_index = load_crop_metadata(config.crop_metadata_path)
    crop_paths = sorted(
        path
        for path in config.crop_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    samples = []
    for crop_path in crop_paths:
        metadata = metadata_index.get(crop_path.stem)
        if metadata is None:
            continue
        metadata_path = Path(str(metadata.get("_metadata_path", config.crop_metadata_path)))
        source_image_path = resolve_original_image_path(
            metadata=metadata,
            original_images_dir=config.original_images_dir,
            crop_metadata_path=config.crop_metadata_path,
        )
        with Image.open(crop_path) as image:
            crop_width, crop_height = image.size
        samples.append(
            CropSample(
                image_stem=str(metadata.get("crop_id") or crop_path.stem),
                crop_image_path=crop_path,
                metadata_path=metadata_path,
                source_image_path=source_image_path,
                crop_size=(crop_height, crop_width),
                transform_crop_to_orig=parse_transform_matrix(metadata),
            )
        )
    if not samples:
        raise RuntimeError(f"No crop images with matching metadata found in {config.crop_images_dir}.")
    return samples


def preprocess_external_model_input(
    crop_image_path: Path,
    external_config: Any,
    external_crop: Any,
    crop_scale_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Preprocess one crop exactly like the external HRNet dataset pipeline."""
    image = Image.open(crop_image_path).convert("RGB")
    crop_width, crop_height = image.size
    image_np = np.asarray(image, dtype=np.float32)
    center = torch.tensor([crop_width / 2.0, crop_height / 2.0], dtype=torch.float32)
    scale_value = max(crop_width, crop_height) / 200.0 * crop_scale_multiplier
    scale = torch.tensor(scale_value, dtype=torch.float32)

    input_size = list(external_config.MODEL.IMAGE_SIZE)
    cropped = external_crop(image_np, center.clone(), float(scale.item()), input_size, rot=0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((cropped.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
    return (
        torch.tensor(tensor.tolist(), dtype=torch.float32),
        center,
        scale,
        (int(input_size[1]), int(input_size[0])),
    )


def decode_external_model_output(
    heatmaps: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    decode_preds: Any,
    heatmap_size: tuple[int, int],
) -> np.ndarray:
    """Decode external HRNet heatmaps into crop-space landmark coordinates."""
    decoded = decode_preds(
        heatmaps.detach().cpu(),
        centers.detach().cpu(),
        scales.detach().cpu(),
        list(heatmap_size),
    )
    return np.asarray(decoded.tolist(), dtype=np.float32)


def reproject_landmarks_to_original_image(
    landmarks_crop: np.ndarray,
    transform_crop_to_orig: np.ndarray,
) -> np.ndarray:
    """Project crop-space landmarks to original-image pixels."""
    return apply_homogeneous_transform(
        landmarks=landmarks_crop,
        transform_matrix=transform_crop_to_orig,
    ).astype(np.float32)


def save_prediction_label_68(output_path: Path, landmarks: np.ndarray) -> None:
    """Save one 68-row prediction txt in original-image pixel coordinates."""
    if landmarks.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68, 2), got {landmarks.shape}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for x_coord, y_coord in landmarks:
            file.write(f"{float(x_coord):.6f} {float(y_coord):.6f}\n")


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    color = color.strip().lstrip("#")
    if len(color) != 6:
        return (0, 255, 0)
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def save_fallback_landmark_overlay_image(
    image_path: Path,
    output_path: Path,
    predicted_landmarks: np.ndarray,
    predicted_visibility: np.ndarray,
    show_indices: bool = False,
    point_radius: int = 4,
    line_width: int = 2,
    line_color: str = "#00FF00",
) -> None:
    """Save a simple 68-landmark overlay without importing matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    rgb_line = _hex_to_rgb(line_color)

    for group in MULTIPIE_68_CONNECTIONS:
        points = []
        for index in group:
            x_coord, y_coord = predicted_landmarks[index]
            if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                points = []
                break
            if int(predicted_visibility[index]) <= 0:
                points = []
                break
            points.append((float(x_coord), float(y_coord)))
        if len(points) >= 2:
            draw.line(points, fill=rgb_line, width=line_width, joint="curve")

    for landmark_index, (x_coord, y_coord) in enumerate(predicted_landmarks):
        if not np.isfinite(x_coord) or not np.isfinite(y_coord):
            continue
        color = (255, 0, 0) if int(predicted_visibility[landmark_index]) > 0 else (0, 0, 255)
        draw.ellipse(
            (
                float(x_coord) - point_radius,
                float(y_coord) - point_radius,
                float(x_coord) + point_radius,
                float(y_coord) + point_radius,
            ),
            fill=color,
            outline=(255, 255, 255),
            width=max(1, line_width),
        )
        if show_indices:
            draw.text(
                (float(x_coord) + point_radius + 2, float(y_coord) + point_radius + 2),
                str(landmark_index),
                fill=(255, 255, 255),
            )
    image.save(output_path, quality=92, optimize=True)


def get_overlay_writer() -> Any:
    """Return the repository overlay writer, falling back to a local PIL renderer."""
    try:
        from scripts.utils.visualization import save_landmark_overlay_image

        return save_landmark_overlay_image
    except Exception as error:
        print(
            "[WARNING] Could not import repository overlay utility; "
            f"using lightweight PIL fallback instead. Reason: {error}"
        )
        return save_fallback_landmark_overlay_image


def write_inference_report(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-image inference report CSV."""
    output_path = output_dir / "inference_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_stem",
        "crop_image_path",
        "original_image_path",
        "output_label_path",
        "output_overlay_path",
        "inference_success",
        "failure_reason",
        "num_landmarks_predicted",
        "coordinate_space_before_reprojection",
        "coordinate_space_after_reprojection",
        "checkpoint_path",
        "external_repo_root",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_inference_summary(
    config: ExternalInfAnFaceInferenceConfig,
    device: torch.device,
    rows: list[dict[str, Any]],
) -> None:
    """Write a human-readable Markdown inference summary."""
    successes = [row for row in rows if row["inference_success"]]
    failures = [row for row in rows if not row["inference_success"]]
    lines = [
        "# External InfAnFace Inference Summary",
        "",
        "## Inputs",
        f"- external repository: `{config.external_repo_root}`",
        f"- checkpoint: `{config.checkpoint_path}`",
        f"- crop images: `{config.crop_images_dir}`",
        f"- crop metadata: `{config.crop_metadata_path}`",
        f"- original images: `{config.original_images_dir}`",
        f"- output directory: `{config.output_dir}`",
        f"- device: `{device}`",
        "",
        "## Results",
        f"- crops processed: {len(rows)}",
        f"- successful predictions: {len(successes)}",
        f"- failed predictions: {len(failures)}",
        "- saved label coordinate convention: original-image pixel coordinates",
        "- label format: 68 rows of `x y` coordinates",
        "",
        "## Assumptions",
        "- The external model is the HRNet-R90JT-style 68-landmark model from the cloned repository.",
        "- Input detector crops are treated as the full face box for the external HRNet crop transform.",
        "- Crop-space predictions are first scaled from the external model input size back to the detector crop size, then transformed to original-image pixels with `transform_crop_to_orig`.",
        f"- crop scale multiplier used for external preprocessing: {config.crop_scale_multiplier}",
    ]
    if failures:
        lines.extend(["", "## Failures"])
        for row in failures[:20]:
            lines.append(f"- `{row['image_stem']}`: {row['failure_reason']}")
        if len(failures) > 20:
            lines.append(f"- ... {len(failures) - 20} additional failures")
    (config.output_dir / "inference_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_infanface_external_inference(
    config: ExternalInfAnFaceInferenceConfig,
) -> dict[str, Any]:
    """Run external infant landmark inference and export benchmark-ready outputs."""
    save_landmark_overlay_image = get_overlay_writer() if config.save_overlays else None

    device = select_inference_device(config.device)
    labels_dir = config.output_dir / "labels"
    overlays_dir = config.output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if config.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading external model on device: {device}")
    model, external_config, external_crop, decode_preds = load_external_infanface_model(
        config=config,
        device=device,
    )
    samples = index_crop_samples(config)
    print(f"[INFO] Crop samples queued: {len(samples)}")

    rows: list[dict[str, Any]] = []
    heatmap_size = tuple(int(value) for value in external_config.MODEL.HEATMAP_SIZE)
    batch_size = max(1, int(config.batch_size))

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        prepared_batch: list[dict[str, Any]] = []

        for sample in batch_samples:
            label_path = labels_dir / f"{sample.image_stem}.txt"
            overlay_path = overlays_dir / f"{sample.image_stem}.jpg"
            row = {
                "image_stem": sample.image_stem,
                "crop_image_path": str(sample.crop_image_path),
                "original_image_path": str(sample.source_image_path),
                "output_label_path": str(label_path),
                "output_overlay_path": str(overlay_path) if config.save_overlays else "",
                "inference_success": False,
                "failure_reason": "",
                "num_landmarks_predicted": 0,
                "coordinate_space_before_reprojection": "external_model_crop_pixels",
                "coordinate_space_after_reprojection": "original_image_pixel_coordinates",
                "checkpoint_path": str(config.checkpoint_path),
                "external_repo_root": str(config.external_repo_root),
            }
            try:
                tensor, center, scale, _transformed_size = preprocess_external_model_input(
                    crop_image_path=sample.crop_image_path,
                    external_config=external_config,
                    external_crop=external_crop,
                    crop_scale_multiplier=config.crop_scale_multiplier,
                )
                prepared_batch.append(
                    {
                        "sample": sample,
                        "row": row,
                        "label_path": label_path,
                        "overlay_path": overlay_path,
                        "tensor": tensor,
                        "center": center,
                        "scale": scale,
                    }
                )
            except Exception as error:
                row["failure_reason"] = str(error)
                rows.append(row)
                print(f"[WARNING] Failed preprocessing {sample.image_stem}: {error}")

        if not prepared_batch:
            continue

        try:
            tensors = torch.stack([item["tensor"] for item in prepared_batch], dim=0)
            centers = torch.stack([item["center"] for item in prepared_batch], dim=0)
            scales = torch.stack([item["scale"] for item in prepared_batch], dim=0)
            with torch.inference_mode():
                heatmaps = model(tensors.to(device))
            decoded_batch = decode_external_model_output(
                heatmaps=heatmaps,
                centers=centers,
                scales=scales,
                decode_preds=decode_preds,
                heatmap_size=heatmap_size,
            )
        except Exception as error:
            for item in prepared_batch:
                row = item["row"]
                row["failure_reason"] = str(error)
                rows.append(row)
                print(f"[WARNING] Failed inference {row['image_stem']}: {error}")
            continue

        for item_index, item in enumerate(prepared_batch):
            sample = item["sample"]
            row = item["row"]
            try:
                landmarks_original = reproject_landmarks_to_original_image(
                    landmarks_crop=decoded_batch[item_index],
                    transform_crop_to_orig=sample.transform_crop_to_orig,
                )
                save_prediction_label_68(item["label_path"], landmarks_original)
                if config.save_overlays and save_landmark_overlay_image is not None:
                    save_landmark_overlay_image(
                        image_path=sample.source_image_path,
                        output_path=item["overlay_path"],
                        predicted_landmarks=landmarks_original,
                        predicted_visibility=np.ones(68, dtype=np.int64),
                        show_indices=config.show_indices,
                        point_radius=config.point_radius,
                        line_width=config.line_width,
                        line_color=config.line_color,
                    )
                row["inference_success"] = True
                row["num_landmarks_predicted"] = 68
            except Exception as error:
                row["failure_reason"] = str(error)
                print(f"[WARNING] Failed export {sample.image_stem}: {error}")
            rows.append(row)

    write_inference_report(config.output_dir, rows)
    write_inference_summary(config, device, rows)
    success_count = sum(1 for row in rows if row["inference_success"])
    print("[INFO] External InfAnFace inference finished.")
    print(f"[INFO] Successful predictions: {success_count}/{len(rows)}")
    print(f"[INFO] Labels dir: {labels_dir}")
    print(f"[INFO] Overlay dir: {overlays_dir if config.save_overlays else 'disabled'}")
    print(f"[INFO] Report: {config.output_dir / 'inference_report.csv'}")
    return {
        "num_samples": len(rows),
        "num_successful": success_count,
        "labels_dir": str(labels_dir),
        "overlays_dir": str(overlays_dir) if config.save_overlays else None,
        "report_path": str(config.output_dir / "inference_report.csv"),
        "summary_path": str(config.output_dir / "inference_summary.md"),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the external model adapter."""
    parser = argparse.ArgumentParser(
        description="Run the external InfAnFace infant landmark model on detector crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML/JSON config path.")
    parser.add_argument("--device", type=str, default=None, help="Override runtime.device.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output.output_dir.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    run_infanface_external_inference(config)


if __name__ == "__main__":
    main()
