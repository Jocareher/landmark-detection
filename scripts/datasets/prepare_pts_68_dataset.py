from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.logging import tee_terminal_output

try:
    from scripts.utils.visualization import (
        render_landmark_preview_image,
        save_overlay_image,
    )
except ModuleNotFoundError:
    render_landmark_preview_image = None
    save_overlay_image = None


NUM_LANDMARKS = 68
VALID_CLASS_INDICES = set(range(5))
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

FALLBACK_68_CONNECTIONS = [
    (range(0, 17), False),
    (range(17, 22), False),
    (range(22, 27), False),
    (range(27, 31), False),
    (range(31, 36), False),
    (range(36, 42), True),
    (range(42, 48), True),
    (range(48, 60), True),
    (range(60, 68), True),
]

REPORT_COLUMNS = [
    "dataset_name",
    "stem",
    "source_image_path",
    "source_pts_path",
    "newborn_prediction_path",
    "output_image_path",
    "output_label_path",
    "output_plot_path",
    "image_found",
    "pts_found",
    "newborn_prediction_found",
    "class_idx",
    "n_points_declared",
    "n_points_parsed",
    "converted",
    "plot_generated",
    "status",
    "warning_message",
]


@dataclass(frozen=True)
class PtsDatasetOptions:
    """Options controlling conversion of one or more 68-landmark `.pts` datasets."""

    overwrite_existing: bool = False
    preserve_image_extension: bool = True
    coordinate_format: str = "pixel"
    allow_missing_class_idx: bool = False
    missing_class_idx_placeholder: int | None = None
    generate_plots: bool = True
    jpeg_quality: int = 95
    show_indices: bool = False
    point_radius: int = 5
    line_width: int = 2
    line_color: str = "#00C853"


@dataclass(frozen=True)
class PtsDatasetConfig:
    """Configuration for preparing one 68-landmark dataset from `.pts` labels."""

    name: str
    source_root: Path
    newborn_predictions_dir: Path
    output_root: Path
    options: PtsDatasetOptions

    @property
    def output_dataset_root(self) -> Path:
        """Return the dataset-specific output root."""
        return self.output_root / self.name


@dataclass(frozen=True)
class PtsParseResult:
    """Parsed `.pts` payload with declared and parsed landmark counts."""

    landmarks: np.ndarray
    n_points_declared: int | None
    n_points_parsed: int


class PtsParseError(ValueError):
    """Malformed `.pts` file error carrying partial parse counts for reports."""

    def __init__(
        self,
        message: str,
        n_points_declared: int | None = None,
        n_points_parsed: int | None = None,
    ) -> None:
        super().__init__(message)
        self.n_points_declared = n_points_declared
        self.n_points_parsed = n_points_parsed


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            logging.warning(
                "PyYAML is not installed; using the built-in parser for the simple "
                "dataset preparation config schema."
            )
            loaded = _load_simple_prepare_yaml(text)
        else:
            loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return loaded


def _parse_simple_yaml_scalar(value: str) -> Any:
    """Parse a scalar from the small YAML subset used by preparation configs."""
    value = value.strip()
    if value == "":
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_simple_yaml_key_value(line: str) -> tuple[str, Any]:
    """Split one simple `key: value` YAML line."""
    key, _, value = line.partition(":")
    if not key.strip():
        raise ValueError(f"Malformed YAML line: {line!r}")
    return key.strip(), _parse_simple_yaml_scalar(value)


def _load_simple_prepare_yaml(text: str) -> dict[str, Any]:
    """Parse the simple YAML subset used by `prepare_cifd_tif.yaml`.

    This fallback intentionally supports only top-level scalars, a top-level
    `options` mapping, and a `datasets` list of mappings with optional nested
    `options`. Install PyYAML for general YAML support.
    """
    result: dict[str, Any] = {}
    current_section: str | None = None
    current_dataset: dict[str, Any] | None = None
    in_dataset_options = False

    for raw_line in text.splitlines():
        stripped_comment = raw_line.split("#", 1)[0].rstrip()
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        line = stripped_comment.strip()

        if indent == 0:
            key, value = _split_simple_yaml_key_value(line)
            current_section = key
            current_dataset = None
            in_dataset_options = False
            if key == "datasets":
                result[key] = []
            elif value is None:
                result[key] = {}
            else:
                result[key] = value
            continue

        if current_section == "datasets":
            if indent == 2 and line.startswith("- "):
                current_dataset = {}
                result.setdefault("datasets", []).append(current_dataset)
                item = line[2:].strip()
                in_dataset_options = False
                if item:
                    key, value = _split_simple_yaml_key_value(item)
                    current_dataset[key] = value
                continue
            if current_dataset is None:
                raise ValueError("Found dataset field before a dataset item.")
            if indent == 4:
                key, value = _split_simple_yaml_key_value(line)
                if key == "options":
                    current_dataset[key] = {}
                    in_dataset_options = True
                else:
                    current_dataset[key] = value
                    in_dataset_options = False
                continue
            if indent == 6 and in_dataset_options:
                key, value = _split_simple_yaml_key_value(line)
                current_dataset.setdefault("options", {})[key] = value
                continue

        if current_section == "options" and indent == 2:
            key, value = _split_simple_yaml_key_value(line)
            result.setdefault("options", {})[key] = value
            continue

        raise ValueError(
            "Unsupported YAML syntax in config. Install PyYAML for full YAML support."
        )
    return result


def _resolve_config_path(value: str | Path, config_dir: Path) -> Path:
    """Resolve config paths relative to the config file directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_dir / path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV, always creating a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _parse_options(raw_options: dict[str, Any] | None) -> PtsDatasetOptions:
    """Parse conversion options from a config mapping."""
    options = raw_options or {}
    placeholder = options.get("missing_class_idx_placeholder")
    if placeholder is not None:
        placeholder = int(placeholder)
        if placeholder not in VALID_CLASS_INDICES:
            raise ValueError(
                "options.missing_class_idx_placeholder must be an integer in [0, 4]."
            )
    parsed = PtsDatasetOptions(
        overwrite_existing=bool(options.get("overwrite_existing", False)),
        preserve_image_extension=bool(options.get("preserve_image_extension", True)),
        coordinate_format=str(options.get("coordinate_format", "pixel")),
        allow_missing_class_idx=bool(options.get("allow_missing_class_idx", False)),
        missing_class_idx_placeholder=placeholder,
        generate_plots=bool(options.get("generate_plots", True)),
        jpeg_quality=int(options.get("jpeg_quality", 95)),
        show_indices=bool(options.get("show_indices", False)),
        point_radius=int(options.get("point_radius", 5)),
        line_width=int(options.get("line_width", 2)),
        line_color=str(options.get("line_color", "#00C853")),
    )
    if parsed.coordinate_format != "pixel":
        raise ValueError(
            "Only coordinate_format='pixel' is supported. The InfAnFace benchmark "
            "loader expects absolute 68-point coordinates."
        )
    return parsed


def _load_configs(config_path: Path) -> list[PtsDatasetConfig]:
    """Load all dataset conversion configs from one YAML or JSON file."""
    config_path = config_path.resolve()
    raw = _load_yaml_or_json(config_path)
    config_dir = config_path.parent
    options = _parse_options(raw.get("options"))
    output_root = _resolve_config_path(raw["output_root"], config_dir)
    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Config must define a non-empty datasets list.")

    configs: list[PtsDatasetConfig] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each dataset entry must be a mapping.")
        dataset_options = _parse_options(
            {**raw.get("options", {}), **dataset.get("options", {})}
        )
        configs.append(
            PtsDatasetConfig(
                name=str(dataset["name"]),
                source_root=_resolve_config_path(dataset["source_root"], config_dir),
                newborn_predictions_dir=_resolve_config_path(
                    dataset["newborn_predictions_dir"], config_dir
                ),
                output_root=output_root,
                options=dataset_options if dataset.get("options") else options,
            )
        )
    return configs


def parse_pts_68_file(label_path: Path) -> PtsParseResult:
    """Parse a 68-point `.pts` landmark file into a `(68, 2)` float array.

    The parser reads the declared `n_points` value, consumes only coordinate rows
    between `{` and `}`, and validates that the final coordinates are finite.
    """
    declared_points: int | None = None
    in_block = False
    coordinates: list[list[float]] = []

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("n_points"):
            parts = line.replace(":", " ").split()
            if len(parts) < 2:
                raise PtsParseError(
                    f"Malformed n_points line at {label_path}:{line_number}.",
                    n_points_parsed=len(coordinates),
                )
            declared_points = int(float(parts[1]))
            continue
        if line == "{":
            in_block = True
            continue
        if line == "}":
            in_block = False
            break
        if not in_block:
            continue

        tokens = line.split()
        if len(tokens) != 2:
            raise PtsParseError(
                f"Malformed coordinate row at {label_path}:{line_number}. "
                f"Expected two numbers, got {raw_line!r}.",
                n_points_declared=declared_points,
                n_points_parsed=len(coordinates),
            )
        coordinates.append([float(tokens[0]), float(tokens[1])])

    if declared_points is None:
        raise PtsParseError(
            f"Missing n_points declaration in {label_path}.",
            n_points_parsed=len(coordinates),
        )
    if declared_points != NUM_LANDMARKS:
        raise PtsParseError(
            f"Expected n_points: {NUM_LANDMARKS} in {label_path}, got {declared_points}.",
            n_points_declared=declared_points,
            n_points_parsed=len(coordinates),
        )

    landmarks = np.asarray(coordinates, dtype=np.float32)
    if landmarks.shape != (NUM_LANDMARKS, 2):
        raise PtsParseError(
            f"Expected exactly {NUM_LANDMARKS} coordinate rows in {label_path}, "
            f"got {landmarks.shape[0]}.",
            n_points_declared=declared_points,
            n_points_parsed=int(landmarks.shape[0]),
        )
    if not np.isfinite(landmarks).all():
        raise PtsParseError(
            f"Non-finite coordinate found in {label_path}.",
            n_points_declared=declared_points,
            n_points_parsed=int(landmarks.shape[0]),
        )
    return PtsParseResult(
        landmarks=landmarks,
        n_points_declared=declared_points,
        n_points_parsed=int(landmarks.shape[0]),
    )


def find_matching_image_by_stem(images_dir: Path, stem: str) -> Path | None:
    """Return the first supported image under `images_dir` matching `stem`."""
    for suffix in sorted(SUPPORTED_IMAGE_SUFFIXES):
        for candidate in (
            images_dir / f"{stem}{suffix}",
            images_dir / f"{stem}{suffix.upper()}",
        ):
            if candidate.is_file():
                return candidate
    matches = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file()
        and path.stem == stem
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return matches[0] if matches else None


def read_newborn_class_idx(prediction_txt_path: Path) -> int | None:
    """Read the first valid NewBORN detection class id from one prediction file.

    NewBORN rows are expected to contain:
    `class_idx x1 y1 x2 y2 x3 y3 x4 y4 angle`.
    """
    for raw_line in prediction_txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 10:
            continue
        try:
            class_value = float(tokens[0])
        except ValueError:
            continue
        class_idx = int(class_value)
        if class_value == float(class_idx) and class_idx in VALID_CLASS_INDICES:
            return class_idx
    return None


def build_newborn_prediction_index(
    newborn_predictions_dir: Path,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Index NewBORN prediction files by stem and parse their first valid class id."""
    index: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    if not newborn_predictions_dir.exists():
        warnings.append(
            f"NewBORN predictions directory does not exist: {newborn_predictions_dir}"
        )
        return index, warnings

    for prediction_path in sorted(newborn_predictions_dir.rglob("*.txt")):
        stem = prediction_path.stem
        if stem in index:
            warnings.append(
                f"Duplicate NewBORN prediction stem {stem!r}; keeping {index[stem]['path']}."
            )
            continue
        class_idx = read_newborn_class_idx(prediction_path)
        index[stem] = {
            "path": prediction_path,
            "class_idx": class_idx,
            "valid": class_idx is not None,
        }
    return index, warnings


def infanface_label_has_class_header(label_path: Path) -> bool:
    """Return whether an InfAnFace-style label starts with a valid class_idx header."""
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return False
    tokens = lines[0].split()
    if len(tokens) != 1:
        return False
    try:
        value = float(tokens[0])
    except ValueError:
        return False
    class_idx = int(value)
    return value == float(class_idx) and class_idx in VALID_CLASS_INDICES


def write_infanface_style_label_68(
    output_path: Path,
    class_idx: int | None,
    landmarks: np.ndarray,
) -> None:
    """Write a 68-landmark InfAnFace-style label with an optional class header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (NUM_LANDMARKS, 2):
        raise ValueError(
            f"Expected landmarks with shape ({NUM_LANDMARKS}, 2), got {landmarks.shape}."
        )
    if not np.isfinite(landmarks).all():
        raise ValueError("Cannot write landmarks containing non-finite coordinates.")
    if class_idx is not None and int(class_idx) not in VALID_CLASS_INDICES:
        raise ValueError(
            f"Invalid class_idx={class_idx}. Expected an integer in [0, 4]."
        )

    with output_path.open("w", encoding="utf-8") as handle:
        if class_idx is not None:
            handle.write(f"{int(class_idx)}\n")
        for x_coord, y_coord in landmarks:
            handle.write(f"{float(x_coord):.6f} {float(y_coord):.6f}\n")


def validate_infanface_style_label_68(
    label_path: Path,
    require_class_header: bool = True,
) -> None:
    """Validate the generated 68-landmark label shape and numeric content."""
    lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_lines = NUM_LANDMARKS + (1 if require_class_header else 0)
    if len(lines) != expected_lines:
        raise ValueError(
            f"Expected {expected_lines} non-empty lines in {label_path}, got {len(lines)}."
        )
    if require_class_header:
        if not infanface_label_has_class_header(label_path):
            raise ValueError(f"Invalid class_idx header in {label_path}.")
        landmark_lines = lines[1:]
    else:
        landmark_lines = lines

    for line_number, raw_line in enumerate(
        landmark_lines, start=(2 if require_class_header else 1)
    ):
        tokens = raw_line.split()
        if len(tokens) != 2:
            raise ValueError(f"Invalid coordinate row at {label_path}:{line_number}.")
        x_coord, y_coord = float(tokens[0]), float(tokens[1])
        if not (math.isfinite(x_coord) and math.isfinite(y_coord)):
            raise ValueError(f"Non-finite coordinate at {label_path}:{line_number}.")


def _index_images(images_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Index supported images by filename stem."""
    index: dict[str, Path] = {}
    warnings: list[str] = []
    if not images_dir.exists():
        warnings.append(f"Images directory does not exist: {images_dir}")
        return index, warnings
    for image_path in sorted(images_dir.rglob("*")):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
        ):
            continue
        if image_path.stem in index:
            warnings.append(
                f"Duplicate image stem {image_path.stem!r}; keeping {index[image_path.stem]}."
            )
            continue
        index[image_path.stem] = image_path
    return index, warnings


def _index_pts_labels(labels_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """Index `.pts` labels by filename stem."""
    index: dict[str, Path] = {}
    warnings: list[str] = []
    if not labels_dir.exists():
        warnings.append(f"Labels directory does not exist: {labels_dir}")
        return index, warnings
    for label_path in sorted(labels_dir.rglob("*.pts")):
        if label_path.stem in index:
            warnings.append(
                f"Duplicate .pts stem {label_path.stem!r}; keeping {index[label_path.stem]}."
            )
            continue
        index[label_path.stem] = label_path
    return index, warnings


def _copy_or_convert_image(
    source_path: Path,
    output_path: Path,
    preserve_extension: bool,
    jpeg_quality: int,
) -> None:
    """Copy an image as-is or convert it to RGB JPEG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_extension:
        shutil.copy2(source_path, output_path)
        return
    with Image.open(source_path) as image:
        ImageOps.exif_transpose(image).convert("RGB").save(
            output_path,
            format="JPEG",
            quality=jpeg_quality,
        )


def draw_landmark_overlay_68(
    image_path: Path,
    landmarks: np.ndarray,
    output_path: Path,
    show_indices: bool = False,
    point_radius: int = 5,
    line_width: int = 2,
    line_color: str = "#00C853",
) -> None:
    """Draw a 68-landmark overlay in pixel coordinates and save it as an image."""
    with Image.open(image_path) as image:
        source_image = ImageOps.exif_transpose(image).convert("RGB")

    landmarks = np.asarray(landmarks, dtype=np.float32)
    if render_landmark_preview_image is not None and save_overlay_image is not None:
        rendered = render_landmark_preview_image(
            image=source_image,
            landmarks=landmarks,
            visibility=np.ones(NUM_LANDMARKS, dtype=np.float32),
            show_indices=show_indices,
            point_radius=point_radius,
            line_width=line_width,
            line_color=line_color,
            draw_all_connections=True,
        )
        save_overlay_image(rendered, output_path)
        return

    draw = ImageDraw.Draw(source_image)
    for landmark_range, close_loop in FALLBACK_68_CONNECTIONS:
        points = [
            (float(landmarks[index, 0]), float(landmarks[index, 1]))
            for index in landmark_range
            if index < len(landmarks)
            and math.isfinite(float(landmarks[index, 0]))
            and math.isfinite(float(landmarks[index, 1]))
        ]
        if len(points) >= 2:
            if close_loop:
                points = [*points, points[0]]
            draw.line(points, fill=line_color, width=line_width, joint="curve")

    for landmark_index, (x_coord, y_coord) in enumerate(landmarks):
        if not (math.isfinite(float(x_coord)) and math.isfinite(float(y_coord))):
            continue
        left = float(x_coord) - point_radius
        top = float(y_coord) - point_radius
        right = float(x_coord) + point_radius
        bottom = float(y_coord) + point_radius
        draw.ellipse(
            (left, top, right, bottom),
            fill="red",
            outline="white",
            width=max(1, line_width // 2),
        )
        if show_indices:
            draw.text(
                (float(x_coord) + point_radius + 2, float(y_coord) + point_radius + 2),
                str(landmark_index),
                fill="red",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_image.save(output_path, format="JPEG", quality=90)


def _output_paths(
    config: PtsDatasetConfig, stem: str, source_image_path: Path | None
) -> tuple[Path, Path, Path]:
    """Return output image, label, and plot paths for one stem."""
    image_suffix = source_image_path.suffix if source_image_path is not None else ".jpg"
    if not config.options.preserve_image_extension:
        image_suffix = ".jpg"
    output_root = config.output_dataset_root
    return (
        output_root / "images" / f"{stem}{image_suffix.lower()}",
        output_root / "labels" / f"{stem}.txt",
        output_root / "plots" / f"{stem}.jpg",
    )


def _empty_report_row(config: PtsDatasetConfig, stem: str) -> dict[str, Any]:
    """Create an initialized conversion report row."""
    return {
        "dataset_name": config.name,
        "stem": stem,
        "source_image_path": "",
        "source_pts_path": "",
        "newborn_prediction_path": "",
        "output_image_path": "",
        "output_label_path": "",
        "output_plot_path": "",
        "image_found": False,
        "pts_found": False,
        "newborn_prediction_found": False,
        "class_idx": "",
        "n_points_declared": "",
        "n_points_parsed": "",
        "converted": False,
        "plot_generated": False,
        "status": "",
        "warning_message": "",
    }


def prepare_single_pts_dataset(config: PtsDatasetConfig) -> dict[str, object]:
    """Prepare one 68-landmark `.pts` dataset and write conversion reports."""
    logging.info("Preparing dataset %s", config.name)
    images_dir = config.source_root / "images"
    labels_dir = config.source_root / "labels"
    output_root = config.output_dataset_root
    reports_dir = output_root / "reports"
    for directory_name in ("images", "labels", "plots", "reports"):
        (output_root / directory_name).mkdir(parents=True, exist_ok=True)

    image_index, image_warnings = _index_images(images_dir)
    pts_index, pts_warnings = _index_pts_labels(labels_dir)
    newborn_index, newborn_warnings = build_newborn_prediction_index(
        config.newborn_predictions_dir
    )

    report_rows: list[dict[str, Any]] = []
    missing_newborn_rows: list[dict[str, Any]] = []
    missing_image_rows: list[dict[str, Any]] = []
    malformed_pts_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    plot_failure_rows: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()

    all_stems = sorted(set(image_index) | set(pts_index) | set(newborn_index))
    for stem in all_stems:
        image_path = image_index.get(stem)
        pts_path = pts_index.get(stem)
        newborn_record = newborn_index.get(stem)
        output_image_path, output_label_path, output_plot_path = _output_paths(
            config, stem, image_path
        )

        row = _empty_report_row(config, stem)
        row.update(
            {
                "source_image_path": str(image_path) if image_path else "",
                "source_pts_path": str(pts_path) if pts_path else "",
                "newborn_prediction_path": str(newborn_record["path"])
                if newborn_record
                else "",
                "output_image_path": str(output_image_path) if image_path else "",
                "output_label_path": str(output_label_path) if pts_path else "",
                "output_plot_path": str(output_plot_path)
                if image_path and pts_path
                else "",
                "image_found": image_path is not None,
                "pts_found": pts_path is not None,
                "newborn_prediction_found": newborn_record is not None,
            }
        )

        warnings: list[str] = []
        if image_path is None and pts_path is not None:
            row["status"] = "missing_image"
            missing_image_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_pts_path": str(pts_path),
                    "reason": "label_without_image",
                }
            )
            unmatched_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_image_path": "",
                    "source_pts_path": str(pts_path),
                    "newborn_prediction_path": row["newborn_prediction_path"],
                    "reason": "label_without_image",
                }
            )
            row["warning_message"] = "Missing source image."
            report_rows.append(row)
            continue

        if pts_path is None and image_path is not None:
            row["status"] = "missing_pts_label"
            unmatched_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_image_path": str(image_path),
                    "source_pts_path": "",
                    "newborn_prediction_path": row["newborn_prediction_path"],
                    "reason": "image_without_label",
                }
            )
            row["warning_message"] = "Missing .pts label."
            report_rows.append(row)
            continue

        if image_path is None or pts_path is None:
            row["status"] = "unmatched_newborn_prediction"
            unmatched_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_image_path": "",
                    "source_pts_path": "",
                    "newborn_prediction_path": row["newborn_prediction_path"],
                    "reason": "prediction_without_image_or_label",
                }
            )
            row["warning_message"] = "NewBORN prediction has no matching image/label."
            report_rows.append(row)
            continue

        try:
            parse_result = parse_pts_68_file(pts_path)
            row["n_points_declared"] = parse_result.n_points_declared
            row["n_points_parsed"] = parse_result.n_points_parsed
            landmarks = parse_result.landmarks
        except PtsParseError as error:
            row["n_points_declared"] = (
                "" if error.n_points_declared is None else error.n_points_declared
            )
            row["n_points_parsed"] = (
                "" if error.n_points_parsed is None else error.n_points_parsed
            )
            row["status"] = "malformed_pts"
            row["warning_message"] = str(error)
            malformed_pts_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_pts_path": str(pts_path),
                    "error": str(error),
                }
            )
            report_rows.append(row)
            continue
        except Exception as error:
            row["status"] = "malformed_pts"
            row["warning_message"] = str(error)
            malformed_pts_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_pts_path": str(pts_path),
                    "error": str(error),
                }
            )
            report_rows.append(row)
            continue

        class_idx = None
        if newborn_record is not None and newborn_record.get("valid"):
            class_idx = int(newborn_record["class_idx"])
            row["class_idx"] = class_idx
        elif config.options.allow_missing_class_idx:
            class_idx = config.options.missing_class_idx_placeholder
            row["class_idx"] = "" if class_idx is None else class_idx
            warnings.append("Missing or invalid NewBORN class_idx; allowed by config.")
            missing_newborn_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_image_path": str(image_path),
                    "source_pts_path": str(pts_path),
                    "newborn_prediction_path": row["newborn_prediction_path"],
                    "reason": "missing_or_invalid_newborn_prediction_allowed",
                }
            )
        else:
            row["status"] = "missing_or_invalid_newborn_prediction"
            row[
                "warning_message"
            ] = "Missing NewBORN prediction or no valid class_idx in [0, 4]."
            missing_newborn_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_image_path": str(image_path),
                    "source_pts_path": str(pts_path),
                    "newborn_prediction_path": row["newborn_prediction_path"],
                    "reason": "missing_or_invalid_newborn_prediction",
                }
            )
            report_rows.append(row)
            continue

        output_candidates = [output_image_path, output_label_path]
        if config.options.generate_plots:
            output_candidates.append(output_plot_path)
        existing_outputs = [path for path in output_candidates if path.exists()]
        if existing_outputs and not config.options.overwrite_existing:
            row["status"] = "output_exists"
            row[
                "warning_message"
            ] = "Output exists and overwrite_existing is false: " + "|".join(
                str(path) for path in existing_outputs
            )
            report_rows.append(row)
            continue

        try:
            _copy_or_convert_image(
                source_path=image_path,
                output_path=output_image_path,
                preserve_extension=config.options.preserve_image_extension,
                jpeg_quality=config.options.jpeg_quality,
            )
            write_infanface_style_label_68(output_label_path, class_idx, landmarks)
            validate_infanface_style_label_68(
                output_label_path,
                require_class_header=class_idx is not None,
            )
        except Exception as error:
            row["status"] = "write_failed"
            row["warning_message"] = str(error)
            report_rows.append(row)
            continue

        row["converted"] = True
        if class_idx is not None:
            class_counts[int(class_idx)] += 1

        if config.options.generate_plots:
            try:
                draw_landmark_overlay_68(
                    image_path=image_path,
                    landmarks=landmarks,
                    output_path=output_plot_path,
                    show_indices=config.options.show_indices,
                    point_radius=config.options.point_radius,
                    line_width=config.options.line_width,
                    line_color=config.options.line_color,
                )
                row["plot_generated"] = True
            except Exception as error:
                warnings.append(f"Plot generation failed: {error}")
                plot_failure_rows.append(
                    {
                        "dataset_name": config.name,
                        "stem": stem,
                        "source_image_path": str(image_path),
                        "output_plot_path": str(output_plot_path),
                        "error": str(error),
                    }
                )
        row["status"] = "converted"
        row["warning_message"] = " | ".join(warnings)
        report_rows.append(row)

    prediction_only_stems = sorted(
        set(newborn_index) - (set(image_index) | set(pts_index))
    )
    for stem in prediction_only_stems:
        if not any(
            row["stem"] == stem and row["status"] == "unmatched_newborn_prediction"
            for row in report_rows
        ):
            unmatched_rows.append(
                {
                    "dataset_name": config.name,
                    "stem": stem,
                    "source_image_path": "",
                    "source_pts_path": "",
                    "newborn_prediction_path": str(newborn_index[stem]["path"]),
                    "reason": "prediction_without_image_or_label",
                }
            )

    converted_count = sum(1 for row in report_rows if row["converted"] is True)
    plot_count = sum(1 for row in report_rows if row["plot_generated"] is True)
    missing_labels_count = sum(
        1 for row in unmatched_rows if row["reason"] == "image_without_label"
    )
    missing_images_count = len(missing_image_rows)
    invalid_newborn_count = len(missing_newborn_rows)
    malformed_pts_count = len(malformed_pts_rows)
    valid_prediction_count = sum(
        1 for record in newborn_index.values() if record.get("valid")
    )

    _write_csv(reports_dir / "conversion_report.csv", report_rows, REPORT_COLUMNS)
    _write_csv(
        reports_dir / "missing_newborn_predictions.csv",
        missing_newborn_rows,
        [
            "dataset_name",
            "stem",
            "source_image_path",
            "source_pts_path",
            "newborn_prediction_path",
            "reason",
        ],
    )
    _write_csv(
        reports_dir / "missing_images.csv",
        missing_image_rows,
        ["dataset_name", "stem", "source_pts_path", "reason"],
    )
    _write_csv(
        reports_dir / "malformed_pts_files.csv",
        malformed_pts_rows,
        ["dataset_name", "stem", "source_pts_path", "error"],
    )
    _write_csv(
        reports_dir / "unmatched_images_or_labels.csv",
        unmatched_rows,
        [
            "dataset_name",
            "stem",
            "source_image_path",
            "source_pts_path",
            "newborn_prediction_path",
            "reason",
        ],
    )
    _write_csv(
        reports_dir / "plot_generation_failures.csv",
        plot_failure_rows,
        ["dataset_name", "stem", "source_image_path", "output_plot_path", "error"],
    )

    summary: dict[str, object] = {
        "dataset_name": config.name,
        "source_root": str(config.source_root),
        "newborn_predictions_dir": str(config.newborn_predictions_dir),
        "output_dir": str(output_root),
        "source_images": len(image_index),
        "source_pts_labels": len(pts_index),
        "newborn_prediction_files_indexed": len(newborn_index),
        "valid_newborn_prediction_files": valid_prediction_count,
        "successfully_converted_labels": converted_count,
        "overlays_generated": plot_count,
        "missing_images": missing_images_count,
        "missing_labels": missing_labels_count,
        "missing_or_invalid_newborn_predictions": invalid_newborn_count,
        "malformed_pts_files": malformed_pts_count,
        "unmatched_prediction_files": len(prediction_only_stems),
        "coordinate_convention": "pixel coordinates in original image space",
        "landmark_order": "original .pts row order; standard 68-point landmark order assumed",
        "preserve_image_extension": config.options.preserve_image_extension,
        "output_image_extension": "source extension"
        if config.options.preserve_image_extension
        else ".jpg",
        "overwrite_existing": config.options.overwrite_existing,
        "allow_missing_class_idx": config.options.allow_missing_class_idx,
        "class_idx_distribution": dict(sorted(class_counts.items())),
        "warnings": image_warnings + pts_warnings + newborn_warnings,
    }
    _write_dataset_summary_md(
        reports_dir / "conversion_summary.md", config, summary, class_counts
    )
    logging.info(
        "Finished %s: %s converted, %s overlays, %s malformed .pts, %s missing/invalid NewBORN.",
        config.name,
        converted_count,
        plot_count,
        malformed_pts_count,
        invalid_newborn_count,
    )
    return summary


def _write_dataset_summary_md(
    output_path: Path,
    config: PtsDatasetConfig,
    summary: dict[str, object],
    class_counts: Counter[int],
) -> None:
    """Write the per-dataset Markdown conversion summary."""
    class_lines = [f"- class_idx {idx}: {class_counts.get(idx, 0)}" for idx in range(5)]
    warnings = summary.get("warnings", [])
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- None"]
    lines = [
        f"# {config.name} `.pts` 68-Landmark Preparation Summary",
        "",
        "## Inputs and outputs",
        f"- dataset name: `{config.name}`",
        f"- source root: `{config.source_root}`",
        f"- NewBORN predictions dir: `{config.newborn_predictions_dir}`",
        f"- output dir: `{config.output_dataset_root}`",
        f"- generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Conversion summary",
        f"- number of source images: {summary['source_images']}",
        f"- number of source `.pts` labels: {summary['source_pts_labels']}",
        f"- number of NewBORN prediction files indexed: {summary['newborn_prediction_files_indexed']}",
        f"- number of valid NewBORN prediction files: {summary['valid_newborn_prediction_files']}",
        f"- number of successfully converted labels: {summary['successfully_converted_labels']}",
        f"- number of overlays generated: {summary['overlays_generated']}",
        f"- number of missing images: {summary['missing_images']}",
        f"- number of missing labels: {summary['missing_labels']}",
        f"- number of missing or invalid NewBORN predictions: {summary['missing_or_invalid_newborn_predictions']}",
        f"- number of malformed `.pts` files: {summary['malformed_pts_files']}",
        f"- number of unmatched NewBORN prediction files: {summary['unmatched_prediction_files']}",
        "",
        "## Class_idx distribution",
        *class_lines,
        "",
        "## Format assumptions",
        f"- coordinate convention: {summary['coordinate_convention']}",
        f"- landmark order: {summary['landmark_order']}",
        "- label rows: one `class_idx` header followed by 68 `x y` rows when NewBORN class_idx is available",
        f"- image extensions preserved: {summary['preserve_image_extension']}",
        f"- output image extension policy: {summary['output_image_extension']}",
        "- visibility flags added: no",
        "- normalization applied: no",
        "",
        "## Diagnostics",
        "- full conversion rows: `conversion_report.csv`",
        "- missing or invalid NewBORN predictions: `missing_newborn_predictions.csv`",
        "- missing images: `missing_images.csv`",
        "- malformed `.pts` files: `malformed_pts_files.csv`",
        "- unmatched images, labels, or predictions: `unmatched_images_or_labels.csv`",
        "- plot generation failures: `plot_generation_failures.csv`",
        "",
        "## Warnings",
        *warning_lines,
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_multiple_pts_datasets(config_path: Path) -> dict[str, object]:
    """Prepare every dataset listed in one YAML or JSON config file."""
    configs = _load_configs(config_path)
    summaries = [prepare_single_pts_dataset(config) for config in configs]
    output_root = configs[0].output_root
    _write_top_level_summary(output_root, summaries)
    return {"output_root": str(output_root), "datasets": summaries}


def _write_top_level_summary(
    output_root: Path, summaries: list[dict[str, object]]
) -> None:
    """Write top-level CSV and Markdown summaries across all processed datasets."""
    fieldnames = [
        "dataset_name",
        "source_root",
        "newborn_predictions_dir",
        "output_dir",
        "source_images",
        "source_pts_labels",
        "newborn_prediction_files_indexed",
        "valid_newborn_prediction_files",
        "successfully_converted_labels",
        "overlays_generated",
        "missing_images",
        "missing_labels",
        "missing_or_invalid_newborn_predictions",
        "malformed_pts_files",
        "unmatched_prediction_files",
        "coordinate_convention",
        "landmark_order",
        "preserve_image_extension",
        "output_image_extension",
        "overwrite_existing",
        "allow_missing_class_idx",
    ]
    rows = [
        {field: summary.get(field, "") for field in fieldnames} for summary in summaries
    ]
    _write_csv(output_root / "preparation_summary.csv", rows, fieldnames)

    total_converted = sum(
        int(summary["successfully_converted_labels"]) for summary in summaries
    )
    total_plots = sum(int(summary["overlays_generated"]) for summary in summaries)
    lines = [
        "# 68-Landmark `.pts` Dataset Preparation Summary",
        "",
        f"- datasets processed: {len(summaries)}",
        f"- total converted labels: {total_converted}",
        f"- total overlays generated: {total_plots}",
        f"- output root: `{output_root}`",
        f"- generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Datasets",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['dataset_name']}",
                f"- source images: {summary['source_images']}",
                f"- source `.pts` labels: {summary['source_pts_labels']}",
                f"- NewBORN predictions indexed: {summary['newborn_prediction_files_indexed']}",
                f"- converted labels: {summary['successfully_converted_labels']}",
                f"- overlays generated: {summary['overlays_generated']}",
                f"- missing images: {summary['missing_images']}",
                f"- missing labels: {summary['missing_labels']}",
                f"- missing or invalid NewBORN predictions: {summary['missing_or_invalid_newborn_predictions']}",
                f"- malformed `.pts` files: {summary['malformed_pts_files']}",
                f"- coordinate convention: {summary['coordinate_convention']}",
                "",
            ]
        )
    (output_root / "preparation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare 68-landmark datasets from .pts labels and NewBORN class predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML or JSON conversion config."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional terminal log path. Defaults to <output_root>/preparation.log.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the dataset preparation workflow from the command line."""
    args = parse_args()
    configs = _load_configs(args.config)
    log_path = args.log_path or configs[0].output_root / "preparation.log"
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with tee_terminal_output(log_path):
        summaries = [prepare_single_pts_dataset(config) for config in configs]
        _write_top_level_summary(configs[0].output_root, summaries)
        print("[INFO] 68-landmark .pts dataset preparation finished.")
        print(f"[INFO] Output root: {configs[0].output_root}")
        print(f"[INFO] Datasets processed: {len(summaries)}")
        print(
            "[INFO] Converted labels: "
            f"{sum(int(summary['successfully_converted_labels']) for summary in summaries)}"
        )
        print(
            f"[INFO] Top-level summary: {configs[0].output_root / 'preparation_summary.md'}"
        )


if __name__ == "__main__":
    main()
