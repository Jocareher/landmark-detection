from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


LANDMARK_NAMES_72 = [
    *(f"face_contour_{index}" for index in range(1, 18)),
    *(f"right_eyebrow_{index}" for index in range(18, 23)),
    *(f"left_eyebrow_{index}" for index in range(23, 28)),
    *(f"nose_bridge_{index}" for index in range(28, 32)),
    *(f"nose_base_{index}" for index in range(32, 37)),
    *(f"right_eye_{index}" for index in range(37, 43)),
    *(f"left_eye_{index}" for index in range(43, 49)),
    *(f"outer_lip_{index}" for index in range(49, 61)),
    *(f"inner_lip_{index}" for index in range(61, 69)),
    "under_lip_69",
    "upper_chin70",
    "left_chin_71",
    "right_chin_72",
]


LANDMARK_CONNECTIONS = [
    # Jaw
    list(range(0, 17)),
    # Right eyebrow
    list(range(17, 22)),
    # Left eyebrow
    list(range(22, 27)),
    # Nose bridge
    list(range(27, 31)),
    # Nose base
    list(range(31, 36)),
    # Right eye (closed loop)
    [36, 37, 38, 39, 40, 41, 36],
    # Left eye
    [42, 43, 44, 45, 46, 47, 42],
    # Outer lip
    list(range(48, 60)) + [48],
    # Inner lip
    list(range(60, 68)) + [60],
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")
ROTATION_LABELS = ("left", "quarter_left", "frontal", "quarter_right", "right")
Landmark = Tuple[float, float, int]
LandmarkInstance = List[Landmark]


@dataclass(frozen=True)
class DatasetSample:
    split: str | None
    image_file: Path
    label_file: Path
    rotation: str | None


def process_landmark_annotations(
    json_path: str | Path,
    output_dir: str | Path,
    images_dir: str | Path,
    normalize: bool = False,
    landmarks_order: Sequence[str] = LANDMARK_NAMES_72,
) -> None:
    """Convert JSON annotations into one `.txt` file per image."""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)

    images_output_dir = output_dir / "images"
    labels_output_dir = output_dir / "labels"
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as handle:
        annotations = json.load(handle)

    for annotation in annotations:
        image_path = annotation["image"].split("?d=")[-1]
        image_name = Path(image_path).name
        original_width = annotation["landmarks"][0]["original_width"]
        original_height = annotation["landmarks"][0]["original_height"]

        landmarks_data = {name: (np.nan, np.nan) for name in landmarks_order}
        for landmark in annotation["landmarks"]:
            label = landmark["keypointlabels"][0]
            x_coord, y_coord = landmark["x"], landmark["y"]

            if 0 <= x_coord <= 100 and 0 <= y_coord <= 100:
                x_coord = (x_coord / 100) * original_width
                y_coord = (y_coord / 100) * original_height

            if normalize:
                x_coord /= original_width
                y_coord /= original_height
                point = (x_coord, y_coord)
            else:
                point = (int(round(x_coord)), int(round(y_coord)))

            landmarks_data[label] = point

        instance_line = " ".join(
            f"{x_coord:.6f} {y_coord:.6f}"
            if normalize
            else f"{x_coord} {y_coord}"
            if not (np.isnan(x_coord) or np.isnan(y_coord))
            else "nan nan"
            for x_coord, y_coord in landmarks_data.values()
        )

        output_file = labels_output_dir / f"{Path(image_name).stem}.txt"
        output_file.write_text(instance_line + "\n", encoding="utf-8")

        source_image = images_dir / image_name
        if source_image.exists():
            shutil.copy(source_image, images_output_dir / image_name)
        else:
            print(f"Warning: image not found, skipping copy: {source_image}")


def remap_visibility(visibility: int) -> int:
    """Map labels from `1/2` to `0/1`."""
    mapping = {1: 0, 2: 1}
    if visibility not in mapping:
        raise ValueError(f"Unexpected visibility label: {visibility}")
    return mapping[visibility]


def parse_annotation_line(line: str, num_landmarks: int = 72) -> LandmarkInstance:
    """Parse `class cx cy w h x1 y1 v1 ...` into `(x, y, v)` tuples."""
    tokens = line.strip().split()
    if not tokens:
        return []

    expected_length = 5 + 3 * num_landmarks
    if len(tokens) != expected_length:
        raise ValueError(
            f"Invalid annotation length. Expected {expected_length}, got {len(tokens)}"
        )

    landmarks: LandmarkInstance = []
    for index in range(5, len(tokens), 3):
        x_coord = float(tokens[index])
        y_coord = float(tokens[index + 1])
        visibility = remap_visibility(int(float(tokens[index + 2])))
        landmarks.append((x_coord, y_coord, visibility))
    return landmarks


def format_landmarks_block(
    landmarks: Sequence[Landmark],
    float_precision: int = 6,
) -> str:
    return "\n".join(
        f"{x_coord:.{float_precision}f} {y_coord:.{float_precision}f} {visibility}"
        for x_coord, y_coord, visibility in landmarks
    )


def convert_label_file(
    input_file: str | Path,
    output_file: str | Path,
    num_landmarks: int = 72,
    float_precision: int = 6,
    separate_instances_with_blank_line: bool = True,
) -> None:
    """Convert one legacy label file into the `x y v` format."""
    input_file = Path(input_file)
    output_file = Path(output_file)

    blocks: List[str] = []
    for raw_line in input_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        blocks.append(
            format_landmarks_block(
                parse_annotation_line(raw_line, num_landmarks=num_landmarks),
                float_precision=float_precision,
            )
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n\n" if separate_instances_with_blank_line else "\n"
    output_text = separator.join(blocks)
    if output_text:
        output_text += "\n"
    output_file.write_text(output_text, encoding="utf-8")


def convert_dataset_labels(
    dataset_root: str | Path,
    splits: Sequence[str] = ("train", "val", "test"),
    num_landmarks: int = 72,
    float_precision: int = 6,
    separate_instances_with_blank_line: bool = True,
    output_labels_dirname: str | None = None,
) -> None:
    dataset_root = Path(dataset_root)
    for split_name in splits:
        split_dir = dataset_root / split_name
        input_labels_dir = split_dir / "labels"
        if not input_labels_dir.exists():
            continue

        output_labels_dir = (
            input_labels_dir
            if output_labels_dirname is None
            else split_dir / output_labels_dirname
        )
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        for input_file in sorted(input_labels_dir.glob("*.txt")):
            output_file = (
                input_file
                if output_labels_dir.resolve() == input_labels_dir.resolve()
                else output_labels_dir / input_file.name
            )
            convert_label_file(
                input_file=input_file,
                output_file=output_file,
                num_landmarks=num_landmarks,
                float_precision=float_precision,
                separate_instances_with_blank_line=separate_instances_with_blank_line,
            )


def preview_converted_label(
    input_file: str | Path,
    num_landmarks: int = 72,
    float_precision: int = 6,
    separate_instances_with_blank_line: bool = True,
) -> str:
    input_file = Path(input_file)
    blocks: List[str] = []
    for raw_line in input_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        blocks.append(
            format_landmarks_block(
                parse_annotation_line(raw_line, num_landmarks=num_landmarks),
                float_precision=float_precision,
            )
        )
    separator = "\n\n" if separate_instances_with_blank_line else "\n"
    return separator.join(blocks)


def extract_rotation_from_filename(filename: str) -> str | None:
    stem = Path(filename).stem
    for rotation in ("quarter_left", "quarter_right", "frontal", "left", "right"):
        if stem.endswith(f"_{rotation}"):
            return rotation
    return None


def detect_split_names(
    dataset_root: str | Path,
    preferred_splits: Sequence[str] = ("train", "val", "test"),
) -> List[str | None]:
    dataset_root = Path(dataset_root)
    available = [split for split in preferred_splits if (dataset_root / split).is_dir()]
    return available or [None]


def get_split_dirs(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
) -> List[Tuple[str | None, Path]]:
    dataset_root = Path(dataset_root)
    split_names = (
        list(splits) if splits is not None else detect_split_names(dataset_root)
    )
    if split_names == [None]:
        return [(None, dataset_root)]
    return [(split_name, dataset_root / split_name) for split_name in split_names]


def find_image_for_label(
    label_file: str | Path,
    images_dir: str | Path,
    allowed_suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> Path:
    label_file = Path(label_file)
    images_dir = Path(images_dir)
    for suffix in allowed_suffixes:
        candidate = images_dir / f"{label_file.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No matching image found for {label_file.name} in {images_dir}"
    )


def iter_dataset_samples(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
    labels_dirname: str = "labels",
    images_dirname: str = "images",
    allowed_suffixes: Sequence[str] = IMAGE_SUFFIXES,
) -> Iterator[DatasetSample]:
    for split_name, split_dir in get_split_dirs(dataset_root, splits=splits):
        labels_dir = split_dir / labels_dirname
        images_dir = split_dir / images_dirname
        if not labels_dir.exists() or not images_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            try:
                image_file = find_image_for_label(
                    label_file, images_dir, allowed_suffixes
                )
            except FileNotFoundError:
                continue
            yield DatasetSample(
                split=split_name,
                image_file=image_file,
                label_file=label_file,
                rotation=extract_rotation_from_filename(label_file.name),
            )


def parse_landmark_label_file(
    label_file: str | Path,
    invisible_strategy: str = "keep",
) -> List[LandmarkInstance]:
    """
    Parse label files in the `x y v` format.

    `invisible_strategy` can be:
    - `keep`: preserve original coordinates
    - `nan`: replace invisible points with `NaN`
    - `zero`: replace invisible points with `(0, 0)`
    """
    label_file = Path(label_file)
    instances: List[LandmarkInstance] = []
    current_instance: LandmarkInstance = []

    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current_instance:
                instances.append(current_instance)
                current_instance = []
            continue

        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(f"Invalid landmark line in {label_file}: {line!r}")

        x_coord = float(tokens[0])
        y_coord = float(tokens[1])
        visibility = int(float(tokens[2]))
        if visibility not in (0, 1):
            raise ValueError(f"Invalid visibility in {label_file}: {visibility}")

        if visibility == 0:
            if invisible_strategy == "nan":
                x_coord, y_coord = np.nan, np.nan
            elif invisible_strategy == "zero":
                x_coord, y_coord = 0.0, 0.0
            elif invisible_strategy != "keep":
                raise ValueError(f"Unknown invisible_strategy: {invisible_strategy}")

        current_instance.append((x_coord, y_coord, visibility))

    if current_instance:
        instances.append(current_instance)
    return instances


def load_landmarks_array(
    label_file: str | Path,
    invisible_strategy: str = "keep",
    instance_index: int = 0,
) -> np.ndarray:
    instances = parse_landmark_label_file(
        label_file, invisible_strategy=invisible_strategy
    )
    if instance_index >= len(instances):
        raise IndexError(f"Instance {instance_index} not found in {label_file}")
    return np.asarray(instances[instance_index], dtype=np.float32)


def draw_landmark_connections(
    draw,
    coords: np.ndarray,
    connections,
    color="green",
    width=2,
):
    for connection in connections:
        points = []
        for idx in connection:
            if np.isnan(coords[idx]).any():
                continue
            points.append(tuple(coords[idx]))

        if len(points) >= 2:
            draw.line(points, fill=color, width=width)


def draw_landmarks_on_image(
    image: Image.Image,
    landmark_instances: Sequence[LandmarkInstance],
    point_radius: int = 3,
    draw_invisible: bool = True,
    visible_color: str = "red",
    invisible_color: str = "blue",
) -> Image.Image:
    from PIL import ImageDraw

    output_image = image.copy()
    drawer = ImageDraw.Draw(output_image)
    width, height = output_image.size

    for instance in landmark_instances:
        for x_coord, y_coord, visibility in instance:
            if np.isnan(x_coord) or np.isnan(y_coord):
                continue
            if visibility == 0 and not draw_invisible:
                continue

            x_pixel = x_coord * width
            y_pixel = y_coord * height
            color = visible_color if visibility == 1 else invisible_color
            drawer.ellipse(
                (
                    x_pixel - point_radius,
                    y_pixel - point_radius,
                    x_pixel + point_radius,
                    y_pixel + point_radius,
                ),
                fill=color,
                outline=color,
            )

            # coords_array = np.array([(x, y) for x, y, _ in instance])
            # coords_array[:, 0] *= width
            # coords_array[:, 1] *= height

            # draw_landmark_connections(drawer, coords_array, LANDMARK_CONNECTIONS)

    return output_image


def plot_single_label_on_image(
    image_file: str | Path,
    label_file: str | Path,
    output_file: str | Path,
    point_radius: int = 3,
    invisible_strategy: str = "keep",
    draw_invisible: bool = True,
) -> None:
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    landmark_instances = parse_landmark_label_file(
        label_file,
        invisible_strategy=invisible_strategy,
    )
    plotted = draw_landmarks_on_image(
        image=image,
        landmark_instances=landmark_instances,
        point_radius=point_radius,
        draw_invisible=draw_invisible,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(plotted)
    plt.show()


def plot_dataset_landmarks(
    dataset_root: str | Path,
    output_root: str | Path = "plots",
    splits: Sequence[str] | None = None,
    point_radius: int = 3,
    invisible_strategy: str = "keep",
    draw_invisible: bool = True,
) -> None:
    output_root = Path(output_root)
    for sample in iter_dataset_samples(dataset_root, splits=splits):
        split_name = sample.split or "all"
        output_file = output_root / split_name / f"{sample.label_file.stem}.jpg"
        plot_single_label_on_image(
            image_file=sample.image_file,
            label_file=sample.label_file,
            output_file=output_file,
            point_radius=point_radius,
            invisible_strategy=invisible_strategy,
            draw_invisible=draw_invisible,
        )


def count_landmarks_visibility(label_file: str | Path) -> Tuple[int, int]:
    visible = 0
    invisible = 0
    for instance in parse_landmark_label_file(label_file, invisible_strategy="keep"):
        for _, _, visibility in instance:
            visible += int(visibility == 1)
            invisible += int(visibility == 0)
    return visible, invisible


def analyze_dataset(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for sample in iter_dataset_samples(dataset_root, splits=splits):
        visible, invisible = count_landmarks_visibility(sample.label_file)
        rows.append(
            {
                "split": sample.split or "all",
                "filename": sample.label_file.stem,
                "rotation": sample.rotation,
                "visible_landmarks": visible,
                "invisible_landmarks": invisible,
                "num_landmarks": visible + invisible,
                "has_rotation": sample.rotation is not None,
            }
        )
    return pd.DataFrame(rows)


def visibility_per_landmark(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
    num_landmarks: int = 72,
) -> np.ndarray:
    counts_visible = np.zeros(num_landmarks, dtype=np.float32)
    counts_total = np.zeros(num_landmarks, dtype=np.float32)

    for sample in iter_dataset_samples(dataset_root, splits=splits):
        instances = parse_landmark_label_file(
            sample.label_file, invisible_strategy="keep"
        )
        for instance in instances:
            if len(instance) != num_landmarks:
                raise ValueError(
                    f"Unexpected number of landmarks in {sample.label_file}: {len(instance)}"
                )
            for index, (_, _, visibility) in enumerate(instance):
                counts_total[index] += 1
                counts_visible[index] += visibility

    return np.divide(
        counts_visible,
        counts_total,
        out=np.zeros_like(counts_visible),
        where=counts_total > 0,
    )


def load_shape_coordinates_from_label_file(
    label_file: str | Path,
    invisible_strategy: str = "nan",
    require_single_instance: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one landmark file as coordinates plus a visibility mask.

    Returns:
        coords: array of shape (num_landmarks, 2)
        visible_mask: boolean array of shape (num_landmarks,)
    """
    instances = parse_landmark_label_file(
        label_file, invisible_strategy=invisible_strategy
    )
    if require_single_instance and len(instances) != 1:
        raise ValueError(
            f"Expected exactly one instance in {label_file}, found {len(instances)}"
        )

    instance = instances[0]
    coords = np.asarray(
        [(x_coord, y_coord) for x_coord, y_coord, _ in instance], dtype=np.float32
    )
    visible_mask = np.asarray(
        [visibility == 1 for _, _, visibility in instance], dtype=bool
    )
    return coords, visible_mask


def collect_dataset_shapes(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
    invisible_strategy: str = "nan",
    min_visible_fraction: float = 0.0,
    min_visible_landmarks: int | None = None,
) -> List[dict]:
    """
    Collect samples and landmark coordinates from one dataset.

    Samples can be filtered by minimum visible landmarks.
    """
    records: List[dict] = []
    for sample in iter_dataset_samples(dataset_root, splits=splits):
        coords, visible_mask = load_shape_coordinates_from_label_file(
            sample.label_file,
            invisible_strategy=invisible_strategy,
        )
        num_visible = int(visible_mask.sum())
        required_visible = (
            min_visible_landmarks
            if min_visible_landmarks is not None
            else int(np.ceil(min_visible_fraction * len(visible_mask)))
        )
        if num_visible < required_visible:
            continue

        records.append(
            {
                "dataset": Path(dataset_root).name,
                "split": sample.split or "all",
                "filename": sample.label_file.stem,
                "rotation": sample.rotation,
                "coords": coords,
                "visible_mask": visible_mask,
                "num_visible": num_visible,
                "visible_fraction": num_visible / float(len(visible_mask)),
            }
        )
    return records


def collect_multiple_dataset_shapes(
    dataset_roots: dict[str, str | Path],
    invisible_strategy: str = "nan",
    min_visible_fraction: float = 0.0,
    min_visible_landmarks: int | None = None,
) -> List[dict]:
    records: List[dict] = []
    for dataset_name, dataset_root in dataset_roots.items():
        dataset_records = collect_dataset_shapes(
            dataset_root=dataset_root,
            invisible_strategy=invisible_strategy,
            min_visible_fraction=min_visible_fraction,
            min_visible_landmarks=min_visible_landmarks,
        )
        for record in dataset_records:
            record["dataset"] = dataset_name
        records.extend(dataset_records)
    return records


def _resolve_alignment_mask(
    visible_mask: np.ndarray,
    alignment_landmark_indices: Sequence[int] | None = None,
) -> np.ndarray:
    mask = visible_mask.copy()
    if alignment_landmark_indices is not None:
        subset_mask = np.zeros_like(mask, dtype=bool)
        subset_mask[list(alignment_landmark_indices)] = True
        mask &= subset_mask
    return mask


def _normalize_shape(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered)
    if scale <= 0:
        raise ValueError("Shape has zero scale and cannot be normalized.")
    return centered / scale


def _estimate_similarity_transform(
    source: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Estimate a similarity transform mapping source points onto target points.

    Returns:
        rotation, scale, translation
    """
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)

    source_centered = source - source_center
    target_centered = target - target_center

    source_scale = np.linalg.norm(source_centered)
    target_scale = np.linalg.norm(target_centered)
    if source_scale <= 0 or target_scale <= 0:
        raise ValueError("Degenerate shape encountered during Procrustes alignment.")

    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale

    covariance = source_normalized.T @ target_normalized
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = u_matrix @ vt_matrix

    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1
        rotation = u_matrix @ vt_matrix

    scale = target_scale / source_scale
    translation = target_center - scale * (source_center @ rotation)
    return rotation, scale, translation


def _apply_similarity_transform(
    coords: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    transformed = np.full_like(coords, np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(coords).any(axis=1)
    transformed[valid_mask] = (scale * (coords[valid_mask] @ rotation)) + translation
    return transformed


def _compute_mean_shape(aligned_shapes: np.ndarray) -> np.ndarray:
    mean_shape = np.nanmean(aligned_shapes, axis=0)
    valid_mask = ~np.isnan(mean_shape).any(axis=1)
    mean_shape[valid_mask] = _normalize_shape(mean_shape[valid_mask])
    return mean_shape.astype(np.float32)


def generalized_procrustes_analysis(
    shapes: np.ndarray,
    visible_masks: np.ndarray,
    alignment_landmark_indices: Sequence[int] | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
    min_alignment_landmarks: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run generalized Procrustes analysis with missing landmarks.

    Shapes are aligned using only landmarks visible in both the sample and the
    current reference shape. Missing points remain as NaN.
    """
    if len(shapes) == 0:
        raise ValueError("At least one shape is required for GPA.")

    shapes = np.asarray(shapes, dtype=np.float32)
    visible_masks = np.asarray(visible_masks, dtype=bool)

    reference_shape = None
    for coords, visible_mask in zip(shapes, visible_masks):
        mask = _resolve_alignment_mask(visible_mask, alignment_landmark_indices)
        if mask.sum() >= min_alignment_landmarks:
            reference_shape = np.full_like(coords, np.nan, dtype=np.float32)
            reference_shape[mask] = _normalize_shape(coords[mask])
            break

    if reference_shape is None:
        raise ValueError("No valid reference shape found for GPA.")

    aligned_shapes = np.full_like(shapes, np.nan, dtype=np.float32)

    for _ in range(max_iterations):
        for sample_index, (coords, visible_mask) in enumerate(
            zip(shapes, visible_masks)
        ):
            sample_mask = _resolve_alignment_mask(
                visible_mask, alignment_landmark_indices
            )
            reference_mask = ~np.isnan(reference_shape).any(axis=1)
            common_mask = sample_mask & reference_mask

            if common_mask.sum() < min_alignment_landmarks:
                continue

            rotation, scale, translation = _estimate_similarity_transform(
                coords[common_mask],
                reference_shape[common_mask],
            )
            aligned_shapes[sample_index] = _apply_similarity_transform(
                coords,
                rotation=rotation,
                scale=scale,
                translation=translation,
            )

        new_reference = _compute_mean_shape(aligned_shapes)
        difference = np.nanmean((new_reference - reference_shape) ** 2)
        reference_shape = new_reference

        if np.isfinite(difference) and difference < tolerance:
            break

    return aligned_shapes, reference_shape


def impute_missing_landmarks(
    aligned_shapes: np.ndarray,
    reference_shape: np.ndarray,
) -> np.ndarray:
    imputed = aligned_shapes.copy()
    for sample_index in range(imputed.shape[0]):
        missing_mask = np.isnan(imputed[sample_index]).any(axis=1)
        imputed[sample_index, missing_mask] = reference_shape[missing_mask]
    return imputed


def build_aligned_shape_dataframe(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
    invisible_strategy: str = "nan",
    min_visible_fraction: float = 0.7,
    min_visible_landmarks: int | None = None,
    alignment_landmark_indices: Sequence[int] | None = None,
    impute_missing: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    import pandas as pd

    records = collect_dataset_shapes(
        dataset_root=dataset_root,
        splits=splits,
        invisible_strategy=invisible_strategy,
        min_visible_fraction=min_visible_fraction,
        min_visible_landmarks=min_visible_landmarks,
    )
    if not records:
        raise ValueError(f"No valid samples found in {dataset_root}")

    shapes = np.stack([record["coords"] for record in records], axis=0)
    visible_masks = np.stack([record["visible_mask"] for record in records], axis=0)
    aligned_shapes, mean_shape = generalized_procrustes_analysis(
        shapes=shapes,
        visible_masks=visible_masks,
        alignment_landmark_indices=alignment_landmark_indices,
    )

    if impute_missing:
        aligned_shapes = impute_missing_landmarks(aligned_shapes, mean_shape)

    rows = []
    for record, aligned_shape in zip(records, aligned_shapes):
        row = {
            "dataset": record["dataset"],
            "split": record["split"],
            "filename": record["filename"],
            "rotation": record["rotation"],
            "num_visible": record["num_visible"],
            "visible_fraction": record["visible_fraction"],
        }
        flattened = aligned_shape.reshape(-1)
        for index, value in enumerate(flattened):
            row[f"f_{index:03d}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    if impute_missing:
        feature_columns = [column for column in df.columns if column.startswith("f_")]
        df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df, mean_shape


def build_joint_aligned_shape_dataframe(
    dataset_roots: dict[str, str | Path],
    invisible_strategy: str = "nan",
    min_visible_fraction: float = 0.7,
    min_visible_landmarks: int | None = None,
    alignment_landmark_indices: Sequence[int] | None = None,
    impute_missing: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    import pandas as pd

    records = collect_multiple_dataset_shapes(
        dataset_roots=dataset_roots,
        invisible_strategy=invisible_strategy,
        min_visible_fraction=min_visible_fraction,
        min_visible_landmarks=min_visible_landmarks,
    )
    if not records:
        raise ValueError("No valid samples found across the provided datasets.")

    shapes = np.stack([record["coords"] for record in records], axis=0)
    visible_masks = np.stack([record["visible_mask"] for record in records], axis=0)
    aligned_shapes, mean_shape = generalized_procrustes_analysis(
        shapes=shapes,
        visible_masks=visible_masks,
        alignment_landmark_indices=alignment_landmark_indices,
    )

    if impute_missing:
        aligned_shapes = impute_missing_landmarks(aligned_shapes, mean_shape)

    rows = []
    for record, aligned_shape in zip(records, aligned_shapes):
        row = {
            "dataset": record["dataset"],
            "split": record["split"],
            "filename": record["filename"],
            "rotation": record["rotation"],
            "num_visible": record["num_visible"],
            "visible_fraction": record["visible_fraction"],
        }
        flattened = aligned_shape.reshape(-1)
        for index, value in enumerate(flattened):
            row[f"f_{index:03d}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    if impute_missing:
        feature_columns = [column for column in df.columns if column.startswith("f_")]
        df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df, mean_shape


def load_shape_vector_from_label_file(
    label_file: str | Path,
    invisible_strategy: str = "keep",
    require_single_instance: bool = True,
) -> np.ndarray:
    instances = parse_landmark_label_file(
        label_file, invisible_strategy=invisible_strategy
    )
    if require_single_instance and len(instances) != 1:
        raise ValueError(
            f"Expected exactly one instance in {label_file}, found {len(instances)}"
        )

    instance = instances[0]
    coords: List[float] = []
    for x_coord, y_coord, _ in instance:
        coords.extend([x_coord, y_coord])
    return np.asarray(coords, dtype=np.float32)


def build_pca_dataframe(
    dataset_root: str | Path,
    splits: Sequence[str] | None = None,
    invisible_strategy: str = "keep",
    drop_rows_with_nan: bool = True,
) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for sample in iter_dataset_samples(dataset_root, splits=splits):
        shape_vector = load_shape_vector_from_label_file(
            sample.label_file,
            invisible_strategy=invisible_strategy,
        )
        row = {
            "split": sample.split or "all",
            "filename": sample.label_file.stem,
            "rotation": sample.rotation,
        }
        for index, value in enumerate(shape_vector):
            row[f"f_{index:03d}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    feature_columns = [column for column in df.columns if column.startswith("f_")]
    if drop_rows_with_nan and feature_columns:
        df = df.dropna(subset=feature_columns).reset_index(drop=True)
    return df


def run_shape_pca(
    df_shapes: pd.DataFrame,
    n_components: int = 10,
) -> Tuple[pd.DataFrame, PCA]:
    from sklearn.decomposition import PCA

    feature_columns = [
        column for column in df_shapes.columns if column.startswith("f_")
    ]
    x_matrix = df_shapes[feature_columns].to_numpy(dtype=np.float32)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(x_matrix)

    metadata_columns = [
        column for column in ("split", "filename", "rotation") if column in df_shapes
    ]
    df_pca = df_shapes[metadata_columns].copy()
    for index in range(transformed.shape[1]):
        df_pca[f"PC{index + 1}"] = transformed[:, index]
    return df_pca, pca


def add_procrustes_distance_column(
    df_shapes: pd.DataFrame,
    reference_vector: np.ndarray | None = None,
    column_name: str = "procrustes_distance",
) -> pd.DataFrame:
    feature_columns = [
        column for column in df_shapes.columns if column.startswith("f_")
    ]
    x_matrix = df_shapes[feature_columns].to_numpy(dtype=np.float32)
    if reference_vector is None:
        reference_vector = x_matrix.mean(axis=0)
    distances = np.linalg.norm(x_matrix - reference_vector[None, :], axis=1)
    df_out = df_shapes.copy()
    df_out[column_name] = distances
    return df_out


def run_pose_specific_pca(
    df_shapes: pd.DataFrame,
    rotation: str,
    n_components: int = 10,
) -> PCA:
    from sklearn.decomposition import PCA

    if "rotation" not in df_shapes.columns:
        raise ValueError("The dataframe does not contain a 'rotation' column.")
    df_pose = df_shapes[df_shapes["rotation"] == rotation]
    if df_pose.empty:
        raise ValueError(f"No samples found for rotation: {rotation}")

    feature_columns = [column for column in df_pose.columns if column.startswith("f_")]
    x_matrix = df_pose[feature_columns].to_numpy(dtype=np.float32)
    pca = PCA(n_components=n_components)
    pca.fit(x_matrix)
    return pca


def plot_pca_scatter(
    df_pca: pd.DataFrame,
    x_component: str = "PC1",
    y_component: str = "PC2",
    color_by: str = "rotation",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    if color_by in df_pca.columns:
        grouped = sorted(df_pca[color_by].fillna("unknown").unique())
        for group in grouped:
            subset = df_pca[df_pca[color_by].fillna("unknown") == group]
            plt.scatter(
                subset[x_component], subset[y_component], label=group, alpha=0.65, s=18
            )
        plt.legend()
    else:
        plt.scatter(df_pca[x_component], df_pca[y_component], alpha=0.65, s=18)

    plt.xlabel(x_component)
    plt.ylabel(y_component)
    plt.title(f"{x_component} vs {y_component}")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_explained_variance(
    pca: PCA,
    figsize: Tuple[int, int] = (8, 4),
) -> None:
    import matplotlib.pyplot as plt

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    x_axis = np.arange(1, len(explained) + 1)

    plt.figure(figsize=figsize)
    plt.plot(x_axis, explained, marker="o", label="Explained variance ratio")
    plt.plot(x_axis, cumulative, marker="o", label="Cumulative explained variance")
    plt.xlabel("Principal component")
    plt.ylabel("Variance ratio")
    plt.title("PCA explained variance")
    plt.xticks(x_axis)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_pose_specific_variance(pca: PCA, title: str) -> None:
    import matplotlib.pyplot as plt

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    x_axis = np.arange(1, len(explained) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(x_axis, explained, marker="o", label="explained")
    plt.plot(x_axis, cumulative, marker="o", label="cumulative")
    plt.xlabel("component")
    plt.ylabel("variance")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_shape_configuration(
    shape: np.ndarray,
    title: str = "Shape",
    figsize: Tuple[int, int] = (5, 5),
    color: str = "black",
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.scatter(shape[:, 0], shape[:, 1], c=color, s=20)
    plt.gca().set_aspect("equal")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()


def visualize_eigenshape(
    pca: PCA,
    component_index: int = 0,
    scale: float = 2.0,
) -> None:
    import matplotlib.pyplot as plt

    mean_shape = pca.mean_.reshape(-1, 2)
    eigenvector = pca.components_[component_index].reshape(-1, 2)
    shape_plus = mean_shape + scale * eigenvector
    shape_minus = mean_shape - scale * eigenvector

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, shape, title, color in zip(
        axes,
        (shape_minus, mean_shape, shape_plus),
        ("- Mode", "Mean shape", "+ Mode"),
        ("red", "black", "blue"),
    ):
        axis.scatter(shape[:, 0], shape[:, 1], c=color, s=20)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.grid(alpha=0.3)

    plt.suptitle(f"PCA Mode {component_index + 1}")
    plt.tight_layout()
    plt.show()
