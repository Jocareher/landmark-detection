from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def denormalize_image_tensor(
    image: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Undo channel-wise normalization on one image tensor."""
    if image.ndim != 3:
        raise ValueError(
            f"Expected image with shape (C, H, W), got {tuple(image.shape)}."
        )
    mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device).view(
        -1, 1, 1
    )
    std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device).view(
        -1, 1, 1
    )
    return image.clone() * std_tensor + mean_tensor


def _resize_heatmap_to_image(
    heatmap: torch.Tensor, image_height: int, image_width: int
) -> np.ndarray:
    """Resize a heatmap tensor to match the spatial size of the source image."""
    resized_heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized_heatmap.squeeze(0).squeeze(0).cpu().numpy()


def visualize_predicted_heatmaps_on_train_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str | Path,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    num_images: int = 4,
    grid_cols: int = 4,
    overlay_alpha: float = 0.45,
    use_max_projection: bool = True,
    normalize_heatmap: bool = True,
    use_wandb: bool = False,
) -> Path:
    """Overlay predicted heatmaps on a fixed set of samples and save the figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()
    dataset = dataloader.dataset
    fixed_sample_indices = [
        index for index in range(min(len(dataset), max(num_images, 8)))
    ]
    if not fixed_sample_indices:
        raise RuntimeError("The dataset is empty. Cannot visualize fixed samples.")

    fixed_sample_indices = fixed_sample_indices[:num_images]
    selected_samples = [dataset[index] for index in fixed_sample_indices]
    images = torch.stack([sample["image"] for sample in selected_samples], dim=0).to(
        device, non_blocking=True
    )
    metadata_list = [sample.get("metadata", {}) for sample in selected_samples]

    with torch.no_grad():
        outputs = model(images)

    predicted_heatmaps = outputs["heatmaps"].detach().cpu()
    images_cpu = images.detach().cpu()
    num_images = min(num_images, images_cpu.shape[0])
    num_rows = math.ceil(num_images / grid_cols)
    fig, axes = plt.subplots(num_rows, grid_cols, figsize=(grid_cols * 4, num_rows * 4))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for image_index in range(num_images):
        # Aggregate all landmark heatmaps into a single overlay for quick inspection.
        image_tensor = denormalize_image_tensor(
            images_cpu[image_index], mean=mean, std=std
        )
        image_np = image_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        image_height, image_width = image_np.shape[:2]
        current_heatmaps = predicted_heatmaps[image_index]
        aggregated_heatmap = (
            current_heatmaps.max(dim=0).values
            if use_max_projection
            else current_heatmaps.sum(dim=0)
        )
        heatmap_np = _resize_heatmap_to_image(
            aggregated_heatmap, image_height, image_width
        )

        if normalize_heatmap:
            heatmap_min = float(heatmap_np.min())
            heatmap_max = float(heatmap_np.max())
            if heatmap_max > heatmap_min:
                heatmap_np = (heatmap_np - heatmap_min) / (heatmap_max - heatmap_min)

        axes[image_index].imshow(image_np)
        axes[image_index].imshow(heatmap_np, cmap="jet", alpha=overlay_alpha)
        axes[image_index].axis("off")
        sample_title = metadata_list[image_index].get(
            "sample_id", f"sample_{fixed_sample_indices[image_index]}"
        )
        axes[image_index].set_title(sample_title, fontsize=10)

    for axis_index in range(num_images, len(axes)):
        axes[axis_index].axis("off")

    plt.tight_layout()
    save_path = output_dir / f"train_heatmaps_epoch_{epoch + 1:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if use_wandb:
        import wandb

        wandb.log(
            {
                "train/predicted_heatmaps_overlay": wandb.Image(
                    str(save_path), caption=f"Epoch {epoch + 1}"
                )
            }
        )

    if model_was_training:
        model.train()
    return save_path


def get_default_landmark_names() -> list[str]:
    """Return the default ordered list of 72 landmark names."""
    return [
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


def get_landmark_region_definitions() -> list[tuple[str, range, str]]:
    """Return landmark groups together with their display colors."""
    return [
        ("Face contour", range(0, 17), "#4C78A8"),
        ("Right eyebrow", range(17, 22), "#F58518"),
        ("Left eyebrow", range(22, 27), "#E45756"),
        ("Nose bridge", range(27, 31), "#72B7B2"),
        ("Nose base", range(31, 36), "#54A24B"),
        ("Right eye", range(36, 42), "#EECA3B"),
        ("Left eye", range(42, 48), "#B279A2"),
        ("Outer lip", range(48, 60), "#FF9DA6"),
        ("Inner lip", range(60, 68), "#9D755D"),
        ("Under lip", range(68, 69), "#BAB0AC"),
        ("Upper chin", range(69, 70), "#2F4B7C"),
        ("Left chin", range(70, 71), "#D45087"),
        ("Right chin", range(71, 72), "#7F7F7F"),
    ]


def get_landmark_connection_definitions() -> list[tuple[range, bool]]:
    """Return landmark index groups and whether they should be drawn as closed loops."""
    return [
        (range(0, 17), False),
        (range(17, 22), False),
        (range(22, 27), False),
        (range(27, 31), False),
        (range(31, 36), False),
        (range(36, 42), True),
        (range(42, 48), True),
        (range(48, 60), True),
        (range(60, 68), True),
        (range(68, 69), False),
        (range(69, 70), False),
        (range(70, 71), False),
        (range(71, 72), False),
    ]


def get_landmark_region_colors(num_landmarks: int) -> list[str]:
    """Assign one display color per landmark based on its semantic region."""
    colors = ["#999999"] * num_landmarks
    for _, landmark_range, color in get_landmark_region_definitions():
        for landmark_index in landmark_range:
            if 0 <= landmark_index < num_landmarks:
                colors[landmark_index] = color
    return colors


def save_landmark_overlay_image(
    image_path: Path,
    output_path: Path,
    predicted_landmarks: np.ndarray,
    predicted_visibility: np.ndarray,
    show_indices: bool = False,
    point_radius: int = 10,
    line_width: int = 4,
    line_color: str = "#FFD400",
) -> None:
    """Render predicted landmarks on top of the original image and save it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for landmark_range, close_loop in get_landmark_connection_definitions():
        connected_points = []
        for landmark_index in landmark_range:
            if landmark_index >= len(predicted_landmarks):
                continue
            x_coord, y_coord = predicted_landmarks[landmark_index]
            connected_points.append((float(x_coord), float(y_coord)))

        if len(connected_points) >= 2:
            draw.line(
                connected_points, fill=line_color, width=line_width, joint="curve"
            )
            if close_loop:
                draw.line(
                    [connected_points[-1], connected_points[0]],
                    fill=line_color,
                    width=line_width,
                    joint="curve",
                )

    for landmark_index, (x_coord, y_coord) in enumerate(predicted_landmarks):
        visibility_value = int(predicted_visibility[landmark_index])
        color = "blue" if visibility_value == 0 else "red"
        outline_color = "white" if visibility_value == 1 else "black"

        left = x_coord - point_radius
        top = y_coord - point_radius
        right = x_coord + point_radius
        bottom = y_coord + point_radius

        draw.ellipse(
            (left, top, right, bottom),
            fill=color,
            outline=outline_color,
            width=max(2, line_width // 2),
        )

        if show_indices:
            draw.text(
                (x_coord + point_radius + 2, y_coord + point_radius + 2),
                str(landmark_index),
                fill=color,
            )

    image.save(output_path)


@contextmanager
def fixed_visualization_seed(seed: int):
    """Temporarily fix RNG states so preview generation stays reproducible."""
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def render_landmark_preview_image(
    image: torch.Tensor | np.ndarray | Image.Image,
    landmarks: np.ndarray,
    visibility: np.ndarray,
    show_indices: bool = False,
    point_radius: int = 6,
    line_width: int = 2,
    line_color: str = "#00C853",
    draw_all_connections: bool = False,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """Render one landmark preview image using the repository visibility colors."""
    if isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got {tuple(image.shape)}.")
        image_tensor = denormalize_image_tensor(
            image.detach().cpu(), mean=mean, std=std
        )
        image_np = image_tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        rendered_image = Image.fromarray((image_np * 255.0).astype(np.uint8))
    elif isinstance(image, np.ndarray):
        rendered_image = Image.fromarray(image.astype(np.uint8))
    elif isinstance(image, Image.Image):
        rendered_image = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}.")

    draw = ImageDraw.Draw(rendered_image)

    for landmark_range, close_loop in get_landmark_connection_definitions():
        connected_points = []
        for landmark_index in landmark_range:
            if landmark_index >= len(landmarks):
                continue
            if not draw_all_connections and int(visibility[landmark_index]) != 1:
                continue
            x_coord, y_coord = landmarks[landmark_index]
            connected_points.append((float(x_coord), float(y_coord)))

        if len(connected_points) >= 2:
            draw.line(
                connected_points, fill=line_color, width=line_width, joint="curve"
            )
            if close_loop:
                draw.line(
                    [connected_points[-1], connected_points[0]],
                    fill=line_color,
                    width=line_width,
                    joint="curve",
                )

    for landmark_index, (x_coord, y_coord) in enumerate(landmarks):
        visibility_value = int(visibility[landmark_index])
        color = "red" if visibility_value == 1 else "blue"
        outline_color = "white" if visibility_value == 1 else "black"
        left = x_coord - point_radius
        top = y_coord - point_radius
        right = x_coord + point_radius
        bottom = y_coord + point_radius
        draw.ellipse(
            (left, top, right, bottom),
            fill=color,
            outline=outline_color,
            width=max(1, line_width // 2),
        )
        if show_indices:
            draw.text(
                (x_coord + point_radius + 2, y_coord + point_radius + 2),
                str(landmark_index),
                fill=color,
            )

    return rendered_image


def save_dataset_preview_grid(
    dataset,
    output_path: Path,
    title: str,
    num_samples: int,
    seed: int,
    show_indices: bool = False,
    point_radius: int = 6,
    line_width: int = 2,
    line_color: str = "#00C853",
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> Path:
    """Save a deterministic preview grid for a fixed subset of one dataset split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_count = min(len(dataset), max(1, int(num_samples)))
    sample_indices = list(range(preview_count))

    with fixed_visualization_seed(seed):
        samples = [dataset[index] for index in sample_indices]

    grid_cols = min(4, preview_count)
    grid_rows = math.ceil(preview_count / grid_cols)
    figure, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(grid_cols * 4.0, grid_rows * 4.0),
    )
    axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

    for axis, sample, sample_index in zip(axes, samples, sample_indices):
        rendered_image = render_landmark_preview_image(
            image=sample["image"],
            landmarks=np.asarray(sample["landmarks"]),
            visibility=np.asarray(sample["visibility"]),
            show_indices=show_indices,
            point_radius=point_radius,
            line_width=line_width,
            line_color=line_color,
            mean=mean,
            std=std,
        )
        axis.imshow(np.asarray(rendered_image))
        axis.axis("off")
        sample_id = sample.get("metadata", {}).get(
            "sample_id", f"sample_{sample_index}"
        )
        axis.set_title(str(sample_id), fontsize=10)

    for axis in axes[len(samples) :]:
        axis.axis("off")

    figure.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_landmark_comparison_overlay_image(
    image_path: Path,
    output_path: Path,
    predicted_landmarks: np.ndarray,
    predicted_visibility: np.ndarray,
    target_landmarks: np.ndarray,
    target_visibility: np.ndarray,
    show_indices: bool = False,
    point_radius: int = 10,
    line_width: int = 4,
    predicted_line_color: str = "#FFD400",
    target_line_color: str = "#00C853",
) -> None:
    """Render GT and predicted landmarks together on top of one source image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    def draw_connections(
        landmarks: np.ndarray,
        visibility: np.ndarray | None,
        color: str,
        draw_all_connections: bool,
    ) -> None:
        for landmark_range, close_loop in get_landmark_connection_definitions():
            connected_points = []
            for landmark_index in landmark_range:
                if landmark_index >= len(landmarks):
                    continue
                if (
                    not draw_all_connections
                    and visibility is not None
                    and int(visibility[landmark_index]) != 1
                ):
                    continue
                x_coord, y_coord = landmarks[landmark_index]
                connected_points.append((float(x_coord), float(y_coord)))

            if len(connected_points) >= 2:
                draw.line(
                    connected_points,
                    fill=color,
                    width=line_width,
                    joint="curve",
                )
                if close_loop:
                    draw.line(
                        [connected_points[-1], connected_points[0]],
                        fill=color,
                        width=line_width,
                        joint="curve",
                    )

    draw_connections(
        landmarks=target_landmarks,
        visibility=target_visibility,
        color=target_line_color,
        draw_all_connections=False,
    )
    draw_connections(
        landmarks=predicted_landmarks,
        visibility=predicted_visibility,
        color=predicted_line_color,
        draw_all_connections=True,
    )

    for landmark_index, (x_coord, y_coord) in enumerate(target_landmarks):
        visibility_value = int(target_visibility[landmark_index])
        if visibility_value != 1:
            continue
        left = x_coord - point_radius
        top = y_coord - point_radius
        right = x_coord + point_radius
        bottom = y_coord + point_radius
        draw.ellipse(
            (left, top, right, bottom),
            fill="#00C853",
            outline="white",
            width=max(2, line_width // 2),
        )
        if show_indices:
            draw.text(
                (x_coord + point_radius + 2, y_coord - point_radius - 2),
                f"gt:{landmark_index}",
                fill="#00C853",
            )

    for landmark_index, (x_coord, y_coord) in enumerate(predicted_landmarks):
        visibility_value = int(predicted_visibility[landmark_index])
        color = "#FF3D00" if visibility_value == 1 else "#2962FF"
        outline_color = "white" if visibility_value == 1 else "black"
        left = x_coord - point_radius
        top = y_coord - point_radius
        right = x_coord + point_radius
        bottom = y_coord + point_radius
        draw.rectangle(
            (left, top, right, bottom),
            fill=color,
            outline=outline_color,
            width=max(2, line_width // 2),
        )
        if show_indices:
            draw.text(
                (x_coord + point_radius + 2, y_coord + point_radius + 2),
                f"pr:{landmark_index}",
                fill=color,
            )

    image.save(output_path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    value_format: str,
) -> None:
    """Plot and save a binary confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")

    class_labels = ["Visible (0)", "Invisible (1)"]
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(class_labels, fontsize=11)
    axis.set_yticklabels(class_labels, fontsize=11)
    axis.set_xlabel("Predicted label", fontsize=12)
    axis.set_ylabel("Ground-truth label", fontsize=12)
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)

    threshold = float(matrix.max()) * 0.55 if matrix.size > 0 else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = "white" if value >= threshold else "black"
            axis.text(
                column_index,
                row_index,
                format(value, value_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=10)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_per_landmark_boxplot(
    per_landmark_errors: list[list[float]],
    output_path: Path,
    use_landmark_names: bool = True,
    title: str = "Per-landmark NME distribution",
    y_limits: tuple[float, float] | None = None,
    y_scale: str = "log",
) -> None:
    """Plot a per-landmark NME boxplot using semantic region colors."""
    landmark_names = get_default_landmark_names()
    number_of_landmarks = len(per_landmark_errors)
    landmark_colors = get_landmark_region_colors(number_of_landmarks)
    displayed_landmark_names = landmark_names[:number_of_landmarks]

    if y_scale not in {"log", "linear"}:
        raise ValueError(f"Unsupported y_scale '{y_scale}'.")

    plotted_errors: list[list[float]] = []
    for landmark_values in per_landmark_errors:
        current_values = [float(value) for value in landmark_values]
        if y_scale == "log":
            current_values = [max(value, 1e-8) for value in current_values]
        plotted_errors.append(current_values)

    figure_width = max(20.0, number_of_landmarks * 0.34)
    figure, axis = plt.subplots(figsize=(figure_width, 7))

    boxplot = axis.boxplot(
        plotted_errors,
        showfliers=True,
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#8B0000", "linewidth": 2.0},
        meanprops={
            "marker": "D",
            "markerfacecolor": "#003366",
            "markeredgecolor": "white",
            "markersize": 4.5,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#4B0082",
            "markeredgecolor": "#4B0082",
            "markersize": 2.8,
            "alpha": 0.35,
        },
        whiskerprops={"color": "#4d4d4d", "linewidth": 1.1},
        capprops={"color": "#4d4d4d", "linewidth": 1.1},
    )

    for patch, color in zip(boxplot["boxes"], landmark_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.0)

    axis.set_yscale(y_scale)
    if y_limits is not None:
        axis.set_ylim(y_limits)

    axis.set_xticks(np.arange(1, number_of_landmarks + 1))
    if use_landmark_names:
        axis.set_xticklabels(displayed_landmark_names, rotation=90, fontsize=8)
    else:
        axis.set_xticklabels(
            [str(index) for index in range(number_of_landmarks)],
            rotation=90,
            fontsize=8,
        )

    axis.set_xlabel("Landmark", fontsize=16, fontweight="bold")
    axis.set_ylabel(
        f"Normalized error ({y_scale} scale)",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_title(title, fontsize=17, fontweight="bold", pad=12)
    axis.tick_params(axis="y", labelsize=11)
    axis.grid(True, axis="y", linestyle="--", alpha=0.35)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    region_legend_handles = [
        Patch(facecolor=color, edgecolor="#333333", alpha=0.65, label=region_name)
        for region_name, _, color in get_landmark_region_definitions()
    ]
    stats_legend_handles = [
        Line2D([0], [0], color="#8B0000", lw=2.0, label="Median"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#003366",
            markeredgecolor="white",
            markersize=6,
            label="Mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4B0082",
            markeredgecolor="#4B0082",
            markersize=5,
            label="Outlier",
        ),
    ]
    axis.legend(
        handles=region_legend_handles + stats_legend_handles,
        loc="upper right",
        fontsize=9,
        frameon=True,
        ncol=2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compute_global_log_y_limits(
    grouped_errors_collection: list[list[list[float]]],
    lower_margin_factor: float = 0.8,
    upper_margin_factor: float = 1.2,
    min_positive_value: float = 1e-8,
) -> tuple[float, float]:
    """Compute shared log-scale y limits across several error groupings."""
    all_values: list[float] = []
    for grouped_errors in grouped_errors_collection:
        for landmark_values in grouped_errors:
            for value in landmark_values:
                all_values.append(max(float(value), min_positive_value))

    if not all_values:
        return min_positive_value, 1.0

    global_min = min(all_values)
    global_max = max(all_values)
    y_min = max(global_min * lower_margin_factor, min_positive_value)
    y_max = max(global_max * upper_margin_factor, y_min * 10.0)
    return y_min, y_max


def compute_global_linear_y_limits(
    grouped_errors_collection: list[list[list[float]]],
    upper_margin_factor: float = 1.05,
) -> tuple[float, float]:
    """Compute shared linear-scale y limits across several error groupings."""
    all_values: list[float] = []
    for grouped_errors in grouped_errors_collection:
        for landmark_values in grouped_errors:
            for value in landmark_values:
                all_values.append(float(value))

    if not all_values:
        return 0.0, 1.0

    global_min = min(all_values)
    global_max = max(all_values)
    y_min = min(0.0, global_min)
    y_max = max(global_max * upper_margin_factor, y_min + 1e-6)
    return y_min, y_max


def plot_yaw_view_boxplots(
    orientation_to_errors: dict[str, list[list[float]]],
    output_dir: Path,
    use_landmark_names: bool = True,
    y_limits: tuple[float, float] | None = None,
    orientation_metrics: dict[str, dict[str, float | None]] | None = None,
    y_scale: str = "log",
    filename_suffix: str = "log",
) -> None:
    """Generate one per-landmark NME boxplot for each face orientation."""
    ordered_orientations = [
        "left",
        "quarter_left",
        "frontal",
        "quarter_right",
        "right",
    ]

    for orientation in ordered_orientations:
        if orientation not in orientation_to_errors:
            continue

        current_errors = orientation_to_errors[orientation]
        if all(len(values) == 0 for values in current_errors):
            continue

        current_metrics = (
            orientation_metrics.get(orientation, {}) if orientation_metrics else {}
        )
        title = (
            f"Per-landmark NME distribution - {orientation.replace('_', ' ').title()}"
        )
        summary_parts = []
        if current_metrics.get("mean_nme_box") is not None:
            summary_parts.append(f"Mean NME box: {current_metrics['mean_nme_box']:.6f}")
        if current_metrics.get("mean_nme_interocular") is not None:
            summary_parts.append(
                f"Mean NME interocular: {current_metrics['mean_nme_interocular']:.6f}"
            )
        if summary_parts:
            title = f"{title}\n" + " | ".join(summary_parts)

        plot_per_landmark_boxplot(
            per_landmark_errors=current_errors,
            output_path=output_dir
            / f"boxplot_nme_per_landmark_{orientation}_{filename_suffix}.png",
            use_landmark_names=use_landmark_names,
            title=title,
            y_limits=y_limits,
            y_scale=y_scale,
        )
