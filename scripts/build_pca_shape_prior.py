from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import build_config
from scripts.dataset import Resize, SyntheticLandmarkDataset
from scripts.engine.pca_shape_prior import (
    build_global_pca_shape_prior_payload,
    flatten_landmark_shapes,
)
from scripts.utils.visualization import save_figure_png_and_pdf


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline global PCA prior creation."""
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Fit one global PCA shape prior from the synthetic train split only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=defaults.dataset_root,
        help="Dataset root containing the train/images and train/labels folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path where the PCA prior .pt file will be saved.",
    )
    parser.add_argument(
        "--pca-num-components",
        type=int,
        default=32,
        help="Maximum number of PCA components to keep when no variance threshold is provided.",
    )
    parser.add_argument(
        "--pca-explained-variance",
        type=float,
        default=None,
        help=(
            "Cumulative explained variance target in (0, 1]. "
            "When provided, it takes precedence over --pca-num-components."
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=defaults.image_size[0],
        help="Training crop height used before fitting PCA.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=defaults.image_size[1],
        help="Training crop width used before fitting PCA.",
    )
    parser.add_argument(
        "--num-landmarks",
        type=int,
        default=defaults.num_landmarks,
        help="Number of landmarks expected in each label file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.cache_dir,
        help="Directory used for the train split sample index cache.",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        default=False,
        help="Disable dataset sample-index cache loading and writing.",
    )
    parser.add_argument(
        "--previews-dir",
        type=Path,
        default=None,
        help=(
            "Directory where PCA analysis plots will be saved. "
            "Defaults to output_path.parent / 'previews'."
        ),
    )
    return parser.parse_args()


def collect_train_landmarks(
    args: argparse.Namespace,
) -> torch.Tensor:
    """Load train-only GT landmarks in resized crop space."""
    transform = Resize(size=(args.image_height, args.image_width))
    cache_file = None
    if args.cache_dir is not None:
        cache_file = args.cache_dir / "pca_train_cache.pth"
    dataset = SyntheticLandmarkDataset(
        root_dir=args.dataset_root,
        split="train",
        transform=transform,
        target_mode="regression",
        num_landmarks=args.num_landmarks,
        validate_labels=True,
        cache_file=cache_file,
        use_cache=not args.disable_cache,
        show_progress=True,
    )
    shapes = []
    for index in range(len(dataset)):
        sample = dataset[index]
        shapes.append(sample["landmarks"].float())
    return torch.stack(shapes, dim=0)


def save_pca_analysis_plots(
    prior_payload: dict[str, Any],
    aligned_shapes: torch.Tensor,
    previews_dir: Path,
) -> None:
    """Save compact global PCA diagnostic plots for quick inspection."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    prior = prior_payload["global_prior"]
    explained_ratio = prior["all_explained_variance_ratio"].cpu()
    cumulative_ratio = explained_ratio.cumsum(dim=0)
    selected_k = int(prior["num_components"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(explained_ratio) + 1), explained_ratio.numpy())
    axes[0].axvline(
        selected_k, color="tab:red", linestyle="--", label=f"K={selected_k}"
    )
    axes[0].set_title("Explained Variance Ratio - Global PCA Prior")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance ratio")
    axes[0].legend()
    axes[1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio.numpy())
    axes[1].axvline(selected_k, color="tab:red", linestyle="--")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Cumulative Explained Variance - Global PCA Prior")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative ratio")
    fig.tight_layout()
    save_figure_png_and_pdf(
        fig, previews_dir / "pca_prior_global_explained_variance.png", dpi=150
    )
    plt.close(fig)

    mean_shape = prior["mean_shape"].reshape(-1, 2).cpu()
    reference_shape = prior["reference_shape"].cpu()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(reference_shape[:, 0], reference_shape[:, 1], s=16)
    axes[0].plot(reference_shape[:, 0], reference_shape[:, 1], alpha=0.35)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Reference Shape - Global PCA Prior")
    axes[1].scatter(mean_shape[:, 0], mean_shape[:, 1], s=16)
    axes[1].plot(mean_shape[:, 0], mean_shape[:, 1], alpha=0.35)
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("PCA Mean Shape - Global PCA Prior")
    fig.tight_layout()
    save_figure_png_and_pdf(
        fig, previews_dir / "pca_prior_global_mean_shapes.png", dpi=150
    )
    plt.close(fig)

    num_modes_to_plot = min(6, selected_k)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.reshape(-1)
    for mode_index, ax in enumerate(axes):
        ax.axis("off")
        if mode_index >= num_modes_to_plot:
            continue
        component = prior["components"][mode_index].reshape(-1, 2).cpu()
        std = torch.sqrt(prior["explained_variance"][mode_index].cpu())
        minus_shape = mean_shape - 2.0 * std * component
        plus_shape = mean_shape + 2.0 * std * component
        ax.plot(minus_shape[:, 0], minus_shape[:, 1], color="tab:blue", alpha=0.8)
        ax.plot(plus_shape[:, 0], plus_shape[:, 1], color="tab:orange", alpha=0.8)
        ax.scatter(mean_shape[:, 0], mean_shape[:, 1], s=8, color="black", alpha=0.5)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"PC {mode_index + 1}: +/-2 std")
        ax.axis("on")
    fig.tight_layout()
    save_figure_png_and_pdf(
        fig, previews_dir / "pca_prior_global_shape_modes.png", dpi=150
    )
    plt.close(fig)

    shape_vectors = flatten_landmark_shapes(aligned_shapes)
    mean_vector = prior["mean_shape"].to(dtype=shape_vectors.dtype)
    centered = shape_vectors - mean_vector
    components = prior["components"].to(dtype=shape_vectors.dtype)
    max_k = components.shape[0]
    reconstruction_errors = []
    for component_count in range(1, max_k + 1):
        current_components = components[:component_count]
        coefficients = centered @ current_components.T
        reconstructed = mean_vector + coefficients @ current_components
        reconstruction_errors.append(
            torch.mean((shape_vectors - reconstructed).square()).item()
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, max_k + 1), reconstruction_errors, marker="o", linewidth=1.5)
    ax.set_title("Train Shape Reconstruction Error - Global PCA Prior")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Mean squared projection residual")
    fig.tight_layout()
    save_figure_png_and_pdf(
        fig, previews_dir / "pca_prior_global_reconstruction_error.png", dpi=150
    )
    plt.close(fig)

    summary_rows = [
        {
            "scope": "global",
            "num_samples": int(prior["num_samples"]),
            "num_components": int(prior["num_components"]),
            "requested_num_components": prior["requested_num_components"],
            "explained_variance_threshold": prior["explained_variance_threshold"],
            "cumulative_explained_variance": float(
                prior["explained_variance_ratio"].sum().item()
            ),
        }
    ]

    summary_path = previews_dir / "pca_prior_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def main() -> None:
    """Fit and save the train-only global PCA shape prior."""
    args = parse_args()
    if args.pca_num_components is not None and args.pca_num_components < 1:
        raise ValueError("--pca-num-components must be at least 1.")

    landmarks = collect_train_landmarks(args)
    prior_payload, aligned_shapes = build_global_pca_shape_prior_payload(
        landmarks=landmarks,
        num_components=args.pca_num_components,
        explained_variance_threshold=args.pca_explained_variance,
    )
    prior_payload["dataset_root"] = str(args.dataset_root)
    prior_payload["source_split"] = "train"
    prior_payload["image_size"] = (int(args.image_height), int(args.image_width))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prior_payload, args.output_path)

    previews_dir = args.previews_dir or (args.output_path.parent / "previews")
    save_pca_analysis_plots(
        prior_payload=prior_payload,
        aligned_shapes=aligned_shapes,
        previews_dir=previews_dir,
    )
    print(f"[INFO] Saved PCA prior: {args.output_path}")
    print(f"[INFO] Saved PCA previews: {previews_dir}")


if __name__ == "__main__":
    main()
