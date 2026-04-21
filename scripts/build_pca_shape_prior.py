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
    fit_pca_shape_prior,
    flatten_landmark_shapes,
    normalize_landmark_shapes,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline PCA shape-prior creation."""
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Fit a PCA shape prior from the synthetic train split only.",
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
        "--num-components",
        type=int,
        default=32,
        help="Number of PCA components to keep in the saved prior.",
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


def collect_train_landmarks(args: argparse.Namespace) -> torch.Tensor:
    """Load train-only GT landmarks in the same resized crop space used by training."""
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
    shapes = [dataset[index]["landmarks"].float() for index in range(len(dataset))]
    return torch.stack(shapes, dim=0)


def save_pca_analysis_plots(
    prior: dict[str, Any],
    landmarks: torch.Tensor,
    previews_dir: Path,
) -> None:
    """Save compact PCA diagnostic plots for quick prior inspection."""
    previews_dir.mkdir(parents=True, exist_ok=True)
    explained_ratio = prior["all_explained_variance_ratio"].cpu()
    cumulative_ratio = explained_ratio.cumsum(dim=0)
    selected_k = int(prior["num_components"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(explained_ratio) + 1), explained_ratio.numpy())
    axes[0].axvline(
        selected_k,
        color="tab:red",
        linestyle="--",
        label=f"K={selected_k}",
    )
    axes[0].set_title("Explained Variance Ratio")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance ratio")
    axes[0].legend()
    axes[1].plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio.numpy())
    axes[1].axvline(selected_k, color="tab:red", linestyle="--")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative ratio")
    fig.tight_layout()
    fig.savefig(previews_dir / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

    mean_shape = prior["mean_shape"].reshape(-1, 2).cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mean_shape[:, 0], mean_shape[:, 1], s=16)
    ax.plot(mean_shape[:, 0], mean_shape[:, 1], alpha=0.35)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mean Normalized Shape")
    fig.tight_layout()
    fig.savefig(previews_dir / "pca_mean_shape.png", dpi=150)
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
    fig.savefig(previews_dir / "pca_shape_modes.png", dpi=150)
    plt.close(fig)

    normalized_shapes, _, _ = normalize_landmark_shapes(landmarks)
    shape_vectors = flatten_landmark_shapes(normalized_shapes)
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
    ax.set_title("Train Shape Reconstruction Error")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Mean squared projection residual")
    fig.tight_layout()
    fig.savefig(previews_dir / "pca_reconstruction_error.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Fit and save the train-only PCA shape prior."""
    args = parse_args()
    if args.num_components < 1:
        raise ValueError("--num-components must be at least 1.")

    landmarks = collect_train_landmarks(args)
    prior = fit_pca_shape_prior(
        landmarks=landmarks,
        num_components=args.num_components,
    )
    prior["dataset_root"] = str(args.dataset_root)
    prior["source_split"] = "train"
    prior["image_size"] = (int(args.image_height), int(args.image_width))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prior, args.output_path)

    previews_dir = args.previews_dir or (args.output_path.parent / "previews")
    save_pca_analysis_plots(prior=prior, landmarks=landmarks, previews_dir=previews_dir)
    summary_path = previews_dir / "pca_prior_summary.json"
    summary = {
        "output_path": str(args.output_path),
        "dataset_root": str(args.dataset_root),
        "source_split": "train",
        "num_train_shapes": prior["num_train_shapes"],
        "num_landmarks": prior["num_landmarks"],
        "num_components": prior["num_components"],
        "requested_num_components": prior["requested_num_components"],
        "image_size": prior["image_size"],
        "normalization": prior["normalization"],
        "cumulative_explained_variance": float(
            prior["explained_variance_ratio"].sum().item()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Saved PCA prior: {args.output_path}")
    print(f"[INFO] Saved PCA previews: {previews_dir}")
    print(
        "[INFO] Cumulative explained variance for saved components: "
        f"{summary['cumulative_explained_variance']:.6f}"
    )


if __name__ == "__main__":
    main()
