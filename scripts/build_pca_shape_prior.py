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
    build_pose_pca_shape_prior_payload,
    flatten_landmark_shapes,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline global or exact-pose PCA prior creation."""
    defaults = build_config()
    parser = argparse.ArgumentParser(
        description="Fit global or exact-angle pose PCA shape priors from the synthetic train split only.",
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
    parser.add_argument("--prior-mode", choices=("global", "pose5", "pose13"), default="global",
                        help="Global or exact-angle pose priors. pose5 selects -75,-45,0,45,75 only.")
    return parser.parse_args()


def collect_train_landmarks(
    args: argparse.Namespace,
    include_yaw: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    yaw = []
    for index in range(len(dataset)):
        sample = dataset[index]
        shapes.append(sample["landmarks"].float())
        if include_yaw:
            yaw.append(float(sample["yaw_angle"]))
    if not shapes:
        raise ValueError("No training samples found.")
    landmarks = torch.stack(shapes, dim=0)
    return (landmarks, torch.tensor(yaw)) if include_yaw else landmarks


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
    fig.savefig(previews_dir / "pca_prior_global_explained_variance.png", dpi=150)
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
    fig.savefig(previews_dir / "pca_prior_global_mean_shapes.png", dpi=150)
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
    fig.savefig(previews_dir / "pca_prior_global_shape_modes.png", dpi=150)
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
    fig.savefig(previews_dir / "pca_prior_global_reconstruction_error.png", dpi=150)
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
    """Fit and save train-only global or pose-conditioned PCA priors."""
    args = parse_args()
    if args.pca_num_components is not None and args.pca_num_components < 1:
        raise ValueError("--pca-num-components must be at least 1.")

    aligned_by_pose = {}
    if args.prior_mode == "global":
        landmarks = collect_train_landmarks(args)
        prior_payload, aligned_shapes = build_global_pca_shape_prior_payload(
            landmarks=landmarks, num_components=args.pca_num_components,
            explained_variance_threshold=args.pca_explained_variance)
        prior_payload["prior_mode"] = "global"
    else:
        landmarks, yaw = collect_train_landmarks(args, include_yaw=True)
        prior_payload, aligned_shapes, aligned_by_pose = build_pose_pca_shape_prior_payload(
            landmarks, yaw, mode=args.prior_mode, num_components=args.pca_num_components,
            explained_variance_threshold=args.pca_explained_variance)
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
    for label, shapes in aligned_by_pose.items():
        prior = prior_payload["pose_priors"][label]
        save_pca_analysis_plots({"global_prior": prior}, shapes, previews_dir / f"yaw_{label}")
        print(f"[INFO] Pose {label}: samples={prior['num_samples']} components={prior['num_components']}")
    if aligned_by_pose:
        summary = {"prior_mode": args.prior_mode, "selection": "exact_yaw",
                   "excluded_samples": prior_payload["num_excluded_samples"],
                   "poses": [{"yaw": label, "num_samples": prior["num_samples"],
                              "num_components": prior["num_components"]}
                             for label, prior in prior_payload["pose_priors"].items()]}
        (previews_dir / "pose_pca_summary.json").write_text(json.dumps(summary, indent=2))
        counts = {prior["num_components"] for prior in prior_payload["pose_priors"].values()}
        if len(counts) > 1:
            print("[WARNING] Pose priors have different component counts; inspect flexibility before comparing minima.")
        print(f"[INFO] Excluded samples: {prior_payload['num_excluded_samples']}")
    print(f"[INFO] Saved PCA prior: {args.output_path}")
    print(f"[INFO] Saved PCA previews: {previews_dir}")


if __name__ == "__main__":
    main()
