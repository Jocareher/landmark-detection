from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.engine.benchmark import (
    benchmark_prediction_directory,
    resolve_prediction_labels_dir,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-model txt-only benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark one landmark prediction directory against GT txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset split root containing images/ and labels/ for the benchmark set.",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Prediction directory for one evaluated model. Supported layouts are either '*.txt' directly under the root or a nested 'labels/' directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for benchmark summaries and plots. If omitted, a sibling '<prediction_root>_benchmark' directory is created.",
    )
    parser.add_argument(
        "--use-landmark-names",
        action="store_true",
        default=False,
        help="Use landmark names instead of indices on the benchmark boxplots.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save the resolved benchmark arguments next to the outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run txt-only benchmarking for one prediction directory."""
    args = parse_args()
    prediction_root, _ = resolve_prediction_labels_dir(args.prediction_root)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else prediction_root.parent / f"{prediction_root.name}_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_config:
        (output_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    "dataset_root": str(args.dataset_root),
                    "prediction_root": str(args.prediction_root),
                    "output_dir": str(output_dir),
                    "use_landmark_names": bool(args.use_landmark_names),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = benchmark_prediction_directory(
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        output_dir=output_dir,
        use_landmark_names_in_boxplot=args.use_landmark_names,
        fixed_log_y_limits=(1e-3, 1.0),
    )

    print("[INFO] Benchmark finished.")
    print(f"[INFO] Model name: {summary['model_name']}")
    print("[INFO] Prediction layout: " f"{summary['model_landmark_format']}-landmark")
    print(f"[INFO] Total images: {summary['total_images']}")
    print(f"[INFO] Images with prediction: {summary['images_with_prediction']}")
    print(f"[INFO] Images without prediction: {summary['images_without_prediction']}")
    print(
        "[INFO] Detection rate: "
        f"{summary['images_with_prediction']}/{summary['total_images']} "
        f"({summary['detection_rate'] * 100.0:.2f}%)"
    )
    if summary.get("mean_nme_box_visible_intersection") is not None:
        print(
            "[INFO] Mean NME box visible-intersection: "
            f"{summary['mean_nme_box_visible_intersection']:.4f}"
        )
    else:
        print("[INFO] Mean NME box visible-intersection: n/a")
    if summary.get("median_nme_box_visible_intersection") is not None:
        print(
            "[INFO] Median NME box visible-intersection: "
            f"{summary['median_nme_box_visible_intersection']:.4f}"
        )
    if summary.get("mean_nme_box_point_to_line_visible_intersection") is not None:
        print(
            "[INFO] Mean NME box point-to-line visible-intersection: "
            f"{summary['mean_nme_box_point_to_line_visible_intersection']:.4f}"
        )
    else:
        print("[INFO] Mean NME box point-to-line visible-intersection: n/a")
    if summary.get("median_nme_box_point_to_line_visible_intersection") is not None:
        print(
            "[INFO] Median NME box point-to-line visible-intersection: "
            f"{summary['median_nme_box_point_to_line_visible_intersection']:.4f}"
        )
    if summary.get("mean_nme_box_gt_valid") is not None:
        print(
            "[INFO] Mean NME box GT-valid: "
            f"{summary['mean_nme_box_gt_valid']:.4f}"
        )
    else:
        print("[INFO] Mean NME box GT-valid: n/a")
    if summary.get("median_nme_box_gt_valid") is not None:
        print(
            "[INFO] Median NME box GT-valid: "
            f"{summary['median_nme_box_gt_valid']:.4f}"
        )
    if summary.get("mean_nme_box_point_to_line_gt_valid") is not None:
        print(
            "[INFO] Mean NME box point-to-line GT-valid: "
            f"{summary['mean_nme_box_point_to_line_gt_valid']:.4f}"
        )
    else:
        print("[INFO] Mean NME box point-to-line GT-valid: n/a")
    if summary.get("median_nme_box_point_to_line_gt_valid") is not None:
        print(
            "[INFO] Median NME box point-to-line GT-valid: "
            f"{summary['median_nme_box_point_to_line_gt_valid']:.4f}"
        )
    print(f"[INFO] Valid landmarks used: {summary['valid_landmarks_used']}")
    if summary.get("gt_valid_landmarks_used") is not None:
        print(f"[INFO] GT-valid landmarks used: {summary['gt_valid_landmarks_used']}")
    print(f"[INFO] Output dir: {output_dir}")
    if summary["images_with_invalid_prediction"] > 0:
        print(
            "[INFO] Invalid prediction files treated as missing: "
            f"{summary['images_with_invalid_prediction']}"
        )
    if summary.get("unmatched_prediction_files_count", 0) > 0:
        print(
            "[INFO] Prediction files without resolved GT match: "
            f"{summary['unmatched_prediction_files_count']}"
        )


if __name__ == "__main__":
    main()
