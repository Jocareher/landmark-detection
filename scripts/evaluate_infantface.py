from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.engine.benchmark import (
    benchmark_infantface_prediction_directory,
    resolve_prediction_labels_dir,
    resolve_text_labels_dir,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one-model-at-a-time InfantFace evaluation."""
    parser = argparse.ArgumentParser(
        description="Benchmark one prediction directory against InfantFace GT txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        required=True,
        help="InfantFace GT root. Supported layouts are either '*.txt' directly under the root or a nested 'labels/' directory.",
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
        help="Output directory for summaries and plots. If omitted, a sibling '<prediction_root>_infantface' directory is created.",
    )
    parser.add_argument(
        "--use-landmark-names",
        action="store_true",
        default=False,
        help="Use landmark names instead of indices on the InfantFace boxplots.",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save the resolved evaluation arguments next to the outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one InfantFace evaluation against a single prediction directory."""
    args = parse_args()
    prediction_root, _ = resolve_prediction_labels_dir(args.prediction_root)
    gt_root, _ = resolve_text_labels_dir(args.gt_root, labels_subdir_name="labels")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else prediction_root.parent / f"{prediction_root.name}_infantface"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_config:
        (output_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    "gt_root": str(gt_root),
                    "prediction_root": str(args.prediction_root),
                    "output_dir": str(output_dir),
                    "use_landmark_names": bool(args.use_landmark_names),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = benchmark_infantface_prediction_directory(
        gt_root=gt_root,
        prediction_root=args.prediction_root,
        output_dir=output_dir,
        use_landmark_names_in_boxplot=args.use_landmark_names,
    )

    print("[INFO] InfantFace evaluation finished.")
    print(f"[INFO] Model name: {summary['model_name']}")
    print(
        "[INFO] Prediction layout: "
        f"{summary['model_landmark_format']}-landmark "
        f"(evaluated as {summary['evaluated_landmark_count']} landmarks)"
    )
    print(f"[INFO] Total samples: {summary['total_images']}")
    print(f"[INFO] Samples with prediction: {summary['images_with_prediction']}")
    print(f"[INFO] Samples without prediction: {summary['images_without_prediction']}")
    print(
        "[INFO] Detection rate: "
        f"{summary['images_with_prediction']}/{summary['total_images']} "
        f"({summary['detection_rate'] * 100.0:.2f}%)"
    )
    if summary["mean_nme_box"] is not None:
        print(f"[INFO] Mean NME box: {summary['mean_nme_box']:.4f}")
    else:
        print("[INFO] Mean NME box: n/a")
    if summary["mean_nme_box_point_to_line"] is not None:
        print(
            "[INFO] Mean NME box point-to-line: "
            f"{summary['mean_nme_box_point_to_line']:.4f}"
        )
    else:
        print("[INFO] Mean NME box point-to-line: n/a")
    print(f"[INFO] Valid landmarks used: {summary['valid_landmarks_used']}")
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
