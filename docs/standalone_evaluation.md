# Standalone checkpoint evaluation

`scripts/evaluate.py` supports three explicit dataset protocols. The legacy
`natural` mode remains an alias for `babyland`.

## SynBaby

```bash
python scripts/evaluate.py \
  --eval-mode synthetic \
  --checkpoint /path/to/full_model_best.pth \
  --dataset-root /path/to/synbaby72 \
  --landmark-loss wasserstein \
  --output-dir /path/to/run/evaluation/synbaby
```

## BabyLand-72

```bash
python scripts/evaluate.py \
  --eval-mode babyland \
  --checkpoint /path/to/full_model_best.pth \
  --dataset-root /path/to/babyland/crops/all_detections \
  --natural-gt-root /path/to/babyland72/labels \
  --natural-source-root /optional/path/to/source/images \
  --landmark-loss wasserstein \
  --output-dir /path/to/run/evaluation/babyland
```

## InfantFace

```bash
python scripts/evaluate.py \
  --eval-mode infanface \
  --checkpoint /path/to/full_model_best.pth \
  --dataset-root /path/to/infanface/crops/all_detections \
  --natural-gt-root /path/to/infanface/labels \
  --natural-source-root /optional/path/to/source/images \
  --landmark-loss wasserstein \
  --output-dir /path/to/run/evaluation/infanface
```

The InfantFace protocol performs crop inference, projection to original-image
coordinates, prediction export, and the dedicated 72-to-68 landmark benchmark
in one command. Its output layout is:

```text
infanface/
  figures/
  predictions/
    images/
    labels/
  metrics_summary.csv
  summary.json
  per_image_nme.csv
  per_image_per_landmark_nme.csv
  normalizer_diagnostics/       # only for normalized models
```

The InfantFace summary includes metrics with and without contour, Hausdorff
statistics, and orientation-dependent results.

## Modular checkpoints

When the landmarker and normalizer were saved separately, use:

```bash
python scripts/evaluate.py \
  --eval-mode infanface \
  --checkpoint /path/to/landmarker_best.pth \
  --normalizer-checkpoint /path/to/normalizer_best.pth \
  --dataset-root /path/to/infanface/crops/all_detections \
  --natural-gt-root /path/to/infanface/labels \
  --landmark-loss wasserstein \
  --output-dir /path/to/run/evaluation/infanface
```

Do not pass `--normalizer-checkpoint` when `--checkpoint` already contains the
full normalized model.
