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


## TTA scopes with LayerNorm or InstanceNorm checkpoints

The evaluator reconstructs the head and normalizer normalization from checkpoint
metadata, including full-model and split landmarker + normalizer checkpoints.
Train one LN model and one IN model first; then run each checkpoint with all
three scopes. No normalization flag is needed at evaluation time. Keep the
metadata produced by the training/export pipeline: LN and IN affine tensors can
have identical shapes, so the weights alone cannot identify the normalization.
Older checkpoints without head metadata default to the original BatchNorm heads.

| `--pca-tta-adaptation-scope` | Trainable parameters |
|---|---|
| `normalizer` | Complete external CNN, including its normalization parameters |
| `normalizer_head_norms` | Complete external CNN + gamma/beta of normalization layers in the three task heads |
| `normalizer_heads` | Complete external CNN + all three task heads, including output predictors |

All modes freeze the HRNet backbone, including BN statistics. Heads run in eval
mode while selected parameters receive gradients: LN and IN without running
statistics behave identically in train/eval. Legacy head BN keeps its source
statistics. Adam, gradient checks, and clipping use the entire selected parameter
set. Every image starts with source weights and a fresh optimizer; all adapted
head and normalizer weights are restored after success or adaptation failure.

Configure dataset paths and the PCA prior in `configs/pca_tta_evaluation.yaml`.
For example, run the following from the repository root, substituting the actual
checkpoint and output parent. Repeat with the InstanceNorm checkpoint and a
different output parent to obtain all six comparisons:

```bash
for scope in normalizer normalizer_head_norms normalizer_heads; do
  python -m scripts.evaluate \
    --config configs/pca_tta_evaluation.yaml \
    --checkpoint /absolute/path/to/layer_run/checkpoints/full_model_best.pth \
    --pca-tta-adaptation-scope "$scope" \
    --output-dir "/absolute/path/to/layer_tta/$scope" \
    --wandb-run-name "layer_tta_$scope"
done
```

For split weights, use `--checkpoint .../landmarker_best.pth` together with
`--normalizer-checkpoint .../normalizer_best.pth` from the matching trained run.
Use the same data, adaptation steps, and learning rate for the initial comparison.
The existing step sweep inherits `pca_tta_adaptation_scope` from its base YAML.

`tta/summary.json` records scope, selected parameter names/count, and both model
architectures. Per-image metrics include the unadapted baseline. The only
adaptation objective remains PCA reconstruction on full-landmark heatmaps; it
does not supervise visibility logits. Parameters exclusive to the visibility
classifier therefore receive no gradient even in the full-head scope. The
shared visibility features can still change through the full-landmark path.
Reduced PCA error alone does not establish improved landmark accuracy: compare
post-hoc NME and Hausdorff metrics, including degraded cases, against the baseline.


## PCA reconstruction loss in image coordinates

The shared training/TTA PCA loss now uses the supervisor's complete inverse
transform approach. Given predicted input-crop landmarks X, estimate the existing
similarity transform T(X) = s X R + t, align to the prior reference, and reconstruct
Z_hat = PCA(T(X)). Reuse exactly that transform to compute:

```text
X_hat = ((Z_hat - t) @ R.T) / s
L_pca = mean((X - X_hat)^2)
```

This replaces the previous aligned-coordinate MSE; it is one regularizer, not
an additional term. Gradients flow through scale, rotation, translation, PCA
projection, and the inverse. No `detach` or stop-gradient is applied to this
path. MSE averages over both coordinates, landmarks, and samples; there is no
division by image dimensions or predicted face size. Its units are input-crop
pixels squared, not original uncropped photograph pixels squared.

The change applies to supervised training whenever `lambda_pca_projection > 0`
and to all three TTA adaptation scopes. The PCA prior and model checkpoint
formats remain compatible; they do not need rebuilding for this loss change.
TTA summary metadata identifies `pca_loss_space: input_image_pixels` and
`pca_alignment_gradient: full`.

For each nondegenerate shape, L_image = L_aligned / s^2. Historical loss values
and weights are therefore not directly comparable. Use separate output runs and
reassess the supervised regularization weight. With full gradients, reducing
predicted shape size can reduce the image-space residual without improving
relative shape; evaluate landmark accuracy as well as reconstruction loss.
The existing handling of degenerate alignment is retained: training raises an
alignment error; TTA restores source weights and returns the baseline prediction.

## Dataset-specific reports and heatmap diagnostics

Training final evaluation and standalone/TTA evaluation share these reports:

- **SynBaby:** all annotated landmarks, regardless of visibility. Box-normalized
  NME/HD use their ordinary names; duplicate `*_gt_valid` and `non_contour`
  entries are omitted.
- **BabyLand:** geometry uses `*_gt_valid` exclusively. Predicted visibility does
  not select landmarks. No visible-intersection, non-contour or interocular
  geometric entries are exported. The visibility classifier report is retained.
- **InfAnFace:** its existing full-face and non-contour protocols are retained.

`metrics_summary.csv` retains `metric,value` and adds a `section` column to group
coverage, NME, Hausdorff, heatmaps, visibility, orientations and setup. Unavailable
scalar entries are omitted. `metrics_by_orientation.csv` has one row per pose.
`summary.json` contains the same protocol and global/per-pose statistics.

For BabyLand and InfAnFace, `std_*` is the **population standard deviation across
finite per-image errors** (`ddof=0`). It is computed globally and within each pose,
using each metric's valid images. One image gives std=0; no valid images gives
null. It is not the standard deviation across pooled individual landmarks.
Geometric values remain fractions, so multiply NME/HD and their std by 100 when
presenting percentages. SynBaby also gets image-level global std.

Heatmap entropy and spread are diagnostics, with fixed spatial softmax
**temperature 1**, averaged over **all output landmark heatmaps** without a GT or
predicted-visibility mask. They do not change the training or TTA loss:

- Entropy: `-sum(p * log(p)) / log(H * W)`.
- Spread: `2 * (Var(x) + Var(y))`, with `x=v/(W-1)` and `y=u/(H-1)`.

Both are bounded by [0,1]. Spread measures spatial variance within each heatmap;
it is not prediction variance across augmentations. Exported errors are still
calculated after the existing coordinate reprojection; heatmap diagnostics use
the heatmap grid before decoding/reprojection.

Each evaluated dataset writes `heatmaps/per_image.csv`, `heatmaps/summary.json`,
`figures/heatmaps/distributions.png` and `figures/heatmaps/by_orientation.png`.
InfAnFace heatmaps are matched to evaluated prediction IDs to obtain GT poses.
Heatmap means/std also appear in the main global and pose reports.

Training records both diagnostics in train/val `results.csv`, checkpoint metrics,
history and W&B, and writes `figures/heatmaps/training_curves.png` when training
finishes. TTA records each image at each step in `trajectories.csv`, initial/final
values and deltas in `image_summary.csv`, and mean/std/median/10th/90th percentiles
in `aggregate_curves.csv`. Its curve is saved at
`figures/heatmaps/adaptation_curves.png`, with the same [0,1] scale.
In batch episodes, `source_mean_heatmap_*` comes from the frozen source model in
eval mode; `initial_mean_heatmap_*` is adaptation step 0, which can differ when
BN switches to current-batch statistics. Final dataset diagnostics describe the
returned adapted predictions, before model restoration affects the next episode.

These outputs apply automatically to new local/HPC runs; existing result files
are not rewritten. A training CSV reused with the older header gains the two
new columns, leaving previous rows blank for those measurements.
