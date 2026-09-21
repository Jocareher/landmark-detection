# Residual image normalizer experiments

## Motivation

These experiments test the supervised prerequisite of a future test-time
adaptation method: whether a small appearance adapter can be inserted before
BabyLand-72 without changing the established landmark pipeline. The separate PCA-guided TTA evaluator adapts the normalizer after supervised
training. DAE guidance, atlas switching, and target-domain training are outside
these experiments.

The effective pipeline is:

`normalized input -> residual normalizer -> existing HRNet landmarker -> existing outputs`

The normalizer receives the channel-normalized tensor already produced by the
current dataset transforms. Consequently, output clamping is disabled by
default. Its output is

`x_normalized = x + residual_scale * tanh(delta(x))`.

The final convolution is zero-initialized by default, making the initial
mapping exactly the identity. ReLU is the default activation because this is a
small convolutional adapter rather than a new backbone. Internal normalization
defaults to `none` to avoid imposing source-batch statistics or changing the
input distribution before the pretrained landmarker.

## Experiment modes

| Mode | Normalizer | Landmarker | Purpose |
|---|---|---|---|
| `normalizer_sanity` | Identity, frozen | Frozen | Verify unchanged images, heatmaps, visibility and decoded landmarks |
| `normalizer_train_frozen_landmarker` | Trained | Fully frozen | Determine whether the adapter alone can improve supervised SynBaby validation |
| `normalizer_joint_finetune` | Trained from identity | Official HRNet weights; transition 3, stage 4, and new task heads trained | Compare adapter-only training with partial HRNet adaptation without requiring a previously trained landmarker |

All modes require the established Wasserstein heatmap loss and barycenter
decoder. The normalizer does not replace or reimplement either path. Training
uses SynBaby labels only. BabyLand-72 and InfAnFace remain evaluation-only.

## Configuration and precedence

Configuration precedence is repository defaults, then YAML, then explicit CLI
arguments. A single comprehensive file is provided at
`configs/normalizer_experiments.yaml`. Every entry under its `arguments` key
matches an argparse destination (except `--config`, which selects the YAML
itself). The experiment mode resolves the trainable modules, so separate YAML
files are unnecessary and cannot silently diverge.

Set `checkpoint.path` and the existing natural-dataset paths before running.
For example:

```bash
python -m scripts.main \
  --config configs/normalizer_experiments.yaml \
  --experiment-mode normalizer_sanity \
  --checkpoint /absolute/path/to/best_model.pth \
  --babyland-crop-root /absolute/path/to/babyland/crops \
  --babyland-gt-root /absolute/path/to/babyland/labels \
  --infanface-crop-root /absolute/path/to/infanface/crops \
  --infanface-gt-root /absolute/path/to/infanface/labels
```

```bash
python -m scripts.main \
  --config configs/normalizer_experiments.yaml \
  --experiment-mode normalizer_train_frozen_landmarker \
  --checkpoint /absolute/path/to/best_model.pth
```

```bash
python -m scripts.main \
  --config configs/normalizer_experiments.yaml \
  --experiment-mode normalizer_joint_finetune \
  --pretrained-weights /absolute/path/to/HR18-300W.pth
```

Joint fine-tuning requires `checkpoint: null`. It initializes the backbone from
the official HRNet weights, creates new task heads and an identity-initialized
normalizer, freezes the stem and stages 1--3 (including frozen BatchNorm
statistics), and trains transition 3, stage 4, the task heads, and the
normalizer.

The same command-line options can override YAML values, for example:

```bash
python -m scripts.main \
  --config configs/normalizer_experiments.yaml \
  --experiment-mode normalizer_train_frozen_landmarker \
  --checkpoint /absolute/path/to/best_model.pth \
  --epochs 30 --lr 5e-5 --use-wandb
```

Runs retain the repository's historical structure:
`runs/<wandb_run_name-or-run_YYYYmmdd_HHMMSS>/`. The experiment mode is stored
in the resolved config and reports; it does not add another directory level.

When `use_wandb: true` (or `--use-wandb`) is selected, epoch-level training and
validation losses, NME, PCA loss, image L1/TV diagnostics, final official
evaluation scalars, and per-dataset normalizer diagnostics are logged to W&B.

## Fixed-probe normalizer monitoring

Trainable-normalizer modes snapshot a fixed set of validation tensors once and
reuse those exact tensors throughout the run. By default, four probes are
captured at initialization and after epochs 1, 5, 10, and 20, plus the final
epoch. This schedule gives substantially more evidence than logging a fresh
batch every epoch while adding only a few extra forward passes to an entire
training run. It can be changed with:

```yaml
normalizer_monitoring: true
normalizer_monitor_probes: 4
normalizer_monitor_steps: [0, 1, 5, 10, 20]
normalizer_tta_monitor_steps: [0, 1, 5, 10, 20]
normalizer_monitor_difference_max: 0.15
```

Each raw panel contains the unchanged input, normalized image, absolute RGB
difference with a fixed scale, landmarks from the detector without the
normalizer, landmarks after normalization, and source ground truth. Ground
truth appears only for synthetic validation probes and is never passed to the
normalizer or used as an adaptation objective.

The monitor records per-channel mean and standard deviation, luminance
contrast, robust dynamic range, high-frequency energy, pixel residuals,
heatmap peak confidence, decoded landmark displacement, and source-only
localization error. Phase-correlation registration and edge correlation are
used to flag possible geometry changes. These checks can reveal translations
or altered edge structure, but visual review remains necessary to identify
color collapse, excessive smoothing, hallucinated texture, or
identity-dependent effects.

Raw outputs are saved directly under `normalizer_monitoring/`, including
per-checkpoint panels, CSV metrics, checkpoint grids, GIF animations, and
professional trajectory/final-probe plots. The same panels and numeric
summaries are sent to W&B when enabled. Logging more than approximately four
probes or capturing every epoch is discouraged unless a short diagnostic run
is being performed, because image encoding and W&B uploads can otherwise
become a noticeable cost.

`NormalizerProbeMonitor.capture(...)` is also ready for a future test-time
adaptation loop. That loop should call it for the same target image at steps 0,
1, 5, 10, 20, and the final step, pass both adaptation and structural-prior
losses, and omit ground truth from the real-image probe batch. The monitor then
saves `adaptation_losses.csv` and `adaptation_losses.png`. No executable TTA
optimization is introduced by the current experiments.

Optional residual regularization is enabled in the shared YAML with
`normalizer_image_regularization`, `normalizer_lambda_l1`, and
`normalizer_lambda_tv`, or with the corresponding CLI options. It is disabled
by default.

## Outputs

Each run keeps the existing evaluation outputs and adds:

- `configs/resolved_config.yaml` and `.json`;
- `checkpoints/full_model_best.pth` and `full_model_last.pth`;
- `checkpoints/normalizer_best.pth` and `normalizer_last.pth`;
- modular landmarker checkpoints with the normalizer excluded;
- `checkpoints/checkpoint_manifest.json`;
- per-dataset image-change and prediction-drift CSV/JSON files in
  `normalizer_diagnostics/diagnostic_tables/`;
- input, normalized, auto-scaled absolute residual, fixed-scale absolute
  residual, signed residual, amplified normalized output, and labeled
  side-by-side examples in
  `normalizer_diagnostics/image_comparisons/<dataset>/`;
- fixed-probe checkpoint panels, grids, GIFs, and raw statistics in
  `normalizer_monitoring/`;
- `normalizer_monitoring/plots/probe_metric_trajectories.png`, showing the
  checkpoint mean and probe min-max range for appearance change, landmark
  displacement, localization-error change, and edge preservation;
- `normalizer_monitoring/plots/final_probe_profile.png`, exposing whether the
  final behavior is uniform or driven by individual probe images;
- normalizer-specific protocol and diagnostic report in
  `normalizer_diagnostics/experiment_report.md`;
- official dataset metrics and consolidated spreadsheet exports only under
  `evaluation/` and `evaluation/reports/`.

The diagnostic report includes mean L1/L2 and maximum image difference,
residual statistics, total variation, changed-pixel fractions, heatmap drift,
decoded landmark displacement, visibility-logit drift, and visibility decision
agreement. Official performance remains the NME, region, pose, visibility,
failure-rate, and detection-rate output of the existing evaluation code.

## Residual visualizations

The diagnostic `side_by_side` panel uses five labeled tiles:

1. original input;
2. normalized output;
3. signed residual, with zero represented by middle gray and one fixed scale
   for every image;
4. absolute residual with the same fixed scale for every image;
5. input plus an amplified signed residual, to make the direction of the
   learned appearance correction visible.

The header reports mean absolute residual, maximum absolute residual, and mean
signed RGB change. The legacy `residual_abs` file remains available but is
auto-scaled independently by each image's maximum and must not be used to
compare magnitudes across images. New comparable outputs are stored under
`residual_abs_fixed`, `residual_signed`, and
`normalized_change_amplified`.

## Standalone evaluation and checkpoint compatibility

Every normalizer run exports three representations for both the best and last
training checkpoints:

- `full_model_best.pth`: landmarker and normalizer together; this is the
  preferred checkpoint for evaluating the complete experiment;
- `landmarker_best.pth`: landmarker only, with every `normalizer.*` tensor
  excluded;
- `normalizer_best.pth`: normalizer only; it cannot run without a compatible
  landmarker.

`evaluate.py` detects the checkpoint representation. When a normalized model
is loaded, it automatically runs the official evaluation and the complete
normalizer diagnostics, including the improved visualizations. For example,
the complete model can be evaluated with:

```bash
python -m scripts.evaluate \
  --eval-mode natural \
  --checkpoint /run/checkpoints/full_model_best.pth \
  --dataset-root /data/babyland/crops/all_detections \
  --natural-gt-root /data/babyland/labels \
  --dataset-name babyland \
  --landmark-loss wasserstein \
  --output-dir /run/standalone_evaluation/babyland
```

The same model can be reconstructed from modular checkpoints:

```bash
python -m scripts.evaluate \
  --eval-mode natural \
  --checkpoint /run/checkpoints/landmarker_best.pth \
  --normalizer-checkpoint /run/checkpoints/normalizer_best.pth \
  --dataset-root /data/babyland/crops/all_detections \
  --natural-gt-root /data/babyland/labels \
  --dataset-name babyland \
  --landmark-loss wasserstein \
  --output-dir /run/standalone_evaluation/babyland_split
```

To measure the exported landmarker without its normalizer, omit
`--normalizer-checkpoint`:

```bash
python -m scripts.evaluate \
  --eval-mode natural \
  --checkpoint /run/checkpoints/landmarker_best.pth \
  --dataset-root /data/babyland/crops/all_detections \
  --natural-gt-root /data/babyland/labels \
  --dataset-name babyland \
  --landmark-loss wasserstein \
  --output-dir /run/standalone_evaluation/landmarker_only
```

A normalizer-only checkpoint cannot be passed as `--checkpoint`. A separate
normalizer also cannot be combined with `full_model_best.pth`, because that
checkpoint already contains one. Legacy landmarker-only checkpoints remain
supported.

## Interpretation and limitations

The sanity mode should produce zero or numerical-noise-level drift because the
normalizer is identity-initialized. A learned normalizer should be judged by
both official landmark metrics and the magnitude/structure of its image
changes. A low shape-prior or drift loss alone does not prove correct image
localization.

These experiments do not establish domain generalization by themselves. They
only establish the architecture, supervised trainability, checkpointing, and
diagnostic baseline needed before considering per-image test-time updates.


## LayerNorm and InstanceNorm comparison

Run two independent supervised experiments, keeping data, seed, losses, and
training budget fixed. Both update the complete residual normalizer, transition3,
stage4, and all three heads. Stem, layer1, stage2, and stage3 stay frozen,
including their BatchNorm running statistics. One unfrozen stage does not mean
exactly 25% of the parameters in HRNet's multibranch architecture.

```bash
python -m scripts.main --config configs/normalizer_experiments.yaml \
  --experiment-mode normalizer_joint_finetune \
  --transfer-mode fine_tuning --num-unfrozen-stages 1 \
  --normalizer-normalization layer --head-normalization layer \
  --wandb-run-name normalization_layer

python -m scripts.main --config configs/normalizer_experiments.yaml \
  --experiment-mode normalizer_joint_finetune \
  --transfer-mode fine_tuning --num-unfrozen-stages 1 \
  --normalizer-normalization instance --head-normalization instance \
  --wandb-run-name normalization_instance
```

Keep `unfreeze_stem: false` and `checkpoint: null` in the shared YAML. Configure
its dataset, pretrained HRNet weights, and output paths for the training machine.
These commands start from official HRNet weights and newly initialized heads,
not a previously trained task checkpoint. The two commands use distinct run names. Use new names when repeating runs.

`layer` means LayerNorm over C at each spatial position (NCHW -> NHWC ->
LayerNorm(C) -> NCHW), with one learned scale and bias per channel. It does not
normalize jointly over C,H,W. `instance` means InstanceNorm2d over H,W per image
and channel, with affine=True and track_running_stats=False. Both work
independently of the other images in a batch and use the same statistics in
train/eval. They have the same number of affine parameters per layer.

Normalization goes between convolution and activation in the normalizer's hidden
blocks, and replaces BN in the visibility, visible-landmark, and full-landmark
feature heads. Backbone BN, the output predictors, the normalizer's final RGB
convolution, and the residual addition are preserved. The zero final convolution
still gives an exact identity normalizer at initialization. This also initially
blocks gradients to its preceding hidden layers until the final convolution
learns; supervised training of the complete normalizer is therefore required.

Checkpoints record `landmarker_architecture` as well as the normalizer
architecture. Full checkpoints and split landmarker/normalizer exports can be
reloaded for evaluation/TTA. Legacy checkpoints without the new metadata retain
BatchNorm heads by default.

After training, use each run's full checkpoint with the existing evaluator:

```bash
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml \
  --checkpoint /absolute/path/to/run/checkpoints/full_model_best.pth
```

The existing episodic PCA TTA updates the complete normalizer only (convolutions
and normalization affine parameters), freezes HRNet and all heads, and resets
normalizer/optimizer state for each image. There are no InstanceNorm running
statistics to adapt. Compare baseline and adapted NME/visibility metrics for
each checkpoint; a lower PCA residual alone is not evidence of better landmarks.

The evaluator also supports `normalizer_head_norms` and `normalizer_heads` via
`--pca-tta-adaptation-scope`. See `docs/standalone_evaluation.md` for the three
TTA ablations. The default remains normalizer-only adaptation.


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
