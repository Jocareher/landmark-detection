# Residual image normalizer experiments

## Motivation

These experiments test the supervised prerequisite of a future test-time
adaptation method: whether a small appearance adapter can be inserted before
BabyLand-72 without changing the established landmark pipeline. Test-time
adaptation, DAE guidance, atlas switching, and target-domain training are
intentionally not implemented here.

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

Raw outputs are saved under
`normalizer_monitoring/source_validation/`, including per-checkpoint panels,
CSV metrics, checkpoint grids, and GIF animations. The same panels and numeric
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
- per-dataset image-change and prediction-drift CSV/JSON files in `metrics/`;
- input, normalized, auto-scaled absolute residual, fixed-scale absolute
  residual, signed residual, amplified normalized output, and labeled
  side-by-side examples in
  `visualizations/<dataset>/`;
- fixed-probe checkpoint panels, grids, GIFs, and raw statistics in
  `normalizer_monitoring/source_validation/`;
- `reports/report.md`.

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
