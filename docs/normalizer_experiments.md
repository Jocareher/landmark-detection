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
| `normalizer_joint_finetune` | Trained | Existing last-stage-plus-heads fine-tuning | Compare adapter-only training with the current partial fine-tuning policy |

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
  --checkpoint /absolute/path/to/best_model.pth
```

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

Optional residual regularization is enabled in the shared YAML with
`normalizer_image_regularization`, `normalizer_lambda_l1`, and
`normalizer_lambda_tv`, or with the corresponding CLI options. It is disabled
by default.

## Outputs

Each run keeps the existing evaluation outputs and adds:

- `configs/resolved_config.yaml` and `.json`;
- `checkpoints/full_model_best.pth` and `full_model_last.pth`;
- `checkpoints/normalizer_best.pth` and `normalizer_last.pth`;
- modular landmarker checkpoints for joint fine-tuning;
- `checkpoints/checkpoint_manifest.json`;
- per-dataset image-change and prediction-drift CSV/JSON files in `metrics/`;
- input, normalized, absolute-residual, and side-by-side examples in
  `visualizations/<dataset>/`;
- `reports/report.md`.

The diagnostic report includes mean L1/L2 and maximum image difference,
residual statistics, total variation, changed-pixel fractions, heatmap drift,
decoded landmark displacement, visibility-logit drift, and visibility decision
agreement. Official performance remains the NME, region, pose, visibility,
failure-rate, and detection-rate output of the existing evaluation code.

## Checkpoint compatibility

Legacy checkpoints containing only the HRNet landmarker state can initialize a
wrapped model. Wrapped checkpoints contain `normalizer.*` and `landmarker.*`
keys. The manifest records the base checkpoint, mode, trainable modules,
decoder, loss path, evaluation protocol, Git revision, and resolved config.

## Interpretation and limitations

The sanity mode should produce zero or numerical-noise-level drift because the
normalizer is identity-initialized. A learned normalizer should be judged by
both official landmark metrics and the magnitude/structure of its image
changes. A low shape-prior or drift loss alone does not prove correct image
localization.

These experiments do not establish domain generalization by themselves. They
only establish the architecture, supervised trainability, checkpointing, and
diagnostic baseline needed before considering per-image test-time updates.
