# GT-grouped PCA training: scale Mahalanobis to reconstruction

## Objective

`L_total = L_other + lambda_pca_reconstruction * R + lambda_pca_mahalanobis * H`

R is PCA reconstruction MSE. H is the raw hinge `max(D² − c, 0)`.
The two weights are independent. Keep the existing supervised loss weights and
reconstruction weight. Adjust ONLY the Mahalanobis weight to match the numerical
scale of the weighted reconstruction contribution. This is not a target of 5%
of supervised loss or gradients.

Explicit training weights must be set as a pair. They override the old
`lambda_pca_projection` / `pca_mahalanobis_weight` product. Existing configurations
that omit both fields retain legacy behavior. TTA weights are unchanged.

## GT grouping

Set `pca_training_prior_mode: gt_grouped`. Read the synthetic `yaw_angle` in
degrees and use saved prior metadata:

| BabyLand class | Synthetic angles | Prior key |
|---|---|---|
| 0 | −75, −65 | −75 |
| 1 | −55, −45, −30 | −45 |
| 2 | −15, 0, +15 | 0 |
| 3 | +30, +45, +55 | 45 |
| 4 | +65, +75 | 75 |

Zero degrees means class 2, not class 0. The assigned group alone contributes,
with its own significant p and chi-square cutoff. Training and synthetic
validation use the same mapping. Missing yaw, unsupported angles, old exact-angle
banks, or required PA metadata cause explicit errors.

## One diagnostic epoch

Use the existing training YAML and keep the existing architecture, initialization,
supervised weights, and reconstruction weight. For an existing reconstruction
weight of 1, the diagnostic fields under `arguments` are:

```yaml
epochs: 1
pca_training_prior_mode: gt_grouped
pca_prior_path: outputs/pose5_grouped_2026_11_12/pose5_grouped_parallel_alpha05.pt
pca_loss_space: aligned
pca_mahalanobis_enabled: true
pca_mahalanobis_percentile: 95.0
lambda_pca_reconstruction: 1.0  # KEEP the existing reconstruction weight
lambda_pca_mahalanobis: 0.0    # measure the new term before applying it
pca_gradient_monitor_interval: 10
wandb_run_name: grouped_pca_weight_diagnostic
```

If the previous reconstruction coefficient was different, use that value instead
of 1. Do not change it to perform this calibration. Keep your existing data paths
and remaining settings. No new YAML is required.

The raw Mahalanobis hinge is calculated even though its diagnostic weight is
zero. The run continues optimizing the existing supervised and reconstruction
objective. No full HRNet diagnostic training was launched during implementation.

Optional probes record raw losses, actual weighted losses, hinge activity, and
gradient norms every ten batches in
`RUN_DIRECTORY/loss_calibration/train_epoch_0001.csv`. Gradients do not need to be
used for the default loss-value scaling. An interval of 0 disables probes.

## Calculate ONLY the new Mahalanobis coefficient

```bash
python -m scripts.analysis.calibrate_training_pca_weights \
  --log-dir RUN_DIRECTORY/loss_calibration
```

The default method is `loss-relative`. It computes image-count-weighted means
of R and H over recorded probe batches, INCLUDING batches where H=0:

`lambda_M = lambda_R * mean(R) / mean(H)`

Consequently, on those diagnostic measurements:

`lambda_M * mean(H) = lambda_R * mean(R)`

The command prints the unchanged reconstruction weight and the proposed new
Mahalanobis weight, and writes `weight_starting_points.json`. If every H is zero,
it returns null for the new weight rather than inventing a value. A diagnostic
with reconstruction weight zero cannot provide the intended positive reference
and is rejected with instructions to keep the existing weight.

Hypothetical example: mean R=.001, mean H=100 and lambda_R=1 give lambda_M=.00001.
The two contributions then average .001 each. The raw Mahalanobis value remains
100; its weighted contribution becomes comparable to reconstruction. These are
illustrative numbers, not measurements from the actual training run.

An optional `--mahalanobis-to-reconstruction 0.5` targets half the weighted
reconstruction magnitude. The default 1.0 targets equal mean magnitudes.

## Validate the fixed coefficient

Copy the proposed Mahalanobis weight into the same YAML while preserving every
other loss weight. Keep this coefficient FIXED during a 3–5 epoch pilot. Do not
calculate R/H separately on every training batch. Compare against the same
initialization and a reconstruction-only baseline. Use held-out validation NME
and stability; do not choose weights from BabyLand test NME.

Mean loss-value matching is a scale calibration, not a proof of optimal weighting.
It does not guarantee equal gradient norms or prevent occasional spikes. Probes
also record gradients so those effects can be inspected separately if needed.
The older gradient-based helper modes remain available explicitly through
`--method gradient-independent` or `--method gradient-legacy`; neither is the
default procedure for the user's requested calibration.

### Selecting the coefficient for the full training run

Use one diagnostic epoch to calculate a starting value h0, then compare four
matched five-epoch pilots with lambda_H equal to 0, 0.3*h0, h0, and 3*h0.
The zero case is the grouped reconstruction-only baseline. Keep every other
loss weight, seed, architecture, trainable scope, data protocol, PA alpha and
chi-square coverage fixed. Restart each pilot from the same original
initialization/checkpoint, not from the preceding pilot's final model.

Compare validation NME at the same epoch and inspect the whole validation curve,
weighted H/R contribution, hinge activity and spikes. Total losses across these
pilots are different objectives and should not rank the candidates. Prefer a
stable candidate that improves validation NME over the zero-H baseline. If no
candidate helps, retain the baseline and investigate rather than assuming that
mean loss matching proves a nonzero H weight is beneficial.

Run the selected fixed coefficient over the full intended schedule and verify
the result again. Five-epoch pilots are a screening budget, not proof that
their ranking persists over the full run. An inconclusive pilot calls for
longer matched runs, not a claim that the numerical calibration is optimal.

## Which curves to read

- `weighted_pca_reconstruction_loss`: actual lambda_R * R.
- `weighted_pca_total_mahalanobis_loss`: actual lambda_M * H.
- `weighted_pca_loss`: sum of those two contributions.
- `pca_reconstruction_loss` and `pca_mahalanobis_penalty`: raw terms.
- `pca_mahalanobis_active_fraction`: fraction above the selected prior's bound.

In explicit-weight mode, `pca_loss` is the unweighted R+H diagnostic. It may be
large even when its actual weighted contribution is small. Read
`weighted_pca_loss` to see what enters the total objective. Epoch metrics go to
CSV/history and W&B. Explicit weights and routing mode are saved in checkpoints.

## Submit the complete baseline using only the YAML

The active `configs/normalizer_experiments.yaml` is configured for the matched
full baseline: all BatchNorm, GT-grouped alpha05 prior, q=95%, lambda_R=1,
lambda_H=0, 60 epochs and batch size32. It skips post-training SynBaby test and
InfanFace while retaining ordinary train/validation and BabyLand evaluation.

From the repository root on Correfoc:

```bash
bash slurm/train_hrnet_landmarks_template.sh \
  --settings configs/correfoc_hpc.example.json \
  --config configs/normalizer_experiments.yaml
```

Use the settings JSON already configured with your cluster paths. Add `--dry-run`
to check the plan first. No explicit normalization or loss-weight flags are
needed: the YAML supplies them. Epochs and batch size use CLI > training YAML >
JSON defaults. A repository-relative PCA prior in the YAML takes precedence
over the JSON prior path; old absolute workstation prior paths retain the HPC
JSON mapping unless overridden explicitly with `--pca-prior-path`. Dataset,
pretrained-weight and output paths, GPU/CPU resources and time limits remain
managed by the HPC settings. Each submission snapshots its own YAML, so later
edits do not change an already submitted job.

## Correfoc pilots with 90% coverage

Changing q from 95 to 90 changes the hinge values, so repeat the diagnostic if
the initial coefficient must still match mean reconstruction magnitude. Keep the
same alpha05 bank: q=90 does not mean PA alpha=.10. The launcher accepts training
weight and percentile overrides, snapshots each resolved job independently, and
can skip SynBaby test evaluation while retaining the usual training validation.
Its default no-normalization-override behavior retains the YAML architecture.

From the repository root on Correfoc, use the settings JSON with your configured
paths (replace the local settings filename below if yours has another name):

```bash
submit_h90() {
  bash slurm/train_hrnet_landmarks_template.sh \
    --settings configs/correfoc_hpc.local.json \
    --config configs/normalizer_experiments.yaml \
    --pca-prior-path "$PWD/outputs/pose5_grouped_2026_11_12/pose5_grouped_parallel_alpha05.pt" \
    --pca-training-prior-mode gt_grouped \
    --pca-loss-space aligned \
    --pca-mahalanobis-enabled \
    --pca-mahalanobis-percentile 90 \
    --lambda-pca-reconstruction 1 \
    --lambda-pca-mahalanobis "$2" \
    --pca-gradient-monitor-interval 10 \
    --epochs "$3" \
    --no-evaluate-synbaby --evaluate-babyland --no-evaluate-infanface \
    --run-name "$1" "${@:4}"
}

# Validate paths/resources first. No job is submitted with --dry-run.
submit_h90 gt_grouped_a05_q90_calibration 0 1 --no-evaluate-babyland --dry-run
# Submit one diagnostic epoch. No natural-dataset evaluation is needed here.
submit_h90 gt_grouped_a05_q90_calibration 0 1 --no-evaluate-babyland
```

After that job completes, activate the lmks environment and replace CAL90 below
with the actual run directory's loss_calibration subdirectory:

```bash
CAL90=/path/to/completed_calibration_run/loss_calibration
python -m scripts.analysis.calibrate_training_pca_weights --log-dir "$CAL90"
H90=$(python -c 'import json,sys; v=json.load(open(sys.argv[1]))["lambda_pca_mahalanobis"]; assert v is not None and v>0; print(v)' "$CAL90/weight_starting_points.json")
H90_LOW=$(python -c 'import sys; print(.3*float(sys.argv[1]))' "$H90")
H90_HIGH=$(python -c 'import sys; print(3*float(sys.argv[1]))' "$H90")

submit_h90 gt_grouped_a05_q90_H0   0           5
submit_h90 gt_grouped_a05_q90_H03 "$H90_LOW"  5
submit_h90 gt_grouped_a05_q90_H1  "$H90"      5
submit_h90 gt_grouped_a05_q90_H3  "$H90_HIGH" 5
```

The old 95%-calibrated weights remain possible exploratory candidates, but are
not guaranteed to match mean reconstruction loss at 90%. Each pilot keeps the
same reconstruction/supervised weights and YAML normalization. BabyLand runs
after training, SynBaby test and InfanFace do not. Validation during training
continues normally. The existing post-training best-checkpoint selection is
unchanged; inspect recorded checkpoints/epochs when comparing BabyLand results.

## Checkpoint selection

Training saves `checkpoints/full_model_best.pth` at the minimum SynBaby validation
total loss and `checkpoints/full_model_best_nme.pth` at the minimum finite
SynBaby validation NME. The last epoch is exported as
`checkpoints/full_model_last.pth`. Automatic post-training evaluation and early
stopping still use validation total loss. Pass either best checkpoint explicitly
to evaluate on BabyLand or to run TTA.

## TTA after the full training runs

The active `configs/pca_tta_evaluation.yaml` applies grouped PCA + Mahalanobis TTA
on BabyLand, using the SAME grouped alpha05 bank and GT orientation routing.
The reconstruction weight is1, the Mahalanobis weight is1.23917647e-6, and
chi-square coverage is95%.
The protocol is all normalization layers, batch size16,1500steps, constant
learning rate.001, seed42, and no entropy/spread losses. Model architecture
comes from the full checkpoint, not from the evaluation YAML.

Compare the two source models under this same fixed adaptation objective:
source-trained lambda_H=0 versus source-trained lambda_H=1.23917647e-6.
To isolate the effect of Mahalanobis during adaptation, additionally run the SAME
checkpoint with `pca_mahalanobis_enabled: false`. The active weight is set
explicitly in the TTA YAML; training weights are not copied automatically.

In the TTA YAML, set `checkpoint` to the new training run's
`checkpoints/full_model_best.pth`, keep `normalizer_checkpoint: null`, and give
`wandb_run_name` a distinct name identifying the source model. Then submit:

```bash
bash slurm/tta_landmarks.sh \
  --settings configs/correfoc_hpc.example.json \
  --config configs/pca_tta_evaluation.yaml
```

Use your configured HPC settings filename. Add `--dry-run` to verify the
checkpoint, grouped prior, scope, batch size, steps and learning rate first.
The launcher now reads those experiment fields and the checkpoint from YAML
unless explicitly overridden by CLI; HPC dataset paths/resources still come
from the settings JSON. Each job snapshots its YAML.

GT pose mode uses BabyLand orientation classes0–4 to choose a fixed prior. GT
landmark coordinates remain evaluation-only. This is an oracle-orientation
protocol, as in the previous GT-conditioned experiments.

The active TTA+Mahalanobis experiment has `pca_mahalanobis_enabled: true` and
`pca_mahalanobis_weight: 1.23917647e-6`. In the TTA engine the coefficients are
`tta_pca_weight * R + tta_pca_weight * pca_mahalanobis_weight * H`. Training's
`lambda_pca_reconstruction` and `lambda_pca_mahalanobis` are not TTA switches.
The stored small beta is only an exploratory starting value, not a coefficient
calibrated on natural-image adaptation. Keep the same source checkpoint when
comparing adaptation objectives.

For local execution, activate your training/evaluation environment, check the
local BabyLand paths in the YAML, and run from the repository root:

```bash
python -m scripts.evaluate \
  --config configs/pca_tta_evaluation.yaml \
  --checkpoint /path/to/new_training_run/checkpoints/full_model_best.pth
```

Alternatively, fill `checkpoint` in that YAML and omit the CLI checkpoint flag.
Use different run/output names when evaluating different source checkpoints.
