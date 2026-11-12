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
