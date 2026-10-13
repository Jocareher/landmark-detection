# TTA with PCA, heatmap entropy and spatial spread

All objectives are available on `research-tta` through
`configs/pca_tta_evaluation.yaml`, `scripts.evaluate`, and `slurm/tta_landmarks.sh`.
The historical `pca_tta: true` switch enables the adapter for **all** signal
combinations. No retraining or new checkpoint architecture is required.
Training objectives are unchanged. The default uses one global PCA; minimum selection across pose priors is
available through `pca_prior_mode: min_pose` (see the pose PCA guide below).

## Definitions

For image b, landmark k, heatmap logits z, and spatial index i:

\[
p_{bki}=\frac{\exp(z_{bki})}{\sum_j\exp(z_{bkj})},\qquad
r_i=\left(\frac{x_i}{W-1},\frac{y_i}{H-1}\right),\qquad
\mu_{bk}=\sum_i p_{bki}r_i.
\]

Temperature is fixed at 1. Softmax is over **spatial positions** independently
for each landmark, not over channels. Stable log-softmax is used internally.

\[
L_H=-\frac{1}{BK\log(HW)}\sum_{b,k,i}p_{bki}\log p_{bki},\qquad
L_S=\frac{2}{BK}\sum_{b,k,i}p_{bki}\|r_i-\mu_{bk}\|_2^2.
\]

B is the episode batch size, K the number of output landmarks, H/W the heatmap
height/width, and x/y the spatial column/row coordinates. Entropy is normalized
by its maximum log(HW). Spread is twice the trace of spatial covariance:
each normalized coordinate has variance at most 1/4, so both losses lie in
[0,1]. Spread is a variance, not a standard deviation or augmentation measure.
All full-landmark heatmaps participate, including occluded landmarks; no GT or
predicted visibility mask is consumed. Gradients flow through probabilities,
spatial means, the chosen trainable parameters, and PCA alignment when enabled.

\[
L_{TTA}=\lambda_{PCA}L_{PCA}+\lambda_H L_H+\lambda_S L_S.
\]

`pca_loss_space: aligned` selects the original aligned reconstruction MSE;
`image` selects inverse-transformed reconstruction MSE in input-crop pixel².
A zero weight skips PCA completely, so its prior path can be null or absent.
Weights must be finite, nonnegative, and at least one must be positive.
There is no automatic balancing or face-size normalization. Aligned PCA can
be much smaller than entropy/spread; equal weights do not imply equal influence.
Choose weights and learning rates using a fixed validation protocol, and report
localization error as well as objective reduction. Heatmaps can become sharp at
incorrect positions, so decreasing these objectives does not prove better NME.

## YAML and local runs

Keep dataset/checkpoint paths and the desired `pca_tta_adaptation_scope` in the
existing YAML, then set:

```yaml
arguments:
  pca_tta: true
  tta_pca_weight: 0.0
  tta_entropy_weight: 1.0
  tta_spread_weight: 0.0
  pca_loss_space: aligned
  pca_prior_path: null
```

This fragment describes objective settings; edit these fields in the complete
YAML rather than replacing the full file with the fragment.

| Experiment | PCA weight | Entropy weight | Spread weight |
|---|---:|---:|---:|
| PCA | positive | 0 | 0 |
| Entropy | 0 | positive | 0 |
| Spread | 0 | 0 | positive |
| PCA + entropy | positive | positive | 0 |
| PCA + spread | positive | 0 | positive |
| Entropy + spread | 0 | positive | positive |
| All three | positive | positive | positive |

```bash
conda activate lmks
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml
```

CLI options override YAML. Use a distinct output directory for each experiment
to avoid overwriting reports. Examples for separate loss ablations:

```bash
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml --tta-pca-weight 0 --tta-entropy-weight 1 --tta-spread-weight 0 --wandb-run-name tta_entropy --output-dir runs/tta_entropy
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml --tta-pca-weight 0 --tta-entropy-weight 0 --tta-spread-weight 1 --wandb-run-name tta_spread --output-dir runs/tta_spread
```

## HPC

The same YAML weights are read by the login-node launcher, recorded in
`metadata/launch.json`, and copied into `metadata/experiment.yaml`. Overrides:

```bash
bash slurm/tta_landmarks.sh --settings configs/upf_hpc.example.json --checkpoint /path/to/full_model_best.pth --dataset babyland --scope normalizer --tta-pca-weight 0 --tta-entropy-weight 1 --tta-spread-weight 0 --run-name tta_entropy
bash slurm/tta_landmarks.sh --settings configs/upf_hpc.example.json --checkpoint /path/to/full_model_best.pth --dataset babyland --scope normalizer --tta-pca-weight 0 --tta-entropy-weight 0 --tta-spread-weight 1 --run-name tta_spread
```

Use `--config` to choose another complete YAML. For heatmap-only runs the HPC
JSON needs dataset paths, but no PCA prior. Resource, environment installation,
source snapshot, checkpoint loading and dataset reprojection remain unchanged.

## Episodes and reports

Adaptation scope remains independent of the objective. `normalizer_all_norms`
adapts a loader batch; other existing scopes adapt one image per episode.
`normalizer_all_instance_norms` requires batch size 1. Adapted source parameters
and buffers are restored and a fresh optimizer is created for each episode.

In `tta/trajectories.csv`, raw components are `pca_reconstruction_loss`,
`heatmap_entropy_loss`, and `heatmap_spread_loss`. Their weighted contributions
are `weighted_pca_loss`, `weighted_entropy_loss`, and `weighted_spread_loss`.
`sample_tta_loss` is their per-image sum; `total_tta_loss` is the episode batch
mean (equal to the sample objective for single-image episodes). Disabled PCA
has a blank/NaN raw loss and a zero weighted contribution.

`image_summary.csv` contains initial/final/minimum objective and component
values. `relative_loss_reduction` refers to the selected objective;
`relative_pca_loss_reduction` refers to raw PCA only. Aggregate curves use
per-image objectives, not repeated batch means. CSV and W&B logs retain both
raw and weighted components. Figures include `tta_loss_components.png`, the
objective curve, entropy/spread curves, and the existing drift/evaluation
figures. PCA-only runs retain the historical PCA curve filename; mixed or
heatmap-only runs use `tta_objective_curve.png`. PCA-versus-NME figures are
created only when PCA is enabled. GT-valid metrics remain post-hoc evaluation
and never enter the optimizer.

Pose-conditioned PCA banks and minimum-loss selection are described in
[pose PCA TTA](pose_pca_tta.md). Select `pca_prior_mode: min_pose`; the minimum
is taken per image before batch averaging, independently of entropy/spread.
