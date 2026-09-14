# Landmark Heatmap Loss Regimes

The training entry point supports three landmark heatmap loss regimes:

```bash
python scripts/main.py --landmark-loss mse
python scripts/main.py --landmark-loss adaptive_wing
python scripts/main.py --landmark-loss wasserstein
```

The default is `mse`, which preserves the original mean-squared heatmap
training objective.

## Decoder Pairings

- `mse` uses argmax heatmap decoding with subpixel refinement.
- `adaptive_wing` uses argmax heatmap decoding with subpixel refinement.
- `wasserstein` uses barycenter decoding.

The selected decoder is stored in the resolved config as `coordinate_decoder` and
is used consistently for train/validation NME, SynBaby evaluation, BabyLand
evaluation, and InfAnFace inference/evaluation.

## Losses

`mse` uses per-landmark heatmap MSE for both the full heatmap branch and the
visible-only heatmap branch.

`adaptive_wing` uses Adaptive Wing Loss for both landmark heatmap branches. The
implementation follows the standard piecewise heatmap regression formulation
from "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression" with
configurable `omega`, `theta`, `epsilon`, and `alpha`.

`wasserstein` treats each heatmap channel as a spatial probability distribution.
Predicted heatmaps are normalized with a spatial softmax, target Gaussian
heatmaps are normalized to sum to one, and the loss compares x/y marginal CDFs.
This is a separable 2D Wasserstein-style approximation chosen to avoid dense
4096x4096 transport matrices for 64x64 heatmaps and 72 landmarks.

The visibility BCE branch and optional global PCA shape regularizer remain part of
the multitask objective for every loss regime.

## Global PCA regularization during training

The existing global Procrustes PCA prior is reused without rebuilding it or
conditioning it on yaw. Inference, evaluation and export use the network's
decoded heatmaps directly; they do not load or apply PCA. Learning this behavior
requires training or fine-tuning with the regularizer enabled. It is a soft
training constraint, not a guarantee that every unseen prediction is plausible.

For each prediction, let `s` be its Procrustes-aligned shape, `mu` the prior mean,
`U` the retained components as rows, and `v` their explained variances:

```text
b = (s - mu) @ U.T
v_safe[j] = max(v[j], pca_variance_floor * max(v))
q = mean(b[j]^2 / v_safe[j])                 # squared Mahalanobis distance / K
L_mahalanobis = max(q / pca_mahalanobis_limit - 1, 0)^2
L_subspace = sum((s - (mu + b @ U))^2) / sum(v_safe)
L_pca = L_subspace + L_mahalanobis
L_total = L_supervised + lambda_pca_projection * L_pca
```

The Mahalanobis term has exactly zero gradient throughout the accepted region,
including nonmean shapes. Outside it, the penalty grows smoothly from zero.
The supervised heatmap losses continue to anchor predictions to the image's
ground truth. The residual term detects deviations outside the retained
subspace, which coefficient-space Mahalanobis cannot see. It penalizes those
deviations without directly shrinking the retained coefficients.

The default limit is `q <= 2.0`, equivalent to `D_M^2 <= 2K`, not `D_M <= 2`.
This is an initial tunable tolerance, **not** a calibrated confidence level.
Tune it and the regularizer weight on validation data, tracking accuracy by yaw
as well as overall accuracy. A global prior can still disfavor underrepresented
poses. The relative variance floor defaults to `1e-4` to stabilize nearly zero
eigenvalues.

The historical `--lambda-pca-projection` flag now weights this combined loss.
Its scale differs from the old unnormalized projection MSE, so previous weights
should be retuned. For example, append these options to your training command,
substituting the path to your existing prior:

```bash
--pca-prior-path /path/to/existing_global_prior.pt \
--lambda-pca-projection 0.01 \
--pca-mahalanobis-limit 2.0 \
--pca-variance-floor 1e-4
```

The weight `0.01` is an example starting value, not a validated optimum. The
default weight remains zero; a positive weight requires a valid prior path.

### Coordinates and gradients

The regularizer's forward pass uses the same coordinates as inference:
argmax with subpixel refinement for MSE/Adaptive Wing, or the barycenter at the
configured softmax temperature for Wasserstein. For argmax, backpropagation uses
a **straight-through soft-argmax surrogate** because argmax has no useful
coordinate gradient. This gradient is an approximation; the forward coordinates
are the actual argmax predictions. Barycenter uses its exact differentiable
decoder. No reconstructed PCA coordinates replace the model outputs.

The shape loss stays in float32 under AMP. The training alignment uses an
analytic 2D rotation, matching the prior alignment on nondegenerate shapes and
avoiding SVD-gradient singularities. Initially collapsed predictions use a
clamped scale and identity rotation when the rotation is undefined.

### Diagnostics

Training/validation history, new `results.csv` files and W&B include:

- `pca_loss`: combined, unweighted regularizer.
- `pca_subspace_loss`: variance-normalized residual outside the subspace.
- `pca_mahalanobis_loss`: penalty only beyond the tolerance.
- `pca_mahalanobis_sq_per_component`: mean `q` before thresholding.
- `pca_outside_fraction`: fraction of shapes exceeding the Mahalanobis limit.
- `pca_projection_mse`: original projection MSE for scale comparison.

These diagnostics are zero when regularization is disabled. Existing CSV
headers are respected when appending to older result files; the extra columns
are included in new files. The resolved configuration records the two new
parameters when using `--save-config`.
