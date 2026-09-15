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
L_subspace = sum((s - (mu + b @ U))^2) / (2 * N)  # /144 for 72 landmarks
L_pca = lambda_pca_projection * L_subspace + lambda_pca_mahalanobis * L_mahalanobis
L_total = L_supervised + L_pca
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

`--lambda-pca-projection` weights only the projection MSE, restoring its original
normalization by the number of coordinates (144 for 72 landmarks). It is no
longer divided by total retained PCA variance. `--lambda-pca-mahalanobis` is an
independent weight for the bounded Mahalanobis penalty. Both default to zero;
either positive weight requires a valid prior path. To use both, explicitly set
both weights. Setting the Mahalanobis weight to zero gives the original residual
normalization (with the current decoder and stable alignment).

For example, append these options to your Wasserstein training command,
substituting the path to your existing prior:

```bash
--landmark-loss wasserstein \
--pca-prior-path /path/to/existing_global_prior.pt \
--lambda-pca-projection 1.0 \
--lambda-pca-mahalanobis 0.0001 \
--pca-mahalanobis-limit 2.0 \
--pca-variance-floor 1e-4
```

These weights illustrate independent control; they are not validated optima.
Reuse your previously validated residual weight where available. Mahalanobis
can still have large gradients at initialization. Restoring the residual scale
does not fix Procrustes scale invariance or guarantee absence of collapse.
No automatic warm-up is enabled by this change.

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

- `pca_loss`: actual weighted contribution to `total_loss`,
  `lambda_pca_projection * pca_subspace_loss + lambda_pca_mahalanobis * pca_mahalanobis_loss`.
- `pca_subspace_loss`: unweighted projection MSE, divided by 2N coordinates.
- `pca_mahalanobis_loss`: penalty only beyond the tolerance.
- `pca_mahalanobis_sq_per_component`: mean `q` before thresholding.
- `pca_outside_fraction`: fraction of shapes exceeding the Mahalanobis limit.
- `pca_projection_mse`: alias of `pca_subspace_loss`, retained for CSV compatibility.

The subspace and Mahalanobis diagnostics are unweighted; both are computed
when either weight is positive, even if the other term has weight zero.
All diagnostics are zero when both weights are zero.

**CSV migration:** older runs logged an unweighted `pca_loss` and a
variance-normalized `pca_subspace_loss`. Do not directly compare those columns
with this version. Use a new run directory for the new formulation. Existing CSV
headers are respected when appending to older result files; the extra columns
are included in new files. The resolved configuration records both weights and the tolerance parameters
when using `--save-config`.
