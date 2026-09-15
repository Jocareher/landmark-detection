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

## PCA reconstruction with bounded coefficients (default)

`--pca-regularization bounded_reconstruction` reuses the existing global PCA
prior and limits each shape coefficient to a configurable number of standard
deviations. It is a training loss; inference still decodes network heatmaps
without PCA. No prior rebuilding or yaw-specific prior is required.

For each decoded prediction aligned by Procrustes, with `U` storing components
as rows and `v` the original explained variances:

```text
b = (s - mu) @ U.T
bounds = pca_coefficient_alpha * sqrt(v)
b_star = clip(b, -bounds, bounds)
s_star = mu + b_star @ U
L_bounded = mean((s - s_star)^2)             # /144 for 72 landmarks
L_total = L_supervised + lambda_pca_projection * L_bounded
```

`--pca-coefficient-alpha` defaults to 3: each coefficient can vary within
+/-3 of its own standard deviations. Alpha is not the loss weight. Accepted
coefficients are unchanged, so there is no continuous attraction to the mean.
Zero-variance components have a zero-width interval. No variance inversion or
variance floor is used to define these bounds.

The squared distance consists of the original projection MSE plus the squared
coefficient clipping residual divided by 2N. Gradients flow through Procrustes,
projection, clipping and reconstruction; the target is not detached. With
orthonormal PCA components this is the squared distance to the coefficient-box
constrained PCA set in aligned coordinates, differentiable also at its boundary.

To enable it, add the following options to your usual training command:

```bash
--landmark-loss wasserstein \
--pca-prior-path /path/to/existing_global_prior.pt \
--pca-regularization bounded_reconstruction \
--pca-coefficient-alpha 3.0 \
--lambda-pca-projection 1.0
```

The weight 1.0 is illustrative; reuse a previously validated projection weight
where possible and validate the new constraint. Regularization remains disabled
by default (`lambda_pca_projection=0`). No automatic warm-up is added. Setting
`lambda_pca_mahalanobis` nonzero in this mode is rejected rather than silently
adding the previous Mahalanobis penalty.

### Decoding and limitations

Wasserstein uses the differentiable barycenter at the configured temperature.
MSE/Adaptive Wing use the inference argmax/subpixel coordinates with a
straight-through soft-argmax gradient approximation. Inference is unchanged.
The shape loss uses float32 under AMP and stable analytic 2D alignment.

This is a soft training constraint, not a hard inference projection. Procrustes
removes scale and position; the loss alone cannot prevent spatial collapse or
ensure correct landmark localization. Independently bounded coefficients also
do not guarantee every joint combination is a real face. Validate NME and
visualizations, including different yaw views, against the no-PCA baseline.

### Metrics

- `pca_loss`: weighted contribution actually added to the training objective.
- `pca_bounded_loss`: unweighted MSE to the bounded reconstruction.
- `pca_subspace_loss`: original unbounded projection MSE.
- `pca_coefficient_loss`: excess coefficient squared distance divided by 2N.
- `pca_clipped_fraction`: fraction of all coefficients that exceeded their bounds.
- `pca_outside_fraction`: fraction of shapes with at least one clipped coefficient.
- `pca_projection_mse`: compatibility alias of `pca_subspace_loss`.

In bounded mode the historical Mahalanobis metrics are zero (not measured).
All PCA diagnostics are zero when regularization is disabled. In the console,
`clipped` refers to coefficients and `outside` refers to whole shapes.

Use a new run directory: historical `outside` referred to the Mahalanobis
ellipsoid, and the oldest `pca_loss` values were unweighted. Existing CSV headers
are respected; new files include the new metrics. `--save-config` records the
selected mode, alpha and weights.

### Previous Mahalanobis mode for comparison

Use `--pca-regularization mahalanobis` explicitly to reproduce the previous
combination (projection MSE /2N plus a separately weighted squared hinge):

```text
v_safe[j] = max(v[j], pca_variance_floor * max(v))
q = mean(b[j]^2 / v_safe[j])
L_mahalanobis = max(q / pca_mahalanobis_limit - 1, 0)^2
L_pca = lambda_pca_projection * L_subspace + lambda_pca_mahalanobis * L_mahalanobis
```

Both weights default to zero. The limit defaults to 2.0 and the relative floor
to 1e-4. Bounded-reconstruction metrics are zero in this legacy mode. The
Mahalanobis distance per component and outside fraction retain their previous
meanings. These limits are tunable tolerances, not guarantees of anatomical
validity or calibrated confidence levels.
