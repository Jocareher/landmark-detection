# Optional PCA coefficient penalty

The optional objective is `L_PCA = R + beta * max(D_squared - c, 0)` per image,
then averaged over images. `R` is the existing reconstruction MSE and
`D_squared = sum_j b_j^2 / max(explained_variance_j, eps)`.

Coefficients `b` come from the aligned, mean-centered predicted shape. Eigenvalues
come from the fixed source prior, not the test batch. No dense inverse is needed.
The distance sums over components without division by their count. `c` is
subtracted directly, not squared or square-rooted. It is an experimental threshold,
with no automatic chi-square confidence-level interpretation.

Use the following under `arguments` in either normalizer_experiments.yaml or
pca_tta_evaluation.yaml:

```yaml
pca_loss_space: aligned
pca_mahalanobis_enabled: true
pca_mahalanobis_c: 10.0
pca_mahalanobis_weight: 1.0
pca_mahalanobis_eps: 1.0e-8
```

Disabled by default. Set `pca_mahalanobis_enabled: false` to preserve the original
loss. Zero beta also removes its contribution, but enabled mode still validates
the prior and records the distance. Beta defaults to 1 as a formula coefficient,
not as a calibrated recommendation: tune it separately from c, especially when
aligned reconstruction MSE is small. c controls activation; beta controls strength.

CLI equivalents: `--pca-mahalanobis-enabled`, `--no-pca-mahalanobis-enabled`,
`--pca-mahalanobis-c`, `--pca-mahalanobis-weight`, `--pca-mahalanobis-eps`.
The Slurm launcher preserves these YAML arguments in the run snapshot, so no
new HPC JSON paths or dependencies are required.

Training multiplies the complete PCA objective by `lambda_pca_projection`; TTA
multiplies it by `tta_pca_weight`. If the outer PCA weight is zero, no coefficient
loss participates in optimization. The checkpoint architecture is unchanged.
Training metadata never overrides the explicit objective selected for TTA.

Both reconstruction modes remain available. With `aligned`, R uses aligned
coordinates. With `image`, R uses input-crop pixels squared. The coefficient term
always uses aligned PCA coefficients and is dimensionless. Beta must account for
the different scales. No alignment, reconstruction or coefficient is detached.

For `min_pose`, each candidate uses its own mean, components and eigenvalues.
Selection minimizes the complete objective per image before batch averaging.

The prior requires `explained_variance` matching its retained components. The
existing builder already saves this field. Rebuilding is unnecessary if present.
Missing or invalid eigenvalues raise an explicit error when enabled. Epsilon
floors zero and tiny variances.

Recorded terms:

- `pca_reconstruction_loss`: R alone.
- `pca_mahalanobis_squared`: raw D squared (zero when disabled).
- `pca_mahalanobis_penalty`: max(D squared - c, 0).
- `weighted_pca_mahalanobis_loss`: beta times that penalty, before the outer weight.
- `pca_loss`: R plus the weighted penalty.
- TTA `weighted_pca_loss`: outer PCA weight times pca_loss.

Training history/curve CSVs, checkpoints and TTA metadata record the settings.
TTA trajectories, image summaries and aggregate curves include these components.
Curve plots include reconstruction, total PCA loss and the weighted coefficient
penalty. Inside the allowed region, the hinge and its coefficient gradient are zero.
