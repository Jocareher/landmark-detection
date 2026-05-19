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

The visibility BCE branch and optional PCA projection regularizer remain part of
the multitask objective for every loss regime.
