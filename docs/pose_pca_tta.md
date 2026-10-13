# Pose-conditioned PCA and minimum-loss TTA

## Build priors from SynBaby train labels

The source dataset must contain numerical yaw headers. Only the **train** split
is used, with resize and no random augmentation. All landmarks are included;
visibility does not filter the shapes. Modes:

- `global` (default): the historical single PCA, all finite training shapes.
- `pose5`: EXACT yaw -75, -45, 0, +45, +75 degrees. Other angles are excluded,
  not merged into nearby classes.
- `pose13`: EXACT yaw -75, -65, -55, -45, -30, -15, 0, +15, +30, +45, +55,
  +65, +75 degrees.

```bash
conda activate lmks
python -m scripts.build_pca_shape_prior --dataset-root /path/to/synbaby --output-path weights/pca_pose5_k32.pt --prior-mode pose5 --pca-num-components 32 --previews-dir weights/previews_pose5
python -m scripts.build_pca_shape_prior --dataset-root /path/to/synbaby --output-path weights/pca_pose13_k32.pt --prior-mode pose13 --pca-num-components 32 --previews-dir weights/previews_pose13
```

Every requested class must have at least two finite shapes; unknown/missing yaw
labels fail explicitly. Each file contains `pose_priors` plus a compatible
`global_prior` fitted to the same selected samples. The global prior remains
usable by the historical training and evaluation code. Training still uses
this global entry; minimum-pose selection is a **TTA** option.

All selected samples undergo shared generalized Procrustes alignment. Each
pose PCA fits its own mean and components on its subset of those aligned
shapes and uses the **same reference shape**. The saved metadata records class
order, sample counts, component counts, exact-angle selection and exclusions.
Previews include global diagnostics and one `yaw_<angle>/` directory per class.
Specify distinct `--previews-dir` values if building multiple files in one folder.

Use the same requested component count across models for a controlled
comparison. It is capped by each class's available samples and vector dimension;
inspect the saved counts/console output. `--pca-explained-variance` is also
supported but can yield different component counts and residual flexibility.

## TTA configuration

Edit these fields in the complete `configs/pca_tta_evaluation.yaml`:

```yaml
arguments:
  pca_tta: true
  pca_prior_path: /path/to/pca_pose5_k32.pt
  pca_prior_mode: min_pose
  pca_loss_space: aligned
  pca_tta_episode_unit: batch
  pca_tta_adaptation_scope: normalizer
  batch_size: 16
  tta_pca_weight: 1.0
  tta_entropy_weight: 0.0
  tta_spread_weight: 0.0
```

`pca_prior_mode: global` preserves single-prior behavior. `min_pose` uses every
pose prior in the file; it needs neither GT pose nor predicted pose. A legacy
file with only `global_prior` fails explicitly in `min_pose` mode. Rebuild it
using the commands above. Disabled PCA (`tta_pca_weight: 0`) does not require
any prior even if `min_pose` remains selected.

For predicted landmarks X_b and candidate pose c:

\[
L_b^c = \mathrm{MSE}(T(X_b),\mathrm{PCA}_c(T(X_b))),\quad
c_b^* = \arg\min_c L_b^c,\quad
L_{PCA}=\frac1B\sum_b L_b^{c_b^*}.
\]

This is the `aligned` mode. `image` compares original coordinates with the
inverse-transformed reconstruction for each candidate before taking the minimum.
The minimum is chosen **per image, before batch averaging**, and recomputed at
EVERY adaptation step. Gradients flow through only the winning candidate for
each image. Priors stay frozen; equal-loss ties select the first stored class.
Entropy/spread can still be added using the existing weighted objective.

`pca_tta_episode_unit: batch` shares one optimizer across the loader batch,
independently of adaptation scope. Source weights and buffers are restored after
the episode, and the next batch gets a fresh optimizer. `auto` preserves the
historical scope behavior; `image` adapts each image independently.
`normalizer_all_norms` requires batch/auto; `normalizer_all_instance_norms`
continues to require batch size 1.

## Launch

Use separate output directories per experiment:

```bash
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml --pca-prior-path weights/pca_pose5_k32.pt --pca-prior-mode min_pose --pca-tta-episode-unit batch --batch-size 16 --output-dir runs/tta_min_pose5_bs16
python -m scripts.evaluate --config configs/pca_tta_evaluation.yaml --pca-prior-path weights/pca_pose13_k32.pt --pca-prior-mode min_pose --pca-tta-episode-unit batch --batch-size 16 --output-dir runs/tta_min_pose13_bs16
```

For HPC, set `paths.pca_prior` in the settings JSON to the desired prior file,
then submit (the YAML options are used unless overridden):

```bash
bash slurm/tta_landmarks.sh --settings configs/upf_hpc.example.json --checkpoint /path/to/full_model_best.pth --dataset babyland --scope normalizer --pca-prior-mode min_pose --pca-tta-episode-unit batch --batch-size 16 --run-name tta_min_pose5_bs16
```

Build the prior ahead of submission in the `lmks` environment; the TTA job loads
it rather than rebuilding it. The launcher records overrides in launch metadata
and the resolved experiment YAML.

## Diagnostics

`trajectories.csv` includes each candidate's `pca_candidate_loss_yaw_<angle>`
and the winner's `pca_selected_yaw` per image/step. `pca_reconstruction_loss` is
the winning residual, not the mean across candidates. The image summary records
initial/final winner and the number of selection changes. Aggregate curves
include candidate statistics and winner counts by yaw; summary JSON records
prior selection mode and the available classes. These diagnostics let you see
if one flexible class dominates or selections switch repeatedly.

Compare GT-valid NME, not only the minimum residual: a smaller residual can
reflect a more permissive bank rather than more accurate landmark locations.
