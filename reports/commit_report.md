# Residual image normalizer implementation report

## Scope

A shallow residual image normalizer was added before the existing BabyLand-72
landmarker. Three supervised experiment modes are available: identity sanity,
normalizer-only training with a frozen landmarker, and joint normalizer plus
last-HRNet-stage/head fine-tuning. Test-time adaptation was not implemented.

## Added files

- `scripts/models/image_normalizer.py`
- `scripts/models/normalized_landmarker.py`
- `scripts/engine/normalizer_experiments.py`
- `configs/normalizer_sanity.yaml`
- `configs/normalizer_train_frozen_landmarker.yaml`
- `configs/normalizer_joint_finetune.yaml`
- `tests/test_image_normalizer.py`
- `docs/normalizer_experiments.md`

## Modified files

- `scripts/config.py`: normalizer settings, YAML loading, resolved config files.
- `scripts/main.py`: CLI/YAML precedence, modes, checkpoint compatibility,
  evaluation, diagnostics, reports.
- `scripts/engine/train.py`: optional residual L1/TV terms and metrics.
- `scripts/models/__init__.py`: public normalizer exports.

## Preserved behavior

The default mode is `none`. The existing HRNet outputs, visibility branch,
Wasserstein loss, barycenter decoder, crop-to-original projection, and official
dataset evaluation functions remain in use. Natural images are never used for
supervised training.

## Reports and visualizations

Runs save resolved configs, modular/full checkpoints and a manifest,
image-change statistics, prediction drift, visibility agreement, and input /
normalized / residual visual examples. Existing evaluation artifacts are
preserved.

## Tests

Lightweight tests cover shape preservation, exact identity initialization,
wrapper output compatibility, freezing behavior, and nested YAML overrides.
The complete lightweight suite passes: `24 passed`. The system Python emits a
pre-existing NumPy 2.x binary-compatibility warning from the installed PyTorch
build; it does not fail the tests, and the new image writer avoids relying on
the optional PyTorch-to-NumPy bridge.

## Known limitations

- No test-time optimization, DAE, PCA-guided TTA, or atlas switching is present.
- Natural-dataset paths and a compatible Wasserstein checkpoint must be supplied.
- Full dataset training/evaluation was not included in the lightweight test suite.

## Reproduction

See `docs/normalizer_experiments.md` and the three YAML files in `configs/`.
