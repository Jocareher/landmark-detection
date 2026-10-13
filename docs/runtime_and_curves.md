# Runtime measurements and loss curves

Enabled automatically for training, standalone evaluation and TTA, locally and
in Slurm snapshots. No new flags or dependencies are required.

## Training

At the training run root:

- `curves/train_val_losses.csv`: one row per split/epoch with EVERY returned
  training loss component (total, full/visible landmarks, visibility, PCA,
  image L1/TV), metrics and epoch/validation-forward timings. Updated every epoch.
- `curves/<loss>.png`: train/val curves for each loss on the same axes, generated
  after training. Existing `results.csv` and heatmap curves remain available.
- `timing/training_summary.json`: total `train_model` duration, sums of train
  and validation epoch times, completed epochs and validation-forward time.
  Includes optimizer work, validation, checkpoints, monitoring and curve exports;
  excludes subsequent test-set evaluation.
- `timing/run_summary.json`: the complete training CLI run, including model and
  data setup, training, final dataset evaluations and diagnostics. Excludes Slurm
  queue time and environment bootstrap. Printed to the terminal as well.

Validation forward measurements exclude loss computation and GT metrics and do
not run any extra forward passes. Epoch times use a monotonic synchronized clock.

## Prediction latency (SynBaby, BabyLand and InfAnFace)

Every evaluation/inference output directory contains:

- `timing/prediction_batches.csv`: sample IDs, batch size, transfer, inference or
  adaptation, decode, reprojection, and measured end-to-end seconds per batch.
- `timing/summary.csv`: ordered stage totals, counts, mean, standard deviation,
  median, P95, and amortized mean seconds per image.
- `timing/summary.json`: the same statistics, device and measurement definitions,
  plus total evaluator duration (including metrics/report work before this save).

**Input boundary:** the dataloader has delivered already prepared CPU crop
images. End-to-end latency starts before device transfer and ends after decoded
landmarks are reprojected to original-image coordinates. It excludes loading,
resizing and normalization that occurred inside the dataset before batch delivery.
The model's external normalizer is included in forward time.

Prediction reprojection is computed for the whole batch before GT metrics,
uncertainty diagnostics, label exports and overlays. Those operations are not
inside prediction end-to-end latency. The complete CLI run includes them.
BabyLand and InfAnFace use the existing crop-to-original affine transformation;
SynBaby uses its original-size scaling. No prediction geometry is changed.

For batch inference, batch latency divided by batch size is **amortized time per
image**, not independent per-image request latency. Summaries divide total batch
seconds by total image count, accounting for a smaller final batch. No extra
benchmark forward or warmup is executed; the first batch is included. Timings use
`perf_counter` and synchronize CUDA/MPS at stage boundaries. CPU is also supported.
These are instrumented wall-clock measurements, with synchronization overhead.

## TTA

The parent evaluation's end-to-end latency includes the entire `adapt_batch`
call instead of treating optimization as a single inference pass. Under `tta/`:

- `curves/tta_losses_per_image.csv`: all trajectory signals and weighted losses,
  including candidate pose-PCA residuals and the selected yaw where applicable.
- `curves/tta_losses_aggregate.csv`: per-step aggregate statistics.
- `curves/<loss>_mean.png`: every finite raw/weighted loss and pose candidate
  curve. Disabled PCA remains blank/NaN in CSV; no fictitious curve is plotted.
- `timing/model_forwards.csv`: normalizer + landmarker inference time at each
  adaptation step, with batch size, step index and gradient-mode flag. These
  forwards use the existing TTA gradient mode; they are not a separate no-grad
  inference benchmark. Source/fallback and visualization-only forwards are not
  included in this table.
- `timing/adaptation_batches.csv`: duration of each adapter call, including
  restoration, loss/backward/optimizer, monitoring and probe work.
- `timing/summary.json`: forward/adaptation distributions, amortized costs and
  elapsed adapter lifetime through artifact creation. Lifetime includes time
  between calls (data delivery, evaluator work); summed adaptation seconds
  isolate the actual adapter calls.

The parent `timing/run_summary.json` measures the complete evaluation/TTA command,
including checkpoint setup, adaptation, metrics, figures and normalizer diagnostics.
All durations are seconds. Parameters, buffers and fresh-optimizer restoration
remain unchanged. No GT is introduced into TTA optimization.
