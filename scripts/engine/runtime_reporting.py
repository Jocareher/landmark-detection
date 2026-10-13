"""Synchronized runtime measurements and loss-curve artifacts."""
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def clock(device):
    synchronize(device)
    return time.perf_counter()


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    return dict(count=int(values.size), mean_seconds=float(values.mean()),
                std_seconds=float(values.std()), median_seconds=float(np.median(values)),
                p95_seconds=float(np.percentile(values, 95)), total_seconds=float(values.sum()))


class PredictionTiming:
    """Batch latency from an already prepared crop tensor to original coordinates."""
    def __init__(self, device):
        self.device = device
        self.started = clock(device)
        self.rows = []
        self.current = None

    def begin(self, sample_ids, adaptation=False):
        self.current = dict(batch_index=len(self.rows), batch_size=len(sample_ids),
                            sample_ids='|'.join(str(x) for x in sample_ids), adaptation=adaptation)
        self.batch_start = clock(self.device)

    @contextmanager
    def stage(self, name):
        start = clock(self.device)
        yield
        self.current[f'{name}_seconds'] = clock(self.device) - start

    def finish(self, original_coordinates=True):
        self.current['end_to_end_seconds'] = clock(self.device) - self.batch_start
        self.current['output_coordinates'] = 'original_image' if original_coordinates else 'crop'
        self.rows.append(self.current)
        self.current = None

    def save(self, output_dir):
        directory = Path(output_dir)/'timing'
        directory.mkdir(parents=True, exist_ok=True)
        write_rows(directory/'prediction_batches.csv', self.rows)
        samples = sum(row['batch_size'] for row in self.rows)
        summary = dict(total_evaluation_seconds=clock(self.device)-self.started,
                       num_images=samples, num_batches=len(self.rows),
                       measurement='synchronized perf_counter; actual execution; includes first batch; no extra warmup',
                       end_to_end_boundary='prepared CPU crop tensor -> transfer -> forward or TTA -> decode -> original-coordinate reprojection',
                       exclusions='image loading/resize/normalization before loader yield, GT metrics, overlays and prediction/report file writes',
                       per_image_mean_definition='total measured batch latency / image count; amortized cost, not independent image latency',
                       device=str(self.device), torch_version=str(torch.__version__))
        for name in ('transfer', 'inference', 'adaptation', 'decode', 'reprojection', 'end_to_end'):
            values = [row[f'{name}_seconds'] for row in self.rows if f'{name}_seconds' in row]
            stats = distribution(values)
            if stats:
                stats['amortized_mean_seconds_per_image'] = stats['total_seconds']/samples if samples else None
            summary[name] = stats
        write_rows(directory/'summary.csv', [dict(stage=name, **summary[name]) for name in
                   ('transfer','inference','adaptation','decode','reprojection','end_to_end') if summary[name] is not None])
        (directory/'summary.json').write_text(json.dumps(summary, indent=2))
        return summary


def save_training_curves(history, output_dir, plot=True):
    directory = Path(output_dir)/'curves'
    rows = [dict(epoch=i+1, split=split, **metrics)
            for split, epochs in history.items() for i, metrics in enumerate(epochs)]
    write_rows(directory/'train_val_losses.csv', rows)
    keys = sorted({key for row in rows for key in row if 'loss' in key})
    if not keys or not plot:
        return
    import matplotlib.pyplot as plt
    for key in keys:
        fig, ax = plt.subplots(figsize=(7,4))
        for split in history:
            selected = [row for row in rows if row['split'] == split]
            ax.plot([row['epoch'] for row in selected], [row.get(key, np.nan) for row in selected], label=split)
        ax.set(xlabel='Epoch', ylabel=key, title=key.replace('_',' ').title())
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory/f'{key}.png', dpi=180)
        plt.close(fig)


def project_prediction_batch(landmarks, metadata, affine=False):
    from .postprocessing import (extract_batched_size, project_landmarks_between_sizes,
                                project_landmarks_to_original_size, apply_homogeneous_transform)
    results = []
    for i, prediction in enumerate(landmarks):
        transformed = extract_batched_size(metadata['transformed_size'], i)
        if affine and 'crop_size' in metadata and 'transform_crop_to_orig' in metadata and 'source_image_path' in metadata:
            crop = project_landmarks_between_sizes(prediction, transformed,
                                                   extract_batched_size(metadata['crop_size'], i))
            original = apply_homogeneous_transform(crop, metadata['transform_crop_to_orig'][i])
        else:
            crop = None
            original = project_landmarks_to_original_size(prediction, transformed,
                        extract_batched_size(metadata['original_size'], i)).numpy()
        results.append((crop, original))
    return results


def save_run_timing(output_dir, started, kind, device=None):
    if device is not None:
        synchronize(device)
    directory = Path(output_dir)/'timing'
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(kind=kind, total_run_seconds=time.perf_counter()-started,
                  boundary='CLI main entry through setup, computation, evaluation, diagnostics and reports; excludes queue/conda/bootstrap')
    (directory/'run_summary.json').write_text(json.dumps(result, indent=2))
    print(f"[TIMING] {kind} complete run: {result['total_run_seconds']:.2f} s")
    return result


def save_tta_curves(trajectory, aggregate, output_dir):
    directory = Path(output_dir)/'curves'
    write_rows(directory/'tta_losses_per_image.csv', trajectory)
    write_rows(directory/'tta_losses_aggregate.csv', aggregate)
    if not aggregate:
        return
    keys = sorted({key for row in aggregate for key in row if 'loss' in key and key.endswith('_mean')})
    import matplotlib.pyplot as plt
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in aggregate])
        if not np.isfinite(values).any():
            continue
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot([row['step'] for row in aggregate], values)
        ax.set(xlabel='TTA step', ylabel=key, title=key.replace('_',' ').title())
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(directory/f'{key}.png', dpi=180)
        plt.close(fig)
