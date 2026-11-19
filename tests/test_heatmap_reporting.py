from __future__ import annotations

import csv

import numpy as np
import torch

from scripts.engine.evaluation_reporting import add_distribution_statistics, clean_metrics, write_dataset_metric_summary
from scripts.engine.heatmap_diagnostics import heatmap_image_rows, save_heatmap_diagnostics
from scripts.engine.metrics import compute_heatmap_uncertainty
from scripts.engine.pca_tta import _aggregate_trajectories


def test_entropy_and_spread_limits_and_logit_offset_invariance():
    uniform = torch.zeros(1, 2, 2, 2, requires_grad=True)
    result = compute_heatmap_uncertainty(uniform)
    assert torch.allclose(result['entropy'], torch.ones(1, 2))
    assert torch.allclose(result['spread'], torch.ones(1, 2))
    assert not result['entropy'].requires_grad
    peaked = torch.full((1, 1, 4, 4), -100.)
    peaked[0, 0, 1, 2] = 100.
    assert compute_heatmap_uncertainty(peaked)['entropy'].max() < 1e-6
    assert compute_heatmap_uncertainty(peaked)['spread'].max() < 1e-6
    logits = torch.randn(3, 72, 5, 7)
    a, b = compute_heatmap_uncertainty(logits), compute_heatmap_uncertainty(logits + 10)
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6)
        assert ((a[key] >= 0) & (a[key] <= 1)).all()


def test_dataset_cleaning_preserves_only_requested_protocols():
    metrics = {'mean_nme_box': .1, 'mean_nme_box_gt_valid': .2, 'mean_nme_box_non_contour': .3, 'mean_nme_interocular': .4, 'mean_nme_box_visible_intersection': .5,
               'orientation_metrics': {'left': {'mean_nme_box_gt_valid': .2, 'mean_nme_box_visible_intersection': .5}},
               'visibility_metrics': {'global': {'f1': .9}}}
    syn = clean_metrics(metrics, 'synbaby')
    assert 'mean_nme_box' in syn and 'mean_nme_box_gt_valid' not in syn
    assert 'mean_nme_box_non_contour' not in syn
    baby = clean_metrics(metrics, 'babyland')
    assert set(baby) == {'mean_nme_box_gt_valid', 'orientation_metrics', 'visibility_metrics'}
    assert set(baby['orientation_metrics']['left']) == {'mean_nme_box_gt_valid'}
    assert clean_metrics(metrics, 'infanface') == metrics


def test_std_is_between_images_and_ignores_missing_errors(tmp_path):
    rows = [{'orientation': 'left', 'mean_nme_box_gt_valid': .1, 'hausdorff_box_gt_valid': .2}, {'orientation': 'left', 'mean_nme_box_gt_valid': .3, 'hausdorff_box_gt_valid': .6}, {'orientation': 'right', 'mean_nme_box_gt_valid': .2, 'hausdorff_box_gt_valid': float('nan')}, {'orientation': 'right', 'mean_nme_box_gt_valid': None}]
    summary = {'orientation_metrics': {'left': {}, 'right': {}, 'frontal': {}}}
    add_distribution_statistics(summary, rows)
    assert np.isclose(summary['std_nme_box_gt_valid'], np.std([.1, .3, .2]))
    assert np.isclose(summary['orientation_metrics']['left']['std_nme_box_gt_valid'], .1)
    assert summary['orientation_metrics']['right']['std_nme_box_gt_valid'] == 0
    assert summary['orientation_metrics']['frontal']['std_nme_box_gt_valid'] is None
    write_dataset_metric_summary(tmp_path / 'metrics_summary.csv', summary)
    assert (tmp_path / 'metrics_by_orientation.csv').is_file()
    with (tmp_path / 'metrics_summary.csv').open() as handle:
        exported = list(csv.DictReader(handle))
    assert all(row['value'] != 'None' for row in exported)


def test_heatmap_artifacts_have_shared_scales_and_pose_stats(tmp_path):
    rows = heatmap_image_rows(torch.zeros(2, 72, 4, 4), ['a', 'b'])
    rows[0]['orientation'], rows[1]['orientation'] = 'left', 'right'
    summary = {}
    save_heatmap_diagnostics(rows, tmp_path, summary)
    assert summary['mean_heatmap_entropy'] == 1
    assert summary['std_heatmap_entropy'] == 0
    assert summary['orientation_metrics']['left']['mean_heatmap_entropy'] == 1
    assert (tmp_path / 'heatmaps/per_image.csv').is_file()
    assert (tmp_path / 'figures/heatmaps/by_orientation.png').is_file()


def test_tta_uncertainty_aggregation_weights_images_not_batches():
    rows = [dict(sample_id=str(i), step=0, failed=False, pca_reconstruction_loss=1., mean_landmark_drift_px=0., max_landmark_drift_px=0., mean_heatmap_entropy=value, mean_heatmap_spread=value / 2) for i, value in enumerate([.1, .3, .8])]
    aggregate = _aggregate_trajectories(rows)[0]
    assert np.isclose(aggregate['heatmap_entropy_mean'], .4)
    assert np.isclose(aggregate['heatmap_spread_mean'], .2)


def test_final_evaluators_export_clean_protocols_even_if_all_predicted_invisible(tmp_path, monkeypatch):
    from scripts.engine import evaluate, evaluate_natural

    class Model(torch.nn.Module):
        def forward(self, images):
            return {'heatmaps': torch.zeros(len(images), 72, 4, 4), 'visibility_logits': torch.full((len(images), 72), -100.)}

    for module in (evaluate, evaluate_natural):
        for name in ('plot_per_landmark_boxplot', 'plot_yaw_view_boxplots', 'plot_confusion_matrix', 'plot_grouped_nme_boxplot'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda **kwargs: None)
    landmarks = torch.stack((torch.arange(72) % 8 + 1, torch.arange(72) // 8 + 1), dim=1).float()
    sample = {'image': torch.zeros(3, 16, 16), 'landmarks': landmarks, 'visibility': torch.ones(72), 'metadata': {
        'sample_id': 'a', 'image_path': str(tmp_path / 'a.png'), 'yaw_angle': 0., 'original_size': (16, 16), 'transformed_size': (16, 16),
        'orientation': 'frontal', 'source_image_path': str(tmp_path / 'a.png'), 'crop_image_path': str(tmp_path / 'a.png'),
        'crop_size': (16, 16), 'transform_crop_to_orig': torch.eye(3), 'class_idx': 2, 'source_image_name': 'a.png'}}
    loader = torch.utils.data.DataLoader([sample], batch_size=1)
    syn = evaluate.evaluate_checkpoint(Model(), loader, torch.device('cpu'), tmp_path / 'syn', save_overlays=False)
    assert 'mean_nme_box_gt_valid' not in syn
    assert syn['num_landmarks'] == 72
    mirrored = {**sample, 'landmarks': landmarks + 2, 'metadata': {**sample['metadata'], 'sample_id': 'flip_a'}}
    baby_loader = torch.utils.data.DataLoader([sample, mirrored], batch_size=1)
    natural = evaluate_natural.evaluate_natural_checkpoint(Model(), baby_loader, torch.device('cpu'), tmp_path / 'baby', save_overlays=False)
    subsets = natural["image_subset_metrics"]
    assert subsets["original"]["num_samples"] == subsets["mirrored"]["num_samples"] == 1
    assert natural["num_samples"] == 2
    assert subsets["original"]["mean_nme_box_gt_valid"] < subsets["mirrored"]["mean_nme_box_gt_valid"]
    with (tmp_path / "baby/metrics_by_image_subset.csv").open() as handle:
        split = {row["image_subset"]: row for row in csv.DictReader(handle)}
    assert set(split) == {"original", "mirrored"}
    with (tmp_path / "baby/metrics_summary.csv").open() as handle:
        assert {"Original images", "Mirrored images"} <= {row["section"] for row in csv.DictReader(handle)}

    assert natural['mean_nme_box_gt_valid'] is not None
    assert 'mean_nme_box' not in natural
    assert 'std_nme_box_gt_valid' in natural
    assert subsets['original']['std_nme_box_gt_valid'] == subsets['mirrored']['std_nme_box_gt_valid'] == 0
    with (tmp_path / 'baby/per_image_per_landmark_nme.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 144
    assert {row['evaluation_landmark_inclusion'] for row in rows} == {'gt_valid'}
    with (tmp_path / 'baby/per_image_nme.csv').open() as handle:
        fields = csv.DictReader(handle).fieldnames
    assert 'mean_nme_box_gt_valid' in fields
    assert 'mean_nme_box' not in fields and 'mean_nme_interocular' not in fields


def test_babyland_tta_monitors_official_nme_and_annotates_gif(tmp_path, monkeypatch):
    from PIL import Image

    from scripts.engine import evaluate_natural, pca_tta
    from scripts.models import NormalizedLandmarker, ResidualImageNormalizer

    class Landmarker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm2d(3)
            self.norm.running_var.fill_(4.0)
            self.head = torch.nn.Conv2d(3, 72, 1)

        def forward(self, images):
            heatmaps = self.head(self.norm(images))
            return {"heatmaps": heatmaps, "visible_heatmaps": heatmaps,
                    "visibility_logits": heatmaps.mean(dim=(2, 3))}

    model = NormalizedLandmarker(
        Landmarker(), ResidualImageNormalizer(hidden_channels=4, num_layers=2)
    )
    landmarks = torch.stack((torch.arange(72) % 8 + 2, torch.arange(72) // 8 + 2), dim=1).float()
    samples = []
    for name, orientation, class_id in (("a", "frontal", 2), ("flip_a", "left", 0)):
        path = str(tmp_path / f"{name}.png")
        samples.append({
            "image": torch.randn(3, 16, 16), "landmarks": landmarks,
            "visibility": torch.ones(72), "metadata": {
                "sample_id": name, "orientation": orientation, "class_idx": class_id,
                "source_image_path": path, "crop_image_path": path,
                "crop_size": (32, 32), "transformed_size": (16, 16),
                "transform_crop_to_orig": torch.tensor([[1., 0., 3.], [0., 1., 2.], [0., 0., 1.]]),
                "source_image_name": f"{name}.png",
            },
        })
    loader = torch.utils.data.DataLoader(samples, batch_size=2)
    for name in ("plot_per_landmark_boxplot", "plot_yaw_view_boxplots", "plot_confusion_matrix",
                 "plot_grouped_nme_boxplot"):
        if hasattr(evaluate_natural, name):
            monkeypatch.setattr(evaluate_natural, name, lambda **kwargs: None)
    labels = []
    build_panel = pca_tta._build_probe_panel

    def track_label(**kwargs):
        labels.append(kwargs["nme_label"])
        return build_panel(**kwargs)

    monkeypatch.setattr(pca_tta, "_build_probe_panel", track_label)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    for episode_unit in ("batch", "image"):
        labels.clear()
        output = tmp_path / episode_unit
        scope = "normalizer_all_norms" if episode_unit == "batch" else "normalizer"
        adapter = pca_tta.PCAGuidedTTA(
            model, None, torch.device("cpu"), output / "tta",
            pca_tta.PCATTAConfig(pca_weight=0, entropy_weight=1, steps=3,
                                 monitor_steps=(1,), probe_count=1, episode_unit=episode_unit,
                                 adaptation_scope=scope),
            dataset_protocol="babyland",
        )
        evaluate_natural.evaluate_natural_checkpoint(
            model, loader, torch.device("cpu"), output / "babyland",
            save_predictions=False, save_overlays=False, tta_adapter=adapter,
            coordinate_decoder="barycenter",
        )
        assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())
        summary = adapter.finalize()["monitored_nme"]
        assert summary["num_complete_trajectories"] == 2
        with (output / "tta/nme_by_image_and_step.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        for item in adapter.summary_rows:
            scored = [row for row in rows if row["sample_id"] == item["sample_id"]]
            assert [row["step"] for row in scored] == ["", "0", "1", "3"]
            assert np.isclose(float(scored[0]["nme_box_gt_valid"]), item["initial_nme_box_gt_valid"])
            assert np.isclose(float(scored[-1]["nme_box_gt_valid"]), item["final_nme_box_gt_valid"])
            if episode_unit == "batch":
                assert not np.isclose(float(scored[1]["nme_box_gt_valid"]), item["initial_nme_box_gt_valid"])
        assert labels == [f"NME (GT): {float(row['nme_box_gt_valid']):.2%}"
                          for row in rows if row["sample_id"] == "a" and row["step"]]
        with Image.open(output / "tta/probes/a/adaptation.gif") as gif:
            assert gif.n_frames == 3
        assert (output / "tta/nme_by_orientation_and_step.csv").is_file()
        assert all((output / "tta/figures" / filename).is_file() for filename in (
            "nme_trajectory_by_orientation.png", "nme_trajectory_by_image.png",
        ))
