import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location('slurm_'+name,ROOT/'slurm'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


launch=module('launch');worker=module('run_job')


def test_cli_and_resolved_worker_preserve_independent_weights_and_90_coverage(monkeypatch):
    monkeypatch.setattr('sys.argv',['launch','train','--settings','cluster.json',
        '--pca-training-prior-mode','gt_grouped','--lambda-pca-reconstruction','1',
        '--lambda-pca-mahalanobis','0.000004','--pca-mahalanobis-percentile','90',
        '--pca-mahalanobis-enabled','--pca-gradient-monitor-interval','10',
        '--no-evaluate-synbaby','--evaluate-babyland','--no-evaluate-infanface'])
    overrides=launch.resolve_training_overrides(launch.arguments())
    plan=dict(mode='train',run_dir='/tmp/run',normalization=None,
        training_overrides=overrides,evaluations={'babyland':True,'infanface':False},
        resources=dict(batch_size=32,workers=4,epochs=5),
        paths=dict(pca_prior='/priors/grouped.pt',train_dataset='/synbaby',pretrained_weights='/hrnet.pt',
                   babyland_crops='/crops',babyland_labels='/labels',babyland_source_root='/images'))
    base={'lambda_pca_reconstruction':1.,'lambda_pca_mahalanobis':0.,'pca_mahalanobis_percentile':95.,
          'evaluate_synbaby':True,'landmark_loss':'wasserstein','normalizer_normalization':'none'}
    resolved=worker.resolved_arguments(plan,base)
    assert resolved['lambda_pca_mahalanobis']==.000004
    assert resolved['lambda_pca_reconstruction']==1. and resolved['pca_mahalanobis_percentile']==90.
    assert resolved['pca_training_prior_mode']=='gt_grouped'
    assert resolved['evaluate_synbaby'] is False and resolved['evaluate_babyland'] is True
    assert resolved['evaluate_infanface'] is False and resolved['epochs']==5
    assert resolved['landmark_loss']=='wasserstein' and base['pca_mahalanobis_percentile']==95.


@pytest.mark.parametrize('key,value',[('lambda_pca_mahalanobis',-1),('lambda_pca_reconstruction',float('nan')),
    ('pca_mahalanobis_percentile',100),('pca_gradient_monitor_interval',-1)])
def test_bad_overrides_fail_on_login_node(key,value):
    with pytest.raises(ValueError):launch.resolve_training_overrides(SimpleNamespace(mode='train',**{key:value}))


def test_training_flags_not_silently_ignored_for_tta():
    with pytest.raises(ValueError,match='train mode'):
        launch.resolve_training_overrides(SimpleNamespace(mode='tta',lambda_pca_mahalanobis=1.))
