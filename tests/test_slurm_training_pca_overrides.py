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


def test_yaml_experiment_sizes_and_portable_prior_override_hpc_defaults():
    source='arguments:\n  epochs: 60\n  batch_size: 32\n  pca_prior_path: "outputs/prior.pt" # shared bank\n'
    resources=launch.training_resource_defaults(source,{'epochs':1,'batch_size':8,'time':'08:00:00'})
    assert resources=={'epochs':60,'batch_size':32,'time':'08:00:00'}
    assert launch.portable_training_prior(source)==ROOT/'outputs/prior.pt'
    assert launch.portable_training_prior('arguments:\n  pca_prior_path: /desktop/prior.pt\n') is None
    for value in ['0','-1','1.5']:
        with pytest.raises(ValueError):launch.training_resource_defaults(f'  epochs: {value}\n',{})


def test_submission_needs_only_yaml_and_settings_for_full_baseline(tmp_path,monkeypatch,capsys):
    import json
    files={}
    for key in ['train_dataset','pretrained_weights','babyland_crops','babyland_labels','babyland_source_root']:
        path=tmp_path/key;path.mkdir();files[key]=str(path)
    # The prior in the HPC JSON must not silently replace the grouped YAML prior.
    files['pca_prior']=str(tmp_path/'wrong_prior_not_present.pt')
    python=tmp_path/'python';python.write_text('')
    settings=dict(cluster='correfoc',partition='std-gpu',runs_root=str(tmp_path/'runs'),
        python_executable=str(python),modules=[],paths=files,
        train=dict(epochs=1,batch_size=8,workers=4,cpus=8,memory='32G',time='08:00:00'))
    settings_file=tmp_path/'settings.json';settings_file.write_text(json.dumps(settings))
    monkeypatch.setattr('sys.argv',['launch','train','--settings',str(settings_file),
        '--config',str(ROOT/'configs/normalizer_experiments.yaml'),'--dry-run'])
    monkeypatch.setattr(launch.subprocess,'check_output',lambda cmd,**kw:
        'gpu:1\n' if cmd[0]=='sinfo' else 'PartitionName=std-gpu MaxTime=1-00:00:00')
    launch.main()
    output=capsys.readouterr().out
    plan=json.loads(output[output.index('{'):])
    assert plan['resources']['epochs']==60 and plan['resources']['batch_size']==32
    assert plan['paths']['pca_prior']==str(ROOT/'outputs/pose5_grouped_2026_11_12/pose5_grouped_parallel_alpha05.pt')
    assert plan['normalization'] is None
    assert plan['evaluations']=={'babyland':True,'infanface':False}
    monkeypatch.setattr('sys.argv',['launch','train','--settings',str(settings_file),
        '--config',str(ROOT/'configs/normalizer_experiments.yaml'),'--dry-run','--epochs','5','--batch-size','16'])
    launch.main();output=capsys.readouterr().out;plan=json.loads(output[output.index('{'):])
    assert plan['resources']['epochs']==5 and plan['resources']['batch_size']==16
