import json
from pathlib import Path

import pytest
import yaml

from test_slurm_training_pca_overrides import launch, worker, ROOT


def test_tta_yaml_controls_adaptation_checkpoint_and_grouped_prior(tmp_path,monkeypatch,capsys):
    paths={}
    for key in ['babyland_crops','babyland_labels','babyland_source_root']:
        p=tmp_path/key;p.mkdir();paths[key]=str(p)
    paths['pca_prior']=str(tmp_path/'wrong_global_prior_not_present.pt')
    python=tmp_path/'python';python.write_text('')
    checkpoint=tmp_path/'new_full_model_best.pth';checkpoint.write_text('')
    settings=dict(cluster='correfoc',partition='std-gpu',runs_root=str(tmp_path/'runs'),
        python_executable=str(python),modules=[],paths=paths,
        tta=dict(steps=20,batch_size=1,learning_rate=.0001,workers=2,cpus=4,memory='16G',time='08:00:00'))
    settings_file=tmp_path/'settings.json';settings_file.write_text(json.dumps(settings))
    base=yaml.safe_load((ROOT/'configs/pca_tta_evaluation.yaml').read_text())['arguments']
    base['checkpoint']=str(checkpoint)
    config=tmp_path/'tta.yaml';config.write_text(yaml.safe_dump({'arguments':base},sort_keys=False))
    argv=['launch','tta','--settings',str(settings_file),'--config',str(config),'--dry-run']
    monkeypatch.setattr('sys.argv',argv)
    monkeypatch.setattr(launch.subprocess,'check_output',lambda cmd,**kw:
        'gpu:1\n' if cmd[0]=='sinfo' else 'PartitionName=std-gpu MaxTime=1-00:00:00')
    launch.main();output=capsys.readouterr().out;plan=json.loads(output[output.index('{'):])
    assert plan['checkpoint']==str(checkpoint)
    assert plan['scope']=='normalizer_all_norms'
    assert plan['resources']['batch_size']==16 and plan['resources']['steps']==1500
    assert plan['resources']['learning_rate']==.001
    assert plan['paths']['pca_prior'].endswith('pose5_grouped_parallel_alpha05.pt')
    resolved=worker.resolved_arguments(plan,base)
    assert resolved['pca_prior_mode']=='gt_pose' and resolved['pca_tta_episode_unit']=='batch'
    assert resolved['pca_mahalanobis_enabled'] and resolved['tta_pca_weight']==1
    assert resolved['pca_mahalanobis_weight']==pytest.approx(1.23917647e-6)
    assert resolved['pca_mahalanobis_percentile']==95
    assert resolved['tta_entropy_weight']==resolved['tta_spread_weight']==0
    monkeypatch.setattr('sys.argv',argv+['--steps','100','--batch-size','8','--tta-learning-rate','.0002','--scope','normalizer'])
    launch.main();output=capsys.readouterr().out;plan=json.loads(output[output.index('{'):])
    assert plan['scope']=='normalizer' and plan['resources']['batch_size']==8
    assert plan['resources']['steps']==100 and plan['resources']['learning_rate']==.0002


@pytest.mark.parametrize('source',['  pca_tta_steps: -1\n','  batch_size: 0\n',
                                  '  pca_tta_learning_rate: nan\n'])
def test_bad_tta_yaml_values_fail_before_submission(source):
    with pytest.raises(ValueError):launch.tta_resource_defaults(source,{})
