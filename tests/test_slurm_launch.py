import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'slurm' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load_script('launch')
worker = load_script('run_job')


def test_snow_queues_and_real_gpu_types():
    assert launch.choose_partition('02:00:00') == 'short'
    assert launch.choose_partition('08:00:00') == 'medium'
    assert launch.choose_partition('24:00:00') == 'high'
    assert launch.choose_gpu('auto', 'gpu:turing:4 gpu:l40s:2(S:0)') == 'l40s'
    assert launch.choose_gpu('auto', 'gpu:rtx6000:2') == 'rtx6000'
    assert launch.choose_gpu('turing', 'gpu:turing:4') == 'turing'
    with pytest.raises(ValueError, match='unambiguous'):
        launch.choose_gpu('auto', 'gpu:turing:4')
    with pytest.raises(ValueError):
        launch.choose_partition('15-00:00:00')


@pytest.mark.parametrize('mode', ['train', 'tta'])
def test_submission_snapshot_and_dependency(tmp_path, monkeypatch, mode):
    settings = json.loads((ROOT / 'configs/upf_hpc.example.json').read_text())
    for key in settings['paths']:
        path = tmp_path / key
        path.mkdir()
        settings['paths'][key] = str(path)
    settings['runs_root'] = str(tmp_path / 'runs')
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps(settings))
    args = SimpleNamespace(mode=mode, settings=settings_path, normalization='instance',
                           scope='normalizer_heads', dataset='babyland', checkpoint=None,
                           normalizer_checkpoint=None, afterok=None, time=None,
                           partition=None, gpu_type=None, epochs=None, steps=None,
                           batch_size=None, dry_run=False)
    if mode == 'tta':
        args.checkpoint = tmp_path / 'future.pth'
        args.afterok = '123'
    commands = []
    def check_output(command, **kwargs):
        commands.append(command)
        if command[0] == 'sinfo':
            return 'gpu:l40s:2(S:0)\n'
        if command[0] == 'scontrol':
            return 'PartitionName=high MaxTime=14-00:00:00'
        return '456\n'
    monkeypatch.setattr(launch, 'arguments', lambda: args)
    monkeypatch.setattr(launch.subprocess, 'check_output', check_output)
    launch.main()
    run, = (tmp_path / 'runs').iterdir()
    plan = json.loads((run / 'metadata/launch.json').read_text())
    assert (run / 'code/scripts/main.py').is_file()
    assert (run / 'code/slurm/run_job.py').is_file()
    assert (run / 'logs').is_dir()
    command = commands[-1]
    assert '--gres=gpu:l40s:1' in command and '--ntasks=1' in command
    assert '--no-requeue' in command
    if mode == 'tta':
        assert '--dependency=afterok:123' in command
        assert '--kill-on-invalid-dep=yes' in command
    base = yaml.safe_load((run / 'metadata/base.yaml').read_text())['arguments']
    resolved = worker.resolved_arguments(plan, base)
    assert resolved['device'] == 'cuda'
    assert resolved['cache_dir'] == str(run / 'dataset_cache')
    assert resolved['num_workers'] < plan['resources']['cpus']
    if mode == 'train':
        assert resolved['normalizer_normalization'] == resolved['head_normalization'] == 'instance'
        assert resolved['num_unfrozen_stages'] == 1
        assert not resolved['evaluate_babyland'] and not resolved['evaluate_infanface']
        assert Path(resolved['output_dir']) / resolved['wandb_run_name'] == run
    else:
        assert resolved['pca_tta_adaptation_scope'] == 'normalizer_heads'
        assert resolved['checkpoint'] == str(args.checkpoint)
        assert resolved['natural_source_root'] == settings['paths']['babyland_source_root']


def test_generated_training_yaml_parses(tmp_path, monkeypatch):
    from scripts.main import parse_args, build_config_from_args
    base = yaml.safe_load((ROOT / 'configs/normalizer_experiments.yaml').read_text())['arguments']
    plan = dict(mode='train', run_dir=str(tmp_path / 'run'), normalization='layer',
                resources=dict(batch_size=8, workers=2, epochs=1),
                paths=dict(pca_prior='/weights/pca.pt', train_dataset='/data/train',
                           pretrained_weights='/weights/hrnet.pt'))
    config_path = tmp_path / 'generated.yaml'
    config_path.write_text(yaml.safe_dump({'arguments': worker.resolved_arguments(plan, base)}))
    monkeypatch.setattr('sys.argv', ['train', '--config', str(config_path)])
    config = build_config_from_args(parse_args())
    assert config.num_workers == 2 and config.batch_size == 8
    assert config.dataset_root == Path('/data/train')
    assert config.head_normalization == 'layer'
