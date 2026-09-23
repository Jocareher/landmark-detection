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
    args = SimpleNamespace(mode=mode, settings=settings_path, config=None, normalization='instance',
                           scope='normalizer_heads', dataset='babyland', checkpoint=None,
                           normalizer_checkpoint=None, afterok=None, time=None,
                           partition=None, gpu_type=None, epochs=None, steps=None,
                           batch_size=None, dry_run=False)
    if mode == 'tta':
        args.checkpoint = tmp_path / 'future.pth'
        args.afterok = '123'
    commands = []
    def check_output(command, **kwargs):
        assert 'text' not in kwargs and 'capture_output' not in kwargs
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
    assert (run / 'code/environments/requirements-hpc.txt').is_file()
    assert (run / 'logs').is_dir()
    command = commands[-1]
    assert '--gres=gpu:l40s:1' in command and '--ntasks=1' in command
    assert '--no-requeue' in command
    if mode == 'tta':
        assert '--dependency=afterok:123' in command
        assert '--kill-on-invalid-dep=yes' in command
    base = yaml.safe_load((run / 'metadata/base.yaml').read_text())['arguments']
    assert run.name.startswith(base['wandb_run_name'] + '_')
    assert '--job-name=' + base['wandb_run_name'] in command
    resolved = worker.resolved_arguments(plan, base)
    assert resolved['device'] == 'cuda'
    assert resolved['cache_dir'] == str(run / 'dataset_cache')
    assert resolved['num_workers'] < plan['resources']['cpus']
    if mode == 'train':
        assert resolved['normalizer_normalization'] == resolved['head_normalization'] == 'instance'
        assert resolved['num_unfrozen_stages'] == 1
        assert resolved['evaluate_babyland'] and resolved['evaluate_infanface']
        assert resolved['babyland_source_root'] == settings['paths']['babyland_source_root']
        assert resolved['infanface_source_root'] == settings['paths']['infanface_source_root']
        assert Path(resolved['output_dir']) / resolved['wandb_run_name'] == run
    else:
        assert resolved['pca_tta_adaptation_scope'] == 'normalizer_heads'
        assert resolved['checkpoint'] == str(args.checkpoint)
        assert resolved['natural_source_root'] == settings['paths']['babyland_source_root']


@pytest.mark.parametrize('variant,normalizer_norm,head_norm', [
    ('baseline', 'none', 'batch'), ('layer', 'layer', 'layer'),
    ('instance', 'instance', 'instance')])
def test_generated_training_yaml_parses(tmp_path, monkeypatch, variant, normalizer_norm, head_norm):
    from scripts.main import parse_args, build_config_from_args
    base = yaml.safe_load((ROOT / 'configs/normalizer_experiments.yaml').read_text())['arguments']
    plan = dict(mode='train', run_dir=str(tmp_path / 'run'), normalization=variant,
                evaluations={'babyland': False, 'infanface': False},
                resources=dict(batch_size=8, workers=2, epochs=1),
                paths=dict(pca_prior='/weights/pca.pt', train_dataset='/data/train',
                           pretrained_weights='/weights/hrnet.pt'))
    config_path = tmp_path / 'generated.yaml'
    config_path.write_text(yaml.safe_dump({'arguments': worker.resolved_arguments(plan, base)}))
    monkeypatch.setattr('sys.argv', ['train', '--config', str(config_path)])
    config = build_config_from_args(parse_args())
    assert config.num_workers == 2 and config.batch_size == 8
    assert config.dataset_root == Path('/data/train')
    assert config.head_normalization == head_norm
    assert config.normalizer_internal_normalization == normalizer_norm


def test_login_launcher_uses_python36_syntax_and_apis():
    import ast
    source = (ROOT / 'slurm/launch.py').read_text()
    ast.parse(source, feature_version=(3, 6))
    assert 'from __future__ import annotations' not in source
    assert 'shlex.join(' not in source
    assert 'capture_output=' not in source
    assert 'text=True' not in source


def test_incomplete_checkout_fails_before_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'REPO', tmp_path)
    with pytest.raises(FileNotFoundError, match='ensure_lmks.sh'):
        launch.validate_runtime_files()


def test_baseline_cli_preset(monkeypatch):
    monkeypatch.setattr('sys.argv', ['launch.py', 'train', '--settings', 'hpc.json',
                                    '--normalization', 'baseline'])
    assert launch.arguments().normalization == 'baseline'


@pytest.mark.parametrize('scalar,expected', [
    ('my_experiment', 'my_experiment'),
    ('"instance run" # comment', 'instance_run'),
    ("'baseline run'", 'baseline_run'),
    ('experiment # comment', 'experiment'),
])
def test_yaml_run_name(scalar, expected):
    assert launch.yaml_run_name('arguments:\n  wandb_run_name: ' + scalar) == expected


@pytest.mark.parametrize('scalar', ['null', '', '|', '../..'])
def test_invalid_yaml_run_name(scalar):
    with pytest.raises(ValueError):
        launch.yaml_run_name('arguments:\n  wandb_run_name: ' + scalar)


def test_training_evaluation_switches_are_independent():
    source = 'arguments:\n  evaluate_babyland: false\n  evaluate_infanface: true\n'
    assert not launch.yaml_evaluation_enabled(source, 'evaluate_babyland')
    assert launch.yaml_evaluation_enabled(source, 'evaluate_infanface')
    plan = dict(mode='train', run_dir='/tmp/run', normalization='adain',
                evaluations={'babyland': False, 'infanface': True},
                resources=dict(batch_size=8, workers=2, epochs=1),
                paths=dict(pca_prior='/prior', train_dataset='/train',
                           pretrained_weights='/weights', infanface_crops='/crops',
                           infanface_labels='/labels', infanface_source_root='/images'))
    resolved = worker.resolved_arguments(plan, {})
    assert not resolved['evaluate_babyland'] and resolved['babyland_source_root'] is None
    assert resolved['evaluate_infanface']
    assert resolved['infanface_crop_root'] == '/crops'


def test_training_can_skip_natural_datasets_without_their_paths(tmp_path, monkeypatch, capsys):
    settings = json.loads((ROOT / 'configs/upf_hpc.example.json').read_text())
    settings['runs_root'] = str(tmp_path / 'runs')
    settings['paths'] = {}
    for key in ('train_dataset', 'pretrained_weights', 'pca_prior'):
        path = tmp_path / key
        path.touch()
        settings['paths'][key] = str(path)
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps(settings))
    config_path = tmp_path / 'train.yaml'
    config_path.write_text('arguments:\n  wandb_run_name: no_natural_eval\n'
                           '  evaluate_babyland: false\n  evaluate_infanface: false\n')
    args = SimpleNamespace(mode='train', settings=settings_path, config=config_path,
                           normalization='adain', scope='normalizer', dataset='babyland',
                           checkpoint=None, normalizer_checkpoint=None, afterok=None,
                           time=None, partition=None, gpu_type=None, epochs=None,
                           steps=None, batch_size=None, dry_run=True)
    monkeypatch.setattr(launch, 'arguments', lambda: args)
    monkeypatch.setattr(launch.subprocess, 'check_output',
                        lambda command, **kwargs: 'gpu:l40s:2' if command[0] == 'sinfo'
                        else 'PartitionName=medium MaxTime=08:00:00')
    launch.main()
    assert 'no_natural_eval_' in capsys.readouterr().out
    assert not (tmp_path / 'runs').exists()


def test_adain_hpc_uses_shared_training_yaml():
    source = (ROOT / 'configs/normalizer_experiments.yaml').read_text()
    base = yaml.safe_load(source)['arguments']
    assert launch.yaml_run_name(source) == base['wandb_run_name']
    plan = dict(mode='train', run_dir='/tmp/train_adain', normalization='adain',
                evaluations={'babyland': False, 'infanface': False},
                resources=dict(batch_size=8, workers=2, epochs=1),
                paths=dict(pca_prior='/weights/pca.pt', train_dataset='/data/train',
                           pretrained_weights='/weights/hrnet.pt'))
    resolved = worker.resolved_arguments(plan, base)
    assert resolved['normalizer_normalization'] == 'adain'
    assert resolved['head_normalization'] == 'adain'
