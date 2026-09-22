"""Submit isolated SNOW jobs. Uses only the Python standard library on login nodes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid

REPO = Path(__file__).resolve().parents[1]
SCOPES = ('normalizer', 'normalizer_head_norms', 'normalizer_heads')


def duration_seconds(value):
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', value)
    if not match:
        raise ValueError('Time must be HH:MM:SS or D-HH:MM:SS.')
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError('Invalid time minutes/seconds.')
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ValueError('Time must be positive.')
    return total


def choose_partition(walltime):
    seconds = duration_seconds(walltime)
    if seconds <= 7200:
        return 'short'
    if seconds <= 28800:
        return 'medium'
    if seconds <= 14 * 86400:
        return 'high'
    raise ValueError('SNOW guide lists a 14-day maximum; request a shorter time.')


def gpu_types(inventory):
    return set(re.findall(r'gpu:([A-Za-z0-9_.-]+):\d+', inventory))


def choose_gpu(requested, inventory):
    available = gpu_types(inventory)
    if requested != 'auto':
        if requested not in available:
            raise ValueError(f'GPU type {requested!r} not in this partition: {sorted(available)}')
        return requested
    # Resolve real GRES tokens; never assume that a physical model is a Slurm type.
    for pattern in (r'l40s|ada|lovelace', r'(?:quadro_?)?rtx_?6000'):
        matches = sorted(x for x in available if re.fullmatch(pattern, x, re.I))
        if matches:
            return matches[0]
    raise ValueError(f'No unambiguous L40S/RTX6000 GRES found: {sorted(available)}. '
                     'Set --gpu-type to an exact sinfo type; generic turing can include T4.')


def required_path(value, *, exists=True):
    path = Path(value).expanduser().resolve()
    if 'REPLACE' in str(path):
        raise ValueError(f'Configure the HPC path first: {path}')
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'tta'))
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--normalization', choices=('baseline', 'layer', 'instance'), default='layer',
                        help='Training: baseline = no normalizer norm + BatchNorm heads; layer/instance = both. TTA architecture comes from checkpoint.')
    parser.add_argument('--scope', choices=SCOPES, default='normalizer')
    parser.add_argument('--dataset', choices=('babyland', 'infanface'), default='babyland')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--normalizer-checkpoint', type=Path)
    parser.add_argument('--afterok', help='Wait for a successful training job ID.')
    parser.add_argument('--time', help='Overrides provisional, unbenchmarked settings.')
    parser.add_argument('--partition', choices=('short', 'medium', 'high'))
    parser.add_argument('--gpu-type', help='Exact type from sinfo; auto prefers L40S then RTX6000.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Check paths/resources and print plan; do not submit.')
    return parser.parse_args()


def validate_runtime_files():
    required = ('slurm/job.sh', 'slurm/ensure_lmks.sh', 'slurm/check_environment.py',
                'slurm/run_job.py', 'scripts/utils/source_images.py',
                'environments/requirements-hpc.txt')
    missing = [name for name in required if not (REPO / name).is_file()]
    if missing:
        raise FileNotFoundError(
            'Incomplete HPC checkout; update the repository before submitting. Missing: '
            + ', '.join(missing))


def main():
    args = arguments()
    validate_runtime_files()
    settings = json.loads(args.settings.read_text())
    resources = dict(settings[args.mode])
    for key in ('time', 'epochs', 'steps', 'batch_size'):
        value = getattr(args, key)
        if value is not None:
            resources[key] = value
    if args.mode == 'tta' and args.checkpoint is None:
        raise ValueError('TTA requires --checkpoint (full model or landmarker + normalizer).')
    if args.afterok and not re.fullmatch(r'\d+', args.afterok):
        raise ValueError('--afterok must be a numeric job ID.')
    for key in ('cpus', 'batch_size'):
        if int(resources[key]) <= 0:
            raise ValueError(f'{key} must be positive.')
    if not 0 <= resources['workers'] < resources['cpus']:
        raise ValueError('Use 0 <= workers < cpus (leave CPU time for the main process).')
    if args.mode == 'train' and resources['epochs'] < 1:
        raise ValueError('epochs must be positive.')
    if args.mode == 'tta' and resources['steps'] < 0:
        raise ValueError('steps must be nonnegative.')
    partition = args.partition or choose_partition(resources['time'])
    inventory = subprocess.check_output(
        ['sinfo', '--noheader', f'--partition={partition}', '--format=%G'], universal_newlines=True)
    gpu = choose_gpu(args.gpu_type or settings['gpu_type'], inventory)
    partition_info = subprocess.check_output(['scontrol', 'show', 'partition', partition], universal_newlines=True)
    maximum = re.search(r'MaxTime=(\S+)', partition_info)
    if maximum and maximum[1] not in ('UNLIMITED', 'INFINITE'):
        if duration_seconds(resources['time']) > duration_seconds(maximum[1]):
            raise ValueError(f'Request exceeds partition MaxTime={maximum[1]}')
    paths = settings['paths']
    needed = ['pca_prior'] + (['train_dataset', 'pretrained_weights'] if args.mode == 'train'
                             else [f'{args.dataset}_crops', f'{args.dataset}_labels',
                                   f'{args.dataset}_source_root'])
    selected_paths = {name: required_path(paths[name]) for name in needed}
    checkpoint = required_path(args.checkpoint, exists=not bool(args.afterok)) if args.checkpoint else None
    separate = required_path(args.normalizer_checkpoint, exists=not bool(args.afterok)) if args.normalizer_checkpoint else None
    root = Path(required_path(settings['runs_root'], exists=False))
    label = f'train_{args.normalization}' if args.mode == 'train' else f'tta_{args.dataset}_{args.scope}'
    name = f'{label}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    run = root / name
    plan = dict(mode=args.mode, normalization=args.normalization, scope=args.scope,
                dataset=args.dataset, paths=selected_paths, checkpoint=checkpoint,
                normalizer_checkpoint=separate, run_dir=str(run), resources=resources,
                partition=partition, gpu_type=gpu, wandb_mode=settings.get('wandb_mode', 'offline'),
                minimum_gpu_memory_gib=settings.get('minimum_gpu_memory_gib', 22))
    command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--no-requeue',
               f'--job-name={label}', f'--partition={partition}', f'--gres=gpu:{gpu}:1',
               f'--cpus-per-task={resources["cpus"]}', f'--mem={resources["memory"]}',
               f'--time={resources["time"]}', f'--chdir={run / "code"}',
               f'--output={run / "logs/slurm-%j.out"}', f'--error={run / "logs/slurm-%j.err"}']
    if settings.get('account'):
        command.append(f'--account={settings["account"]}')
    if args.afterok:
        command.extend([f'--dependency=afterok:{args.afterok}', '--kill-on-invalid-dep=yes'])
    command.extend([str(run / 'code/slurm/job.sh'), str(run / 'metadata/launch.json'),
                    settings.get('conda_module', ''), settings.get('conda_env', 'lmks')])
    print(' '.join(shlex.quote(part) for part in command))
    print(f'Run directory: {run}')
    if args.mode == 'train':
        print(f'Future checkpoint: {run / "checkpoints/full_model_best.pth"}')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for subdir in ('metadata', 'logs', 'code'):
        (run / subdir).mkdir(parents=True, exist_ok=False)
    # Freeze executable sources at submission; parallel/queued jobs do not follow checkout edits.
    for directory in ('scripts', 'slurm', 'environments'):
        shutil.copytree(REPO / directory, run / 'code' / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    template = 'normalizer_experiments.yaml' if args.mode == 'train' else 'pca_tta_evaluation.yaml'
    shutil.copy2(REPO / 'configs' / template, run / 'metadata/base.yaml')
    for filename, git_args in [('git_commit.txt', ['rev-parse', 'HEAD']),
                               ('git_status.txt', ['status', '--short']),
                               ('git_diff.patch', ['diff', 'HEAD'])]:
        result = subprocess.run(['git', '-C', str(REPO), *git_args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        (run / 'metadata' / filename).write_text(result.stdout)
    (run / 'metadata/launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    (run / 'metadata/submit_command.txt').write_text(' '.join(shlex.quote(part) for part in command) + '\n')
    (run / 'metadata/partition.txt').write_text(partition_info)
    (run / 'metadata/gres.txt').write_text(inventory)
    submitted = subprocess.check_output(command, universal_newlines=True).strip()
    (run / 'metadata/job_id.txt').write_text(submitted + '\n')
    print(f'Submitted job: {submitted}')


if __name__ == '__main__':
    main()
