"""Compute-node entry point: preflight, resolve YAML, record provenance, run one job."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def resolved_arguments(plan, base):
    args = dict(base)
    run = Path(plan['run_dir'])
    resources, paths = plan['resources'], plan['paths']
    args.update(device='cuda', batch_size=resources['batch_size'],
                num_workers=resources['workers'], cache_dir=str(run / 'dataset_cache'),
                pca_prior_path=paths['pca_prior'], save_config=True,
                wandb_run_name=run.name)
    if plan['mode'] == 'train':
        args.update(experiment_mode='normalizer_joint_finetune',
                    dataset_root=paths['train_dataset'],
                    pretrained_weights=paths['pretrained_weights'], checkpoint=None,
                    output_dir=str(run.parent), epochs=resources['epochs'],
                    normalizer_normalization=('none' if plan['normalization'] == 'baseline'
                                              else plan['normalization']),
                    head_normalization=('batch' if plan['normalization'] == 'baseline'
                                        else plan['normalization']), transfer_mode='fine_tuning',
                    num_unfrozen_stages=1, unfreeze_stem=False,
                    eval_batch_size=resources['batch_size'], evaluate_synbaby=True)
        for dataset_name in ('babyland', 'infanface'):
            enabled = plan['evaluations'][dataset_name]
            args[f'evaluate_{dataset_name}'] = enabled
            args[f'{dataset_name}_crop_root'] = (
                paths[f'{dataset_name}_crops'] if enabled else None)
            args[f'{dataset_name}_gt_root'] = (
                paths[f'{dataset_name}_labels'] if enabled else None)
            args[f'{dataset_name}_source_root'] = (
                paths[f'{dataset_name}_source_root'] if enabled else None)
    else:
        args.update(eval_mode=plan['dataset'], checkpoint=plan['checkpoint'],
                    normalizer_checkpoint=plan['normalizer_checkpoint'],
                    dataset_root=paths[f'{plan["dataset"]}_crops'],
                    natural_gt_root=paths[f'{plan["dataset"]}_labels'],
                    natural_source_root=paths[f'{plan["dataset"]}_source_root'],
                    output_dir=str(run), pca_tta=True,
                    pca_tta_adaptation_scope=plan['scope'], pca_tta_steps=resources['steps'],
                    pca_tta_learning_rate=resources['learning_rate'])
    return args


def record_command(command, path):
    with path.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    run = Path(plan['run_dir'])
    metadata = run / 'metadata'
    os.environ.update(WANDB_MODE=plan['wandb_mode'], WANDB_DIR=str(run),
                      WANDB_CACHE_DIR=str(run / 'wandb_cache'),
                      MPLCONFIGDIR=str(run / 'matplotlib_cache'))
    status = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running'}
    status_file = metadata / 'execution.json'
    status_file.write_text(json.dumps(status, indent=2))
    monitor = None
    monitor_stream = None
    exit_code = 1
    try:
        import yaml
        import numpy as np
        import torch
        import matplotlib
        import pandas
        import PIL
        import wandb
        import torchinfo
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Check Slurm GPU allocation and the CUDA PyTorch wheel/driver.')
        props = torch.cuda.get_device_properties(0)
        memory_gib = props.total_memory / 1024**3
        if memory_gib < plan['minimum_gpu_memory_gib']:
            raise RuntimeError(f'Allocated {props.name} has {memory_gib:.1f} GiB; '
                               f'configured minimum is {plan["minimum_gpu_memory_gib"]}. '
                               'Choose another GRES or explicitly lower minimum and batch size.')
        # Execute kernels rather than trusting CUDA availability alone.
        x = torch.ones((32, 32), device='cuda', requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        torch.from_numpy(np.zeros(2, dtype=np.float32)).numpy()
        runtime = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                       cuda=torch.version.cuda, gpu=props.name, gpu_memory_gib=memory_gib,
                       capability=torch.cuda.get_device_capability(0),
                       compiled_architectures=torch.cuda.get_arch_list(),
                       slurm={k: os.environ.get(k) for k in (
                           'SLURM_JOB_ID', 'SLURM_JOB_PARTITION', 'SLURM_JOB_NODELIST',
                           'SLURM_CPUS_PER_TASK', 'CUDA_VISIBLE_DEVICES')})
        (metadata / 'runtime.json').write_text(json.dumps(runtime, indent=2))
        print(json.dumps(runtime, indent=2), flush=True)
        record_command([sys.executable, '-m', 'pip', 'freeze'], metadata / 'pip-freeze.txt')
        record_command(['nvidia-smi'], metadata / 'nvidia-smi.txt')
        record_command(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID']], metadata / 'slurm-job.txt')
        # Hash source snapshot files so even uncommitted code is identifiable.
        hashes = {str(p.relative_to(run / 'code')): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (run / 'code').rglob('*') if p.suffix in ('.py', '.sh')}
        (metadata / 'source-sha256.json').write_text(json.dumps(hashes, indent=2))
        for path in [plan['checkpoint'], plan['normalizer_checkpoint'], *plan['paths'].values()]:
            if path is not None and not Path(path).exists():
                raise FileNotFoundError(path)
        base = yaml.safe_load((metadata / 'base.yaml').read_text())['arguments']
        config = metadata / 'experiment.yaml'
        config.write_text(yaml.safe_dump({'arguments': resolved_arguments(plan, base)}, sort_keys=False))
        module = 'scripts.main' if plan['mode'] == 'train' else 'scripts.evaluate'
        command = [sys.executable, '-u', '-m', module, '--config', str(config)]
        (metadata / 'python_command.txt').write_text(shlex.join(command) + '\n')
        print(shlex.join(command), flush=True)
        gpu_uuid = getattr(props, 'uuid', None)
        if gpu_uuid:
            monitor_stream = (run / 'logs/gpu.csv').open('w')
            monitor = subprocess.Popen([
                'nvidia-smi', f'--id={gpu_uuid}',
                '--query-gpu=timestamp,uuid,name,memory.used,memory.total,utilization.gpu',
                '--format=csv', '--loop=60'], stdout=monitor_stream, stderr=subprocess.DEVNULL)
        exit_code = subprocess.run(command, cwd=run / 'code').returncode
        status['status'] = 'completed' if exit_code == 0 else 'failed'
    except Exception as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        if monitor_stream is not None:
            monitor_stream.close()
        status.update(exit_code=exit_code, ended_at=datetime.now(timezone.utc).isoformat())
        status_file.write_text(json.dumps(status, indent=2) + '\n')
    return exit_code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
