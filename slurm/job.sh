#!/bin/bash
# Resources, working directory and log destinations are supplied by launch.py.
set -euo pipefail
plan=$1
conda_module=$2
conda_env=$3
if [[ -n "$conda_module" ]]; then
    module load "$conda_module"
fi
eval "$(conda shell.bash hook)"
conda activate "$conda_env"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_COLOR=1 TERM=dumb TQDM_DISABLE=1
unset PYTHONPATH
# Preserve Slurm's CUDA_VISIBLE_DEVICES. No dependency installation inside jobs.
srun --unbuffered --ntasks=1 python -u slurm/run_job.py "$plan"
