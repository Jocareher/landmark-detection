#!/bin/bash
# Run ONCE inside an interactive compute allocation, not on the login node.
set -euo pipefail
: "${SLURM_JOB_ID:?First request an interactive compute allocation; see README.md}"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
module load "${LMKS_CONDA_MODULE:-Miniconda3/4.9.2}"
eval "$(conda shell.bash hook)"
# Fails if lmks already exists instead of silently mutating an active environment.
conda env create -f "$repo_root/environments/lmks-hpc.yml"
conda activate lmks
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$repo_root/environments/requirements-hpc.txt"
python -m pip check
