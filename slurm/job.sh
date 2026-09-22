#!/bin/bash
# Resources, working directory and log destinations are supplied by launch.py.
set -euo pipefail
plan=$1
conda_module=$2
conda_env=$3
# Module/Conda startup scripts may read unset interactive variables such as PS1.
# Keep errexit/pipefail enabled, and restore nounset after activation.
set +u
if [[ -n "$conda_module" ]]; then
    module load "$conda_module"
fi
conda_hook=$(conda shell.bash hook)
eval "$conda_hook"
unset conda_hook
set -u
# launch.py sets --chdir to the snapshot; Slurm runs this script from its spool.
repo_root=$PWD
source "$repo_root/slurm/ensure_lmks.sh"
ensure_lmks "$conda_env" "$repo_root"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_COLOR=1 TERM=dumb TQDM_DISABLE=1
unset PYTHONPATH
# Use the activated environment explicitly: step startup must not resolve "python"
# using the node's default PATH. Preserve CUDA_VISIBLE_DEVICES and Conda exports.
lmks_python="${CONDA_PREFIX:?Conda activation did not set CONDA_PREFIX}/bin/python"
if [[ ! -x "$lmks_python" ]]; then
    echo "[ENV] Missing executable: $lmks_python" >&2
    exit 1
fi
echo "[ENV] Launching with $lmks_python"
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --unbuffered --ntasks=1 "$lmks_python" -u "$repo_root/slurm/run_job.py" "$plan"
