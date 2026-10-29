#!/bin/bash
# Resources, working directory and log destinations are supplied by launch.py.
set -euo pipefail
plan=$1
conda_module=$2
conda_env=$3
python_executable=${4:-}
if (( $# >= 4 )); then
    shift 4
else
    shift "$#"
fi
# Module/Conda startup scripts may read unset interactive variables such as PS1.
# Keep errexit/pipefail enabled, and restore nounset after activation.
set +u
if [[ -n "$conda_module" ]]; then
    module load "$conda_module"
fi
for module_name in "$@"; do
    module load "$module_name"
done
if [[ -z "$python_executable" ]]; then
    conda_hook=$(conda shell.bash hook)
    eval "$conda_hook"
    unset conda_hook
fi
set -u
# launch.py sets --chdir to the snapshot; Slurm runs this script from its spool.
repo_root=$PWD
if [[ -n "$python_executable" ]]; then
    lmks_python=$python_executable
    "$lmks_python" "$repo_root/slurm/check_environment.py"
else
    source "$repo_root/slurm/ensure_lmks.sh"
    ensure_lmks "$conda_env" "$repo_root"
    lmks_python="${CONDA_PREFIX:?Conda activation did not set CONDA_PREFIX}/bin/python"
fi
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_COLOR=1 TERM=dumb TQDM_DISABLE=1
unset PYTHONPATH
# Use the selected environment explicitly: step startup must not resolve
# "python" using the node's default PATH. Preserve CUDA_VISIBLE_DEVICES.
if [[ ! -x "$lmks_python" ]]; then
    echo "[ENV] Missing executable: $lmks_python" >&2
    exit 1
fi
echo "[ENV] Launching with $lmks_python"
export SLURM_EXPORT_ENV=ALL
srun --export=ALL --unbuffered --ntasks=1 "$lmks_python" -u "$repo_root/slurm/run_job.py" "$plan"
