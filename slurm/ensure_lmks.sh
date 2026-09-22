#!/bin/bash
# Source after initializing Conda. Installation is serialized across this user's jobs.
ensure_lmks() {
    local env_name=$1
    local dependency_root=$2
    local lock_dir="${LMKS_ENV_LOCK_DIR:-$HOME/.cache/lmks-hpc}"
    mkdir -p "$lock_dir"
    command -v flock >/dev/null || { echo 'flock is required for safe concurrent environment setup.' >&2; return 1; }
    exec 9>"$lock_dir/environment.lock"
    echo "[ENV] Waiting for environment setup lock..."
    flock 9
    set +u
    if ! conda activate "$env_name" >/dev/null 2>&1; then
        echo "[ENV] Creating $env_name (Python 3.11)..."
        conda create --yes --name "$env_name" --override-channels -c conda-forge python=3.11 pip
        conda activate "$env_name"
    fi
    set -u
    export PYTHONNOUSERSITE=1
    unset PYTHONPATH
    # Existing complete environments are reused without reinstalling packages.
    if ! python "$dependency_root/slurm/check_environment.py" || ! python -m pip check; then
        echo '[ENV] Installing training and TTA dependencies...'
        python -c 'import sys; assert sys.version_info >= (3, 11), "The existing environment requires Python >=3.11; select a new conda_env name."'
        python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
        python -m pip install -r "$dependency_root/environments/requirements-hpc.txt"
        python -m pip check
        python "$dependency_root/slurm/check_environment.py"
    fi
    echo "[ENV] Ready: $CONDA_PREFIX"
    flock -u 9
    exec 9>&-
}
