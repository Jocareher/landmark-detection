#!/bin/bash
# Optional environment preparation inside a compute allocation. Safe to repeat.
set -euo pipefail
: "${SLURM_JOB_ID:?First request an interactive compute allocation; see README.md}"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Module/Conda startup scripts may read unset interactive variables such as PS1.
# Keep errexit/pipefail enabled, and restore nounset after activation.
set +u
module load "${LMKS_CONDA_MODULE:-Miniconda3/4.9.2}"
conda_hook=$(conda shell.bash hook)
eval "$conda_hook"
unset conda_hook
set -u
source "$repo_root/slurm/ensure_lmks.sh"
ensure_lmks "${LMKS_CONDA_ENV:-lmks}" "$repo_root"
