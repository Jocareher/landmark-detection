#!/bin/bash
# Submit from a login node. See README.md; do not run this wrapper via sbatch.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 "$repo_root/slurm/launch.py" train "$@"
