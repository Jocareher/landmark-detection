#!/bin/bash
# Submit one independent TTA experiment (one checkpoint, dataset and scope).
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${LMKS_LAUNCH_PYTHON:-python3}" "$repo_root/slurm/launch.py" tta "$@"
