"""Exercise noninteractive shell startup without requiring Slurm or Conda."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', ['job.sh', 'install_lmks.sh'])
@pytest.mark.parametrize('hook_fails', [False, True])
def test_conda_startup_without_prompt(script, hook_fails, tmp_path):
    shell = r'''
set -e
unset PS1
export SLURM_JOB_ID=123 CONDA_PREFIX=/mock/lmks
flock() { :; }
module() { local prompt="$PS1"; }
conda() {
    if [[ "$1" == shell.bash ]]; then
        if [[ "$HOOK_FAILS" == yes ]]; then return 17; fi
        printf '%s\n' 'startup_prompt="$PS1"'
    else
        local prompt="$PS1"
    fi
}
srun() { [[ -o nounset ]] && printf 'WORK_STARTED\n'; }
python() { [[ -o nounset ]] && printf 'WORK_STARTED\n'; }
source "$1" plan.json Miniconda3/4.9.2 lmks
'''
    import os
    env = dict(os.environ, HOOK_FAILS='yes' if hook_fails else 'no',
               LMKS_ENV_LOCK_DIR=str(tmp_path))
    result = subprocess.run(['bash', '-c', shell, 'test', str(ROOT / 'slurm' / script)],
                            env=env, capture_output=True, text=True)
    if hook_fails:
        assert result.returncode == 17, result.stderr
        assert 'WORK_STARTED' not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert 'WORK_STARTED' in result.stdout


@pytest.mark.parametrize('existing,complete', [(False, False), (True, True), (True, False)])
def test_environment_created_or_reused(tmp_path, existing, complete):
    import os
    shell = r"""
set -euo pipefail
flock() { :; }
conda() {
    echo "conda $*" >> "$TRACE"
    if [[ "$1" == activate ]]; then
        [[ -f "$STATE/exists" ]] || return 1
        export CONDA_PREFIX="$STATE/env"
    else
        touch "$STATE/exists"
    fi
}
python() {
    echo "python $*" >> "$TRACE"
    if [[ "$1" == *check_environment.py ]]; then
        [[ -f "$STATE/complete" ]]
    elif [[ "$*" == *'pip install -r'* ]]; then
        touch "$STATE/complete"
    fi
}
source "$1"
ensure_lmks lmks "$2"
"""
    if existing:
        (tmp_path / 'exists').touch()
    if complete:
        (tmp_path / 'complete').touch()
    env = dict(os.environ, STATE=str(tmp_path), TRACE=str(tmp_path / 'trace'),
               LMKS_ENV_LOCK_DIR=str(tmp_path / 'locks'))
    result = subprocess.run(['bash', '-c', shell, 'test', str(ROOT / 'slurm/ensure_lmks.sh'), str(ROOT)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    trace = (tmp_path / 'trace').read_text()
    assert ('conda create' in trace) == (not existing)
    assert ('pip install' in trace) == (not complete)
    assert '[ENV] Ready' in result.stdout
