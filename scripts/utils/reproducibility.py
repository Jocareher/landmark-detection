from __future__ import annotations

import datetime
import getpass
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


def get_git_repo_root(start_path: Path | None = None) -> Path | None:
    """Find the nearest parent directory that contains a `.git` folder."""
    start_path = Path.cwd() if start_path is None else Path(start_path).resolve()
    for parent in [start_path, *start_path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def run_shell_command(command: list[str], cwd: Path | None = None) -> str:
    """Execute a shell command and return stdout, or an empty string on failure."""
    try:
        return (
            subprocess.check_output(command, stderr=subprocess.DEVNULL, cwd=cwd)
            .decode()
            .strip()
        )
    except Exception:
        return ""


def save_reproducibility_metadata(
    output_dir: Path,
    parsed_args: dict[str, Any],
    include_git_diff: bool = True,
    include_pip_freeze: bool = True,
) -> Path:
    """Write Git, environment, and CLI metadata for one experiment run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "metadata.txt"

    repo_root = get_git_repo_root()
    git_commit = "N/A"
    git_branch = "N/A"
    git_dirty = False
    git_diff_content = ""

    if repo_root is not None:
        git_commit = (
            run_shell_command(["git", "rev-parse", "HEAD"], cwd=repo_root)[:10] or "N/A"
        )
        git_branch = (
            run_shell_command(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root
            )
            or "N/A"
        )
        git_dirty = bool(
            run_shell_command(["git", "status", "--porcelain"], cwd=repo_root)
        )
        if include_git_diff and git_commit != "N/A":
            git_diff_content = run_shell_command(["git", "diff"], cwd=repo_root)

    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "working_dir": str(Path.cwd()),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "python_ver": sys.version.replace("\n", " "),
        "torch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda or "cpu",
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    }

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in parsed_args.items()
    }

    with metadata_file.open("w", encoding="utf-8") as handle:
        handle.write(
            "# --- RUN REPRODUCIBILITY METADATA -----------------------------------\n"
        )
        for key, value in metadata.items():
            handle.write(f"{key:15}: {value}\n")

        handle.write("\n# Parsed arguments\n")
        json.dump(serializable_args, handle, indent=2)
        handle.write("\n")

        if git_diff_content:
            handle.write(
                "\n# --- GIT DIFF -----------------------------------------------------\n"
            )
            handle.write(git_diff_content + "\n")

        if include_pip_freeze:
            pip_output = run_shell_command([sys.executable, "-m", "pip", "freeze"])
            handle.write(
                "\n# --- PIP FREEZE ---------------------------------------------------\n"
            )
            handle.write(pip_output + "\n")

    print(f"[INFO] Saved reproducibility metadata to {metadata_file}")
    return metadata_file
