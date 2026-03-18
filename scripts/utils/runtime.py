from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed_value: int = 42) -> None:
    """Seed all relevant random generators and enable deterministic execution."""
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker using the worker-specific PyTorch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_default_device(preferred: str = "auto") -> torch.device:
    """Return the requested device or the best available fallback."""
    if preferred == "cuda":
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    if (
        preferred == "mps"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
