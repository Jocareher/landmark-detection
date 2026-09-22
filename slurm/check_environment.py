"""Verify the runtime imports before reusing an existing HPC environment."""
import sys

assert sys.version_info >= (3, 11), 'Python >=3.11 is required'
import matplotlib
import numpy
import pandas
import PIL
import torch
import torchvision
import torchinfo
import tqdm
import wandb
import yaml

assert torch.version.cuda is not None, 'A CUDA build of PyTorch is required'
# Detect ABI mismatches as well as missing packages; GPU checks run in run_job.py.
assert torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)).numpy().shape == (1,)
print('[ENV] Required imports and Torch/NumPy bridge verified.')
