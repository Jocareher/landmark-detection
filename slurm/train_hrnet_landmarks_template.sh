#!/bin/bash
#SBATCH -J hrnet_landmarks
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/logs/slurm_%j.out
#SBATCH --error=/path/to/logs/slurm_%j.err
#SBATCH --chdir=/path/to/landmarks_detection

# Optional module setup. Adjust to match the target cluster.
module load CUDA/12.1
module load Miniconda3/4.9.2

PROJECT_ROOT=/path/to/landmarks_detection
CONDA_ENV_NAME=landmarks
PYTHON_VERSION=3.10
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
DATASET_ROOT="${PROJECT_ROOT}/data/synthetic_lmks_vis_dataset"
RUNS_DIR="${PROJECT_ROOT}/runs_hpc"
RUN_NAME="hrnet_aug_experiment"

eval "$(conda shell.bash hook)"

if ! conda info --envs | grep -q "${CONDA_ENV_NAME}"; then
    echo "Creating conda environment '${CONDA_ENV_NAME}'..."
    conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
    conda activate "${CONDA_ENV_NAME}"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        echo "Installing repository requirements..."
        pip install -r "${REQUIREMENTS_FILE}"
    else
        echo "requirements.txt not found; install dependencies manually before use."
    fi
else
    echo "Using existing conda environment '${CONDA_ENV_NAME}'."
    conda activate "${CONDA_ENV_NAME}"
fi

cd "${PROJECT_ROOT}" || exit 1

python3 scripts/main.py \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${RUNS_DIR}" \
    --wandb-run-name "${RUN_NAME}" \
    --batch-size 16 \
    --eval-batch-size 16 \
    --epochs 60 \
    --lr 1e-4 \
    --transfer-mode fine_tuning \
    --num-unfrozen-stages 2 \
    --enable-photometric-augmentations \
    --enable-geometric-augmentations \
    --brightness-jitter 0.15 \
    --contrast-jitter 0.15 \
    --saturation-jitter 0.10 \
    --blur-probability 0.20 \
    --noise-probability 0.20 \
    --noise-std 0.02 \
    --jpeg-probability 0.15 \
    --rgb-shift-probability 0.15 \
    --geometric-probability 0.50 \
    --geometric-max-translation 0.05 \
    --geometric-scale-min 0.95 \
    --geometric-scale-max 1.05 \
    --geometric-max-rotation-deg 8.0 \
    --save-config
