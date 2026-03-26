from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ExperimentConfig = SimpleNamespace


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic_lmks_vis_dataset"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "default_run"
CACHE_DIR = PROJECT_ROOT / "dataset_cache"
BATCH_SIZE = 16
EVAL_BATCH_SIZE = None
NUM_WORKERS = 4
PIN_MEMORY = True
TARGET_MODE = "both"
VALIDATE_LABELS = False
USE_CACHE = True
SHOW_DATASET_PROGRESS = True


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
PRETRAINED_WEIGHTS = PROJECT_ROOT / "weights" / "HR18-300W.pth"
NUM_LANDMARKS = 72
IMAGE_SIZE = (256, 256)
HEATMAP_SIZE = (64, 64)
HEATMAP_SIGMA = 2.0
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
TRANSFER_MODE = "feature_extractor"
NUM_UNFROZEN_STAGES = 0
UNFREEZE_STEM = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
LR_MILESTONES = (30, 50)
LR_GAMMA = 0.1
LAMBDA_HEATMAP = 1.0
LAMBDA_VISIBILITY = 1.0
PATIENCE = 15
USE_AMP = True
RUN_SMOKE_TEST = False


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "auto"


# ---------------------------------------------------------------------------
# Tracking / Metadata
# ---------------------------------------------------------------------------
USE_WANDB = False
WANDB_PROJECT = "BabyLMKS"
WANDB_RUN_NAME = None
INCLUDE_GIT_DIFF = True
INCLUDE_PIP_FREEZE = True


# ---------------------------------------------------------------------------
# Evaluation / Inference
# ---------------------------------------------------------------------------
EVALUATION_DIRNAME = "evaluation"
INFERENCE_DIRNAME = "inference"
VISIBILITY_THRESHOLD = 0.5
SAVE_EVALUATION_PREDICTIONS = True
SAVE_EVALUATION_OVERLAYS = True
SAVE_INFERENCE_OVERLAYS = True
SAVE_TEST_PREDICTIONS_AFTER_TRAINING = True
SAVE_TEST_OVERLAYS_AFTER_TRAINING = True
SHOW_LANDMARK_INDICES = False
USE_LANDMARK_NAMES_IN_BOXPLOT = False


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
VISUALIZE_EVERY_N_EPOCHS = 5
NUM_VISUALIZATION_IMAGES = 5
OVERLAY_POINT_RADIUS = 10
OVERLAY_LINE_WIDTH = 4
OVERLAY_CONNECTION_COLOR = "#FFD400"



DEFAULT_CONFIG_VALUES: dict[str, Any] = {
    "dataset_root": DATASET_ROOT,
    "runs_dir": RUNS_DIR,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "cache_dir": CACHE_DIR,
    "batch_size": BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
    "target_mode": TARGET_MODE,
    "validate_labels": VALIDATE_LABELS,
    "use_cache": USE_CACHE,
    "show_dataset_progress": SHOW_DATASET_PROGRESS,
    "pretrained_weights": PRETRAINED_WEIGHTS,
    "num_landmarks": NUM_LANDMARKS,
    "image_size": IMAGE_SIZE,
    "heatmap_size": HEATMAP_SIZE,
    "heatmap_sigma": HEATMAP_SIGMA,
    "normalization_mean": NORMALIZATION_MEAN,
    "normalization_std": NORMALIZATION_STD,
    "transfer_mode": TRANSFER_MODE,
    "num_unfrozen_stages": NUM_UNFROZEN_STAGES,
    "unfreeze_stem": UNFREEZE_STEM,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "lr_milestones": LR_MILESTONES,
    "lr_gamma": LR_GAMMA,
    "lambda_heatmap": LAMBDA_HEATMAP,
    "lambda_visibility": LAMBDA_VISIBILITY,
    "patience": PATIENCE,
    "use_amp": USE_AMP,
    "run_smoke_test": RUN_SMOKE_TEST,
    "seed": SEED,
    "device": DEVICE,
    "use_wandb": USE_WANDB,
    "wandb_project": WANDB_PROJECT,
    "wandb_run_name": WANDB_RUN_NAME,
    "include_git_diff": INCLUDE_GIT_DIFF,
    "include_pip_freeze": INCLUDE_PIP_FREEZE,
    "evaluation_dirname": EVALUATION_DIRNAME,
    "inference_dirname": INFERENCE_DIRNAME,
    "visibility_threshold": VISIBILITY_THRESHOLD,
    "save_evaluation_predictions": SAVE_EVALUATION_PREDICTIONS,
    "save_evaluation_overlays": SAVE_EVALUATION_OVERLAYS,
    "save_inference_overlays": SAVE_INFERENCE_OVERLAYS,
    "save_test_predictions_after_training": SAVE_TEST_PREDICTIONS_AFTER_TRAINING,
    "save_test_overlays_after_training": SAVE_TEST_OVERLAYS_AFTER_TRAINING,
    "show_landmark_indices": SHOW_LANDMARK_INDICES,
    "use_landmark_names_in_boxplot": USE_LANDMARK_NAMES_IN_BOXPLOT,
    "visualize_every_n_epochs": VISUALIZE_EVERY_N_EPOCHS,
    "num_visualization_images": NUM_VISUALIZATION_IMAGES,
    "overlay_point_radius": OVERLAY_POINT_RADIUS,
    "overlay_line_width": OVERLAY_LINE_WIDTH,
    "overlay_connection_color": OVERLAY_CONNECTION_COLOR,
}


def build_config() -> ExperimentConfig:
    """Return a mutable config object populated with the module defaults."""
    return ExperimentConfig(**deepcopy(DEFAULT_CONFIG_VALUES))


def resolve_output_dir(config: ExperimentConfig) -> Path:
    """Resolve the final training run output path under `runs/<run_name>`."""
    run_name = (
        config.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    config.wandb_run_name = run_name
    config.output_dir = config.runs_dir / run_name
    return config.output_dir


def resolve_evaluation_output_dir(
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> Path:
    """Resolve the default evaluation output directory for a checkpoint."""
    config.output_dir = checkpoint_path.parent / (
        f"{checkpoint_path.stem}_{config.evaluation_dirname}"
    )
    return config.output_dir


def resolve_inference_output_dir(
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> Path:
    """Resolve the default inference output directory for a checkpoint."""
    config.output_dir = checkpoint_path.parent / (
        f"{checkpoint_path.stem}_{config.inference_dirname}"
    )
    return config.output_dir


def config_to_serializable_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert the config object into JSON-serializable values."""
    return {key: _serialize_value(value) for key, value in vars(config).items()}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value
