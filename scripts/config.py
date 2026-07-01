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
ENABLE_PHOTOMETRIC_AUGMENTATIONS = False
ENABLE_GEOMETRIC_AUGMENTATIONS = False
COLOR_JITTER_BRIGHTNESS = 0.15
COLOR_JITTER_CONTRAST = 0.15
COLOR_JITTER_SATURATION = 0.10
COLOR_JITTER_PROBABILITY = 0.5
BLUR_PROBABILITY = 0.50
BLUR_RADIUS_MIN = 0.1
BLUR_RADIUS_MAX = 1.2
NOISE_PROBABILITY = 0.50
NOISE_STD = 0.02
JPEG_PROBABILITY = 0.15
JPEG_QUALITY_MIN = 60
JPEG_QUALITY_MAX = 95
RGB_SHIFT_PROBABILITY = 0.15
RGB_SHIFT_LIMIT = 0.04
GEOMETRIC_PROBABILITY = 0.50
GEOMETRIC_MAX_TRANSLATION = 0.05
GEOMETRIC_SCALE_MIN = 0.95
GEOMETRIC_SCALE_MAX = 1.05
GEOMETRIC_MAX_ROTATION_DEG = 8.0


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
LAMBDA_VIS = 1.0
LAMBDA_LMK_VIS = 1.0
LAMBDA_LMK_FULL = 1.0
LANDMARK_LOSS = "mse"
COORDINATE_DECODER = "argmax_subpixel"
ADAPTIVE_WING_OMEGA = 14.0
ADAPTIVE_WING_THETA = 0.5
ADAPTIVE_WING_EPSILON = 1.0
ADAPTIVE_WING_ALPHA = 2.1
WASSERSTEIN_SOFTMAX_TEMPERATURE = 1.0
WASSERSTEIN_EPSILON = 1e-8
WASSERSTEIN_VALIDATE_NORMALIZATION = False
PCA_PRIOR_PATH = None
LAMBDA_PCA_PROJECTION = 0.0
APPLY_PCA_INFERENCE = False
PCA_INFERENCE_NUM_COMPONENTS = None
PCA_INFERENCE_ALPHA = 1.0
PATIENCE = 15
USE_AMP = True
RUN_SMOKE_TEST = False
SAVE_PREVIEW_BATCHES = True
PREVIEW_SEED = 12345


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
SAVE_NATURAL_CROP_OVERLAYS = False
SAVE_TEST_PREDICTIONS_AFTER_TRAINING = True
SAVE_TEST_OVERLAYS_AFTER_TRAINING = True
SHOW_LANDMARK_INDICES = False
USE_LANDMARK_NAMES_IN_BOXPLOT = False
EVALUATE_SYNBABY = True
EVALUATE_BABYLAND = True
EVALUATE_INFANFACE = True
BABYLAND_CROP_ROOT = None
BABYLAND_GT_ROOT = None
BABYLAND_SOURCE_ROOT = None
INFANFACE_CROP_ROOT = None
INFANFACE_GT_ROOT = None
INFANFACE_SOURCE_ROOT = None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
VISUALIZE_EVERY_N_EPOCHS = 5
NUM_VISUALIZATION_IMAGES = 8
OVERLAY_POINT_RADIUS = 10
OVERLAY_LINE_WIDTH = 10
OVERLAY_CONNECTION_COLOR = "#A9A9A9"


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
    "enable_photometric_augmentations": ENABLE_PHOTOMETRIC_AUGMENTATIONS,
    "enable_geometric_augmentations": ENABLE_GEOMETRIC_AUGMENTATIONS,
    "color_jitter_brightness": COLOR_JITTER_BRIGHTNESS,
    "color_jitter_contrast": COLOR_JITTER_CONTRAST,
    "color_jitter_saturation": COLOR_JITTER_SATURATION,
    "color_jitter_probability": COLOR_JITTER_PROBABILITY,
    "blur_probability": BLUR_PROBABILITY,
    "blur_radius_min": BLUR_RADIUS_MIN,
    "blur_radius_max": BLUR_RADIUS_MAX,
    "noise_probability": NOISE_PROBABILITY,
    "noise_std": NOISE_STD,
    "jpeg_probability": JPEG_PROBABILITY,
    "jpeg_quality_min": JPEG_QUALITY_MIN,
    "jpeg_quality_max": JPEG_QUALITY_MAX,
    "rgb_shift_probability": RGB_SHIFT_PROBABILITY,
    "rgb_shift_limit": RGB_SHIFT_LIMIT,
    "geometric_probability": GEOMETRIC_PROBABILITY,
    "geometric_max_translation": GEOMETRIC_MAX_TRANSLATION,
    "geometric_scale_min": GEOMETRIC_SCALE_MIN,
    "geometric_scale_max": GEOMETRIC_SCALE_MAX,
    "geometric_max_rotation_deg": GEOMETRIC_MAX_ROTATION_DEG,
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
    "lambda_vis": LAMBDA_VIS,
    "lambda_lmk_vis": LAMBDA_LMK_VIS,
    "lambda_lmk_full": LAMBDA_LMK_FULL,
    "landmark_loss": LANDMARK_LOSS,
    "coordinate_decoder": COORDINATE_DECODER,
    "adaptive_wing_omega": ADAPTIVE_WING_OMEGA,
    "adaptive_wing_theta": ADAPTIVE_WING_THETA,
    "adaptive_wing_epsilon": ADAPTIVE_WING_EPSILON,
    "adaptive_wing_alpha": ADAPTIVE_WING_ALPHA,
    "wasserstein_softmax_temperature": WASSERSTEIN_SOFTMAX_TEMPERATURE,
    "wasserstein_epsilon": WASSERSTEIN_EPSILON,
    "wasserstein_validate_normalization": WASSERSTEIN_VALIDATE_NORMALIZATION,
    "pca_prior_path": PCA_PRIOR_PATH,
    "lambda_pca_projection": LAMBDA_PCA_PROJECTION,
    "apply_pca_inference": APPLY_PCA_INFERENCE,
    "pca_inference_num_components": PCA_INFERENCE_NUM_COMPONENTS,
    "pca_inference_alpha": PCA_INFERENCE_ALPHA,
    "patience": PATIENCE,
    "use_amp": USE_AMP,
    "run_smoke_test": RUN_SMOKE_TEST,
    "save_preview_batches": SAVE_PREVIEW_BATCHES,
    "preview_seed": PREVIEW_SEED,
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
    "save_natural_crop_overlays": SAVE_NATURAL_CROP_OVERLAYS,
    "save_test_predictions_after_training": SAVE_TEST_PREDICTIONS_AFTER_TRAINING,
    "save_test_overlays_after_training": SAVE_TEST_OVERLAYS_AFTER_TRAINING,
    "show_landmark_indices": SHOW_LANDMARK_INDICES,
    "use_landmark_names_in_boxplot": USE_LANDMARK_NAMES_IN_BOXPLOT,
    "evaluate_synbaby": EVALUATE_SYNBABY,
    "evaluate_babyland": EVALUATE_BABYLAND,
    "evaluate_infanface": EVALUATE_INFANFACE,
    "babyland_crop_root": BABYLAND_CROP_ROOT,
    "babyland_gt_root": BABYLAND_GT_ROOT,
    "babyland_source_root": BABYLAND_SOURCE_ROOT,
    "infanface_crop_root": INFANFACE_CROP_ROOT,
    "infanface_gt_root": INFANFACE_GT_ROOT,
    "infanface_source_root": INFANFACE_SOURCE_ROOT,
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
