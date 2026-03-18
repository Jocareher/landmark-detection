from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    dataset_root: Path = PROJECT_ROOT / "data" / "synthetic_lmks_vis_dataset"
    runs_dir: Path = PROJECT_ROOT / "runs"
    output_dir: Path = PROJECT_ROOT / "runs" / "default_run"
    cache_dir: Path | None = PROJECT_ROOT / "dataset_cache"
    pretrained_weights: Path | None = PROJECT_ROOT / "weights" / "HR18-300W.pth"
    batch_size: int = 16
    eval_batch_size: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_milestones: tuple[int, ...] = (30, 50)
    lr_gamma: float = 0.1
    seed: int = 42
    device: str = "auto"
    num_landmarks: int = 72
    image_size: tuple[int, int] = (256, 256)
    heatmap_size: tuple[int, int] = (64, 64)
    heatmap_sigma: float = 2.0
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    target_mode: str = "both"
    validate_labels: bool = False
    use_cache: bool = True
    show_dataset_progress: bool = True
    lambda_heatmap: float = 1.0
    lambda_visibility: float = 1.0
    include_git_diff: bool = True
    include_pip_freeze: bool = True
    patience: int = 15
    use_amp: bool = True
    use_wandb: bool = False
    wandb_project: str | None = "BabyLMKS"
    wandb_run_name: str | None = None
    transfer_mode: str = "feature_extractor"
    num_unfrozen_stages: int = 0
    unfreeze_stem: bool = False
    visualize_every_n_epochs: int = 5
    num_visualization_images: int = 4
    run_smoke_test: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def resolve_output_dir(self) -> Path:
        run_name = (
            self.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.wandb_run_name = run_name
        self.output_dir = self.runs_dir / run_name
        return self.output_dir
