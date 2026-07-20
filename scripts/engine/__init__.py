"""Engine modules for training, evaluation, and inference.

The public symbols are imported lazily so lightweight entry points such as
standalone inference do not pay the import cost of evaluation plotting or
training-only modules.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "AverageMeter",
    "compute_box_normalized_nme",
    "compute_multitask_loss",
    "decode_heatmaps_to_image_coords",
    "evaluate_checkpoint",
    "evaluate_natural_checkpoint",
    "export_inference_outputs",
    "NormalizerProbeMonitor",
    "run_epoch",
    "run_full_evaluation",
    "run_inference",
    "smoke_test_single_batch",
    "train_model",
]

_LAZY_IMPORTS = {
    "AverageMeter": (".metrics", "AverageMeter"),
    "compute_box_normalized_nme": (".metrics", "compute_box_normalized_nme"),
    "compute_multitask_loss": (".losses", "compute_multitask_loss"),
    "decode_heatmaps_to_image_coords": (".metrics", "decode_heatmaps_to_image_coords"),
    "evaluate_checkpoint": (".evaluate", "evaluate_checkpoint"),
    "evaluate_natural_checkpoint": (".evaluate_natural", "evaluate_natural_checkpoint"),
    "export_inference_outputs": (".inference", "export_inference_outputs"),
    "NormalizerProbeMonitor": (".normalizer_monitoring", "NormalizerProbeMonitor"),
    "run_epoch": (".train", "run_epoch"),
    "run_full_evaluation": (".full_evaluation", "run_full_evaluation"),
    "run_inference": (".inference", "run_inference"),
    "smoke_test_single_batch": (".train", "smoke_test_single_batch"),
    "train_model": (".train", "train_model"),
}


def __getattr__(name: str) -> Any:
    """Import engine symbols on first access."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
