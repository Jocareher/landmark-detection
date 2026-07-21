from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .hrn import HRNetLandmarkVisibility
from .image_normalizer import ResidualImageNormalizer


class NormalizedLandmarker(nn.Module):
    """Compose an optional residual image normalizer with an existing landmarker."""

    def __init__(
        self,
        landmarker: nn.Module,
        normalizer: ResidualImageNormalizer | None = None,
    ) -> None:
        """Store the landmarker and optional image normalizer."""
        super().__init__()
        self.landmarker = landmarker
        self.normalizer = normalizer
        self._landmarker_frozen = False

    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the normalizer, or return the original tensor when disabled."""
        if self.normalizer is None:
            return images
        return self.normalizer(images)

    def forward_normalized(
        self, normalized_images: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run the existing landmarker on already-normalized images."""
        outputs = self.landmarker(normalized_images)
        if not isinstance(outputs, dict):
            raise TypeError("The wrapped landmarker must return a dictionary.")
        return outputs

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Normalize images when enabled and preserve the landmarker output format."""
        return self.forward_normalized(self.normalize_images(images))

    def train(self, mode: bool = True) -> NormalizedLandmarker:
        """Set module modes while keeping a frozen landmarker in evaluation mode."""
        super().train(mode)
        if self._landmarker_frozen:
            self.landmarker.eval()
        return self

    def freeze_landmarker(self) -> None:
        """Freeze every landmarker parameter and its training-time state."""
        for parameter in self.landmarker.parameters():
            parameter.requires_grad = False
        self._landmarker_frozen = True
        self.landmarker.eval()

    def unfreeze_landmarker(self) -> None:
        """Mark the landmarker as mode-trainable before applying a transfer policy."""
        self._landmarker_frozen = False

    def freeze_normalizer(self) -> None:
        """Freeze the normalizer when it is enabled."""
        if self.normalizer is not None:
            for parameter in self.normalizer.parameters():
                parameter.requires_grad = False
            self.normalizer.eval()

    def unfreeze_normalizer(self) -> None:
        """Unfreeze the normalizer when it is enabled."""
        if self.normalizer is None:
            raise RuntimeError("Cannot train a disabled normalizer.")
        for parameter in self.normalizer.parameters():
            parameter.requires_grad = True

    def configure_normalizer_only(self) -> None:
        """Train only the normalizer and keep the full landmarker frozen."""
        self.freeze_landmarker()
        self.unfreeze_normalizer()

    def configure_joint_finetune(
        self,
        num_unfrozen_stages: int = 1,
        unfreeze_stem: bool = False,
    ) -> None:
        """Train the normalizer with the existing partial landmarker policy."""
        self.unfreeze_landmarker()
        self.unfreeze_normalizer()
        set_transfer_learning_mode = getattr(
            self.landmarker, "set_transfer_learning_mode", None
        )
        if set_transfer_learning_mode is None:
            raise TypeError(
                "The wrapped landmarker does not expose set_transfer_learning_mode."
            )
        set_transfer_learning_mode(
            mode="fine_tuning",
            num_unfrozen_stages=num_unfrozen_stages,
            unfreeze_stem=unfreeze_stem,
        )

    def parameter_counts(self) -> dict[str, dict[str, int]]:
        """Return total, trainable, and frozen parameter counts by module."""
        return {
            "normalizer": _count_module_parameters(self.normalizer),
            "landmarker": _count_module_parameters(self.landmarker),
            "full_model": _count_module_parameters(self),
        }


def _count_module_parameters(module: nn.Module | None) -> dict[str, int]:
    """Count total, trainable, and frozen parameters in one optional module."""
    if module is None:
        return {"total": 0, "trainable": 0, "frozen": 0}
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def load_normalized_checkpoint(
    model: NormalizedLandmarker,
    checkpoint: Mapping[str, object],
    strict: bool = True,
    load_normalizer: bool = True,
) -> None:
    """Load either a legacy landmarker checkpoint or a wrapped-model checkpoint."""
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a valid model_state_dict.")
    state_keys = list(state.keys())
    if any(str(key).startswith("landmarker.") for key in state_keys):
        if load_normalizer:
            model.load_state_dict(state, strict=strict)
        else:
            landmarker_state = {
                str(key).removeprefix("landmarker."): value
                for key, value in state.items()
                if str(key).startswith("landmarker.")
            }
            model.landmarker.load_state_dict(landmarker_state, strict=strict)
        return
    model.landmarker.load_state_dict(state, strict=strict)


def build_model_from_checkpoints(
    checkpoint: Mapping[str, object],
    *,
    num_landmarks: int = 72,
    normalizer_checkpoint: Mapping[str, object] | None = None,
    fallback_normalizer_architecture: Mapping[str, object] | None = None,
    strict: bool = True,
) -> nn.Module:
    """Reconstruct a landmarker or normalized full model from saved payloads.

    A full-model checkpoint is sufficient by itself. A landmarker-only
    checkpoint may optionally be combined with a normalizer-only checkpoint.
    """
    primary_state = checkpoint.get("model_state_dict")
    if not isinstance(primary_state, Mapping):
        primary_state = checkpoint.get("landmarker_state_dict")
    if not isinstance(primary_state, Mapping):
        if isinstance(checkpoint.get("normalizer_state_dict"), Mapping):
            raise ValueError(
                "A normalizer-only checkpoint cannot run independently. Pass a "
                "full-model checkpoint, or use it alongside a landmarker checkpoint."
            )
        raise ValueError("Checkpoint does not contain model weights.")

    state_keys = [str(key) for key in primary_state]
    is_full_model = any(key.startswith("landmarker.") for key in state_keys)
    if is_full_model:
        if normalizer_checkpoint is not None:
            raise ValueError(
                "A separate normalizer checkpoint cannot be combined with an "
                "already wrapped full-model checkpoint."
            )
        architecture = _resolve_normalizer_architecture(
            checkpoint,
            fallback=fallback_normalizer_architecture,
        )
        model = NormalizedLandmarker(
            landmarker=HRNetLandmarkVisibility(num_landmarks=num_landmarks),
            normalizer=ResidualImageNormalizer(**architecture),
        )
        model.load_state_dict(primary_state, strict=strict)
        return model

    landmarker = HRNetLandmarkVisibility(num_landmarks=num_landmarks)
    landmarker.load_state_dict(primary_state, strict=strict)
    if normalizer_checkpoint is None:
        return landmarker

    normalizer_state = normalizer_checkpoint.get("normalizer_state_dict")
    if not isinstance(normalizer_state, Mapping):
        raise ValueError(
            "The separate normalizer checkpoint lacks normalizer_state_dict."
        )
    architecture = _resolve_normalizer_architecture(
        normalizer_checkpoint,
        fallback=fallback_normalizer_architecture,
    )
    model = NormalizedLandmarker(
        landmarker=landmarker,
        normalizer=ResidualImageNormalizer(**architecture),
    )
    model.normalizer.load_state_dict(normalizer_state, strict=strict)
    return model


def _resolve_normalizer_architecture(
    checkpoint: Mapping[str, object],
    *,
    fallback: Mapping[str, object] | None,
) -> dict[str, object]:
    """Resolve constructor arguments stored with a normalizer checkpoint."""
    architecture = checkpoint.get("normalizer_architecture")
    if not isinstance(architecture, Mapping):
        architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        architecture = fallback
    if not isinstance(architecture, Mapping):
        raise ValueError(
            "The checkpoint contains normalizer weights but no architecture metadata. "
            "Provide fallback normalizer configuration."
        )
    resolved = dict(architecture)
    normalizer_type = resolved.pop("type", "residual")
    if normalizer_type != "residual":
        raise ValueError(f"Unsupported normalizer type: {normalizer_type}")
    return resolved
