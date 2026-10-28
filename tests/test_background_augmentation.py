from __future__ import annotations

import random

import numpy as np
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from scripts.config import build_config
from scripts.dataset.builders import build_transforms
from scripts.dataset.transforms import CaptureWhiteBackgroundReference, RandomWhiteBackground


def _sample() -> dict:
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    image[25:71, 25:71] = (185, 135, 105)  # face-like foreground
    image[47:51, 47:51] = (255, 255, 255)  # bright facial highlight
    image[2:12, 2:12] = (20, 70, 100)  # non-white object outside face
    landmarks = np.array([[35, 35], [48, 35], [60, 35], [35, 50],
                          [48, 50], [60, 50], [35, 62], [48, 62], [60, 62]],
                         dtype=np.float32)
    return {'image': Image.fromarray(image), 'landmarks': landmarks,
            'visibility': np.ones(len(landmarks), dtype=np.float32),
            'metadata': {'sample_id': 'synthetic_example'}}


def test_background_augmentation_preserves_face_box_and_nonwhite_pixels() -> None:
    random.seed(3)
    np.random.seed(3)
    sample = _sample()
    original = np.asarray(sample['image']).copy()
    landmarks = sample['landmarks'].copy()
    transform = RandomWhiteBackground(probability=1)
    result = transform(sample)
    modified = np.asarray(result['image'])
    assert np.array_equal(modified[25:71, 25:71], original[25:71, 25:71])
    assert np.array_equal(modified[2:12, 2:12], original[2:12, 2:12])
    assert not np.array_equal(modified[85:95, 85:95], original[85:95, 85:95])
    assert np.array_equal(result['landmarks'], landmarks)


def test_background_augmentation_fails_closed_without_reliable_face_points() -> None:
    sample = _sample()
    sample['visibility'][:] = 0
    original = np.asarray(sample['image']).copy()
    result = RandomWhiteBackground(probability=1)(sample)
    assert np.array_equal(np.asarray(result['image']), original)
    assert result['metadata']['background_augmentation']['applied'] is False


def test_background_mask_is_captured_before_other_photometric_changes() -> None:
    random.seed(4)
    np.random.seed(4)
    sample = CaptureWhiteBackgroundReference()(_sample())
    # Stand-in for preceding brightness, blur, noise, or JPEG transforms:
    # the originally white background is now gray.
    dimmed = (np.asarray(sample['image'], dtype=np.float32) * .7).astype(np.uint8)
    sample['image'] = Image.fromarray(dimmed)
    result = RandomWhiteBackground(probability=1)(sample)
    modified = np.asarray(result['image'])
    assert np.array_equal(modified[25:71, 25:71], dimmed[25:71, 25:71])
    assert not np.array_equal(modified[85:95, 85:95], dimmed[85:95, 85:95])
    assert '_white_background_reference' not in result


def test_background_augmentation_is_train_only_and_opt_in() -> None:
    config = build_config()
    config.background_augmentation_enabled = True
    train, evaluation = build_transforms(config)
    assert any(isinstance(t, RandomWhiteBackground) for t in train.transforms)
    assert not any(isinstance(t, RandomWhiteBackground) for t in evaluation.transforms)
    config.background_augmentation_enabled = False
    train, _ = build_transforms(config)
    assert not any(isinstance(t, RandomWhiteBackground) for t in train.transforms)


def test_skipped_and_augmented_samples_have_matching_metadata_keys() -> None:
    skipped = RandomWhiteBackground(probability=0)(_sample())
    augmented = RandomWhiteBackground(probability=1)(_sample())
    assert skipped['metadata'].keys() == augmented['metadata'].keys()
    assert (skipped['metadata']['background_augmentation'].keys()
            == augmented['metadata']['background_augmentation'].keys())
    default_collate([skipped['metadata'], augmented['metadata']])
