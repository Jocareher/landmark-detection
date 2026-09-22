from pathlib import Path

import pytest

from scripts.utils.source_images import resolve_source_image


def resolve(raw, root, tmp):
    return resolve_source_image(raw, source_root=root, export_root=tmp,
                                metadata_path=tmp / 'metadata/crop.json')


def test_relocated_nested_and_flat_images(tmp_path):
    root = tmp_path / 'originals'
    image = root / 'subject/image.jpg'
    image.parent.mkdir(parents=True)
    image.touch()
    assert resolve('/old/mac/dataset/subject/image.jpg', root, tmp_path) == image
    assert resolve('subject/image.jpg', root, tmp_path) == image
    flat = root / 'other.jpg'
    flat.touch()
    assert resolve('/old/mac/other.jpg', root, tmp_path) == flat


def test_ambiguous_suffixes_fail(tmp_path):
    (tmp_path / 'subject').mkdir()
    (tmp_path / 'subject/image.jpg').touch()
    (tmp_path / 'image.jpg').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        resolve('/old/subject/image.jpg', tmp_path, tmp_path)


def test_explicit_root_overrides_old_location_and_never_falls_back(tmp_path):
    old = tmp_path / 'old/image.jpg'
    old.parent.mkdir()
    old.touch()
    root = tmp_path / 'new'
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve(old, root, tmp_path)
    new = root / 'image.jpg'
    new.touch()
    assert resolve(old, root, tmp_path) == new
    assert resolve(old, None, tmp_path) == old
