"""Resolve detector metadata after moving datasets between machines."""
from pathlib import Path


def resolve_source_image(raw_path, *, source_root, export_root, metadata_path):
    if not raw_path:
        raise KeyError(f"Missing 'source_image_path' in {metadata_path}.")
    source = Path(str(raw_path))
    if source_root is not None:
        root = Path(source_root)
        # Try preserved trailing directories, including a flat image directory.
        # An explicit root takes precedence over paths on the exporting machine.
        parts = source.parts[1:] if source.is_absolute() else source.parts
        candidates = [root.joinpath(*parts[index:]) for index in range(len(parts))]
        matches = {path.resolve() for path in candidates if path.is_file()}
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous source image '{raw_path}' under {root}: "
                f"{sorted(map(str, matches))}. Preserve an unambiguous directory layout."
            )
        if matches:
            return matches.pop()
    else:
        candidates = ([source] if source.is_absolute() else
                      [Path(export_root) / source, Path(metadata_path).parent / source])
        for path in candidates:
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError(
        f"Could not resolve source image '{raw_path}' from {metadata_path} "
        f"with source_root={source_root}. Copy the original images and check their layout."
    )
