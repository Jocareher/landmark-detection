"""Preview and audit the train-only white-background augmentation.

Example:
python -m scripts.analysis.preview_background_randomization \
  --config configs/normalizer_background_randomization.yaml \
  --output outputs/background_preview.png
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.config import load_yaml_config
from scripts.dataset.dataset import SyntheticLandmarkDataset
from scripts.dataset.transforms import RandomWhiteBackground, Resize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=None,
                        help='Override dataset_root from the YAML config.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')

    config = load_yaml_config(args.config)
    dataset_root = args.dataset_root or Path(config.dataset_root)
    if not dataset_root.exists():
        parser.error(f'synthetic dataset not found: {dataset_root}')
    dataset = SyntheticLandmarkDataset(
        root_dir=dataset_root, split='train', transform=None,
        target_mode='regression', num_landmarks=config.num_landmarks,
        use_cache=False, show_progress=False,
    )
    chosen = random.Random(args.seed).sample(range(len(dataset)), min(args.count, len(dataset)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    transform = RandomWhiteBackground(
        probability=1.0,
        white_threshold=config.background_white_threshold,
        face_margin_px=config.background_face_margin_px,
        texture_std=config.background_texture_std,
        fine_noise_std=config.background_fine_noise_std,
    )
    resize = Resize(size=config.image_size)
    tile_w, tile_h = config.image_size[1], config.image_size[0]
    gap, label_h = 12, 42
    grid = Image.new('RGB', (4 * tile_w + 5 * gap,
                             len(chosen) * (tile_h + label_h + gap) + gap), 'white')
    draw = ImageDraw.Draw(grid)
    rows = []
    for row_number, index in enumerate(chosen):
        info = dataset.samples[index]
        image = dataset._load_image(Path(info['image_path']))
        landmarks, visibility, _, _ = dataset._load_label(
            Path(info['label_path']), *image.size)
        sample = resize({'image': image, 'landmarks': landmarks,
                         'visibility': visibility, 'metadata': {}})
        before = np.asarray(sample['image']).copy()
        result = transform(sample)
        after = np.asarray(result['image'])
        changed = np.any(before != after, axis=2)
        details = result.get('metadata', {}).get('background_augmentation', {})
        landmark_box = details.get('landmark_box_xyxy')
        diff = np.clip(np.abs(after.astype(np.int16) - before.astype(np.int16)) * 3,
                       0, 255).astype(np.uint8)
        mask = np.zeros_like(before)
        mask[changed] = (25, 150, 150)
        preview = Image.fromarray(before)
        if landmark_box is not None:
            x0, y0, x1, y1 = landmark_box
            ImageDraw.Draw(preview).rectangle((x0, y0, x1 - 1, y1 - 1),
                                              outline='#D44A38', width=2)
        views = (preview, Image.fromarray(after), Image.fromarray(diff), Image.fromarray(mask))
        y = gap + row_number * (tile_h + label_h + gap)
        for column, view in enumerate(views):
            x = gap + column * (tile_w + gap)
            grid.paste(view, (x, y + label_h))
            label = ('original + landmark box', 'augmented', 'RGB difference ×3',
                     'changed pixels')[column]
            draw.text((x, y + 6), label, fill='#263C4B')
        rows.append({'sample_id': Path(info['image_path']).stem,
                     'changed_fraction': float(changed.mean()),
                     'eligible_fraction': details.get('eligible_fraction', 0.0),
                     'landmark_box_xyxy': str(landmark_box),
                     'applied': bool(details.get('applied', False))})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    csv_path = args.output.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Preview: {args.output}')
    print(f'Coverage: {csv_path}')
    print(f'Mean changed fraction: {np.mean([r["changed_fraction"] for r in rows]):.3f}')


if __name__ == '__main__':
    main()
