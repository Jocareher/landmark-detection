# BabyLand-72 Dataset Notes

BabyLand-72 label files may optionally begin with a one-line `class_idx` header.
When present, this header is orientation metadata only. The landmark model does
not predict it.

Each landmark row follows:

```text
x y visibility
```

Missing or unannotated landmarks are represented as:

```text
nan nan 0
```

Evaluation code treats these rows as invalid landmarks. BabyLand metrics use a
validity mask built from finite coordinates and visibility, so only landmarks
with finite `x y` values and `visibility = 1` enter visibility-aware metrics.

The Label Studio regeneration script writes the test split as:

```text
output_dataset_root/
├── test/
│   ├── images/
│   ├── labels/
│   └── plots/
└── reports/
```

The `test/plots/` directory contains landmark overlay JPEGs for visual
inspection. Missing landmarks encoded as `nan nan 0` are skipped in these
overlays.
