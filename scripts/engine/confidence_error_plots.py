"""Plotting utilities for the standard confidence-error analysis pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GROUPED_REGIONS = ["contour", "eyebrows", "eyes", "nose", "mouth"]
POSE_ORDER = ["frontal", "quarter_left", "quarter_right", "left", "right"]
LANDMARK_SIGNALS = [
    "heatmap_max",
    "heatmap_entropy",
    "heatmap_variance",
    "peak_sharpness",
    "tta_variance",
]
STRONG_RETENTION_SIGNALS = [
    "heatmap_variance",
    "tta_variance",
    "heatmap_max",
    "heatmap_entropy",
]
UNCERTAINTY_SIGNALS = {
    "heatmap_entropy",
    "heatmap_variance",
    "tta_variance",
    "pca_reconstruction_error",
}
SIGNAL_LABELS = {
    "heatmap_max": "Heatmap max",
    "heatmap_entropy": "Heatmap entropy",
    "heatmap_variance": "Heatmap variance",
    "peak_sharpness": "Peak sharpness",
    "tta_variance": "TTA variance",
    "pca_reconstruction_error": "PCA reconstruction error",
}

FIGURE_DESCRIPTIONS = {
    "fig01_nme_by_pose": "Mean official image NME by pose, with image counts.",
    "fig02_region_nme_and_validity": "Grouped-region NME beside valid/invalid GT landmark counts.",
    "fig03_spearman_pearson_correlation_ranking": "Global Spearman and Pearson confidence-error correlations.",
    "fig04_region_spearman_correlation_heatmap": "Grouped-region Spearman correlations by landmark confidence signal.",
    "fig05_failure_detection_auroc_ranking": "AUROC ranking for detecting landmark failures at NME > 5%.",
    "fig06_retention_curves_strong_signals": "Mean retained NME versus retained landmark percentage.",
    "fig07_region_pseudo_label_viability_matrix": "Region-level pseudo-label viability matrix.",
    "fig08_top25_retained_nme_by_region": "Top-25 retained NME by grouped region.",
    "fig09_pose_pseudo_label_risk": "Pose-stratified pseudo-label risk under strict filtering.",
    "fig10_heatmap_variance_vs_error_density": "Density plot of heatmap variance versus NME.",
    "fig11_tta_variance_vs_error_density": "Density plot of TTA variance versus NME.",
    "fig12_heatmap_max_vs_error_density": "Density plot of heatmap maximum versus NME.",
    "fig13_pca_error_image_level_analysis": "Image-level PCA reconstruction error diagnostic.",
    "fig14_uda_strategy_recommendation": "Compact UDA strategy recommendation matrix.",
}

COLORS = {
    "background": (255, 255, 255),
    "axis": (50, 55, 60),
    "grid": (222, 225, 228),
    "text": (36, 42, 48),
    "muted": (110, 116, 122),
    "blue": (76, 120, 168),
    "orange": (245, 133, 24),
    "green": (84, 162, 75),
    "red": (228, 87, 86),
    "purple": (178, 121, 162),
    "teal": (114, 183, 178),
    "yellow": (236, 164, 0),
    "light_gray": (214, 214, 214),
    "dark": (47, 59, 69),
    "pale_green": (220, 236, 203),
    "pale_yellow": (254, 232, 168),
    "pale_red": (244, 199, 195),
    "table": (247, 247, 247),
}


def save_confidence_error_figure_set(
    *,
    per_landmark_rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    summary_by_region: list[dict[str, Any]],
    pose_summary_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    retention_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    viability_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    """Save the full standard confidence-error figure set as PNG and PDF."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    evaluable = [
        row
        for row in per_landmark_rows
        if bool(row.get("evaluable_for_error"))
        and _is_finite_number(row.get("normalized_error"))
    ]
    outputs: list[Path] = []
    outputs += _plot_nme_by_pose(per_image_rows, pose_summary_rows, figures_dir)
    outputs += _plot_region_nme_and_validity(summary_by_region, figures_dir)
    outputs += _plot_correlation_ranking(correlations, figures_dir)
    outputs += _plot_region_spearman_heatmap(correlations, figures_dir)
    outputs += _plot_failure_auroc(failure_rows, figures_dir)
    outputs += _plot_retention_curves(retention_rows, figures_dir)
    outputs += _plot_viability_matrix(viability_rows, figures_dir)
    outputs += _plot_top25_region_nme(viability_rows, figures_dir)
    outputs += _plot_pose_pseudo_label_risk(evaluable, figures_dir)
    outputs += _plot_density(evaluable, "heatmap_variance", "fig10_heatmap_variance_vs_error_density", "Heatmap variance", figures_dir, log_x=True)
    outputs += _plot_density(evaluable, "tta_variance", "fig11_tta_variance_vs_error_density", "TTA variance", figures_dir, log_x=True)
    outputs += _plot_density(evaluable, "heatmap_max", "fig12_heatmap_max_vs_error_density", "Heatmap max", figures_dir, log_x=False)
    outputs += _plot_pca_image_level(per_image_rows, per_landmark_rows, figures_dir)
    outputs += _plot_uda_strategy(viability_rows, figures_dir)
    return outputs


def figure_markdown_lines(plot_outputs: list[Path]) -> list[str]:
    """Return Markdown bullets describing generated figure files."""
    lines: list[str] = []
    for stem in sorted({path.stem for path in plot_outputs if path.stem.startswith("fig")}):
        description = FIGURE_DESCRIPTIONS.get(stem, "Confidence-error diagnostic figure.")
        png = f"{stem}.png"
        pdf = f"{stem}.pdf"
        lines.append(f"- `{png}` / `{pdf}`: {description}")
    return lines


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = _font(42, bold=True)
FONT_SUBTITLE = _font(30, bold=True)
FONT_BODY = _font(26)
FONT_SMALL = _font(22)
FONT_TINY = _font(18)


def _canvas(width: int = 2400, height: int = 1500) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), COLORS["background"])
    return image, ImageDraw.Draw(image)


def _save(image: Image.Image, figures_dir: Path, name: str) -> list[Path]:
    png = figures_dir / f"{name}.png"
    pdf = figures_dir / f"{name}.pdf"
    image.save(png, dpi=(350, 350))
    image.save(pdf, "PDF", resolution=350)
    return [png, pdf]


def _text_size(draw: ImageDraw.ImageDraw, text: str, used_font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=used_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    used_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = COLORS["text"],
) -> None:
    width, height = _text_size(draw, text, used_font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=used_font, fill=fill)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    used_font: ImageFont.ImageFont,
    width_px: int,
    fill: tuple[int, int, int] = COLORS["text"],
    line_spacing: int = 8,
) -> int:
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if _text_size(draw, candidate, used_font)[0] <= width_px or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, font=used_font, fill=fill)
        y += _text_size(draw, line, used_font)[1] + line_spacing
    return y


def _scale(value: float, vmin: float, vmax: float, start: float, end: float) -> float:
    if not math.isfinite(value):
        return start
    if vmax == vmin:
        return (start + end) / 2
    return start + (value - vmin) * (end - start) / (vmax - vmin)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    right: int,
    bottom: int,
    y_max: float,
    y_label: str,
    x_label: str = "",
    y_ticks: int = 5,
) -> None:
    y_max = max(float(y_max), 1e-8)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    for i in range(y_ticks + 1):
        value = y_max * i / y_ticks
        y = _scale(value, 0, y_max, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((left - 110, y - 14), f"{value:.0f}", font=FONT_SMALL, fill=COLORS["muted"])
    _draw_centered(draw, ((left + right) / 2, bottom + 90), x_label, FONT_BODY)
    draw.text((left, top - 58), y_label, font=FONT_BODY, fill=COLORS["text"])


def _bar_plot(
    title: str,
    categories: list[str],
    values: list[float],
    name: str,
    y_label: str,
    figures_dir: Path,
    annotations: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    y_max: float | None = None,
    subtitle: str | None = None,
) -> list[Path]:
    image, draw = _canvas()
    draw.text((120, 70), title, font=FONT_TITLE, fill=COLORS["text"])
    if subtitle:
        draw.text((120, 130), subtitle, font=FONT_BODY, fill=COLORS["muted"])
    left, top, right, bottom = 260, 245, 2260, 1240
    y_max = y_max or (max(values) * 1.25 if values else 1.0)
    _draw_axes(draw, left, top, right, bottom, y_max, y_label)
    count = len(categories)
    gap = 55
    slot = (right - left - gap * (count + 1)) / max(count, 1)
    colors = colors or [COLORS["blue"]] * count
    annotations = annotations or [f"{value:.1f}" for value in values]
    for i, (category, value) in enumerate(zip(categories, values)):
        x0 = left + gap + i * (slot + gap)
        x1 = x0 + slot
        y = _scale(value, 0, y_max, bottom, top)
        draw.rectangle((x0, y, x1, bottom), fill=colors[i])
        _draw_centered(draw, ((x0 + x1) / 2, bottom + 38), category.replace("_", "\n"), FONT_SMALL)
        for j, line in enumerate(annotations[i].split("\n")):
            _draw_centered(draw, ((x0 + x1) / 2, y - 54 + j * 30), line, FONT_SMALL)
    return _save(image, figures_dir, name)


def _plot_nme_by_pose(
    per_image_rows: list[dict[str, Any]],
    pose_summary_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    rows = [row for row in pose_summary_rows if row.get("pose") in POSE_ORDER]
    if not rows:
        rows = _summarize_pose_from_images(per_image_rows)
    rows = sorted(rows, key=lambda row: POSE_ORDER.index(row["pose"]) if row["pose"] in POSE_ORDER else 99)
    values = [_percent(row.get("mean_nme")) for row in rows]
    return _bar_plot(
        "Official BabyLand-72 NME by Pose",
        [str(row["pose"]) for row in rows],
        values,
        "fig01_nme_by_pose",
        "Mean image NME (%)",
        figures_dir,
        annotations=[f"{value:.1f}%\nn={int(row.get('image_count', 0))}" for value, row in zip(values, rows)],
        colors=[COLORS["red"] if str(row["pose"]) in {"left", "right"} else COLORS["blue"] for row in rows],
        y_max=max(values) * 1.3 if values else 1.0,
        subtitle="Official GT-valid landmark mask; predicted visibility is not used for error masking",
    )


def _plot_region_nme_and_validity(summary_by_region: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    image, draw = _canvas(width=2600, height=1500)
    draw.text((120, 70), "Region-Level NME and Official Evaluation Mask", font=FONT_TITLE, fill=COLORS["text"])
    draw.text((120, 130), "Green bars: valid GT landmark rows; gray segments: invalid/excluded GT rows", font=FONT_BODY, fill=COLORS["muted"])
    rows = [row for row in summary_by_region if row.get("region") in GROUPED_REGIONS]
    rows = sorted(rows, key=lambda row: GROUPED_REGIONS.index(row["region"]))
    left1, top, right1, bottom = 230, 250, 1230, 1240
    values = [_percent(row.get("mean_nme")) for row in rows]
    y_max = max(values) * 1.28 if values else 1.0
    _draw_axes(draw, left1, top, right1, bottom, y_max, "Mean NME (%)", "Grouped region")
    slot = (right1 - left1 - 70 * (len(rows) + 1)) / max(len(rows), 1)
    for i, row in enumerate(rows):
        x0 = left1 + 70 + i * (slot + 70)
        x1 = x0 + slot
        value = _percent(row.get("mean_nme"))
        y = _scale(value, 0, y_max, bottom, top)
        color = COLORS["red"] if row["region"] == "contour" else COLORS["blue"]
        draw.rectangle((x0, y, x1, bottom), fill=color)
        _draw_centered(draw, ((x0 + x1) / 2, y - 25), f"{value:.1f}%", FONT_SMALL)
        _draw_centered(draw, ((x0 + x1) / 2, bottom + 40), str(row["region"]).title(), FONT_SMALL)
    _draw_centered(draw, ((left1 + right1) / 2, top - 60), "Region-Level Error", FONT_SUBTITLE)

    left2, right2 = 1480, 2480
    totals = [
        float(row.get("valid_gt_landmark_count", 0) or 0) + float(row.get("invalid_gt_landmark_count", 0) or 0)
        for row in rows
    ]
    total_max = max(totals) * 1.15 if totals else 1.0
    _draw_axes(draw, left2, top, right2, bottom, total_max, "Landmark rows", "Grouped region")
    for i, row in enumerate(rows):
        x0 = left2 + 70 + i * (slot + 70)
        x1 = x0 + slot
        valid = float(row.get("valid_gt_landmark_count", 0) or 0)
        invalid = float(row.get("invalid_gt_landmark_count", 0) or 0)
        y_valid = _scale(valid, 0, total_max, bottom, top)
        y_total = _scale(valid + invalid, 0, total_max, bottom, top)
        draw.rectangle((x0, y_valid, x1, bottom), fill=COLORS["green"])
        draw.rectangle((x0, y_total, x1, y_valid), fill=COLORS["light_gray"])
        _draw_centered(draw, ((x0 + x1) / 2, bottom + 40), str(row["region"]).title(), FONT_SMALL)
        _draw_centered(draw, ((x0 + x1) / 2, y_total - 45), f"valid {int(valid)}\ninvalid {int(invalid)}", FONT_TINY)
    _draw_centered(draw, ((left2 + right2) / 2, top - 60), "Official Evaluation Mask", FONT_SUBTITLE)
    return _save(image, figures_dir, "fig02_region_nme_and_validity")


def _plot_correlation_ranking(correlations: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    rows = [row for row in correlations if row.get("scope") == "global" and row.get("confidence_signal") in LANDMARK_SIGNALS]
    rows = sorted(rows, key=lambda row: abs(float(row.get("spearman", 0.0) or 0.0)))
    image, draw = _canvas(width=2700, height=1450)
    draw.text((120, 70), "Spearman and Pearson Confidence-Error Correlations", font=FONT_TITLE, fill=COLORS["text"])
    draw.text((120, 130), "Spearman ranks pseudo-label candidates; Pearson reflects linear error magnitude.", font=FONT_BODY, fill=COLORS["muted"])
    _draw_hbar_panel(draw, rows, "spearman", 650, 260, 1530, 1220, "Spearman", -0.75, 0.75)
    _draw_hbar_panel(draw, rows, "pearson", 1720, 260, 2600, 1220, "Pearson", -0.75, 0.75)
    draw.rectangle((1220, 1250, 1250, 1280), fill=COLORS["blue"])
    draw.text((1265, 1248), "Higher = more confidence", font=FONT_SMALL, fill=COLORS["text"])
    draw.rectangle((1640, 1250, 1670, 1280), fill=COLORS["orange"])
    draw.text((1685, 1248), "Higher = more uncertainty", font=FONT_SMALL, fill=COLORS["text"])
    return _save(image, figures_dir, "fig03_spearman_pearson_correlation_ranking")


def _draw_hbar_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, Any]],
    metric: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    title: str,
    x_min: float,
    x_max: float,
) -> None:
    _draw_centered(draw, ((left + right) / 2, top - 55), title, FONT_SUBTITLE)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    for i in range(7):
        value = x_min + (x_max - x_min) * i / 6
        x = _scale(value, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        _draw_centered(draw, (x, bottom + 38), f"{value:.2f}", FONT_TINY, fill=COLORS["muted"])
    zero = _scale(0, x_min, x_max, left, right)
    draw.line((zero, top, zero, bottom), fill=COLORS["axis"], width=3)
    row_h = (bottom - top) / max(len(rows), 1)
    for i, row in enumerate(rows):
        signal = str(row.get("confidence_signal"))
        value = float(row.get(metric, 0.0) or 0.0)
        cy = top + row_h * (i + 0.5)
        if left < 800:
            draw.text((120, cy - 16), SIGNAL_LABELS.get(signal, signal), font=FONT_BODY, fill=COLORS["text"])
        x = _scale(value, x_min, x_max, left, right)
        color = COLORS["orange"] if signal in UNCERTAINTY_SIGNALS else COLORS["blue"]
        draw.rectangle((min(zero, x), cy - row_h * 0.25, max(zero, x), cy + row_h * 0.25), fill=color)
        draw.text((x + (12 if value >= 0 else -72), cy - 14), f"{value:.2f}", font=FONT_SMALL, fill=COLORS["text"])


def _plot_region_spearman_heatmap(correlations: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    grouped = _grouped_region_correlations(correlations)
    image, draw = _canvas(width=2500, height=1350)
    draw.text((120, 70), "Region Spearman Correlation Heatmap", font=FONT_TITLE, fill=COLORS["text"])
    draw.text((120, 130), "Positive values are expected for uncertainty signals; negative values are expected for confidence signals.", font=FONT_BODY, fill=COLORS["muted"])
    left, top = 420, 280
    cell_w, cell_h = 330, 145
    for j, signal in enumerate(LANDMARK_SIGNALS):
        _draw_centered(draw, (left + j * cell_w + cell_w / 2, top - 45), SIGNAL_LABELS[signal].replace(" ", "\n"), FONT_SMALL)
    for i, region in enumerate(GROUPED_REGIONS):
        draw.text((120, top + i * cell_h + 50), region.title(), font=FONT_BODY, fill=COLORS["text"])
        for j, signal in enumerate(LANDMARK_SIGNALS):
            value = grouped.get((region, signal))
            color = _diverging_color(value)
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 4, y0 + cell_h - 4), fill=color)
            label = "n/a" if value is None or not math.isfinite(value) else f"{value:.2f}"
            _draw_centered(draw, (x0 + cell_w / 2, y0 + cell_h / 2), label, FONT_BODY)
    draw.text((420, 1080), "Blue: negative correlation. Red: positive correlation. Spearman is the ranking metric used for pseudo-label selection.", font=FONT_SMALL, fill=COLORS["muted"])
    return _save(image, figures_dir, "fig04_region_spearman_correlation_heatmap")


def _plot_failure_auroc(failure_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    rows = [
        row
        for row in failure_rows
        if row.get("confidence_signal") in LANDMARK_SIGNALS
        and math.isclose(float(row.get("failure_threshold", -1.0)), 0.05)
    ]
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[str(row["confidence_signal"])] = row
    rows = sorted(dedup.values(), key=lambda row: float(row.get("auroc", 0.0)))
    return _horizontal_bar(
        "Failure-Detection AUROC Ranking",
        [SIGNAL_LABELS.get(str(row["confidence_signal"]), str(row["confidence_signal"])) for row in rows],
        [float(row.get("auroc", 0.0) or 0.0) for row in rows],
        "fig05_failure_detection_auroc_ranking",
        "AUROC for detecting NME > 5%",
        figures_dir,
        x_min=0.45,
        x_max=0.90,
        reference=0.5,
        reference_label="Random",
    )


def _horizontal_bar(
    title: str,
    labels: list[str],
    values: list[float],
    name: str,
    x_label: str,
    figures_dir: Path,
    x_min: float | None = None,
    x_max: float | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
) -> list[Path]:
    image, draw = _canvas(width=2400, height=1400)
    draw.text((120, 70), title, font=FONT_TITLE, fill=COLORS["text"])
    left, top, right, bottom = 720, 220, 2220, 1190
    x_min = min(values + [0.0]) if x_min is None else x_min
    x_max = max(values + [1.0]) if x_max is None else x_max
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    for i in range(6):
        value = x_min + (x_max - x_min) * i / 5
        x = _scale(value, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        _draw_centered(draw, (x, bottom + 40), f"{value:.2f}", FONT_SMALL, fill=COLORS["muted"])
    if reference is not None:
        x = _scale(reference, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["red"], width=4)
        if reference_label:
            draw.text((x + 10, top + 10), reference_label, font=FONT_SMALL, fill=COLORS["red"])
    row_h = (bottom - top) / max(len(labels), 1)
    for i, (label, value) in enumerate(zip(labels, values)):
        cy = top + row_h * (i + 0.5)
        x0 = _scale(0, x_min, x_max, left, right)
        x1 = _scale(value, x_min, x_max, left, right)
        draw.rectangle((min(x0, x1), cy - row_h * 0.28, max(x0, x1), cy + row_h * 0.28), fill=COLORS["teal"])
        draw.text((120, cy - 16), label, font=FONT_BODY, fill=COLORS["text"])
        draw.text((x1 + 12, cy - 15), f"{value:.2f}", font=FONT_SMALL, fill=COLORS["text"])
    _draw_centered(draw, ((left + right) / 2, bottom + 95), x_label, FONT_BODY)
    return _save(image, figures_dir, name)


def _plot_retention_curves(retention_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    image, draw = _canvas(width=2400, height=1450)
    draw.text((120, 70), "Retention Curves for Strong Confidence Signals", font=FONT_TITLE, fill=COLORS["text"])
    left, top, right, bottom = 260, 230, 2220, 1180
    rows = [row for row in retention_rows if row.get("region") == "global" and row.get("confidence_signal") in STRONG_RETENTION_SIGNALS]
    y_values = [_percent(row.get("mean_nme")) for row in rows]
    y_max = max(y_values) * 1.18 if y_values else 1.0
    _draw_axes(draw, left, top, right, bottom, y_max, "Mean NME (%)", "Retained landmarks (%)")
    palette = [COLORS["green"], COLORS["purple"], COLORS["blue"], COLORS["orange"]]
    for signal, color in zip(STRONG_RETENTION_SIGNALS, palette):
        signal_rows = sorted([row for row in rows if row.get("confidence_signal") == signal], key=lambda row: float(row.get("retained_fraction", 0.0)))
        points = [
            (
                _scale(float(row.get("retained_fraction", 0.0)) * 100, 0, 100, left, right),
                _scale(_percent(row.get("mean_nme")), 0, y_max, bottom, top),
            )
            for row in signal_rows
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=6)
        for point in points:
            draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=color)
    for i, (signal, color) in enumerate(zip(STRONG_RETENTION_SIGNALS, palette)):
        draw.line((1490, 250 + i * 45, 1545, 250 + i * 45), fill=color, width=7)
        draw.text((1560, 235 + i * 45), SIGNAL_LABELS[signal], font=FONT_SMALL, fill=COLORS["text"])
    return _save(image, figures_dir, "fig06_retention_curves_strong_signals")


def _plot_viability_matrix(viability_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    rows = _ordered_viability_rows(viability_rows)
    cell_rows = [
        [
            str(row.get("region", "")).title(),
            SIGNAL_LABELS.get(str(row.get("best_signal_at_25pct", "")), str(row.get("best_signal_at_25pct", ""))),
            f"{_percent(row.get('mean_nme')):.1f}%",
            f"{_percent(row.get('failure_rate')):.1f}%",
            _recommendation_label(str(row.get("recommendation", ""))),
        ]
        for row in rows
    ]
    row_colors = [_recommendation_color(str(row.get("recommendation", ""))) for row in rows]
    return _table_figure(
        "Region-Wise Pseudo-Label Viability Matrix",
        ["Region", "Best signal", "Top-25 NME", "Top-25 failure", "Recommendation"],
        cell_rows,
        "fig07_region_pseudo_label_viability_matrix",
        row_colors,
        figures_dir,
        width=2600,
        height=1350,
    )


def _plot_top25_region_nme(viability_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    rows = _ordered_viability_rows(viability_rows)
    values = [_percent(row.get("mean_nme")) for row in rows]
    return _bar_plot(
        "Top-25% Retained NME by Region",
        [str(row.get("region", "")) for row in rows],
        values,
        "fig08_top25_retained_nme_by_region",
        "Mean NME in best top-25% (%)",
        figures_dir,
        annotations=[
            f"{value:.1f}%\n{SIGNAL_LABELS.get(str(row.get('best_signal_at_25pct')), str(row.get('best_signal_at_25pct')))}\nfail {_percent(row.get('failure_rate')):.0f}%\nn={int(row.get('retained_landmarks', 0) or 0)}"
            for value, row in zip(values, rows)
        ],
        colors=[_recommendation_color(str(row.get("recommendation", ""))) for row in rows],
        y_max=max(values) * 1.38 if values else 1.0,
    )


def _plot_pose_pseudo_label_risk(evaluable_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    pose_rows = []
    for pose in POSE_ORDER:
        rows = [row for row in evaluable_rows if row.get("pose") == pose and _is_finite_number(row.get("heatmap_variance"))]
        if not rows:
            continue
        ranked = sorted(rows, key=lambda row: float(row["heatmap_variance"]))
        top = ranked[: max(1, int(math.ceil(len(ranked) * 0.25)))]
        pose_rows.append(
            {
                "pose": pose,
                "top25_mean": _mean(row["normalized_error"] for row in top),
                "top25_failure": _mean(float(row["normalized_error"]) > 0.05 for row in top),
                "n": len(top),
            }
        )
    values = [_percent(row["top25_mean"]) for row in pose_rows]
    return _bar_plot(
        "Pose-Stratified Pseudo-Label Risk",
        [row["pose"] for row in pose_rows],
        values,
        "fig09_pose_pseudo_label_risk",
        "Top-25 mean NME (%)",
        figures_dir,
        annotations=[f"{value:.1f}%\nfail {_percent(row['top25_failure']):.0f}%\nn={row['n']}" for value, row in zip(values, pose_rows)],
        colors=[COLORS["red"] if row["pose"] in {"left", "right"} else COLORS["green"] for row in pose_rows],
        y_max=max(values) * 1.35 if values else 1.0,
        subtitle="Top-25 landmarks selected by lowest heatmap variance within each pose",
    )


def _plot_density(
    rows: list[dict[str, Any]],
    signal: str,
    name: str,
    label: str,
    figures_dir: Path,
    log_x: bool,
) -> list[Path]:
    pairs = [
        (float(row[signal]), float(row["normalized_error"]) * 100)
        for row in rows
        if _is_finite_number(row.get(signal)) and _is_finite_number(row.get("normalized_error"))
    ]
    image, draw = _canvas(width=2200, height=1450)
    draw.text((120, 70), f"{label} vs Landmark Error", font=FONT_TITLE, fill=COLORS["text"])
    if not pairs:
        _draw_centered(draw, (1100, 720), f"{label} is unavailable in this run.", FONT_SUBTITLE)
        return _save(image, figures_dir, name)
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if log_x:
        positive = x[x > 0]
        min_pos = float(np.min(positive)) if positive.size else 1e-8
        x = np.log10(np.clip(x, min_pos, None))
        x_label = f"log10({label})"
    else:
        x_label = label
    left, top, right, bottom = 260, 220, 1980, 1180
    x_min, x_max = np.percentile(x, [0.5, 99.5])
    y_min, y_max = 0.0, float(np.percentile(y, 99.0))
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[70, 50], range=[[x_min, x_max], [y_min, y_max]])
    max_log = math.log10(float(hist.max()) + 1.0)
    for ix in range(hist.shape[0]):
        for iy in range(hist.shape[1]):
            count = hist[ix, iy]
            if count <= 0:
                continue
            intensity = math.log10(float(count) + 1.0) / max(max_log, 1e-8)
            color = (int(245 - 170 * intensity), int(248 - 95 * intensity), int(250 - 95 * intensity))
            px0 = _scale(float(x_edges[ix]), x_min, x_max, left, right)
            px1 = _scale(float(x_edges[ix + 1]), x_min, x_max, left, right)
            py0 = _scale(float(y_edges[iy]), y_min, y_max, bottom, top)
            py1 = _scale(float(y_edges[iy + 1]), y_min, y_max, bottom, top)
            draw.rectangle((px0, py1, px1 + 1, py0 + 1), fill=color)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    for i in range(6):
        xv = float(x_min + (x_max - x_min) * i / 5)
        px = _scale(xv, x_min, x_max, left, right)
        draw.line((px, top, px, bottom), fill=COLORS["grid"], width=1)
        _draw_centered(draw, (px, bottom + 42), f"{xv:.1f}", FONT_TINY, fill=COLORS["muted"])
        yv = y_min + (y_max - y_min) * i / 5
        py = _scale(float(yv), y_min, y_max, bottom, top)
        draw.line((left, py, right, py), fill=COLORS["grid"], width=1)
        draw.text((left - 90, py - 14), f"{yv:.0f}", font=FONT_TINY, fill=COLORS["muted"])
    trend = _binned_median_points(x, y, x_min, x_max, y_min, y_max, left, right, bottom, top)
    if len(trend) > 1:
        draw.line(trend, fill=(255, 255, 255), width=10)
        draw.line(trend, fill=COLORS["red"], width=5)
    _draw_centered(draw, ((left + right) / 2, bottom + 95), x_label, FONT_BODY)
    draw.text((left, top - 58), "NME (%)", font=FONT_BODY, fill=COLORS["text"])
    draw.text((right + 45, top + 10), "Darker = more points", font=FONT_SMALL, fill=COLORS["muted"])
    draw.line((right + 45, top + 65, right + 105, top + 65), fill=COLORS["red"], width=5)
    draw.text((right + 125, top + 50), "Binned median", font=FONT_SMALL, fill=COLORS["text"])
    return _save(image, figures_dir, name)


def _plot_pca_image_level(
    per_image_rows: list[dict[str, Any]],
    per_landmark_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    pca_values = [row.get("mean_pca_reconstruction_error") for row in per_image_rows]
    if not any(_is_finite_number(value) for value in pca_values):
        return _table_figure(
            "Image-Level PCA Reconstruction Error Analysis",
            ["Component", "Status", "Evidence", "Use"],
            [
                ["PCA reconstruction error", "Unavailable", "All image-level values are NaN or missing", "Do not use"],
                ["Landmark-level filtering", "Invalid", "PCA is image-level, not landmark-level", "Exclude"],
            ],
            "fig13_pca_error_image_level_analysis",
            [COLORS["pale_red"], COLORS["pale_red"]],
            figures_dir,
            width=2400,
            height=900,
        )
    image, draw = _canvas(width=2600, height=1450)
    draw.text((120, 70), "Image-Level PCA Reconstruction Error Analysis", font=FONT_TITLE, fill=COLORS["text"])
    _draw_scatter_panel(
        draw,
        per_image_rows,
        "mean_pca_reconstruction_error",
        "mean_nme",
        180,
        250,
        1180,
        1180,
        "PCA vs Mean NME",
        "PCA reconstruction error",
        "Mean NME (%)",
        y_percent=True,
    )
    failed_by_image = _failed_landmark_counts(per_landmark_rows)
    rows = [dict(row, failed_landmarks=failed_by_image.get(row.get("image_id"), 0)) for row in per_image_rows]
    _draw_scatter_panel(
        draw,
        rows,
        "mean_pca_reconstruction_error",
        "failed_landmarks",
        1420,
        250,
        2420,
        1180,
        "PCA vs Failed Landmarks",
        "PCA reconstruction error",
        "Failed landmarks per image",
        y_percent=False,
    )
    return _save(image, figures_dir, "fig13_pca_error_image_level_analysis")


def _plot_uda_strategy(viability_rows: list[dict[str, Any]], figures_dir: Path) -> list[Path]:
    rows_by_region = {str(row.get("region")): row for row in viability_rows}
    order = ["mouth", "eyes", "nose", "eyebrows", "contour"]
    table_rows = []
    row_colors = []
    for region in order:
        row = rows_by_region.get(region, {})
        recommendation = _recommendation_label(str(row.get("recommendation", "exclude_or_delay")))
        evidence = (
            f"Top-25 NME {_percent(row.get('mean_nme')):.1f}%, "
            f"failure {_percent(row.get('failure_rate')):.1f}%"
            if row
            else "No viability row available"
        )
        risk = _risk_from_recommendation(str(row.get("recommendation", "exclude_or_delay")))
        table_rows.append([region.title(), recommendation, evidence, risk])
        row_colors.append(_recommendation_color(str(row.get("recommendation", "exclude_or_delay"))))
    return _table_figure(
        "Recommended UDA Strategy from Confidence-Error Diagnostics",
        ["Region", "Recommendation", "Evidence", "Risk"],
        table_rows,
        "fig14_uda_strategy_recommendation",
        row_colors,
        figures_dir,
        width=2500,
        height=1300,
    )


def _table_figure(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    name: str,
    row_colors: list[tuple[int, int, int]],
    figures_dir: Path,
    width: int = 2600,
    height: int = 1350,
) -> list[Path]:
    image, draw = _canvas(width=width, height=height)
    draw.text((120, 70), title, font=FONT_TITLE, fill=COLORS["text"])
    left, top = 120, 210
    table_width = width - 240
    header_h = 95
    row_h = (height - top - 130 - header_h) / max(len(rows), 1)
    if len(columns) == 5:
        weights = [0.16, 0.22, 0.17, 0.17, 0.28]
    else:
        weights = [0.18, 0.24, 0.42, 0.16]
    col_widths = [table_width * weight / sum(weights) for weight in weights]
    x_positions = [left]
    for width_i in col_widths[:-1]:
        x_positions.append(x_positions[-1] + width_i)
    draw.rounded_rectangle((left, top, left + table_width, top + header_h), radius=12, fill=COLORS["dark"])
    for column, x, width_i in zip(columns, x_positions, col_widths):
        draw.text((x + 22, top + 31), column, font=FONT_BODY, fill=(255, 255, 255))
    for row_index, row in enumerate(rows):
        y0 = top + header_h + row_index * row_h
        draw.rectangle((left, y0, left + table_width, y0 + row_h - 2), fill=row_colors[row_index])
        for cell, x, width_i in zip(row, x_positions, col_widths):
            _draw_wrapped(draw, (int(x + 22), int(y0 + 22)), cell, FONT_SMALL, int(width_i - 44), line_spacing=6)
    return _save(image, figures_dir, name)


def _draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    title: str,
    x_label: str,
    y_label: str,
    y_percent: bool,
) -> None:
    pairs = [
        (float(row[x_key]), float(row[y_key]) * (100.0 if y_percent else 1.0))
        for row in rows
        if _is_finite_number(row.get(x_key)) and _is_finite_number(row.get(y_key))
    ]
    if not pairs:
        _draw_centered(draw, ((left + right) / 2, (top + bottom) / 2), "No finite values", FONT_BODY)
        return
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = 0.0, float(y.max() * 1.15)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    for xi, yi in pairs:
        px = _scale(xi, x_min, x_max, left, right)
        py = _scale(yi, y_min, y_max, bottom, top)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=COLORS["blue"])
    _draw_centered(draw, ((left + right) / 2, top - 45), title, FONT_SUBTITLE)
    _draw_centered(draw, ((left + right) / 2, bottom + 75), x_label, FONT_SMALL)
    draw.text((left, top - 85), y_label, font=FONT_SMALL, fill=COLORS["text"])


def _grouped_region_correlations(correlations: list[dict[str, Any]]) -> dict[tuple[str, str], float | None]:
    output: dict[tuple[str, str], float | None] = {}
    for region in GROUPED_REGIONS:
        for signal in LANDMARK_SIGNALS:
            rows = [
                row
                for row in correlations
                if row.get("scope") == region and row.get("confidence_signal") == signal
            ]
            output[(region, signal)] = float(rows[0]["spearman"]) if rows and _is_finite_number(rows[0].get("spearman")) else None
    return output


def _diverging_color(value: float | None) -> tuple[int, int, int]:
    if value is None or not math.isfinite(value):
        return (238, 238, 238)
    value = max(-0.8, min(0.8, value)) / 0.8
    if value >= 0:
        return (255, int(245 - 100 * value), int(245 - 110 * value))
    value = abs(value)
    return (int(245 - 95 * value), int(248 - 105 * value), 255)


def _binned_median_points(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: int,
    right: int,
    bottom: int,
    top: int,
) -> list[tuple[float, float]]:
    quantiles = np.quantile(x, np.linspace(0.03, 0.97, 24))
    trend = []
    for low, high in zip(quantiles[:-1], quantiles[1:]):
        mask = (x >= low) & (x < high)
        if int(mask.sum()) >= 20:
            trend.append(
                (
                    _scale(float((low + high) / 2), x_min, x_max, left, right),
                    _scale(float(np.median(y[mask])), y_min, y_max, bottom, top),
                )
            )
    return trend


def _summarize_pose_from_images(per_image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for pose in POSE_ORDER:
        rows = [row for row in per_image_rows if row.get("pose") == pose and _is_finite_number(row.get("mean_nme"))]
        if rows:
            output.append({"pose": pose, "mean_nme": _mean(row["mean_nme"] for row in rows), "image_count": len(rows)})
    return output


def _failed_landmark_counts(per_landmark_rows: list[dict[str, Any]]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for row in per_landmark_rows:
        if bool(row.get("evaluable_for_error")) and _is_finite_number(row.get("normalized_error")) and float(row["normalized_error"]) > 0.05:
            counts[row.get("image_id")] = counts.get(row.get("image_id"), 0) + 1
    return counts


def _ordered_viability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region = {str(row.get("region")): row for row in rows}
    return [by_region[region] for region in GROUPED_REGIONS if region in by_region]


def _recommendation_label(value: str) -> str:
    labels = {
        "early": "use early",
        "use_early": "use early",
        "late": "use late",
        "use_late": "use late",
        "use_with_strict_filtering": "use with strict filtering",
        "exclude_or_delay": "exclude or delay",
    }
    return labels.get(value, value.replace("_", " "))


def _recommendation_color(value: str) -> tuple[int, int, int]:
    if value in {"early", "use_early"}:
        return COLORS["pale_green"]
    if value in {"late", "use_late", "use_with_strict_filtering"}:
        return COLORS["pale_yellow"]
    return COLORS["pale_red"]


def _risk_from_recommendation(value: str) -> str:
    if value in {"early", "use_early"}:
        return "low"
    if value in {"late", "use_late", "use_with_strict_filtering"}:
        return "medium"
    return "high"


def _percent(value: Any) -> float:
    return float(value) * 100.0 if _is_finite_number(value) else float("nan")


def _mean(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if _is_finite_number(value)]
    return float(np.mean(finite)) if finite else None


def _is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
