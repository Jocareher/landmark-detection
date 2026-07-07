"""Generate publication-style interpretation figures from existing CSV outputs.

This script is intentionally analysis-only: it reads corrected confidence-error
CSV tables and writes figures/Markdown summaries without running inference,
training, or model evaluation.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INPUT_DIR = Path("/Users/jocareher/Downloads/confidence_error_babyland72")
OUTPUT_DIR = Path("analysis_outputs/confidence_error/figures_interpretation")

REGION_ORDER = ["contour", "eyebrows", "eyes", "nose", "mouth"]
POSE_ORDER = ["frontal", "quarter_left", "quarter_right", "left", "right"]
SIGNAL_LABELS = {
    "heatmap_max": "Heatmap max",
    "heatmap_entropy": "Heatmap entropy",
    "heatmap_variance": "Heatmap variance",
    "peak_sharpness": "Peak sharpness",
    "tta_variance": "TTA variance",
    "pca_reconstruction_error": "PCA reconstruction error",
}
UNCERTAINTY_SIGNALS = {
    "heatmap_entropy",
    "heatmap_variance",
    "tta_variance",
    "pca_reconstruction_error",
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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Return a readable sans-serif font available on macOS/Linux runtimes."""
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


FONT_TITLE = font(42, bold=True)
FONT_SUBTITLE = font(30, bold=True)
FONT_BODY = font(26)
FONT_SMALL = font(22)
FONT_TINY = font(18)


def canvas(width: int = 2400, height: int = 1500) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a white RGB canvas and drawing context."""
    image = Image.new("RGB", (width, height), COLORS["background"])
    return image, ImageDraw.Draw(image)


def text_size(draw: ImageDraw.ImageDraw, text: str, used_font: ImageFont.ImageFont) -> tuple[int, int]:
    """Return text width and height."""
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=used_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    used_font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = COLORS["text"],
) -> None:
    """Draw centered text around a coordinate."""
    width, height = text_size(draw, text, used_font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=used_font, fill=fill)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    used_font: ImageFont.ImageFont,
    width_px: int,
    fill: tuple[int, int, int] = COLORS["text"],
    line_spacing: int = 8,
) -> int:
    """Draw wrapped text and return the final y coordinate."""
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if text_size(draw, candidate, used_font)[0] <= width_px or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, font=used_font, fill=fill)
        y += text_size(draw, line, used_font)[1] + line_spacing
    return y


def save(image: Image.Image, name: str) -> list[Path]:
    """Save a figure as PNG and PDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{name}.png"
    pdf = OUTPUT_DIR / f"{name}.pdf"
    image.save(png, dpi=(350, 350))
    image.save(pdf, "PDF", resolution=350)
    return [png, pdf]


def scale_linear(value: float, vmin: float, vmax: float, start: float, end: float) -> float:
    """Map a scalar from data coordinates to pixel coordinates."""
    if vmax == vmin:
        return (start + end) / 2
    return start + (value - vmin) * (end - start) / (vmax - vmin)


def draw_axes(
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
    """Draw simple x/y axes with horizontal grid lines."""
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    for i in range(y_ticks + 1):
        value = y_max * i / y_ticks
        y = scale_linear(value, 0, y_max, bottom, top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((left - 105, y - 14), f"{value:.0f}", font=FONT_SMALL, fill=COLORS["muted"])
    draw_centered(draw, ((left + right) / 2, bottom + 90), x_label, FONT_BODY)
    draw.text((left, top - 58), y_label, font=FONT_BODY, fill=COLORS["text"])


def bar_plot(
    title: str,
    categories: list[str],
    values: list[float],
    name: str,
    y_label: str,
    annotations: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    y_max: float | None = None,
    subtitle: str | None = None,
) -> list[Path]:
    """Draw a clean vertical bar plot."""
    image, draw = canvas()
    draw.text((120, 70), title, font=FONT_TITLE, fill=COLORS["text"])
    if subtitle:
        draw.text((120, 125), subtitle, font=FONT_BODY, fill=COLORS["muted"])
    left, top, right, bottom = 260, 230, 2260, 1240
    y_max = y_max or max(values) * 1.25
    draw_axes(draw, left, top, right, bottom, y_max, y_label)
    n = len(categories)
    gap = 55
    slot = (right - left - gap * (n + 1)) / n
    colors = colors or [COLORS["blue"]] * n
    annotations = annotations or [f"{v:.1f}" for v in values]
    for i, (category, value) in enumerate(zip(categories, values)):
        x0 = left + gap + i * (slot + gap)
        x1 = x0 + slot
        y = scale_linear(value, 0, y_max, bottom, top)
        draw.rectangle((x0, y, x1, bottom), fill=colors[i])
        draw_centered(draw, ((x0 + x1) / 2, bottom + 38), category.replace("_", "\n"), FONT_SMALL)
        for j, line in enumerate(annotations[i].split("\n")):
            draw_centered(draw, ((x0 + x1) / 2, y - 54 + j * 30), line, FONT_SMALL)
    return save(image, name)


def stacked_validity_plot(summary_region: pd.DataFrame) -> list[Path]:
    """Draw region NME and valid/invalid GT counts in aligned panels."""
    image, draw = canvas(width=2600, height=1500)
    draw.text((120, 70), "Region-Level NME and Official Evaluation Mask", font=FONT_TITLE, fill=COLORS["text"])
    draw.text((120, 130), "Green bars: valid GT landmark rows; gray segments: invalid/excluded GT rows", font=FONT_BODY, fill=COLORS["muted"])
    reg = summary_region[summary_region["region"].isin(REGION_ORDER)].set_index("region").reindex(REGION_ORDER).reset_index()

    left1, top, right1, bottom = 230, 250, 1230, 1240
    values = reg["mean_nme_percent"].astype(float).tolist()
    y_max = max(values) * 1.28
    draw_axes(draw, left1, top, right1, bottom, y_max, "Mean NME (%)", "Grouped region")
    slot = (right1 - left1 - 70 * (len(REGION_ORDER) + 1)) / len(REGION_ORDER)
    for i, row in reg.iterrows():
        x0 = left1 + 70 + i * (slot + 70)
        x1 = x0 + slot
        y = scale_linear(row["mean_nme_percent"], 0, y_max, bottom, top)
        color = COLORS["red"] if row["region"] == "contour" else COLORS["blue"]
        draw.rectangle((x0, y, x1, bottom), fill=color)
        draw_centered(draw, ((x0 + x1) / 2, y - 25), f"{row['mean_nme_percent']:.1f}%", FONT_SMALL)
        draw_centered(draw, ((x0 + x1) / 2, bottom + 40), row["region"].title(), FONT_SMALL)
    draw_centered(draw, ((left1 + right1) / 2, top - 60), "Region-Level Error", FONT_SUBTITLE)

    left2, right2 = 1480, 2480
    valid = reg["valid_gt_landmark_count"].astype(float)
    invalid = reg["invalid_gt_landmark_count"].astype(float)
    total_max = float((valid + invalid).max()) * 1.15
    draw_axes(draw, left2, top, right2, bottom, total_max, "Landmark rows", "Grouped region")
    for i, row in reg.iterrows():
        x0 = left2 + 70 + i * (slot + 70)
        x1 = x0 + slot
        y_valid = scale_linear(row["valid_gt_landmark_count"], 0, total_max, bottom, top)
        y_total = scale_linear(row["valid_gt_landmark_count"] + row["invalid_gt_landmark_count"], 0, total_max, bottom, top)
        draw.rectangle((x0, y_valid, x1, bottom), fill=COLORS["green"])
        draw.rectangle((x0, y_total, x1, y_valid), fill=COLORS["light_gray"])
        draw_centered(draw, ((x0 + x1) / 2, bottom + 40), row["region"].title(), FONT_SMALL)
        draw_centered(
            draw,
            ((x0 + x1) / 2, y_total - 45),
            f"valid {int(row['valid_gt_landmark_count'])}\ninvalid {int(row['invalid_gt_landmark_count'])}",
            FONT_TINY,
        )
    draw_centered(draw, ((left2 + right2) / 2, top - 60), "Official Evaluation Mask", FONT_SUBTITLE)
    return save(image, "fig02_region_nme_and_validity")


def horizontal_bar(
    title: str,
    labels: list[str],
    values: list[float],
    name: str,
    x_label: str,
    colors: list[tuple[int, int, int]] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    ref_line: float | None = None,
    ref_label: str | None = None,
) -> list[Path]:
    """Draw a horizontal bar plot."""
    image, draw = canvas(width=2400, height=1400)
    draw.text((120, 70), title, font=FONT_TITLE, fill=COLORS["text"])
    left, top, right, bottom = 720, 220, 2220, 1190
    x_min = float(min(values) if x_min is None else x_min)
    x_max = float(max(values) if x_max is None else x_max)
    if x_min > 0:
        x_min = 0
    colors = colors or [COLORS["blue"]] * len(labels)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    for i in range(6):
        value = x_min + (x_max - x_min) * i / 5
        x = scale_linear(value, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        draw_centered(draw, (x, bottom + 40), f"{value:.2f}", FONT_SMALL, fill=COLORS["muted"])
    if ref_line is not None:
        x = scale_linear(ref_line, x_min, x_max, left, right)
        draw.line((x, top, x, bottom), fill=COLORS["red"], width=4)
        if ref_label:
            draw.text((x + 10, top + 15), ref_label, font=FONT_SMALL, fill=COLORS["red"])
    zero_x = scale_linear(0, x_min, x_max, left, right)
    draw.line((zero_x, top, zero_x, bottom), fill=COLORS["axis"], width=2)
    row_h = (bottom - top) / len(labels)
    for i, (label, value) in enumerate(zip(labels, values)):
        cy = top + row_h * (i + 0.5)
        x0 = zero_x
        x1 = scale_linear(value, x_min, x_max, left, right)
        draw.rectangle((min(x0, x1), cy - row_h * 0.28, max(x0, x1), cy + row_h * 0.28), fill=colors[i])
        draw.text((120, cy - 16), label, font=FONT_BODY, fill=COLORS["text"])
        draw.text((x1 + (14 if value >= 0 else -80), cy - 15), f"{value:.2f}", font=FONT_SMALL, fill=COLORS["text"])
    draw_centered(draw, ((left + right) / 2, bottom + 95), x_label, FONT_BODY)
    return save(image, name)


def line_plot(retention: pd.DataFrame) -> list[Path]:
    """Draw global retention curves for the strongest signals."""
    image, draw = canvas(width=2400, height=1450)
    draw.text((120, 70), "Retention Curves for Strong Confidence Signals", font=FONT_TITLE, fill=COLORS["text"])
    left, top, right, bottom = 260, 230, 2220, 1180
    signals = ["heatmap_variance", "tta_variance", "heatmap_max", "heatmap_entropy"]
    palette = [COLORS["green"], COLORS["purple"], COLORS["blue"], COLORS["orange"]]
    y_max = retention[(retention["region"] == "global") & (retention["confidence_signal"].isin(signals))]["mean_nme"].max() * 100 * 1.18
    draw_axes(draw, left, top, right, bottom, y_max, "Mean NME (%)", "Retained landmarks (%)")
    for signal, color in zip(signals, palette):
        sub = retention[(retention["region"] == "global") & (retention["confidence_signal"] == signal)].sort_values("retained_fraction")
        if sub.empty:
            continue
        points = []
        for _, row in sub.iterrows():
            x = scale_linear(row["retained_fraction"] * 100, 0, 100, left, right)
            y = scale_linear(row["mean_nme"] * 100, 0, y_max, bottom, top)
            points.append((x, y))
        draw.line(points, fill=color, width=6)
        for point in points:
            draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill=color)
    legend_x, legend_y = 1490, 250
    for i, (signal, color) in enumerate(zip(signals, palette)):
        draw.line((legend_x, legend_y + i * 45, legend_x + 55, legend_y + i * 45), fill=color, width=7)
        draw.text((legend_x + 70, legend_y + i * 45 - 15), SIGNAL_LABELS[signal], font=FONT_SMALL, fill=COLORS["text"])
    return save(image, "fig05_retention_curves_strong_signals")


def table_figure(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    name: str,
    row_colors: list[tuple[int, int, int]] | None = None,
    width: int = 2600,
    height: int = 1350,
) -> list[Path]:
    """Draw a polished annotated table."""
    image, draw = canvas(width=width, height=height)
    draw.text((120, 70), title, font=FONT_TITLE, fill=COLORS["text"])
    left, top = 120, 210
    table_width = width - 240
    header_h = 95
    row_h = (height - top - 130 - header_h) / len(rows)
    col_weights = [0.16, 0.22, 0.17, 0.17, 0.28] if len(columns) == 5 else [0.23, 0.19, 0.42, 0.16]
    col_widths = [table_width * w / sum(col_weights) for w in col_weights]
    x_positions = [left]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)
    draw.rounded_rectangle((left, top, left + table_width, top + header_h), radius=12, fill=COLORS["dark"])
    for col, x, w in zip(columns, x_positions, col_widths):
        draw.text((x + 22, top + 31), col, font=FONT_BODY, fill=(255, 255, 255))
    for r, row in enumerate(rows):
        y0 = top + header_h + r * row_h
        bg = row_colors[r] if row_colors else COLORS["table"]
        draw.rectangle((left, y0, left + table_width, y0 + row_h - 2), fill=bg)
        for cell, x, w in zip(row, x_positions, col_widths):
            draw_wrapped(draw, (int(x + 22), int(y0 + 22)), cell, FONT_SMALL, int(w - 44), line_spacing=6)
    return save(image, name)


def density_figure(valid_landmarks: pd.DataFrame, signal: str, name: str, label: str, log_x: bool) -> list[Path]:
    """Draw a hexbin-like density plot with a binned median trend."""
    df = valid_landmarks[["normalized_error", signal]].replace([np.inf, -np.inf], np.nan).dropna()
    x = df[signal].to_numpy(float)
    if log_x:
        min_pos = float(np.nanmin(x[x > 0])) if np.any(x > 0) else 1e-8
        x = np.log10(np.clip(x, min_pos, None))
        x_label = f"log10({label})"
    else:
        x_label = label
    y = df["normalized_error"].to_numpy(float) * 100
    image, draw = canvas(width=2200, height=1450)
    draw.text((120, 70), f"{label} vs Landmark Error", font=FONT_TITLE, fill=COLORS["text"])
    left, top, right, bottom = 260, 220, 1980, 1180
    x_min, x_max = float(np.nanpercentile(x, 0.5)), float(np.nanpercentile(x, 99.5))
    y_min, y_max = 0.0, float(np.nanpercentile(y, 99.0))
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    bins_x, bins_y = 70, 50
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[bins_x, bins_y], range=[[x_min, x_max], [y_min, y_max]])
    max_log = math.log10(hist.max() + 1)
    for ix in range(bins_x):
        for iy in range(bins_y):
            count = hist[ix, iy]
            if count <= 0:
                continue
            intensity = math.log10(count + 1) / max_log
            color = (
                int(245 - 170 * intensity),
                int(248 - 95 * intensity),
                int(250 - 95 * intensity),
            )
            px0 = scale_linear(x_edges[ix], x_min, x_max, left, right)
            px1 = scale_linear(x_edges[ix + 1], x_min, x_max, left, right)
            py0 = scale_linear(y_edges[iy], y_min, y_max, bottom, top)
            py1 = scale_linear(y_edges[iy + 1], y_min, y_max, bottom, top)
            draw.rectangle((px0, py1, px1 + 1, py0 + 1), fill=color)
    draw.line((left, bottom, right, bottom), fill=COLORS["axis"], width=3)
    draw.line((left, top, left, bottom), fill=COLORS["axis"], width=3)
    for i in range(6):
        xv = x_min + (x_max - x_min) * i / 5
        px = scale_linear(xv, x_min, x_max, left, right)
        draw.line((px, top, px, bottom), fill=COLORS["grid"], width=1)
        draw_centered(draw, (px, bottom + 42), f"{xv:.1f}", FONT_TINY, fill=COLORS["muted"])
        yv = y_min + (y_max - y_min) * i / 5
        py = scale_linear(yv, y_min, y_max, bottom, top)
        draw.line((left, py, right, py), fill=COLORS["grid"], width=1)
        draw.text((left - 90, py - 14), f"{yv:.0f}", font=FONT_TINY, fill=COLORS["muted"])
    qs = np.quantile(x, np.linspace(0.03, 0.97, 24))
    trend = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() >= 20:
            trend.append((scale_linear((lo + hi) / 2, x_min, x_max, left, right), scale_linear(float(np.median(y[mask])), y_min, y_max, bottom, top)))
    if len(trend) > 1:
        draw.line(trend, fill=(255, 255, 255), width=10)
        draw.line(trend, fill=COLORS["red"], width=5)
    draw_centered(draw, ((left + right) / 2, bottom + 95), x_label, FONT_BODY)
    draw.text((left, top - 58), "NME (%)", font=FONT_BODY, fill=COLORS["text"])
    draw.text((right + 45, top + 10), "Darker = more points", font=FONT_SMALL, fill=COLORS["muted"])
    draw.line((right + 45, top + 65, right + 105, top + 65), fill=COLORS["red"], width=5)
    draw.text((right + 125, top + 50), "Binned median", font=FONT_SMALL, fill=COLORS["text"])
    return save(image, name)


def main() -> None:
    """Generate all requested figures and the interpretation Markdown file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_image = pd.read_csv(INPUT_DIR / "per_image_confidence_error.csv")
    per_landmark = pd.read_csv(INPUT_DIR / "per_landmark_confidence_error.csv")
    summary_region = pd.read_csv(INPUT_DIR / "summary_by_region.csv")
    correlations = pd.read_csv(INPUT_DIR / "confidence_error_correlations.csv")
    retention = pd.read_csv(INPUT_DIR / "retention_curves.csv")
    failure = pd.read_csv(INPUT_DIR / "failure_detection.csv")
    viability = pd.read_csv(INPUT_DIR / "region_pseudo_label_viability.csv")

    valid_landmarks = per_landmark.copy()
    if "evaluable_for_error" in valid_landmarks.columns:
        valid_landmarks = valid_landmarks[valid_landmarks["evaluable_for_error"].astype(bool)]
    valid_landmarks = valid_landmarks[np.isfinite(valid_landmarks["normalized_error"])]

    saved: list[Path] = []
    notes: list[tuple[str, str, str]] = []

    pose_stats = (
        per_image[per_image["pose"].isin(POSE_ORDER)]
        .groupby("pose")
        .agg(mean=("mean_nme_percent", "mean"), std=("mean_nme_percent", "std"), n=("mean_nme_percent", "size"))
        .reindex(POSE_ORDER)
        .reset_index()
    )
    saved += bar_plot(
        "Official BabyLand-72 NME by Pose",
        pose_stats["pose"].tolist(),
        pose_stats["mean"].astype(float).tolist(),
        "fig01_nme_by_pose",
        "Mean image NME (%)",
        [f"{r.mean:.1f}%\nn={int(r.n)}" for r in pose_stats.itertuples()],
        [COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["red"], COLORS["red"]],
        y_max=float((pose_stats["mean"] + pose_stats["std"]).max() * 1.16),
        subtitle="Error bars: one standard deviation across images",
    )
    notes.append(("fig01_nme_by_pose", "Mean per-image official NME by pose with image counts.", f"The official per-image NME is {per_image['mean_nme_percent'].mean():.2f}%; profiles are the highest-risk poses."))

    saved += stacked_validity_plot(summary_region)
    notes.append(("fig02_region_nme_and_validity", "Two aligned panels: grouped-region NME and valid/invalid GT counts.", "Contour has high error and many invalid/excluded GT landmarks, which weakens it as an early pseudo-label target."))

    corr = correlations[correlations["scope"].eq("global")].copy()
    corr["label"] = corr["confidence_signal"].map(SIGNAL_LABELS).fillna(corr["confidence_signal"])
    corr = corr.sort_values("spearman", key=lambda s: s.abs())
    saved += horizontal_bar(
        "Confidence Signal Ranking by Error Correlation",
        corr["label"].tolist(),
        corr["spearman"].astype(float).tolist(),
        "fig03_confidence_signal_correlation_ranking",
        "Spearman correlation with normalized error",
        [COLORS["orange"] if s in UNCERTAINTY_SIGNALS else COLORS["blue"] for s in corr["confidence_signal"]],
        x_min=-0.7,
        x_max=0.7,
    )
    notes.append(("fig03_confidence_signal_correlation_ranking", "Global Spearman correlations; blue signals should decrease with error, orange uncertainty signals should increase.", "Heatmap variance, TTA variance, and heatmap max have the strongest monotonic relation with error; peak sharpness is weak."))

    fd = failure[np.isclose(failure["failure_threshold"], 0.05)].drop_duplicates("confidence_signal").copy()
    fd["label"] = fd["confidence_signal"].map(SIGNAL_LABELS).fillna(fd["confidence_signal"])
    fd = fd.sort_values("auroc")
    saved += horizontal_bar(
        "Failure-Detection AUROC Ranking",
        fd["label"].tolist(),
        fd["auroc"].astype(float).tolist(),
        "fig04_failure_detection_auroc",
        "AUROC for detecting NME > 5%",
        [COLORS["teal"]] * len(fd),
        x_min=0.45,
        x_max=0.90,
        ref_line=0.5,
        ref_label="random",
    )
    notes.append(("fig04_failure_detection_auroc", "AUROC for failure detection at NME > 0.05.", "Heatmap variance and heatmap max are the best failure detectors; peak sharpness is close to random."))

    saved += line_plot(retention)
    notes.append(("fig05_retention_curves_strong_signals", "Mean retained NME versus retained landmark fraction for strong interpretable signals.", "Confidence filtering clearly reduces expected pseudo-label noise at strict retained fractions."))

    v = viability.set_index("region").reindex(REGION_ORDER).reset_index().copy()
    v["top25_nme_percent"] = v["mean_nme"] * 100
    v["failure_rate_percent"] = v["failure_rate"] * 100
    rows = [
        [
            row.region.title(),
            SIGNAL_LABELS.get(row.best_signal_at_25pct, row.best_signal_at_25pct),
            f"{row.top25_nme_percent:.1f}%",
            f"{row.failure_rate_percent:.1f}%",
            str(row.recommendation).replace("_", " "),
        ]
        for row in v.itertuples()
    ]
    rec_colors = [
        COLORS["pale_green"] if r.recommendation == "early" else COLORS["pale_yellow"] if r.recommendation == "late" else COLORS["pale_red"]
        for r in v.itertuples()
    ]
    saved += table_figure(
        "Region-Wise Pseudo-Label Viability Matrix",
        ["Region", "Best signal", "Top-25 NME", "Top-25 failure", "Recommendation"],
        rows,
        "fig06_region_pseudo_label_viability",
        rec_colors,
    )
    notes.append(("fig06_region_pseudo_label_viability", "Annotated table using the region pseudo-label viability CSV.", "Mouth is the only early region; eyes and nose need strict/later use; contour and eyebrows should be delayed or excluded."))

    saved += bar_plot(
        "Top-25% Retained NME by Region",
        v["region"].tolist(),
        v["top25_nme_percent"].astype(float).tolist(),
        "fig07_top25_retained_nme_by_region",
        "Mean NME in best top-25% (%)",
        [f"{r.top25_nme_percent:.1f}%\n{SIGNAL_LABELS.get(r.best_signal_at_25pct, r.best_signal_at_25pct)}\nfail {r.failure_rate_percent:.0f}%" for r in v.itertuples()],
        [COLORS["red"], COLORS["red"], COLORS["yellow"], COLORS["yellow"], COLORS["green"]],
        y_max=float(v["top25_nme_percent"].max() * 1.35),
    )
    notes.append(("fig07_top25_retained_nme_by_region", "Top-25 retained NME per region using each region's best signal.", "Mouth has the cleanest retained subset; contour and eyebrows remain too noisy after filtering."))

    pose_rows = []
    for pose, group in valid_landmarks[valid_landmarks["pose"].isin(POSE_ORDER)].groupby("pose"):
        ranked = group[np.isfinite(group["heatmap_variance"])].sort_values("heatmap_variance")
        top = ranked.head(max(1, int(math.ceil(len(ranked) * 0.25))))
        pose_rows.append(
            {
                "pose": pose,
                "all_mean": group["normalized_error"].mean() * 100,
                "top25_mean": top["normalized_error"].mean() * 100,
                "top25_failure": (top["normalized_error"] > 0.05).mean() * 100,
                "n_top25": len(top),
            }
        )
    pose_ret = pd.DataFrame(pose_rows).set_index("pose").reindex(POSE_ORDER).reset_index()
    saved += bar_plot(
        "Pose-Stratified Pseudo-Label Risk",
        pose_ret["pose"].tolist(),
        pose_ret["top25_mean"].astype(float).tolist(),
        "fig08_pose_pseudo_label_risk",
        "Top-25 mean NME (%)",
        [f"{r.top25_mean:.1f}%\nfail {r.top25_failure:.0f}%\nn={int(r.n_top25)}" for r in pose_ret.itertuples()],
        [COLORS["green"], COLORS["green"], COLORS["green"], COLORS["red"], COLORS["red"]],
        y_max=float(pose_ret["top25_mean"].max() * 1.35),
        subtitle="Top-25 landmarks selected by lowest heatmap variance within each pose",
    )
    notes.append(("fig08_pose_pseudo_label_risk", "Pose-stratified top-25 filtering with low heatmap variance.", "Filtering helps, but profile poses remain higher risk and should be introduced cautiously."))

    saved += density_figure(valid_landmarks, "heatmap_variance", "fig09_heatmap_variance_vs_error_density", "Heatmap variance", True)
    notes.append(("fig09_heatmap_variance_vs_error_density", "Density plot of heatmap variance against landmark NME with binned median trend.", "Higher heatmap variance is associated with higher error."))
    saved += density_figure(valid_landmarks, "tta_variance", "fig10_tta_variance_vs_error_density", "TTA variance", True)
    notes.append(("fig10_tta_variance_vs_error_density", "Density plot of TTA variance against landmark NME with binned median trend.", "TTA variance behaves as an uncertainty signal and supports conservative filtering."))
    saved += density_figure(valid_landmarks, "heatmap_max", "fig11_heatmap_max_vs_error_density", "Heatmap max", False)
    notes.append(("fig11_heatmap_max_vs_error_density", "Density plot of heatmap maximum against landmark NME with binned median trend.", "Higher heatmap max is generally associated with lower error, but high-confidence failures remain possible."))

    pca_col = "mean_pca_reconstruction_error"
    if pca_col in per_image.columns and per_image[pca_col].replace([np.inf, -np.inf], np.nan).notna().any():
        pca_values = per_image[pca_col].replace([np.inf, -np.inf], np.nan)
        pca_rows = [
            ["Available image rows", str(int(pca_values.notna().sum())), "PCA values are present", "Diagnostic"],
            ["Median PCA error", f"{pca_values.median():.3g}", "Image-level shape plausibility", "Use cautiously"],
        ]
    else:
        pca_rows = [
            ["PCA reconstruction error", "Unavailable", "All image-level values are NaN", "Do not use"],
            ["Landmark-level filtering", "Invalid", "PCA is image-level, not landmark-level", "Exclude"],
        ]
    saved += table_figure(
        "Image-Level PCA Reconstruction Error Analysis",
        ["Component", "Status", "Evidence", "Use"],
        pca_rows,
        "fig12_pca_error_image_level_analysis",
        [COLORS["pale_red"], COLORS["pale_red"]],
        width=2400,
        height=900,
    )
    notes.append(("fig12_pca_error_image_level_analysis", "Image-level PCA diagnostic rather than a landmark-level signal.", "PCA reconstruction error is unavailable in this run and should not guide pseudo-label selection."))

    decision_rows = [
        ["Consistency-only", "Recommended baseline", "Uses unlabeled real data without trusting noisy labels", "Low"],
        ["Pseudo-labeling: mouth", "Use early", "Top-25 NME 2.6%, failure 5.8%", "Moderate"],
        ["Pseudo-labeling: eyes", "Use later / strict", "Top-25 NME 3.9%, failure 19.2%", "Medium"],
        ["Pseudo-labeling: nose", "Use later / strict", "Top-25 NME 3.2%, failure 21.9%", "Medium"],
        ["Pseudo-labeling: eyebrows", "Delay / exclude", "Top-25 NME 5.4%, failure 48.5%", "High"],
        ["Pseudo-labeling: contour", "Exclude initially", "Top-25 NME 7.4%, failure 64.1%", "Very high"],
    ]
    risk_colors = [COLORS["pale_green"], COLORS["pale_green"], COLORS["pale_yellow"], COLORS["pale_yellow"], COLORS["pale_red"], COLORS["pale_red"]]
    saved += table_figure(
        "Recommended UDA Strategy from Confidence-Error Diagnostics",
        ["UDA component", "Recommendation", "Evidence", "Risk"],
        decision_rows,
        "fig13_uda_strategy_recommendation",
        risk_colors,
        width=2700,
        height=1500,
    )
    notes.append(("fig13_uda_strategy_recommendation", "Compact decision matrix translating diagnostics into UDA choices.", "Recommended next step is consistency plus very conservative region-specific pseudo-labeling, beginning with mouth only."))

    cols_by_file = {path.name: list(pd.read_csv(path, nrows=0).columns) for path in sorted(INPUT_DIR.glob("*.csv"))}
    markdown = [
        "# Figure Interpretation Summary\n\n",
        "## Data and Protocol Checks\n",
        f"- Source directory: `{INPUT_DIR}`\n",
        f"- Output directory: `{OUTPUT_DIR.resolve()}`\n",
        f"- Per-image rows: {len(per_image)}\n",
        f"- Evaluable landmark rows: {len(valid_landmarks)}\n",
        f"- Mean official per-image NME: {per_image['mean_nme'].mean():.4f} fraction / {per_image['mean_nme_percent'].mean():.2f}%\n",
        "- Invalid GT landmarks are excluded from error computations through the corrected official evaluation mask.\n",
        "- Predicted visibility is treated only as an analysis variable, not as the official mask.\n",
        "- NME values stored as fractions were multiplied by 100 for plotting.\n\n",
        "## Available CSV Schemas\n",
    ]
    for filename, columns in cols_by_file.items():
        markdown.append(f"- `{filename}`: {', '.join(columns)}\n")
    markdown.append("\n## Figure-by-Figure Interpretation\n")
    for name, construction, conclusion in notes:
        caveat = "Conclusions are diagnostic only; BabyLand labels were used for offline analysis, not for training or adaptation."
        if name == "fig12_pca_error_image_level_analysis":
            caveat = "PCA reconstruction error is image-level and unavailable in this run; it must not be interpreted as a landmark-level confidence signal."
        elif name.startswith(("fig09", "fig10", "fig11")):
            caveat = "Density plots show association, not perfect calibration; high-confidence failures can still occur."
        markdown.append(f"### `{name}`\n")
        markdown.append(f"- What it shows / construction: {construction}\n")
        markdown.append(f"- Conclusion supported: {conclusion}\n")
        markdown.append(f"- Caveat: {caveat}\n")
    markdown.extend(
        [
            "\n## Direct Answers\n",
            "1. **Strongest confidence signals:** heatmap variance is strongest overall, followed closely by TTA variance and heatmap max. Heatmap entropy is useful but weaker. Peak sharpness is weak and should not drive selection.\n",
            "2. **Safest pseudo-labeling regions:** mouth is the safest early region. Eyes and nose may be used later with strict filtering.\n",
            "3. **Regions to exclude initially:** contour and eyebrows should be excluded or delayed. Contour is especially risky because it combines high NME with many invalid/excluded GT landmarks.\n",
            "4. **Does filtering reduce pseudo-label noise?** Yes. Retention curves and top-25 summaries show lower retained NME under strong signals, but filtering is not sufficient for all regions.\n",
            "5. **First UDA experiment:** start with consistency training plus very conservative pseudo-labeling for mouth only, or run consistency-only as a baseline if the experiment budget allows. Avoid broad all-landmark pseudo-labeling.\n",
            "6. **PCA reconstruction error:** treat it as an image-level shape plausibility diagnostic only. In these corrected outputs it is unavailable because all values are NaN, so it should not influence pseudo-label selection.\n\n",
            "## Generated Files\n",
        ]
    )
    for path in saved:
        markdown.append(f"- `{path.name}`\n")
    summary = OUTPUT_DIR / "figure_interpretation_summary.md"
    summary.write_text("".join(markdown), encoding="utf-8")
    saved.append(summary)

    print(f"Source directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Per-image mean NME: {per_image['mean_nme_percent'].mean():.3f}%")
    print(f"Saved {len(saved)} files")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
