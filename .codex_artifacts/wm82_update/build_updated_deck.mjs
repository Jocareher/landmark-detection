import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "/Users/jocareher/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const WORKSPACE_DIR = "/Users/jocareher/Library/CloudStorage/OneDrive-Personal/Educacion/PhD_UPF_2023/landmarks_detection";
const SOURCE_PPTX = "/Users/jocareher/Downloads/weekly_meeting_82.pptx";
const DATA_DIR = "/Users/jocareher/Downloads/tta";
const BUILD_DIR = path.join(WORKSPACE_DIR, ".codex_artifacts/wm82_update");
const OUTPUT_DIR = path.join(WORKSPACE_DIR, "artifacts");
const FINAL_PPTX = path.join(OUTPUT_DIR, "weekly_meeting_82_updated.pptx");

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

await fs.mkdir(BUILD_DIR, { recursive: true });
await fs.mkdir(OUTPUT_DIR, { recursive: true });

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

const [lr1e4Rows, lr1e3Rows, aggregateRows, imageRows] = await Promise.all([
  fs.readFile(path.join(DATA_DIR, "sweep_orientation_results.csv"), "utf8").then(parseCsv),
  fs.readFile(path.join(DATA_DIR, "sweep_orientation_results_lr-1e-3.csv"), "utf8").then(parseCsv),
  fs.readFile(path.join(DATA_DIR, "aggregate_curves.csv"), "utf8").then(parseCsv),
  fs.readFile(path.join(DATA_DIR, "image_summary.csv"), "utf8").then(parseCsv),
]);

const initialByOrientation = {
  left: 0.19417372009574588,
  quarter_left: 0.052600015595677556,
  frontal: 0.04730543839917397,
  quarter_right: 0.052627874823522484,
  right: 0.18702342153191465,
};
const sampleCounts = { left: 174, quarter_left: 54, frontal: 166, quarter_right: 54, right: 174 };
const initialGlobalNme = 0.12839760691336435;
const initialGlobalHd = 0.2307;

function weightedMetric(rows, key) {
  const numerator = rows.reduce((sum, row) => sum + Number(row[key]) * Number(row.num_samples), 0);
  const denominator = rows.reduce((sum, row) => sum + Number(row.num_samples), 0);
  return numerator / denominator;
}

function globalSweep(rows) {
  const byStep = new Map();
  for (const row of rows) {
    const step = Number(row.steps);
    if (!byStep.has(step)) byStep.set(step, []);
    byStep.get(step).push(row);
  }
  return [...byStep.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([step, stepRows]) => ({
      step,
      nme: weightedMetric(stepRows, "mean_nme") * 100,
      hd: weightedMetric(stepRows, "mean_hausdorff") * 100,
    }));
}

const lr1e4Global = [{ step: 0, nme: initialGlobalNme * 100, hd: initialGlobalHd * 100 }, ...globalSweep(lr1e4Rows)];
const lr1e3Global = [{ step: 0, nme: initialGlobalNme * 100, hd: initialGlobalHd * 100 }, ...globalSweep(lr1e3Rows)];

function orientationSeries(name) {
  const values = [{ step: 0, value: initialByOrientation[name] * 100 }];
  for (const row of lr1e3Rows.filter((item) => item.orientation === name)) {
    values.push({ step: Number(row.steps), value: Number(row.mean_nme) * 100 });
  }
  return values;
}

const aggregateSelectedSteps = [0, 50, 100, 250, 500, 750, 1000, 1250, 1500];
const aggregateByStep = new Map(aggregateRows.map((row) => [Number(row.step), row]));
const aggregate = aggregateSelectedSteps.map((step) => ({
  step,
  reductionMedian: Number(aggregateByStep.get(step).relative_reduction_median) * 100,
  reductionP25: Number(aggregateByStep.get(step).relative_reduction_p25) * 100,
  reductionP75: Number(aggregateByStep.get(step).relative_reduction_p75) * 100,
  driftMedian: Number(aggregateByStep.get(step).mean_drift_median_px),
  driftP25: Number(aggregateByStep.get(step).mean_drift_p25_px),
  driftP75: Number(aggregateByStep.get(step).mean_drift_p75_px),
}));

const probe = imageRows.find((row) => row.sample_id === "face_bcn_09__det_000");
if (!probe) throw new Error("Probe face_bcn_09__det_000 was not found.");

const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE_PPTX));

const FONT = "Helvetica Neue";
const NAVY = "#0E2841";
const BLUE = "#156082";
const ORANGE = "#E97132";
const GREEN = "#196B24";
const CYAN = "#0F9ED5";
const PURPLE = "#A02B93";
const LIGHT_BLUE = "#EAF3F7";
const GRID = "#D9E2E7";
const MUTED = "#486675";
const WHITE = "#FFFFFF";

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: options.geometry ?? "textbox",
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 18,
    color: options.color ?? NAVY,
    bold: options.bold ?? false,
    italic: options.italic ?? false,
    alignment: options.alignment ?? "left",
    autoFit: options.autoFit ?? "shrinkText",
  };
  return shape;
}

function addTitle(slide, title) {
  return addText(slide, title, { left: 52, top: 29, width: 1176, height: 53 }, {
    fontSize: 26,
    color: BLUE,
    bold: true,
    autoFit: "none",
  });
}

function addFindingStrip(slide, text) {
  return addText(slide, text, { left: 62, top: 590, width: 1156, height: 58 }, {
    geometry: "roundRect",
    fill: LIGHT_BLUE,
    line: { fill: "#D5E8F0", width: 1 },
    fontSize: 15,
    color: NAVY,
    bold: true,
    alignment: "center",
  });
}

function chartTextStyle(size = 15, bold = false, color = NAVY) {
  return { typeface: FONT, fontSize: size, bold, fill: color };
}

function styleChart(chart) {
  applyPresentationChartFont(chart, { fontFamily: FONT });
}

function addScatterChart(slide, position, series, xTitle, yTitle, yMin, yMax, legendPosition = "bottom") {
  const chart = slide.charts.add("scatter", {
    position,
    series,
    hasLegend: true,
    legend: { position: legendPosition, overlay: false, textStyle: chartTextStyle(13) },
    scatterOptions: { style: "lineWithMarkers" },
    xAxis: {
      title: { text: xTitle, textStyle: chartTextStyle(14, true) },
      min: 0,
      max: 1500,
      numberFormatCode: "0",
      majorGridlines: { line: { fill: GRID, width: 1 } },
      textStyle: chartTextStyle(12),
    },
    yAxis: {
      title: { text: yTitle, textStyle: chartTextStyle(14, true) },
      min: yMin,
      max: yMax,
      numberFormatCode: "0.0",
      majorGridlines: { line: { fill: GRID, width: 1 } },
      textStyle: chartTextStyle(12),
    },
    chartFill: WHITE,
    plotAreaFill: WHITE,
    chartLine: { fill: "none", width: 0 },
    plotAreaLine: { fill: "none", width: 0 },
  });
  styleChart(chart);
  return chart;
}

// Cover date.
presentation.resolve("sh/65g3298r").text.replace("03/09/2026", "16/09/2026");

// Slide 2: protocol table now reports both learning-rate sweeps.
const protocolTable = presentation.resolve("tb/ux4vyxcr");
protocolTable.cells.set(6, 1, "1 × 10⁻⁴ and 1 × 10⁻³");
protocolTable.cells.set(7, 1, "BabyLand: 0–1250 at 10⁻⁴; 0–1500 at 10⁻³; InfAnFace: 0 and 1500 at 10⁻³");
presentation.resolve("nt/hwbqtkby").setText(
  "The normalizer is the only trainable module at test time. The landmarker and PCA prior remain frozen. The 10⁻⁴ and 10⁻³ sweeps share the same step-0 checkpoint.\n[Sources]\n- /Users/jocareher/Downloads/tta/resolved_config.json\n- /Users/jocareher/Downloads/tta/sweep_orientation_results.csv\n- /Users/jocareher/Downloads/tta/sweep_orientation_results_lr-1e-3.csv\n- /Users/jocareher/Downloads/tta/infanface_sweep_orientation_results_lr-1e-3.csv\n[/Sources]"
);

// Slide 3: make the learning rate explicit in the final TTA column.
const summaryTable = presentation.resolve("tb/na9wral8");
summaryTable.cells.set(0, 4, "TTA 1500\n(10⁻³)");
presentation.resolve("sh/r6xcjq5o").text = "At 10⁻³, BabyLand recovers the paper-level mean NME while improving median NME and HD. InfAnFace improves all three aggregate metrics.";

// Slide 4: learning-rate comparison.
const slide4 = presentation.resolve("sl/jyx0ra1s");
slide4.charts.deleteById(presentation.resolve("ch/q50fq5wr").id);
slide4.charts.deleteById(presentation.resolve("ch/r69gjadw").id);
slide4.shapes.deleteAll();
addTitle(slide4, "A Higher Learning Rate Accelerates BabyLand Adaptation");
addText(slide4, "Mean NME", { left: 70, top: 104, width: 520, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
addText(slide4, "Mean normalized Hausdorff distance", { left: 690, top: 104, width: 520, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
addScatterChart(
  slide4,
  { left: 55, top: 140, width: 570, height: 420 },
  [
    {
      name: "lr = 10⁻⁴",
      xValues: lr1e4Global.map((item) => item.step),
      values: lr1e4Global.map((item) => item.nme),
      line: { fill: BLUE, width: 3 },
      marker: { symbol: "circle", size: 7 },
    },
    {
      name: "lr = 10⁻³",
      xValues: lr1e3Global.map((item) => item.step),
      values: lr1e3Global.map((item) => item.nme),
      line: { fill: ORANGE, width: 3 },
      marker: { symbol: "diamond", size: 7 },
    },
  ],
  "TTA updates per image",
  "Mean NME (%)",
  10.0,
  13.2,
);
addScatterChart(
  slide4,
  { left: 655, top: 140, width: 570, height: 420 },
  [
    {
      name: "lr = 10⁻⁴",
      xValues: lr1e4Global.map((item) => item.step),
      values: lr1e4Global.map((item) => item.hd),
      line: { fill: BLUE, width: 3 },
      marker: { symbol: "circle", size: 7 },
    },
    {
      name: "lr = 10⁻³",
      xValues: lr1e3Global.map((item) => item.step),
      values: lr1e3Global.map((item) => item.hd),
      line: { fill: ORANGE, width: 3 },
      marker: { symbol: "diamond", size: 7 },
    },
  ],
  "TTA updates per image",
  "Mean normalized HD (%)",
  19.2,
  23.5,
);
addFindingStrip(slide4, "At 750 updates, lr = 10⁻³ reaches 10.57% NME, already below lr = 10⁻⁴ at 1250 updates (11.04%). The higher rate also lowers HD more quickly.");
slide4.speakerNotes.textFrame.setText(
  "Global values are weighted by the number of samples in each orientation bin. Step 0 is the shared unadapted checkpoint. The 10⁻³ sweep approaches a plateau after 750 updates, whereas 10⁻⁴ remains above it at 1250 updates.\n[Sources]\n- /Users/jocareher/Downloads/tta/sweep_orientation_results.csv\n- /Users/jocareher/Downloads/tta/sweep_orientation_results_lr-1e-3.csv\n- /Users/jocareher/Downloads/tta/image_summary.csv\n[/Sources]"
);

// Slide 5: pose-specific stopping behavior.
const slide5 = presentation.resolve("sl/i107q5of");
slide5.charts.deleteById(presentation.resolve("ch/76dkbetk").id);
slide5.shapes.deleteAll();
addTitle(slide5, "Central Views Plateau by 750 Steps while Profiles Keep Improving");
addText(slide5, "Profile views", { left: 70, top: 104, width: 520, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
addText(slide5, "Frontal and three-quarter views", { left: 690, top: 104, width: 520, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
const leftProfile = orientationSeries("left");
const rightProfile = orientationSeries("right");
addScatterChart(
  slide5,
  { left: 55, top: 140, width: 570, height: 420 },
  [
    { name: "Left", xValues: leftProfile.map((item) => item.step), values: leftProfile.map((item) => item.value), line: { fill: BLUE, width: 3 }, marker: { symbol: "circle", size: 7 } },
    { name: "Right", xValues: rightProfile.map((item) => item.step), values: rightProfile.map((item) => item.value), line: { fill: ORANGE, width: 3 }, marker: { symbol: "diamond", size: 7 } },
  ],
  "TTA updates per image",
  "Mean NME (%)",
  14.0,
  20.0,
);
const ql = orientationSeries("quarter_left");
const frontal = orientationSeries("frontal");
const qr = orientationSeries("quarter_right");
addScatterChart(
  slide5,
  { left: 655, top: 140, width: 570, height: 420 },
  [
    { name: "Quarter left", xValues: ql.map((item) => item.step), values: ql.map((item) => item.value), line: { fill: BLUE, width: 3 }, marker: { symbol: "circle", size: 7 } },
    { name: "Frontal", xValues: frontal.map((item) => item.step), values: frontal.map((item) => item.value), line: { fill: GREEN, width: 3 }, marker: { symbol: "square", size: 7 } },
    { name: "Quarter right", xValues: qr.map((item) => item.step), values: qr.map((item) => item.value), line: { fill: ORANGE, width: 3 }, marker: { symbol: "diamond", size: 7 } },
  ],
  "TTA updates per image",
  "Mean NME (%)",
  4.4,
  5.4,
);
addFindingStrip(slide5, "At lr = 10⁻³, 750 steps capture essentially all gains in frontal and three-quarter views. Both profile means continue decreasing through 1500 steps.");
slide5.speakerNotes.textFrame.setText(
  "The central-view curves remain flat or worsen slightly after 750 updates: quarter-left changes from 5.00% to 5.01%, frontal stays at 4.61% at reported precision, and quarter-right changes from 5.19% to 5.20%. Left and right profiles continue improving to 15.03% and 14.77% at 1500 updates. This supports a pose-dependent descriptive interpretation, not an implemented stopping policy.\n[Sources]\n- /Users/jocareher/Downloads/tta/sweep_orientation_results_lr-1e-3.csv\n- /Users/jocareher/Downloads/tta/orientation_tta_summary.csv\n[/Sources]"
);

// Slide 7: information contained in aggregate_curves.csv.
const slide7 = presentation.resolve("sl/gnmp4jqx");
slide7.tables.deleteById(presentation.resolve("tb/lovu90bu").id);
slide7.shapes.deleteAll();
addTitle(slide7, "Aggregate PCA Convergence and Landmark Drift");
addText(slide7, "Median PCA-loss reduction", { left: 70, top: 104, width: 520, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
addText(slide7, "Mean landmark drift from step 0", { left: 690, top: 104, width: 520, height: 34 }, { fontSize: 18, bold: true, alignment: "center" });
addScatterChart(
  slide7,
  { left: 55, top: 140, width: 570, height: 420 },
  [
    { name: "P25", xValues: aggregate.map((item) => item.step), values: aggregate.map((item) => item.reductionP25), line: { fill: "#9BC2D3", width: 2 }, marker: { symbol: "none", size: 2 } },
    { name: "Median", xValues: aggregate.map((item) => item.step), values: aggregate.map((item) => item.reductionMedian), line: { fill: BLUE, width: 4 }, marker: { symbol: "circle", size: 7 } },
    { name: "P75", xValues: aggregate.map((item) => item.step), values: aggregate.map((item) => item.reductionP75), line: { fill: "#5B9DB8", width: 2 }, marker: { symbol: "none", size: 2 } },
  ],
  "TTA updates per image",
  "Relative reduction (%)",
  0,
  100,
);
addScatterChart(
  slide7,
  { left: 655, top: 140, width: 570, height: 420 },
  [
    { name: "P25", xValues: aggregate.map((item) => item.step), values: aggregate.map((item) => item.driftP25), line: { fill: "#F4B183", width: 2 }, marker: { symbol: "none", size: 2 } },
    { name: "Median", xValues: aggregate.map((item) => item.step), values: aggregate.map((item) => item.driftMedian), line: { fill: ORANGE, width: 4 }, marker: { symbol: "diamond", size: 7 } },
    { name: "P75", xValues: aggregate.map((item) => item.step), values: aggregate.map((item) => item.driftP75), line: { fill: "#C65911", width: 2 }, marker: { symbol: "none", size: 2 } },
  ],
  "TTA updates per image",
  "Mean drift (pixels)",
  0,
  16,
);
addFindingStrip(slide7, "By 750 steps, the median PCA loss falls 84.1% and median drift reaches 4.65 px. From 750 to 1500, loss reduction gains only 4.7 pp while drift rises to 4.90 px.");
slide7.speakerNotes.textFrame.setText(
  "aggregate_curves.csv describes the optimization objective and prediction movement across all 622 BabyLand images. It does not contain NME at every step. Therefore these curves show convergence and drift, not localization accuracy. P25 and P75 indicate the wide heterogeneity across images.\n[Sources]\n- /Users/jocareher/Downloads/tta/aggregate_curves.csv\n[/Sources]"
);

// Slide 8: selected qualitative probe with animated GIF.
const slide8 = presentation.resolve("sl/fu1gfa1s");
slide8.tables.deleteById(presentation.resolve("tb/cnydkj29").id);
slide8.shapes.deleteAll();
addTitle(slide8, "Qualitative Adaptation in a Successful Left-Profile Case");
const gifBytes = new Uint8Array(await fs.readFile(path.join(DATA_DIR, "probes/face_bcn_09__det_000/adaptation.gif")));
slide8.images.add({
  blob: gifBytes,
  contentType: "image/gif",
  alt: "Animated PCA-guided TTA probe for face_bcn_09__det_000 from step 0 to step 1500",
  fit: "contain",
  position: { left: 62, top: 112, width: 1156, height: 230 },
});
addText(slide8, "Observed change", { left: 70, top: 375, width: 330, height: 34 }, { fontSize: 18, bold: true, color: BLUE });
addText(
  slide8,
  "The magenta landmarks move from a scattered profile prediction toward a coherent eye, nose, mouth, and jaw configuration. The input and normalized image remain visually similar, while the enhanced difference map reveals where the normalizer changes image evidence.",
  { left: 70, top: 410, width: 520, height: 126 },
  { fontSize: 16, color: NAVY },
);
addText(slide8, "Step 0 → step 1500", { left: 675, top: 375, width: 330, height: 34 }, { fontSize: 18, bold: true, color: BLUE });
addText(
  slide8,
  `PCA loss: ${Number(probe.initial_pca_reconstruction_loss).toExponential(2)} → ${Number(probe.final_pca_reconstruction_loss).toExponential(2)}\nNME: ${(Number(probe.initial_nme_box_gt_valid) * 100).toFixed(2)}% → ${(Number(probe.final_nme_box_gt_valid) * 100).toFixed(2)}%\nHausdorff: ${(Number(probe.initial_hausdorff_box_gt_valid) * 100).toFixed(2)}% → ${(Number(probe.final_hausdorff_box_gt_valid) * 100).toFixed(2)}%\nMean landmark drift: ${Number(probe.final_mean_landmark_drift_px).toFixed(2)} px`,
  { left: 675, top: 410, width: 470, height: 132 },
  { fontSize: 17, color: NAVY, bold: true },
);
addFindingStrip(slide8, "The GIF plays in slideshow mode. It is a selected successful example, not a representative estimate. Enhanced-difference colors are rescaled per frame and should not be compared as absolute magnitudes.");
slide8.speakerNotes.textFrame.setText(
  "The six panels show input crop, normalizer output, fixed-scale absolute RGB difference, enhanced P99-scaled difference, step-0 landmarks, and current landmarks. The enhanced map makes subtle changes visible but rescales each frame independently. This probe was selected because it clearly illustrates the mechanism and its metrics improve; it should be presented as an illustrative success case. Ground truth is used only for the post-hoc NME and Hausdorff values.\n[Sources]\n- /Users/jocareher/Downloads/tta/probes/face_bcn_09__det_000/adaptation.gif\n- /Users/jocareher/Downloads/tta/image_summary.csv\n- /Users/jocareher/Downloads/tta/trajectories.csv\n[/Sources]"
);

const candidatePath = path.join(BUILD_DIR, "candidate_updated.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const sourceBytes = await fs.readFile(SOURCE_PPTX);
const crypto = await import("node:crypto");
const referenceSha256 = crypto.createHash("sha256").update(sourceBytes).digest("hex");

const requirements = {
  explicitTotalSlideCount: 16,
  requiredNativeChartOwnerSlides: [4, 5, 6, 7],
  requiredNativeTableOwnerSlides: [2, 3, 14, 15, 16],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
};
const fontPolicy = {
  basis: "reference",
  families: [FONT, "Aptos", "Segoe UI Semibold"],
  referencePath: SOURCE_PPTX,
  referenceSha256,
};

const stagingDir = path.join(BUILD_DIR, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const result = await finalizePresentation({
  ...requirements,
  workspaceDir: WORKSPACE_DIR,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: "/Users/jocareher/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "2",
    "--require-native-table-slide", "3",
    "--require-native-table-slide", "14",
    "--require-native-table-slide", "15",
    "--require-native-table-slide", "16",
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  requiredNativeChartOwnerSlides: requirements.requiredNativeChartOwnerSlides,
  materializeLiteralChartWorkbooks: true,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "weekly_meeting_82_updated.validation.json"),
});

console.log(JSON.stringify({ finalPath: FINAL_PPTX, result }, null, 2));
