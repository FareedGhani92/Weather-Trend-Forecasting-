/* Build the Weather Trend Forecasting presentation (PM Accelerator assessment).
   Run:  NODE_PATH="$(npm root -g)" node build_deck.js  */
const fs = require("fs");
const pptxgen = require("pptxgenjs");

// ---- Palette (Midnight Executive: navy + blue + ice) ----------------------
const NAVY = "1E2761", BLUE = "2563EB", SKY = "0EA5E9";
const INK = "0F172A", SLATE = "475569", MUTE = "64748B";
const ICE = "EAF1FB", LIGHT = "F8FAFC", WHITE = "FFFFFF", LINE = "E2E8F0";
const HEAD = "Cambria", BODY = "Calibri";
const FIG = "outputs/figures/";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.33" x 7.5"
pres.author = "PM Accelerator Assessment";
pres.title = "Weather Trend Forecasting";
const W = 13.33, H = 7.5;

const shadow = () => ({ type: "outer", color: "94A3B8", blur: 8, offset: 3, angle: 90, opacity: 0.25 });
const pngSize = p => { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };

// Place an image fit inside a box on a white card, with caption below.
function figure(slide, file, bx, by, bw, bh, caption) {
  const { w: pw, h: ph } = pngSize(FIG + file);
  const ir = pw / ph, br = bw / bh;
  let w, h;
  if (ir > br) { w = bw; h = bw / ir; } else { h = bh; w = bh * ir; }
  const x = bx + (bw - w) / 2, y = by + (bh - h) / 2;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: by, w: bw, h: bh,
    fill: { color: WHITE }, line: { color: LINE, width: 1 }, rectRadius: 0.06, shadow: shadow() });
  slide.addImage({ path: FIG + file, x: x + 0.06, y: y + 0.06, w: w - 0.12, h: h - 0.12 });
  if (caption) slide.addText(caption, { x: bx, y: by + bh + 0.04, w: bw, h: 0.3,
    fontFace: BODY, fontSize: 10.5, color: MUTE, italic: true, align: "center", margin: 0 });
}

// Content-slide header (kicker + title), light background.
function header(slide, kicker, title) {
  slide.background = { color: LIGHT };
  slide.addText(kicker.toUpperCase(), { x: 0.6, y: 0.34, w: 12, h: 0.3,
    fontFace: BODY, fontSize: 12, bold: true, color: BLUE, charSpacing: 2, margin: 0 });
  slide.addText(title, { x: 0.6, y: 0.62, w: 12.1, h: 0.7,
    fontFace: HEAD, fontSize: 28, bold: true, color: NAVY, margin: 0 });
}

// Small white stat card.
function stat(slide, x, y, w, value, label) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.15,
    fill: { color: WHITE }, line: { color: LINE, width: 1 }, rectRadius: 0.08, shadow: shadow() });
  slide.addText(value, { x: x + 0.1, y: y + 0.14, w: w - 0.2, h: 0.6,
    fontFace: HEAD, fontSize: 26, bold: true, color: NAVY, align: "center", margin: 0 });
  slide.addText(label.toUpperCase(), { x: x + 0.1, y: y + 0.74, w: w - 0.2, h: 0.3,
    fontFace: BODY, fontSize: 10.5, bold: true, color: MUTE, align: "center", charSpacing: 1, margin: 0 });
}

// ===========================================================================
// 1 — Title
// ===========================================================================
let s = pres.addSlide();
s.background = { color: NAVY };
s.addText("PM ACCELERATOR  ·  TECHNICAL ASSESSMENT", { x: 0.9, y: 1.0, w: 11.5, h: 0.4,
  fontFace: BODY, fontSize: 14, bold: true, color: SKY, charSpacing: 3, margin: 0 });
s.addText("Weather Trend Forecasting", { x: 0.85, y: 1.5, w: 11.6, h: 1.3,
  fontFace: HEAD, fontSize: 50, bold: true, color: WHITE, margin: 0 });
s.addText("End-to-end analysis of the Global Weather Repository — cleaning, EDA, multi-model forecasting, and advanced climate, air-quality & spatial intelligence.",
  { x: 0.9, y: 2.85, w: 11.0, h: 0.8, fontFace: BODY, fontSize: 17, color: "CADCFC", margin: 0 });
// Mission card
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.9, y: 3.95, w: 11.5, h: 1.5,
  fill: { color: "27306B" }, line: { color: "3A4490", width: 1 }, rectRadius: 0.08 });
s.addText([
  { text: "PM Accelerator Mission\n", options: { bold: true, fontSize: 15, color: SKY } },
  { text: "Supporting product managers through every career stage — from entry-level to leadership — via certified training, hands-on projects and career services, while fostering a diverse tech industry. Through PMA Kids it offers free PM education to underserved teens.",
    options: { fontSize: 12.5, color: "DCE6FA" } },
], { x: 1.15, y: 4.12, w: 11.0, h: 1.2, fontFace: BODY, valign: "top", margin: 0, lineSpacingMultiple: 1.05 });
s.addText("149,879 records   ·   211 countries   ·   268 cities   ·   May 2024 – Jun 2026",
  { x: 0.9, y: 5.75, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: WHITE, margin: 0 });

// ===========================================================================
// 2 — Overview
// ===========================================================================
s = pres.addSlide();
header(s, "Project Overview", "A four-stage analytics pipeline");
stat(s, 0.6, 1.55, 2.9, "149,879", "Clean records");
stat(s, 3.65, 1.55, 2.9, "211", "Countries");
stat(s, 6.7, 1.55, 2.9, "268", "Cities tracked");
stat(s, 9.75, 1.55, 2.98, "~2 yrs", "Daily history");
const stages = [
  ["1  Clean", "Parse last_updated, de-duplicate, impute, winsorize outliers, normalize"],
  ["2  Explore", "Trends, correlations, seasonality; temperature & precipitation visuals"],
  ["3  Forecast", "5 models + ensemble on the daily global-mean series; MAE/RMSE/MAPE/R²"],
  ["4  Advanced", "Anomalies, climate, air quality, feature importance, spatial patterns"],
];
let yy = 3.15;
stages.forEach(([t, d]) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: yy, w: 12.13, h: 0.92,
    fill: { color: WHITE }, line: { color: LINE, width: 1 }, rectRadius: 0.06, shadow: shadow() });
  s.addText(t, { x: 0.85, y: yy + 0.12, w: 2.4, h: 0.68, fontFace: HEAD, fontSize: 18, bold: true, color: BLUE, valign: "middle", margin: 0 });
  s.addText(d, { x: 3.3, y: yy + 0.12, w: 9.2, h: 0.68, fontFace: BODY, fontSize: 14, color: SLATE, valign: "middle", margin: 0 });
  yy += 1.04;
});

// ===========================================================================
// 3 — Data Cleaning
// ===========================================================================
s = pres.addSlide();
header(s, "Basic · Step 1", "Data Cleaning & Preprocessing");
const steps = [
  ["Timestamps", "last_updated parsed to datetime for time-series indexing"],
  ["Duplicates", "Exact and (location, time) duplicates removed — kept latest"],
  ["Missing values", "Median impute (numeric), mode impute (categorical)"],
  ["Outliers", "Winsorized to the Tukey IQR fence — keeps every record"],
  ["Normalization", "Z-score standardized copies added for modelling"],
  ["Feature engineering", "year, month, dayofyear, season derived"],
];
let cx = 0.6, cy = 1.6;
steps.forEach(([t, d], i) => {
  const x = 0.6 + (i % 2) * 6.13, y = 1.6 + Math.floor(i / 2) * 1.28;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 6.0, h: 1.12,
    fill: { color: WHITE }, line: { color: LINE, width: 1 }, rectRadius: 0.07, shadow: shadow() });
  s.addText(t, { x: x + 0.25, y: y + 0.13, w: 5.5, h: 0.36, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText(d, { x: x + 0.25, y: y + 0.5, w: 5.5, h: 0.55, fontFace: BODY, fontSize: 12.5, color: SLATE, margin: 0 });
});
s.addText([
  { text: "Result:  ", options: { bold: true, color: NAVY } },
  { text: "149,880 → 149,879 clean rows   ·   0 missing   ·   0 duplicates", options: { color: SLATE } },
], { x: 0.6, y: 5.55, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 15, align: "center", margin: 0 });

// ===========================================================================
// 4 — EDA: Temperature & Precipitation
// ===========================================================================
s = pres.addSlide();
header(s, "Basic · Step 2", "EDA — Temperature & Precipitation");
figure(s, "temperature_distribution.png", 0.6, 1.55, 6.0, 4.6, "Temperature distribution & spread by season");
figure(s, "precipitation_overview.png", 6.73, 1.55, 6.0, 4.6, "Precipitation distribution (log) & monthly mean");

// ===========================================================================
// 5 — EDA: Correlations & Trend
// ===========================================================================
s = pres.addSlide();
header(s, "Basic · Step 2", "EDA — Correlations & Global Trend");
figure(s, "correlation_heatmap.png", 0.6, 1.55, 5.6, 4.4, "Correlation matrix of weather variables");
figure(s, "global_temp_precip_trend.png", 6.4, 1.55, 6.33, 3.0, "Daily global-mean temperature & precipitation");
s.addText([
  { text: "Key correlations with temperature:  ", options: { bold: true, color: NAVY } },
  { text: "feels-like +0.98 · UV +0.49 · pressure −0.44 · humidity −0.35", options: { color: SLATE } },
], { x: 6.4, y: 4.95, w: 6.33, h: 1.0, fontFace: BODY, fontSize: 13.5, valign: "top", margin: 0, lineSpacingMultiple: 1.1 });

// ===========================================================================
// 6 — Forecasting models & evaluation
// ===========================================================================
s = pres.addSlide();
header(s, "Basic + Advanced · Step 3", "Forecasting — Models & Evaluation");
figure(s, "forecast_comparison.png", 0.6, 1.6, 6.7, 4.5, "Each model vs. actuals on the held-out test window");
const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle" }, o) });
const hd = t => cell(t, { bold: true, color: WHITE, fill: { color: NAVY }, align: "center" });
const rows = [
  [hd("Model"), hd("MAE"), hd("RMSE"), hd("MAPE"), hd("R²")],
  [cell("Naive (lag-1)", { bold: true }), cell("0.190", { align: "center" }), cell("0.242", { align: "center" }), cell("0.83", { align: "center" }), cell("0.767", { align: "center", bold: true })],
  [cell("Ensemble"), cell("0.212", { align: "center" }), cell("0.249", { align: "center" }), cell("0.93", { align: "center" }), cell("0.752", { align: "center" })],
  [cell("SARIMAX"), cell("0.193", { align: "center" }), cell("0.265", { align: "center" }), cell("0.84", { align: "center" }), cell("0.719", { align: "center" })],
  [cell("Random Forest"), cell("0.234", { align: "center" }), cell("0.274", { align: "center" }), cell("1.03", { align: "center" }), cell("0.701", { align: "center" })],
  [cell("Linear Regression"), cell("0.237", { align: "center" }), cell("0.281", { align: "center" }), cell("1.04", { align: "center" }), cell("0.686", { align: "center" })],
];
s.addText("5 models + ensemble · 1-step-ahead walk-forward (fair comparison)",
  { x: 7.5, y: 1.65, w: 5.2, h: 0.4, fontFace: BODY, fontSize: 12.5, italic: true, color: MUTE, margin: 0 });
s.addTable(rows, { x: 7.5, y: 2.15, w: 5.25, colW: [2.05, 0.8, 0.85, 0.78, 0.77], rowH: 0.5,
  border: { pt: 0.5, color: LINE }, fill: { color: WHITE } });
s.addText("On a smooth planetary-mean series the naive baseline is strong; SARIMAX & the ensemble track within ~0.02 RMSE.",
  { x: 7.5, y: 5.5, w: 5.25, h: 0.9, fontFace: BODY, fontSize: 12.5, color: SLATE, valign: "top", margin: 0, lineSpacingMultiple: 1.1 });

// ===========================================================================
// 7 — Forecasting results
// ===========================================================================
s = pres.addSlide();
header(s, "Advanced · Step 3", "Forecasting — Errors & 30-Day Outlook");
figure(s, "model_metrics.png", 0.6, 1.6, 6.0, 4.0, "MAE / RMSE by model (lower is better)");
figure(s, "future_forecast.png", 6.73, 1.6, 6.0, 4.0, "30-day-ahead global mean temperature forecast");
s.addText([
  { text: "Best RMSE 0.242 °C   ·   ", options: { bold: true, color: NAVY } },
  { text: "30-day forecast averages ≈ 23.8 °C", options: { color: SLATE } },
], { x: 0.6, y: 5.85, w: 12.1, h: 0.4, fontFace: BODY, fontSize: 14, align: "center", margin: 0 });

// ===========================================================================
// 8 — Anomaly detection
// ===========================================================================
s = pres.addSlide();
header(s, "Advanced · Anomaly Detection", "Outlier & Anomaly Analysis");
figure(s, "anomaly_detection.png", 0.6, 1.6, 8.4, 4.6, "IsolationForest anomalies in temperature–humidity space");
stat(s, 9.3, 1.95, 3.43, "2,998", "Anomalies (2.0%)");
s.addText([
  { text: "Three methods\n", options: { bold: true, color: NAVY, fontSize: 14 } },
  { text: "IsolationForest (multivariate) cross-checked with Z-score (>3σ) and the IQR rule. Agreement across methods raises confidence in each flag.",
    options: { color: SLATE, fontSize: 12.5 } },
], { x: 9.3, y: 3.3, w: 3.43, h: 2.4, fontFace: BODY, valign: "top", margin: 0, lineSpacingMultiple: 1.12 });

// ===========================================================================
// 9 — Climate analysis
// ===========================================================================
s = pres.addSlide();
header(s, "Advanced · Unique Analysis", "Climate Analysis");
figure(s, "climate_analysis.png", 0.6, 1.6, 9.0, 4.7, "Monthly climatology & seasonality by latitude band");
s.addText([
  { text: "What it shows\n", options: { bold: true, color: NAVY, fontSize: 14 } },
  { text: "Tropical zones stay warm year-round; temperate & polar zones swing strongly with the seasons — and the two hemispheres move in opposite phase.",
    options: { color: SLATE, fontSize: 12.5 } },
], { x: 9.85, y: 2.1, w: 2.9, h: 3.5, fontFace: BODY, valign: "top", margin: 0, lineSpacingMultiple: 1.15 });

// ===========================================================================
// 10 — Air quality
// ===========================================================================
s = pres.addSlide();
header(s, "Advanced · Environmental Impact", "Air Quality vs. Weather");
figure(s, "air_quality_correlation.png", 0.6, 1.6, 8.4, 4.7, "Pollutants vs. weather parameters (correlation)");
s.addText([
  { text: "Weather shapes air quality\n", options: { bold: true, color: NAVY, fontSize: 14 } },
  { text: "Strongest links: humidity ↔ ozone −0.40 and UV ↔ ozone +0.35 — photochemical ozone forms in dry, sunny conditions. Humidity also anti-correlates with the EPA/DEFRA indices.",
    options: { color: SLATE, fontSize: 12.5 } },
], { x: 9.3, y: 2.0, w: 3.43, h: 3.8, fontFace: BODY, valign: "top", margin: 0, lineSpacingMultiple: 1.15 });

// ===========================================================================
// 11 — Feature importance & spatial
// ===========================================================================
s = pres.addSlide();
header(s, "Advanced · Feature Importance + Spatial", "What Drives Temperature & Where");
figure(s, "feature_importance.png", 0.6, 1.6, 6.0, 4.3, "Permutation importance for temperature");
figure(s, "spatial_temperature_map.png", 6.73, 1.6, 6.0, 4.3, "Global temperature — latest reading per city");
s.addText([
  { text: "Latitude (0.83) dominates, then UV (0.29), day-of-year (0.26) & pressure (0.19)", options: { color: SLATE } },
], { x: 0.6, y: 6.0, w: 12.1, h: 0.4, fontFace: BODY, fontSize: 13.5, align: "center", margin: 0 });

// ===========================================================================
// 12 — Geographical patterns
// ===========================================================================
s = pres.addSlide();
header(s, "Advanced · Geographical Patterns", "How Weather Differs Across Countries");
figure(s, "country_temperature_ranking.png", 0.6, 1.6, 6.0, 4.5, "Hottest & coolest countries by mean temperature");
figure(s, "geographical_patterns.png", 6.73, 1.6, 6.0, 4.5, "Temperature spread & precipitation by country");

// ===========================================================================
// 13 — Conclusions
// ===========================================================================
s = pres.addSlide();
s.background = { color: NAVY };
s.addText("KEY INSIGHTS & CONCLUSIONS", { x: 0.9, y: 0.7, w: 11.5, h: 0.5,
  fontFace: BODY, fontSize: 14, bold: true, color: SKY, charSpacing: 3, margin: 0 });
s.addText([
  { text: "Persistence dominates the global daily-mean series — lag-1 R² ≈ 0.77; SARIMAX/ensemble track closely.", options: { bullet: true, breakLine: true } },
  { text: "Geography is the master variable — latitude alone is the top temperature driver (0.83).", options: { bullet: true, breakLine: true } },
  { text: "Weather shapes air quality — humidity is the strongest pollutant correlate (ozone −0.40).", options: { bullet: true, breakLine: true } },
  { text: "Raw data is already complete (0 missing) — cleaning centres on de-duplication, IQR capping & features.", options: { bullet: true, breakLine: true } },
  { text: "Anomaly detection isolates 2.0% of records for data-quality / extreme-event review.", options: { bullet: true } },
], { x: 1.0, y: 1.4, w: 8.0, h: 3.6, fontFace: BODY, fontSize: 15, color: "EAF1FB", margin: 0, paraSpaceAfter: 10, lineSpacingMultiple: 1.05 });

// Requirements completed card
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.2, y: 1.4, w: 3.3, h: 3.9,
  fill: { color: "27306B" }, line: { color: "3A4490", width: 1 }, rectRadius: 0.08 });
s.addText("Requirements ✓", { x: 9.45, y: 1.55, w: 2.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: WHITE, margin: 0 });
s.addText([
  "Cleaning · outliers · normalize",
  "EDA + temp/precip visuals",
  "Forecast + metrics (last_updated)",
  "Anomaly detection",
  "Multiple models + ensemble",
  "Climate analysis",
  "Air-quality impact",
  "Feature importance",
  "Spatial + geographical",
].map((t, i, a) => ({ text: "✓  " + t, options: { breakLine: i < a.length - 1, color: "CADCFC", fontSize: 11.5 } })),
  { x: 9.45, y: 2.05, w: 2.85, h: 3.1, fontFace: BODY, valign: "top", margin: 0, paraSpaceAfter: 5 });

s.addText("Full report: report/REPORT.md   ·   Interactive dashboard: streamlit run dashboard.py   ·   pmaccelerator.io",
  { x: 0.9, y: 6.7, w: 11.5, h: 0.4, fontFace: BODY, fontSize: 12, color: "9FB0D0", margin: 0 });

pres.writeFile({ fileName: "report/Weather_Trend_Forecasting.pptx" }).then(f => console.log("WROTE", f));
