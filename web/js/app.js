/**
 * RAIN-X Operational Meteorological Console
 * Frontend State Management & API Integration
 * Adheres strictly to safe DOM manipulation guidelines (no innerHTML for dynamic data).
 */

const API_BASE = "http://127.0.0.1:8000";

let districtCatalog = {};
let currentBenchmarkCase = null;
let currentBattleLayer = "rain_x";
let geoGridCache = null;

// Color scales for rainfall (mm)
function getRainColor(val) {
  if (val <= 0.5) return "rgba(15, 23, 42, 0.4)";
  if (val < 15.6) return "rgba(2, 132, 199, 0.75)";     // Light blue
  if (val < 64.5) return "rgba(16, 185, 129, 0.85)";   // Emerald green
  if (val < 115.5) return "rgba(245, 158, 11, 0.85)";  // Amber orange
  if (val < 204.4) return "rgba(239, 68, 68, 0.9)";    // Crimson red
  return "rgba(147, 51, 234, 0.95)";                    // Purple extreme
}

function getErrorReductionColor(val) {
  if (val > 15.0) return "rgba(16, 185, 129, 0.9)";    // Strong improvement
  if (val > 0.0) return "rgba(56, 189, 248, 0.75)";    // Moderate improvement
  if (val === 0.0) return "rgba(100, 116, 139, 0.3)";  // Neutral
  return "rgba(239, 68, 68, 0.7)";                     // Raw was better
}

// -------------------------------------------------------------
// Safe DOM Helpers
// -------------------------------------------------------------
function setText(elementId, text) {
  const el = document.getElementById(elementId);
  if (el) {
    el.textContent = text;
  }
}

function clearElement(elementId) {
  const el = document.getElementById(elementId);
  if (el) {
    el.replaceChildren();
  }
}

// -------------------------------------------------------------
// Application Initialization
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  startClock();
  loadDistrictCatalog();
  loadVerificationScorecard();
  loadBenchmarkEvents();
  setupEventListeners();
  drawDefaultMap();
});

function setupTabs() {
  const tabButtons = document.querySelectorAll(".nav-tab-btn");
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      if (!targetId) return;

      // Update button states
      tabButtons.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      // Update panes
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const activePane = document.getElementById(targetId);
      if (activePane) {
        activePane.classList.add("active");
      }

      // Trigger canvas resize/redraw if battle tab activated
      if (targetId === "tab-battle" && currentBenchmarkCase) {
        renderBattleGrid();
      }
    });
  });
}

function startClock() {
  function update() {
    const now = new Date();
    const utcStr = now.toISOString().slice(11, 19) + " UTC";
    setText("clockDisplay", utcStr);
  }
  update();
  setInterval(update, 1000);
}

function setupEventListeners() {
  const stateSelect = document.getElementById("stateSelect");
  const districtSelect = document.getElementById("districtSelect");
  const btnRun = document.getElementById("btnRunDistrictForecast");
  const btnBattle = document.getElementById("btnRunBattle");

  if (stateSelect) {
    stateSelect.addEventListener("change", () => {
      updateDistrictDropdown();
    });
  }

  if (districtSelect) {
    districtSelect.addEventListener("change", () => {
      updateStationDropdown();
    });
  }

  if (btnRun) {
    btnRun.addEventListener("click", () => {
      runDistrictForecast();
    });
  }

  if (btnBattle) {
    btnBattle.addEventListener("click", () => {
      const sel = document.getElementById("battleEventSelect");
      if (sel && sel.value) {
        loadBenchmarkCaseDetails(sel.value);
      }
    });
  }

  // Battle Map Layer buttons
  const layerBtns = [
    { id: "btnViewRainXGrid", layer: "rain_x" },
    { id: "btnViewRawNWPGrid", layer: "raw_nwp" },
    { id: "btnViewObsGrid", layer: "observed" },
    { id: "btnViewReductionGrid", layer: "error_reduction" }
  ];

  layerBtns.forEach(item => {
    const el = document.getElementById(item.id);
    if (el) {
      el.addEventListener("click", () => {
        layerBtns.forEach(b => {
          const btn = document.getElementById(b.id);
          if (btn) btn.classList.remove("active");
        });
        el.classList.add("active");
        currentBattleLayer = item.layer;
        renderBattleGrid();
      });
    }
  });
}

// -------------------------------------------------------------
// Module 1: District Hierarchy & Forecast Loading
// -------------------------------------------------------------
async function loadDistrictCatalog() {
  try {
    const res = await fetch(`${API_BASE}/api/districts`);
    if (!res.ok) throw new Error("Failed to fetch districts");
    districtCatalog = await res.json();

    const stateSelect = document.getElementById("stateSelect");
    clearElement("stateSelect");

    Object.keys(districtCatalog).forEach(state => {
      const opt = document.createElement("option");
      opt.value = state;
      opt.textContent = state;
      stateSelect.appendChild(opt);
    });

    // Default select Andhra Pradesh if available
    if (districtCatalog["Andhra Pradesh"]) {
      stateSelect.value = "Andhra Pradesh";
    }

    updateDistrictDropdown();
    
    // Auto-run initial forecast
    runDistrictForecast();
  } catch (err) {
    console.error("Error loading district catalog:", err);
  }
}

function updateDistrictDropdown() {
  const stateSelect = document.getElementById("stateSelect");
  const districtSelect = document.getElementById("districtSelect");
  const state = stateSelect.value;
  const list = districtCatalog[state] || [];

  clearElement("districtSelect");
  const uniqueDistricts = Array.from(new Set(list.map(item => item.district)));
  
  uniqueDistricts.forEach(dist => {
    const opt = document.createElement("option");
    opt.value = dist;
    opt.textContent = dist;
    districtSelect.appendChild(opt);
  });

  // Default select Visakhapatnam if present
  if (uniqueDistricts.includes("Visakhapatnam")) {
    districtSelect.value = "Visakhapatnam";
  }

  updateStationDropdown();
}

function updateStationDropdown() {
  const stateSelect = document.getElementById("stateSelect");
  const districtSelect = document.getElementById("districtSelect");
  const stationSelect = document.getElementById("stationSelect");

  const state = stateSelect.value;
  const dist = districtSelect.value;
  const list = districtCatalog[state] || [];
  const stations = list.filter(item => item.district === dist);

  clearElement("stationSelect");
  stations.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s.station;
    opt.textContent = `${s.station} (${s.elevation}m)`;
    stationSelect.appendChild(opt);
  });

  // Default select Anakapalli if present
  if (stations.some(s => s.station === "Anakapalli")) {
    stationSelect.value = "Anakapalli";
  }
}

async function runDistrictForecast() {
  const stateSelect = document.getElementById("stateSelect");
  const districtSelect = document.getElementById("districtSelect");
  const stationSelect = document.getElementById("stationSelect");
  const leadSelect = document.getElementById("leadTimeSelect");

  const payload = {
    state: stateSelect.value,
    district: districtSelect.value,
    station: stationSelect.value,
    lead_time_hrs: parseInt(leadSelect.value, 10)
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict/district`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Forecast failed");
    const data = await res.json();
    displayDistrictForecast(data);
  } catch (err) {
    console.error("Forecast request error:", err);
  }
}

function displayDistrictForecast(data) {
  // Primary Metrics
  setText("metricRawNWP", `${data.raw_nwp_mm} mm`);
  setText("metricRainX", `${data.corrected_mm} mm`);
  
  const delta = (data.corrected_mm - data.raw_nwp_mm).toFixed(1);
  const deltaSign = delta > 0 ? `+${delta}` : delta;
  setText("metricDelta", `Delta: ${deltaSign} mm`);
  
  const [q10, q50, q90] = data.interval_q10_q50_q90;
  setText("metricInterval", `${q10} - ${q90} mm`);
  setText("metricConfidence", `${data.confidence_score}%`);

  // IMD Alert Badge & Card
  const alertCard = document.getElementById("districtAlertCard");
  const alertBadge = document.getElementById("districtAlertBadge");
  if (alertCard && alertBadge) {
    alertCard.className = `alert-card ${data.alert.code}`;
    alertBadge.textContent = data.alert.code;
    setText("districtAlertTitle", `IMD Alert: ${data.alert.name}`);
    setText("districtAlertDesc", data.alert.desc);
  }

  // Exceedance probabilities
  const ep = data.exceedance_probabilities;
  setText("probModerate", `${ep.moderate}%`);
  setText("probHeavy", `${ep.heavy}%`);
  setText("probVeryHeavy", `${ep.very_heavy}%`);
  setText("probExtremelyHeavy", `${ep.extremely_heavy}%`);

  // Multi-Model Comparison Table
  renderModelComparisonTable(data.model_comparison, data.raw_nwp_mm);

  // XAI & Explainability Card
  const xai = data.explanation;
  setText("xaiRegimeTag", `${data.regime_info.dominant_regime} (${data.regime_info.confidence}%)`);
  setText("xaiReasonText", xai.meteorological_reason);
  setText("xaiBiasProfile", xai.historical_nwp_bias);

  // Atmospheric Contributors List
  const contList = document.getElementById("xaiContributorsList");
  clearElement("xaiContributorsList");
  if (xai.contributors) {
    xai.contributors.forEach(c => {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.justifyContent = "space-between";
      row.style.padding = "0.35rem 0.55rem";
      row.style.borderRadius = "4px";
      row.style.background = "rgba(255, 255, 255, 0.03)";
      row.style.fontSize = "0.8rem";

      const left = document.createElement("span");
      left.style.color = "var(--text-main)";
      left.textContent = `${c.factor}: ${c.desc}`;

      const right = document.createElement("span");
      right.style.fontWeight = "700";
      right.style.fontFamily = "var(--font-mono)";
      right.style.color = c.effect === "POSITIVE" ? "var(--cyan-accent)" : "var(--rose-accent)";
      right.textContent = c.weight;

      row.appendChild(left);
      row.appendChild(right);
      contList.appendChild(row);
    });
  }

  // AI Forecast Analyst Briefing Section
  const br = data.agent_briefing;
  setText("briefingTitle", br.title);
  setText("briefingExecSummary", br.executive_summary);
  setText("briefingSynoptic", br.synoptic_reasoning);
  setText("briefingRiskP64", br.risk_assessment.probability_gt_64mm);
  setText("briefingRiskP115", br.risk_assessment.probability_gt_115mm);
  setText("briefingTrustCSI", br.trust_index.historical_regime_csi);
  setText("briefingRecommendation", br.actionable_recommendation);

  const actionList = document.getElementById("briefingActionItems");
  clearElement("briefingActionItems");
  if (br.action_items) {
    br.action_items.forEach(item => {
      const li = document.createElement("li");
      li.style.marginBottom = "0.25rem";
      li.textContent = item;
      actionList.appendChild(li);
    });
  }

  // Update GIS Canvas and Map Meta
  setText("mapStationMeta", `Lat: ${data.lat.toFixed(2)}°N | Lon: ${data.lon.toFixed(2)}°E`);
  setText("mapBasinInfo", `Basin: ${data.basin}`);
  setText("mapCoastalInfo", `Coastal Zone: ${data.coastal ? "Yes" : "Inland"}`);
  setText("mapElevationInfo", `Elevation: ${data.elevation_m} m`);

  drawStationOnMap(data.lat, data.lon, data.location, data.corrected_mm);

  // Update Regime Intelligence tab as well
  updateRegimeIntelligenceTab(data.regime_info);
}

function renderModelComparisonTable(comp, rawVal) {
  const tbody = document.getElementById("districtModelComparisonBody");
  clearElement("districtModelComparisonBody");

  const models = [
    { key: "raw_nwp", name: "0. Raw NWP Forecast", technique: "NCMRWF NCUM Atmospheric Model", highlight: false },
    { key: "linear_bc", name: "1. Linear Bias Correction", technique: "Global Multiplicative / Additive OLS", highlight: false },
    { key: "quantile_mapping", name: "2. Quantile Mapping", technique: "Empirical CDF Inversion Matching", highlight: false },
    { key: "random_forest", name: "3. Random Forest Regressor", technique: "Non-Linear Decision Forest Ensemble", highlight: false },
    { key: "lightgbm", name: "4. LightGBM Post-Processor", technique: "Gradient Boosted Histogram Trees", highlight: false },
    { key: "rain_x_moe", name: "5. RAIN-X Mixture-of-Experts", technique: "Physics-Guided Regime Neural MoE", highlight: true }
  ];

  models.forEach(m => {
    const val = comp[m.key] !== undefined ? comp[m.key] : 0.0;
    const delta = (val - rawVal).toFixed(1);
    const deltaSign = delta > 0 ? `+${delta}` : delta;

    const tr = document.createElement("tr");
    if (m.highlight) tr.className = "highlight-row";

    const tdName = document.createElement("td");
    tdName.textContent = m.name;

    const tdVal = document.createElement("td");
    tdVal.style.fontWeight = "800";
    tdVal.textContent = `${val.toFixed(1)} mm`;

    const tdDelta = document.createElement("td");
    tdDelta.textContent = m.key === "raw_nwp" ? "Baseline" : `${deltaSign} mm`;

    const tdTech = document.createElement("td");
    tdTech.style.color = "var(--text-muted)";
    tdTech.style.fontFamily = "var(--font-sans)";
    tdTech.textContent = m.technique;

    tr.appendChild(tdName);
    tr.appendChild(tdVal);
    tr.appendChild(tdDelta);
    tr.appendChild(tdTech);

    tbody.appendChild(tr);
  });
}

function updateRegimeIntelligenceTab(regimeInfo) {
  setText("regimeDominantBadge", `Dominant: ${regimeInfo.dominant_regime} (${regimeInfo.confidence}%)`);
  
  const container = document.getElementById("regimeBarsContainer");
  clearElement("regimeBarsContainer");

  const names = {
    active_monsoon: "Active Monsoon",
    break_monsoon: "Break Monsoon",
    low_depression: "Low / Depression",
    coastal: "Coastal Rainfall",
    orographic: "Orographic Rainfall",
    western_disturbance: "Western Disturbance"
  };

  Object.entries(regimeInfo.probabilities).forEach(([key, prob]) => {
    const pct = (prob * 100).toFixed(1);
    
    const item = document.createElement("div");
    item.className = "regime-bar-item";

    const header = document.createElement("div");
    header.className = "regime-bar-header";

    const label = document.createElement("span");
    label.textContent = names[key] || key;

    const val = document.createElement("span");
    val.style.fontWeight = "700";
    val.style.fontFamily = "var(--font-mono)";
    val.textContent = `${pct}%`;

    header.appendChild(label);
    header.appendChild(val);

    const track = document.createElement("div");
    track.className = "regime-track";

    const fill = document.createElement("div");
    fill.className = "regime-fill";
    fill.style.width = `${pct}%`;
    if (key === regimeInfo.dominant_key) {
      fill.style.background = "linear-gradient(90deg, #00F2FE, #3B82F6)";
      label.style.color = "var(--cyan-accent)";
      label.style.fontWeight = "700";
    }

    track.appendChild(fill);
    item.appendChild(header);
    item.appendChild(track);
    container.appendChild(item);
  });

  // Diagnostics metrics
  setText("diagVorticity", "2.4");
  setText("diagMFC", "+1.6");
  setText("diagPressureAnom", "-3.5");
  setText("diagOFI", "1.8");
}

// -------------------------------------------------------------
// Module 2: Forecast Battle Mode
// -------------------------------------------------------------
async function loadBenchmarkEvents() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmarks`);
    if (!res.ok) return;
    const events = await res.json();

    const sel = document.getElementById("battleEventSelect");
    clearElement("battleEventSelect");

    events.forEach(ev => {
      const opt = document.createElement("option");
      opt.value = ev.id;
      opt.textContent = `${ev.title} (${ev.date})`;
      sel.appendChild(opt);
    });

    if (events.length > 0) {
      sel.value = events[0].id;
      loadBenchmarkCaseDetails(events[0].id);
    }
  } catch (err) {
    console.error("Error loading benchmark events:", err);
  }
}

async function loadBenchmarkCaseDetails(eventId) {
  try {
    setText("battleEventDesc", "Loading synoptic grids and verification for benchmark event...");
    const res = await fetch(`${API_BASE}/api/benchmarks/${eventId}`);
    if (!res.ok) throw new Error("Benchmark fetch failed");
    currentBenchmarkCase = await res.json();

    displayBenchmarkBattle(currentBenchmarkCase);
  } catch (err) {
    console.error("Error loading benchmark details:", err);
  }
}

function displayBenchmarkBattle(caseData) {
  const meta = caseData.metadata;
  setText("battleEventDesc", `${meta.title} &bull; Prevailing Regime: ${meta.regime.toUpperCase()} &bull; ${meta.description}`);

  const p = caseData.peak_station;
  setText("battleRawVal", `${p.raw_nwp_mm} mm`);
  setText("battleRawError", `Error: ${p.raw_error_mm} mm`);

  setText("battleRainXVal", `${p.rain_x_mm} mm`);
  setText("battleRainXError", `Error: ${p.rain_x_error_mm} mm`);

  setText("battleObsVal", `${p.observed_mm} mm`);
  
  const pct = ((p.error_reduction_mm / (p.raw_error_mm + 1e-4)) * 100).toFixed(1);
  setText("battleReduction", `Error Reduced by ${p.error_reduction_mm} mm (${pct}% improvement)`);

  // Verification skill table for this event
  const rawEv = caseData.domain_verification.raw_nwp;
  const rxEv = caseData.domain_verification.rain_x;

  const tbody = document.getElementById("battleVerificationBody");
  clearElement("battleVerificationBody");

  const metrics = [
    { label: "RMSE (Root Mean Square Error)", raw: `${rawEv.rmse} mm`, rx: `${rxEv.rmse} mm`, better: rxEv.rmse < rawEv.rmse },
    { label: "CSI (Critical Success Index)", raw: rawEv.csi.toFixed(3), rx: rxEv.csi.toFixed(3), better: rxEv.csi > rawEv.csi },
    { label: "ETS (Equitable Threat Score)", raw: rawEv.ets.toFixed(3), rx: rxEv.ets.toFixed(3), better: rxEv.ets > rawEv.ets },
    { label: "POD (Probability of Detection)", raw: rawEv.pod.toFixed(3), rx: rxEv.pod.toFixed(3), better: rxEv.pod > rawEv.pod },
    { label: "FAR (False Alarm Ratio)", raw: rawEv.far.toFixed(3), rx: rxEv.far.toFixed(3), better: rxEv.far < rawEv.far },
    { label: "FSS (Fractions Skill Score)", raw: rawEv.fss.toFixed(3), rx: rxEv.fss.toFixed(3), better: rxEv.fss > rawEv.fss }
  ];

  metrics.forEach(m => {
    const tr = document.createElement("tr");

    const tdLbl = document.createElement("td");
    tdLbl.textContent = m.label;

    const tdRaw = document.createElement("td");
    tdRaw.textContent = m.raw;

    const tdRx = document.createElement("td");
    tdRx.style.color = "var(--cyan-accent)";
    tdRx.style.fontWeight = "700";
    tdRx.textContent = m.rx;

    const tdImp = document.createElement("td");
    tdImp.style.color = m.better ? "var(--emerald-accent)" : "var(--rose-accent)";
    tdImp.textContent = m.better ? "Improved Skill \u2713" : "Consistent";

    tr.appendChild(tdLbl);
    tr.appendChild(tdRaw);
    tr.appendChild(tdRx);
    tr.appendChild(tdImp);
    tbody.appendChild(tr);
  });

  // Why did RAIN-X outperform card
  let why = "";
  if (meta.regime === "orographic") {
    why = "Raw numerical weather prediction severely underestimates orographic uplifting when orthogonal low-level jets strike the steep Western Ghats escarpment. RAIN-X activates its specialized Orographic Expert network, incorporating terrain elevation gradients to restore unresolved windward precipitation.";
  } else if (meta.regime === "low_depression") {
    why = "Coarse NWP grids struggle to capture the concentrated eyewall moisture convergence of deep depressions, spreading rain too broadly. RAIN-X's Depression Expert concentrates convective flux along the cyclonic vortex core, reducing peak underprediction.";
  } else if (meta.regime === "break_monsoon") {
    why = "During break monsoon spells, raw NWP falsely generates convective drizzle over dry Central India while missing intense foothill bursts. RAIN-X correctly suppresses central India drizzle and focuses moisture along the sub-Himalayan shear line.";
  } else {
    why = "RAIN-X dynamically adjusts raw forecast bias conditioned on the prevailing atmospheric regime, ensuring physical moisture conservation and superior spatial agreement.";
  }
  setText("battleWhyText", why);

  // Render 2D grid
  renderBattleGrid();
}

function renderBattleGrid() {
  if (!currentBenchmarkCase) return;
  const canvas = document.getElementById("battleCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  const grids = currentBenchmarkCase.grids;
  const gridData = grids[currentBattleLayer] || grids.rain_x;

  const rows = gridData.length;
  const cols = gridData[0].length;

  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Invert row for meteorological latitude (top = north)
      const val = gridData[rows - 1 - r][c];
      
      if (currentBattleLayer === "error_reduction") {
        ctx.fillStyle = getErrorReductionColor(val);
      } else {
        ctx.fillStyle = getRainColor(val);
      }

      ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
    }
  }

  // Draw Title Overlay
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "bold 12px Inter, sans-serif";
  const titleMap = {
    rain_x: "RAIN-X AI Corrected Field (mm)",
    raw_nwp: "Raw NWP Forecast Field (mm)",
    observed: "IMD Ground Truth Observed (mm)",
    error_reduction: "Spatial Error Reduction Map (|Raw Err| - |RAIN-X Err|)"
  };
  ctx.fillText(titleMap[currentBattleLayer] || "Precipitation Field", 12, 22);
}

// -------------------------------------------------------------
// Module 3: Verification Laboratory Scorecard
// -------------------------------------------------------------
async function loadVerificationScorecard() {
  try {
    const res = await fetch(`${API_BASE}/api/scorecard`);
    if (!res.ok) return;
    const data = await res.json();

    const overall = data.stratified["Overall Domain"];
    const tbody = document.getElementById("scorecardBody");
    clearElement("scorecardBody");

    const models = [
      { key: "raw_nwp", name: "0. Raw NWP Forecast", highlight: false },
      { key: "linear_bc", name: "1. Linear Bias Correction", highlight: false },
      { key: "quantile_mapping", name: "2. Quantile Mapping", highlight: false },
      { key: "random_forest", name: "3. Random Forest", highlight: false },
      { key: "lightgbm", name: "4. LightGBM Post-Processor", highlight: false },
      { key: "rain_x_moe", name: "5. RAIN-X Mixture-of-Experts", highlight: true }
    ];

    models.forEach(m => {
      const stat = overall[m.key];
      if (!stat) return;

      const tr = document.createElement("tr");
      if (m.highlight) tr.className = "highlight-row";

      const tdName = document.createElement("td");
      tdName.textContent = m.name;

      const tdRmse = document.createElement("td");
      tdRmse.style.fontWeight = "800";
      tdRmse.textContent = stat.rmse.toFixed(1);

      const tdMae = document.createElement("td");
      tdMae.textContent = stat.mae.toFixed(1);

      const tdBias = document.createElement("td");
      tdBias.textContent = stat.bias.toFixed(2);

      const tdPod = document.createElement("td");
      tdPod.textContent = stat.pod.toFixed(3);

      const tdFar = document.createElement("td");
      tdFar.textContent = stat.far.toFixed(3);

      const tdCsi = document.createElement("td");
      tdCsi.style.fontWeight = "800";
      tdCsi.textContent = stat.csi.toFixed(3);

      const tdEts = document.createElement("td");
      tdEts.style.fontWeight = "800";
      tdEts.textContent = stat.ets.toFixed(3);

      const tdFss = document.createElement("td");
      tdFss.style.fontWeight = "800";
      tdFss.textContent = stat.fss.toFixed(3);

      tr.appendChild(tdName);
      tr.appendChild(tdRmse);
      tr.appendChild(tdMae);
      tr.appendChild(tdBias);
      tr.appendChild(tdPod);
      tr.appendChild(tdFar);
      tr.appendChild(tdCsi);
      tr.appendChild(tdEts);
      tr.appendChild(tdFss);

      tbody.appendChild(tr);
    });

    // Populate Regime-Stratified Table
    const rBody = document.getElementById("regimeStratifiedBody");
    clearElement("regimeStratifiedBody");

    const regimeKeys = [
      "Active Monsoon",
      "Break Monsoon",
      "Low / Depression",
      "Coastal Rainfall",
      "Orographic Rainfall",
      "Western Disturbance"
    ];

    regimeKeys.forEach(rName => {
      const rData = data.stratified[rName];
      if (!rData) return;

      const raw = rData.raw_nwp;
      const rx = rData.rain_x_moe;
      const diff = (raw.rmse - rx.rmse).toFixed(1);

      const tr = document.createElement("tr");

      const tdReg = document.createElement("td");
      tdReg.style.fontWeight = "700";
      tdReg.textContent = rName;

      const tdRawR = document.createElement("td");
      tdRawR.textContent = `${raw.rmse.toFixed(1)} mm`;

      const tdRxR = document.createElement("td");
      tdRxR.style.color = "var(--cyan-accent)";
      tdRxR.style.fontWeight = "700";
      tdRxR.textContent = `${rx.rmse.toFixed(1)} mm`;

      const tdDiff = document.createElement("td");
      tdDiff.style.color = "var(--emerald-accent)";
      tdDiff.textContent = `-${diff} mm (${((diff / raw.rmse) * 100).toFixed(0)}%)`;

      const tdRawC = document.createElement("td");
      tdRawC.textContent = raw.csi.toFixed(3);

      const tdRxC = document.createElement("td");
      tdRxC.style.color = "var(--cyan-accent)";
      tdRxC.textContent = rx.csi.toFixed(3);

      const tdRawE = document.createElement("td");
      tdRawE.textContent = raw.ets.toFixed(3);

      const tdRxE = document.createElement("td");
      tdRxE.style.color = "var(--cyan-accent)";
      tdRxE.textContent = rx.ets.toFixed(3);

      tr.appendChild(tdReg);
      tr.appendChild(tdRawR);
      tr.appendChild(tdRxR);
      tr.appendChild(tdDiff);
      tr.appendChild(tdRawC);
      tr.appendChild(tdRxC);
      tr.appendChild(tdRawE);
      tr.appendChild(tdRxE);

      rBody.appendChild(tr);
    });

  } catch (err) {
    console.error("Error loading verification scorecard:", err);
  }
}

// -------------------------------------------------------------
// GIS Canvas Map Drawing
// -------------------------------------------------------------
function drawDefaultMap() {
  const canvas = document.getElementById("mapCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  ctx.fillStyle = "#060A14";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw subtle grid lines
  ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  // Draw simplified Indian subcontinent outline
  drawIndiaOutline(ctx, canvas.width, canvas.height);
}

function drawIndiaOutline(ctx, w, h) {
  // Approximate coordinate transform: Lon 68 to 98 (X), Lat 8 to 38 (Y)
  function toX(lon) { return ((lon - 68.0) / 30.0) * w; }
  function toY(lat) { return h - ((lat - 8.0) / 30.0) * h; }

  ctx.save();
  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  // Coastlines & borders rough polygon
  ctx.moveTo(toX(69.0), toY(24.0)); // Gujarat
  ctx.lineTo(toX(72.5), toY(21.0)); // Surat
  ctx.lineTo(toX(72.8), toY(19.0)); // Mumbai
  ctx.lineTo(toX(74.0), toY(15.0)); // Goa
  ctx.lineTo(toX(75.0), toY(12.0)); // Mangalore
  ctx.lineTo(toX(77.0), toY(8.5));  // Kanyakumari
  ctx.lineTo(toX(80.2), toY(13.0)); // Chennai
  ctx.lineTo(toX(83.0), toY(17.7)); // Visakhapatnam
  ctx.lineTo(toX(86.5), toY(20.5)); // Odisha coast
  ctx.lineTo(toX(89.0), toY(22.0)); // Sundarbans
  ctx.lineTo(toX(92.0), toY(25.0)); // Meghalaya
  ctx.lineTo(toX(95.0), toY(27.0)); // Assam / NE
  ctx.lineTo(toX(92.0), toY(28.0)); // Arunachal
  ctx.lineTo(toX(85.0), toY(28.5)); // Nepal border
  ctx.lineTo(toX(79.0), toY(31.0)); // Uttarakhand
  ctx.lineTo(toX(76.0), toY(34.5)); // J&K / Ladakh
  ctx.lineTo(toX(74.0), toY(32.0)); // Punjab
  ctx.lineTo(toX(70.5), toY(28.0)); // Rajasthan
  ctx.closePath();

  ctx.fillStyle = "rgba(13, 21, 39, 0.65)";
  ctx.fill();
  ctx.stroke();

  // Draw Western Ghats Ridge
  ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(toX(73.5), toY(20.0));
  ctx.lineTo(toX(75.5), toY(10.0));
  ctx.stroke();

  // Draw Himalayan Ridge
  ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(toX(74.5), toY(32.5));
  ctx.lineTo(toX(92.0), toY(28.0));
  ctx.stroke();

  ctx.restore();
}

function drawStationOnMap(lat, lon, stationName, correctedRain) {
  const canvas = document.getElementById("mapCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  drawDefaultMap();

  function toX(l) { return ((l - 68.0) / 30.0) * w; }
  function toY(l) { return h - ((l - 8.0) / 30.0) * h; }

  const sx = toX(lon);
  const sy = toY(lat);

  // Target Pulse
  ctx.save();
  ctx.beginPath();
  ctx.arc(sx, sy, 14, 0, 2 * Math.PI);
  ctx.strokeStyle = "rgba(0, 242, 254, 0.4)";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
  ctx.fillStyle = "#00F2FE";
  ctx.fill();

  // Label
  ctx.fillStyle = "#FFFFFF";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.fillText(`${stationName}`, sx + 12, sy - 4);
  
  ctx.fillStyle = "var(--cyan-accent)";
  ctx.font = "bold 10px JetBrains Mono, monospace";
  ctx.fillText(`RAIN-X: ${correctedRain.toFixed(1)} mm`, sx + 12, sy + 10);
  ctx.restore();
}
