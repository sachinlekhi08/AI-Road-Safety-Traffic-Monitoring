/**
 * dashboard/static/js/app.js
 * ==========================
 * Front-end logic for the AI Road Safety & Traffic Monitoring dashboard.
 *
 * Responsibilities:
 *  - File upload (drag-and-drop + browse)
 *  - Triggering analysis (uploaded or built-in video)
 *  - Polling /status/<id> and updating the progress banner
 *  - Loading and displaying results (summary cards, charts, video player)
 *  - History table
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let currentAnalysisId = null;
let pollingTimer       = null;
let pieChart           = null;
let lineChart          = null;
let densityChart       = null;

// ---------------------------------------------------------------------------
// On load
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  loadBuiltinVideos();
  loadHistory();
  setupDropZone();
  setupFileInput();

  document.getElementById("btnAnalyze").addEventListener("click", startAnalysis);
});

// ---------------------------------------------------------------------------
// Built-in video list
// ---------------------------------------------------------------------------
async function loadBuiltinVideos() {
  try {
    const resp = await fetch("/videos");
    const videos = await resp.json();
    const container = document.getElementById("builtinVideoList");
    if (!videos.length) {
      container.innerHTML = '<span class="text-muted">No videos found in data/videos/</span>';
      return;
    }
    container.innerHTML = videos.map(v => `
      <button class="list-group-item list-group-item-action builtin-item px-3 py-2"
              data-filename="${v}" onclick="selectBuiltinVideo(this, '${v}')">
        <i class="bi bi-film me-2 text-secondary"></i>${v}
      </button>
    `).join("");
  } catch (e) {
    console.error("Failed to load built-in videos:", e);
  }
}

function selectBuiltinVideo(el, filename) {
  // Deselect all, select clicked
  document.querySelectorAll(".builtin-item").forEach(b => b.classList.remove("selected"));
  el.classList.add("selected");

  // Store intent
  el.closest(".card").dataset.selectedFilename = filename;
  document.getElementById("btnAnalyze").disabled = false;
  document.getElementById("uploadedFilename").textContent = `Selected: ${filename}`;

  // Clear any uploaded file state
  document.getElementById("videoFile").value = "";
  window._uploadedAnalysisId = null;
  window._builtinFilename = filename;
}

// ---------------------------------------------------------------------------
// File upload: drag-and-drop + browse
// ---------------------------------------------------------------------------
function setupDropZone() {
  const zone = document.getElementById("dropZone");
  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length) handleFileSelected(files[0]);
  });
  // Click anywhere on zone to open file picker
  zone.addEventListener("click", () => document.getElementById("videoFile").click());
}

function setupFileInput() {
  document.getElementById("videoFile").addEventListener("change", e => {
    if (e.target.files.length) handleFileSelected(e.target.files[0]);
  });
}

function handleFileSelected(file) {
  window._builtinFilename = null;
  // Deselect built-in
  document.querySelectorAll(".builtin-item").forEach(b => b.classList.remove("selected"));

  document.getElementById("uploadedFilename").textContent = `Selected: ${file.name}`;
  document.getElementById("btnAnalyze").disabled = false;

  // Upload immediately
  uploadFile(file);
}

async function uploadFile(file) {
  const bar        = document.getElementById("uploadBar");
  const progress   = document.getElementById("uploadProgress");
  const statusTxt  = document.getElementById("uploadStatusText");
  const btnAnalyze = document.getElementById("btnAnalyze");

  progress.classList.remove("d-none");
  bar.style.width = "10%";
  statusTxt.textContent = "Uploading…";
  btnAnalyze.disabled = true;

  const formData = new FormData();
  formData.append("video", file);

    try {
    const resp = await fetch("/upload", { method: "POST", body: formData });

    // Guest — Flask redirected this request to the login page.
    if (
      resp.redirected &&
      new URL(resp.url, window.location.origin).pathname === "/login"
    ) {
      window.location.href = resp.url;
      return;
    }

    const data = await resp.json();
    if (resp.ok) {
      window._uploadedAnalysisId = data.analysis_id;
      bar.style.width = "100%";
      statusTxt.textContent = "Upload complete ✓";
      btnAnalyze.disabled = false;
    } else {
      showError(data.error || "Upload failed");
      progress.classList.add("d-none");
    }
  } catch (e) {
    showError("Network error during upload");
    progress.classList.add("d-none");
  }
}

// ---------------------------------------------------------------------------
// Start analysis
// ---------------------------------------------------------------------------
async function startAnalysis() {
  const btnAnalyze = document.getElementById("btnAnalyze");

  // Guard against double-clicks / repeated submits while a request is in flight.
  if (btnAnalyze.disabled) return;

  let endpoint, body;

  if (window._builtinFilename) {
    // Built-in video
    endpoint = "/analyze_existing";
    body = JSON.stringify({
      filename: window._builtinFilename
    });
  } else if (window._uploadedAnalysisId) {
    // Uploaded video
    endpoint = `/analyze/${window._uploadedAnalysisId}`;
    body = JSON.stringify({});
  } else {
    showError("Please select or upload a video first.");
    return;
  }

  btnAnalyze.disabled = true;
  showStatusBanner("Starting analysis…", 0);
  document.getElementById("summarySection").classList.add("d-none");

  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body,
    });

    // User is not logged in.
    // Flask redirected the request to the login page.
    if (
      resp.redirected &&
      new URL(resp.url, window.location.origin).pathname === "/login"
    ) {
      window.location.href = resp.url;
      return;
    }

    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || "Failed to start analysis");
      btnAnalyze.disabled = false;
      return;
    }

    currentAnalysisId = data.analysis_id;
    startPolling(currentAnalysisId);
    // Left disabled on purpose — re-enabled in pollStatus() once the run finishes.

  } catch (e) {
    showError("Network error: " + e.message);
    btnAnalyze.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Status polling
// ---------------------------------------------------------------------------
function startPolling(id) {
  if (pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(() => pollStatus(id), 2000);
}

async function pollStatus(id) {
  try {
    const resp = await fetch(`/status/${id}`);
    const data = await resp.json();

    const pct    = data.progress || 0;
    const status = data.status   || "unknown";

    showStatusBanner(`Processing… (${pct}%)`, pct);

    if (status === "completed") {
      clearInterval(pollingTimer);
      document.getElementById("btnAnalyze").disabled = false;
      showStatusBanner("Analysis complete! Loading results…", 100);
      await loadResults(id);
      loadHistory();
    } else if (status === "failed") {
      clearInterval(pollingTimer);
      document.getElementById("btnAnalyze").disabled = false;
      showError("Processing failed: " + (data.error || "unknown error").split("\n")[0]);
    }
  } catch (e) {
    console.warn("Polling error:", e);
  }
}

// ---------------------------------------------------------------------------
// Load and display results
// ---------------------------------------------------------------------------
async function loadResults(id) {
  const resp = await fetch(`/results/${id}`);
  const data = await resp.json();
  if (!resp.ok) { showError("Failed to load results"); return; }

  renderSummaryCards(data);
  renderDetailsPanel(data);
  renderOutputVideo(id);

  // Fetch time-series for charts
  const tsResp = await fetch(`/timeseries/${id}`);
  const ts = await tsResp.json();
  renderCharts(data, ts);

  document.getElementById("summarySection").classList.remove("d-none");
  document.getElementById("statusBanner").classList.add("d-none");
  // Scroll to results
  document.getElementById("summarySection").scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Render summary stat cards
// ---------------------------------------------------------------------------
function renderSummaryCards(data) {
  const density = (data.traffic_density || "LOW").toLowerCase();
  const densityClass = `stat-card-density ${density}`;

  const crossing = data.crossed_vehicles || 0;

  const cards = [
    {
      label: "Unique Detected Vehicles",
      value: data.total_vehicles || 0,
      cls: "stat-card-vehicles",
      icon: "bi-car-front"
    },
    {
      label: "Vehicles Crossing Line",
      value: crossing,
      cls: "stat-card-vehicles",
      icon: "bi-arrow-down-circle"
    },
    {
      label: "Cars",
      value: data.cars || 0,
      cls: "stat-card-cars",
      icon: "bi-car-front-fill"
    },
    {
      label: "Motorcycles",
      value: data.motorcycles || 0,
      cls: "stat-card-motos",
      icon: "bi-bicycle"
    },
    {
      label: "Buses",
      value: data.buses || 0,
      cls: "stat-card-buses",
      icon: "bi-bus-front"
    },
    {
      label: "Trucks",
      value: data.trucks || 0,
      cls: "stat-card-trucks",
      icon: "bi-truck"
    },
    {
      label: "Pedestrians",
      value: data.pedestrians || 0,
      cls: "stat-card-peds",
      icon: "bi-person-walking"
    },
    {
      label: "Traffic Density",
      value: data.traffic_density || "N/A",
      cls: densityClass,
      icon: "bi-speedometer"
    },
  ];

  const container = document.getElementById("summaryCards");

  container.innerHTML = cards.map(c => `
    <div class="col-6 col-md-4 col-lg-3 col-xl-auto flex-fill">
      <div class="stat-card ${c.cls} position-relative shadow-sm">
        <div class="stat-label">${c.label}</div>
        <div class="stat-value">${c.value}</div>
        <i class="bi ${c.icon} stat-icon"></i>
      </div>
    </div>
  `).join("");
}

// ---------------------------------------------------------------------------
// Render details panel
// ---------------------------------------------------------------------------
function renderDetailsPanel(data) {
  const fmtDur = secs => {
    const m = Math.floor(secs / 60), s = Math.floor(secs % 60);
    return `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
  };
  document.getElementById("detailsPanel").innerHTML = `
    <table class="table table-sm small">
      <tbody>
        <tr><td class="text-muted">Video</td><td><strong>${data.video_filename || "-"}</strong></td></tr>
        <tr><td class="text-muted">Analysed at</td><td>${data.created_at || "-"}</td></tr>
        <tr><td class="text-muted">Duration</td><td>${fmtDur(data.duration_sec || 0)}</td></tr>
        <tr><td class="text-muted">Total Frames</td><td>${data.total_frames || "-"}</td></tr>
        <tr><td class="text-muted">Source FPS</td><td>${(data.fps || 0).toFixed(1)}</td></tr>
        <tr><td class="text-muted">Processing Time</td><td>${(data.processing_time || 0).toFixed(1)} s</td></tr>
        <tr><td class="text-muted">Traffic Density</td>
            <td><span class="badge badge-density-${(data.traffic_density||"low").toLowerCase()}">${data.traffic_density||"N/A"}</span></td></tr>
        <tr><td class="text-muted">Bicycles</td><td>${data.bicycles || 0}</td></tr>
        <tr><td class="text-muted">Traffic Lights</td><td>${data.traffic_lights || 0}</td></tr>
        <tr><td class="text-muted">Vehicles Crossing Line</td><td><strong>${data.crossed_vehicles || 0}</strong></td></tr>
        <tr><td class="text-muted">Cars Crossing</td><td>${data.crossed_cars || 0}</td></tr>
        <tr><td class="text-muted">Motorcycles Crossing</td><td>${data.crossed_motorcycles || 0}</td></tr>
        <tr><td class="text-muted">Buses Crossing</td><td>${data.crossed_buses || 0}</td></tr>
        <tr><td class="text-muted">Trucks Crossing</td><td>${data.crossed_trucks || 0}</td></tr>
      </tbody>
    </table>
  `;
}

// ---------------------------------------------------------------------------
// Render output video
// ---------------------------------------------------------------------------
function renderOutputVideo(id) {
  const vid = document.getElementById("outputVideo");
  vid.src = `/video/${id}?t=${Date.now()}`;
  vid.load();
}

// ---------------------------------------------------------------------------
// Render Charts
// ---------------------------------------------------------------------------
function renderCharts(data, ts) {
  // 1. Pie chart – vehicle distribution
  if (pieChart) pieChart.destroy();
  const pieCtx = document.getElementById("pieChart").getContext("2d");
  pieChart = new Chart(pieCtx, {
    type: "doughnut",
    data: {
      labels: ["Cars", "Motorcycles", "Buses", "Trucks", "Bicycles", "Pedestrians"],
      datasets: [{
        data: [
          data.cars || 0,
          data.motorcycles || 0,
          data.buses || 0,
          data.trucks || 0,
          data.bicycles || 0,
          data.pedestrians || 0,
        ],
        backgroundColor: ["#0dcaf0","#fd7e14","#6f42c1","#dc3545","#20c997","#198754"],
        borderWidth: 2,
        borderColor: "#fff",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { font: { size: 11 } } } },
    },
  });

  // 2. Line chart – vehicle count over time
  if (lineChart) lineChart.destroy();
  const lineCtx = document.getElementById("lineChart").getContext("2d");
  // Sample every 5th data point to avoid chart overload
  const sampled = ts.filter((_, i) => i % 5 === 0);
  lineChart = new Chart(lineCtx, {
    type: "line",
    data: {
      labels: sampled.map(d => `F${d.frame}`),
      datasets: [{
        label: "Vehicles in Frame",
        data: sampled.map(d => d.vehicle_count),
        borderColor: "#0d6efd",
        backgroundColor: "rgba(13,110,253,0.1)",
        borderWidth: 2,
        pointRadius: 2,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxTicksLimit: 10, font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 10 } } },
      },
    },
  });

  // 3. Density over time (bar chart, coloured by density level)
  if (densityChart) densityChart.destroy();
  const densityCtx = document.getElementById("densityChart").getContext("2d");
  const densityColors = sampled.map(d => {
    if (d.density === "HIGH")   return "#dc3545";
    if (d.density === "MEDIUM") return "#fd7e14";
    return "#198754";
  });
  densityChart = new Chart(densityCtx, {
    type: "bar",
    data: {
      labels: sampled.map(d => `F${d.frame}`),
      datasets: [{
        label: "Vehicles in Frame",
        data: sampled.map(d => d.vehicle_count),
        backgroundColor: densityColors,
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: ctx => `Density: ${sampled[ctx.dataIndex]?.density || ""}`,
          },
        },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 12, font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 10 } } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// History table
// ---------------------------------------------------------------------------
async function loadHistory() {
  try {
    const resp = await fetch("/history");
    const rows = await resp.json();
    const tbody = document.getElementById("historyTableBody");
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="11" class="text-center text-muted py-3">No analyses yet.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const den = (r.traffic_density || "LOW").toLowerCase();
      const statusBadge = r.status === "completed"
        ? '<span class="badge bg-success">Completed</span>'
        : r.status === "failed"
        ? '<span class="badge bg-danger">Failed</span>'
        : `<span class="badge bg-warning text-dark">${r.status}</span>`;
      return `
        <tr>
          <td>${r.created_at || "-"}</td>
          <td title="${r.video_filename}">${truncate(r.video_filename, 25)}</td>
          <td>${r.total_vehicles || 0}</td>
          <td>${r.cars || 0}</td>
          <td>${r.motorcycles || 0}</td>
          <td>${r.buses || 0}</td>
          <td>${r.trucks || 0}</td>
          <td>${r.pedestrians || 0}</td>
          <td><span class="badge badge-density-${den}">${r.traffic_density || "-"}</span></td>
          <td>${statusBadge}</td>
          <td>
            ${r.status === "completed"
              ? `<button class="btn btn-sm btn-outline-primary py-0" onclick="viewAnalysis('${r.analysis_id}')">View</button>`
              : ""}
          </td>
        </tr>`;
    }).join("");
  } catch (e) {
    console.error("Failed to load history:", e);
  }
}

async function viewAnalysis(id) {
  await loadResults(id);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
function showStatusBanner(message, pct) {
  const banner = document.getElementById("statusBanner");
  banner.classList.remove("d-none", "alert-danger");
  banner.classList.add("alert-info");
  document.getElementById("statusText").textContent = message;
  document.getElementById("progressBar").style.width = pct + "%";
}

function showError(message) {
  if (pollingTimer) clearInterval(pollingTimer);
  const banner = document.getElementById("statusBanner");
  banner.classList.remove("d-none", "alert-info");
  banner.classList.add("alert-danger");
  document.getElementById("statusText").innerHTML =
    `<i class="bi bi-exclamation-triangle-fill me-2"></i>${message}`;
  document.getElementById("progressBar").style.width = "0%";
}

function truncate(str, max) {
  if (!str) return "-";
  return str.length > max ? str.slice(0, max) + "…" : str;
}

