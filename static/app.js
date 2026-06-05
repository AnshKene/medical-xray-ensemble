/**
 * app.js — Frontend logic for Ensemble X-Ray AI
 * Handles: drag-and-drop, file selection, /predict fetch, result rendering
 */

"use strict";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const btnBrowse       = document.getElementById("btn-browse");
const btnChange       = document.getElementById("btn-change");
const btnAnalyze      = document.getElementById("btn-analyze");

const dropIdleState   = document.getElementById("drop-idle-state");
const dropPreviewState= document.getElementById("drop-preview-state");
const previewImg      = document.getElementById("preview-img");
const previewFilename = document.getElementById("preview-filename");

const loadingOverlay  = document.getElementById("loading-overlay");
const resultsSection  = document.getElementById("results-section");

const statusPill      = document.getElementById("status-pill");
const statusText      = statusPill.querySelector(".status-text");

const resultUploadImg = document.getElementById("result-upload-img");
const verdictBadge    = document.getElementById("verdict-badge");
const confPct         = document.getElementById("conf-pct");
const confBar         = document.getElementById("conf-bar");
const confBarWrap     = document.getElementById("conf-bar-wrap");
const probBars        = document.getElementById("prob-bars");
const weightBars      = document.getElementById("weight-bars");
const gradcamGrid     = document.getElementById("gradcam-grid");
const metaList        = document.getElementById("meta-list");

const errorToast      = document.getElementById("error-toast");
const errorMsg        = document.getElementById("error-msg");
const toastClose      = document.getElementById("toast-close");

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── Utility helpers ───────────────────────────────────────────────────────────

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.classList.remove("hidden");
}

function hideError() {
  errorToast.classList.add("hidden");
}

function setStatus(state, text) {
  statusPill.className = `status-pill status-${state}`;
  statusText.textContent = text;
}

function setLoading(on) {
  loadingOverlay.classList.toggle("hidden", !on);
}

// Animate a loading step indicator (cycles through the 4 model names)
let _stepTimer = null;
function startStepAnimation() {
  const steps = [
    document.getElementById("step-1"),
    document.getElementById("step-2"),
    document.getElementById("step-3"),
    document.getElementById("step-4"),
  ];
  let idx = 0;
  steps.forEach(s => s.classList.remove("active"));
  _stepTimer = setInterval(() => {
    steps.forEach(s => s.classList.remove("active"));
    steps[idx % steps.length].classList.add("active");
    idx++;
  }, 600);
}

function stopStepAnimation() {
  clearInterval(_stepTimer);
  document.querySelectorAll(".step").forEach(s => s.classList.remove("active"));
}

// ── Status check on load ──────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const res  = await fetch("/status");
    const data = await res.json();
    if (data.ready) {
      const gpu = data.device.startsWith("cuda") ? `GPU (${data.device})` : "CPU";
      setStatus("ready", `Ready · ${data.models.length} models · ${gpu}`);
    } else {
      setStatus("error", "Models not loaded");
      showError(data.error || "Models not found — run main.py first.");
      btnAnalyze.disabled = true;
    }
  } catch {
    setStatus("error", "Server unreachable");
  }
}

// ── File selection ────────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Please select an image file (PNG, JPEG, BMP, or WebP).");
    return;
  }

  selectedFile = file;
  hideError();

  // Show preview
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewFilename.textContent = file.name;

  dropIdleState.classList.add("hidden");
  dropPreviewState.classList.remove("hidden");
  btnAnalyze.disabled = false;
}

// Browse button — trigger hidden file input
btnBrowse.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

btnChange.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  dropPreviewState.classList.add("hidden");
  dropIdleState.classList.remove("hidden");
  btnAnalyze.disabled = true;
  resultsSection.classList.add("hidden");
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// Keyboard accessibility for drop zone
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

// ── Drag-and-drop ─────────────────────────────────────────────────────────────
["dragenter", "dragover"].forEach(ev =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("drag-over");
  })
);

["dragleave", "dragend"].forEach(ev =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
  })
);

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove("drag-over");
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

// ── Inference request ─────────────────────────────────────────────────────────
btnAnalyze.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  setLoading(true);
  startStepAnimation();
  resultsSection.classList.add("hidden");
  btnAnalyze.disabled = true;

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);

    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Server error ${res.status}`);
    }

    renderResults(data);
  } catch (err) {
    showError(err.message || "Unexpected error during analysis.");
  } finally {
    stopStepAnimation();
    setLoading(false);
    btnAnalyze.disabled = false;
  }
});

// ── Result rendering ──────────────────────────────────────────────────────────
const CLASS_META = {
  covid:     { label: "COVID-19", fillClass: "fill-covid" },
  normal:    { label: "Normal",   fillClass: "fill-normal" },
  pneumonia: { label: "Pneumonia",fillClass: "fill-pneumonia" },
};

function renderResults(data) {
  // Upload image
  resultUploadImg.src = data.upload_url;
  resultUploadImg.alt = `Analyzed X-ray — predicted: ${data.prediction}`;

  // Verdict badge
  const predKey  = data.prediction.toLowerCase();
  const predMeta = CLASS_META[predKey] || { label: data.prediction, fillClass: "fill-model" };
  verdictBadge.textContent  = predMeta.label;
  verdictBadge.className    = `verdict-badge ${predKey}`;
  verdictBadge.setAttribute("aria-label", `Prediction: ${predMeta.label}`);

  // Confidence bar (animated)
  const pct = Math.round(data.confidence * 100);
  confPct.textContent = `${pct}%`;
  confBarWrap.setAttribute("aria-valuenow", pct);
  confBarWrap.setAttribute("aria-label", `Confidence: ${pct}%`);
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confBar.style.width = `${pct}%`;
    });
  });

  // Probability bars
  probBars.innerHTML = "";
  Object.entries(data.probabilities).forEach(([cls, prob]) => {
    const meta   = CLASS_META[cls.toLowerCase()] || { label: cls, fillClass: "fill-model" };
    const pctVal = Math.round(prob * 100);
    const row    = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">${meta.label}</span>
      <div class="prob-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pctVal}" aria-label="${meta.label} probability">
        <div class="prob-fill ${meta.fillClass}" data-target="${pctVal}" style="width:0%"></div>
      </div>
      <span class="prob-pct">${pctVal}%</span>
    `;
    probBars.appendChild(row);
  });

  // Ensemble weight bars
  weightBars.innerHTML = "";
  Object.entries(data.model_weights).forEach(([model, w]) => {
    const pctVal = Math.round(w * 100);
    const row    = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label">${model}</span>
      <div class="prob-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pctVal}" aria-label="${model} ensemble weight">
        <div class="prob-fill fill-weight" data-target="${pctVal}" style="width:0%"></div>
      </div>
      <span class="prob-pct">${pctVal}%</span>
    `;
    weightBars.appendChild(row);
  });

  // Meta info
  metaList.innerHTML = "";
  const metaItems = [
    ["Prediction",    predMeta.label],
    ["Confidence",    `${pct}%`],
    ["Device",        data.device],
    ["Models loaded", Object.keys(data.model_weights).length],
  ];
  if (data.missing_models?.length) {
    metaItems.push(["Missing models", data.missing_models.join(", ")]);
  }
  metaItems.forEach(([k, v]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="meta-key">${k}</span><span class="meta-val">${v}</span>`;
    metaList.appendChild(li);
  });

  // Grad-CAM grid
  gradcamGrid.innerHTML = "";
  const modelNames = Object.keys(data.model_weights);
  modelNames.forEach(name => {
    const cell = document.createElement("div");
    cell.className = "gradcam-cell";

    if (data.gradcam_urls && data.gradcam_urls[name]) {
      const cacheBust = `?t=${Date.now()}`;
      cell.innerHTML = `
        <img src="${data.gradcam_urls[name]}${cacheBust}"
             alt="Grad-CAM heatmap for ${name}"
             class="gradcam-img"
             loading="lazy" />
        <div class="gradcam-label">${name}</div>
      `;
    } else {
      const errMsg = data.gradcam_errors?.[name] || "Grad-CAM unavailable";
      cell.innerHTML = `
        <div class="gradcam-error">⚠ ${errMsg}</div>
        <div class="gradcam-label">${name}</div>
      `;
    }
    gradcamGrid.appendChild(cell);
  });

  // Show results
  resultsSection.classList.remove("hidden");

  // Animate bars after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll(".prob-fill[data-target]").forEach(el => {
        el.style.width = `${el.dataset.target}%`;
      });
    });
  });

  // Scroll into view
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 120);
}

// ── Toast close ───────────────────────────────────────────────────────────────
toastClose.addEventListener("click", hideError);

// ── Init ──────────────────────────────────────────────────────────────────────
checkStatus();
