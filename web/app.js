"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
const HISTORY_KEY = "dermaai.history.v1";

const state = { file: null, previewUrl: null, result: null };

// ---------- navigation ----------
function route() {
  const id = (location.hash || "#check").slice(1);
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== id));
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.hash === `#${id}`));
  if (id === "history") renderHistory();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

// ---------- bootstrap ----------
async function init() {
  route();
  try {
    const [health, opts, conds, model] = await Promise.all(
      ["/api/health", "/api/options", "/api/conditions", "/api/model"].map((u) => fetch(u).then((r) => r.json())));
    $("#demo-banner").hidden = !health.demo_mode;
    const sel = $("#localization");
    opts.localization.forEach((l) => sel.insertAdjacentHTML("beforeend", `<option value="${esc(l)}">${esc(l[0].toUpperCase() + l.slice(1))}</option>`));
    $("#conditions").innerHTML = conds.conditions.map((c) => `
      <div class="card"><span class="pill ${esc(c.malignancy)}">${esc(c.malignancy)}</span>
      <h3 style="margin-top:8px">${esc(c.name)}</h3><p>${esc(c.summary)}</p><p><b>What to do:</b> ${esc(c.action)}</p></div>`).join("");
    $("#disclaimer").textContent = conds.disclaimer;
    renderModelCard(model);
  } catch (e) {
    console.error(e);
  }
}

function renderModelCard(m) {
  const rows = [["Architecture", m.arch], ["Input size", `${m.img_size}px`], ["Calibration temperature", m.temperature],
    ["Status", m.demo_mode ? "Demo (untrained)" : `Trained ${m.trained_at ?? ""}`]];
  const met = m.metrics || {};
  if (met.balanced_accuracy != null) rows.push(["Balanced accuracy (test)", pct(met.balanced_accuracy, 1)]);
  if (met.macro_auc != null) rows.push(["Macro AUC", met.macro_auc.toFixed(3)]);
  if (met.melanoma_auc != null) rows.push(["Melanoma AUC", met.melanoma_auc.toFixed(3)]);
  if (met.malignant_auc != null) rows.push(["Malignant vs benign AUC", met.malignant_auc.toFixed(3)]);
  if (met.melanoma_at_90_sensitivity) rows.push(["Specificity @ 90% melanoma sensitivity", pct(met.melanoma_at_90_sensitivity.specificity, 1)]);
  $("#model-card").innerHTML = `<h3>Model card</h3><table>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>`;
}

// ---------- input ----------
const drop = $("#drop"), fileInput = $("#file");
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && setFile(e.dataTransfer.files[0]));
fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
$("#consent").addEventListener("change", updateSubmit);

function setFile(f) {
  if (!f.type.startsWith("image/")) return alert("Please choose an image file.");
  state.file = f;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(f);
  $("#preview").src = state.previewUrl;
  $("#preview").hidden = false;
  $("#drop-hint").hidden = true;
  updateSubmit();
}
function updateSubmit() { $("#submit").disabled = !(state.file && $("#consent").checked); }

$("#form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.file) return;
  const fd = new FormData();
  fd.append("image", state.file);
  const age = $("#age").value, sex = $("#sex").value, loc = $("#localization").value;
  if (age) fd.append("age", age);
  if (sex) fd.append("sex", sex);
  if (loc) fd.append("localization", loc);
  $("#submit").disabled = true;
  $("#result").innerHTML = `<div class="card"><div class="spinner"></div><p class="muted" style="text-align:center">Analyzing…</p></div>`;
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    state.result = body;
    renderResult(body);
    saveHistory(body);
  } catch (err) {
    $("#result").innerHTML = `<div class="card"><h3>Analysis failed</h3><p class="muted">${esc(err.message)}</p></div>`;
  } finally {
    updateSubmit();
  }
});

// ---------- result ----------
function renderResult(r) {
  const t = r.triage, top = r.prediction.top, a = r.abcde, q = r.quality;
  const retake = t.level === "retake";
  const bars = r.prediction.probabilities.map((p) => `
    <div class="bar"><span>${esc(p.name)}</span><span class="v">${pct(p.probability, 1)}</span>
    <div class="track"><div class="fill ${esc(p.malignancy)}" style="width:${(p.probability * 100).toFixed(1)}%"></div></div></div>`).join("");
  const meter = (label, v, hint) => `<div class="meter"><div class="head"><span>${label}</span><b>${pct(v)}</b></div>
    <div class="track"><div class="fill" style="width:${pct(v)}"></div></div>${hint ? `<div class="muted">${hint}</div>` : ""}</div>`;
  const issues = q.issues.length
    ? q.issues.map((i) => `<li>${i.severity === "error" ? "⛔" : "⚠️"} ${esc(i.message)}</li>`).join("")
    : `<li>✅ Sharp, well-lit and well-framed.</li>`;

  $("#result").innerHTML = `
    <div class="triage ${esc(t.level)}">
      <div class="level">${retake ? "Retake needed" : `${esc(t.level)} priority`}</div>
      <h2>${esc(t.title)}</h2><div>${esc(t.message)}</div>
      ${t.reasons.length ? `<ul>${t.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    </div>
    <div class="result-grid">
      ${retake ? "" : `<div class="card top-dx">
        <h3>Most likely</h3>
        <span class="pill ${esc(top.malignancy)}">${esc(top.malignancy)}</span>
        <div class="name">${esc(top.name)}</div>
        <div class="pct">${pct(top.probability)}</div>
        <p class="muted">${esc(top.summary)}</p>
        <p><b>Next step:</b> ${esc(top.action)}</p>
        ${meter("Model uncertainty", r.uncertainty.score, r.uncertainty.level === "high" ? "High — treat this result with caution." : "")}
      </div>`}
      <div class="card viewer">
        <h3>What the AI looked at</h3>
        <div class="tabs">
          <button type="button" data-mode="heat" class="on">Attention map</button>
          <button type="button" data-mode="outline">Lesion outline</button>
          <button type="button" data-mode="photo">Photo</button>
        </div>
        <canvas id="canvas" width="448" height="448"></canvas>
      </div>
      ${retake ? "" : `<div class="card"><h3>All conditions</h3><div class="bars">${bars}</div></div>`}
      <div class="card">
        <h3>ABCDE analysis</h3>
        ${a ? `${meter("<b>A</b>symmetry", a.asymmetry)}${meter("<b>B</b>order irregularity", a.border_irregularity)}
          ${meter("<b>C</b>olour variety", a.color_variegation, a.colors_detected.length ? `Colours: ${a.colors_detected.map(esc).join(", ")}` : "")}
          ${meter("<b>D</b>iameter (share of photo)", Math.min(1, a.diameter_relative))}
          <div class="meter"><b>E</b>volution — compare with earlier photos in <a href="#history">History</a>.</div>`
          : `<p class="muted">Could not isolate the lesion. Centre it and fill more of the frame.</p>`}
      </div>
      <div class="card ${retake ? "wide" : ""}">
        <h3>Image quality · ${q.score}/100</h3><ul class="issues">${issues}</ul>
      </div>
    </div>
    <div class="actions">
      <button type="button" class="ghost" onclick="window.print()">Print / save report</button>
      <button type="button" class="ghost" id="new-check">New check</button>
    </div>
    <p class="disclaimer">${esc(r.disclaimer)} · Model ${esc(r.model.arch)} v${esc(r.model.version)} · ${r.processing_ms} ms · ID ${esc(r.id.slice(0, 8))}</p>`;

  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    drawView(b.dataset.mode);
  }));
  $("#new-check").addEventListener("click", () => { location.hash = "#check"; window.scrollTo({ top: 0, behavior: "smooth" }); fileInput.click(); });
  drawView(r.explanation ? "heat" : "outline");
  if (!r.explanation) $('.tabs [data-mode="heat"]').remove();
}

function loadImage(src) {
  return new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });
}

async function drawView(mode) {
  const c = $("#canvas"), ctx = c.getContext("2d"), r = state.result;
  const src = mode === "heat" ? r.explanation.heatmap : state.previewUrl;
  const img = await loadImage(src);
  const scale = Math.min(c.width / img.width, c.height / img.height);
  const w = img.width * scale, h = img.height * scale, x = (c.width - w) / 2, y = (c.height - h) / 2;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(img, x, y, w, h);
  if (mode === "outline" && r.abcde?.outline?.length) {
    ctx.beginPath();
    r.abcde.outline.forEach(([px, py], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x + px * w, y + py * h));
    ctx.closePath();
    ctx.lineWidth = 3; ctx.strokeStyle = "#22d3ee"; ctx.setLineDash([8, 5]); ctx.stroke(); ctx.setLineDash([]);
  }
}

// ---------- history (local only) ----------
function readHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; } }
function writeHistory(h) { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, 40))); } catch { /* quota or private mode */ } }

async function saveHistory(r) {
  try {
    const img = await loadImage(state.previewUrl), c = document.createElement("canvas"), s = 200 / Math.max(img.width, img.height);
    c.width = img.width * s; c.height = img.height * s;
    c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
    writeHistory([{ id: r.id, ts: r.timestamp, thumb: c.toDataURL("image/jpeg", 0.75), level: r.triage.level,
      top: r.prediction.top.name, p: r.prediction.top.probability, mal: r.triage.malignancy_probability,
      location: $("#localization").value || null }, ...readHistory()]);
  } catch (e) { console.warn(e); }
}

function renderHistory() {
  const h = readHistory();
  $("#history-list").innerHTML = h.length ? h.map((e) => `
    <div class="card item"><img src="${esc(e.thumb)}" alt="">
      <div class="meta"><span class="pill ${e.level === "high" ? "malignant" : e.level === "moderate" ? "pre-malignant" : "benign"}">${esc(e.level)}</span>
      <div><b>${esc(e.top)}</b> · ${pct(e.p)}</div>
      <div class="muted">${new Date(e.ts).toLocaleString()}${e.location ? ` · ${esc(e.location)}` : ""}</div>
      <div class="muted">Malignancy risk ${pct(e.mal)}</div></div></div>`).join("")
    : `<p class="muted">No checks yet.</p>`;
}
$("#clear-history").addEventListener("click", () => { if (confirm("Delete all saved checks on this device?")) { writeHistory([]); renderHistory(); } });

init();
