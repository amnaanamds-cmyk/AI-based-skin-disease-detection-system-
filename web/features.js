"use strict";
// Live camera photo coach, body map (mole mapping) and dermatologist referrals.
// Relies on app.js (state, setFile, $, esc, pct, loadImage) and care.js (care, store, showPreview).
const SPOTS_KEY = "dermaai.spots.v1";
const REFERRALS_KEY = "dermaai.referrals.v1";

async function dataUrlFromUrl(url, maxSide, quality = 0.85) {
  const img = await loadImage(url), c = document.createElement("canvas"), s = Math.min(1, maxSide / Math.max(img.width, img.height));
  c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
  c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", quality);
}
const blobFromDataUrl = (u) => fetch(u).then((r) => r.blob());
const levelPill = (l) => ({ high: "malignant", moderate: "pre-malignant", low: "benign" }[l] || "");

// =====================================================================
// 1. Live camera with real-time photo coach
// =====================================================================
const cam = { stream: null, target: null, timer: null, goodFrames: 0, busy: false };
const probe = document.createElement("canvas");

function frameMetrics(video, target) {
  const w = 192, h = Math.round((w * video.videoHeight) / video.videoWidth) || 144;
  probe.width = w; probe.height = h;
  const ctx = probe.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data, n = w * h, gray = new Float32Array(n);
  let sum = 0, glare = 0, skin = 0, centre = 0, centreN = 0, edge = 0, edgeN = 0;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = d[p], g = d[p + 1], b = d[p + 2], y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[i] = y; sum += y;
    if (Math.min(r, g, b) >= 250) glare++;
    const cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b, cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b;
    if (cr >= 133 && cr <= 178 && cb >= 70 && cb <= 130) skin++;
    const x = i % w, yy = (i / w) | 0, dx = Math.abs(x - w / 2) / w, dy = Math.abs(yy - h / 2) / h;
    if (dx < 0.15 && dy < 0.15) { centre += y; centreN++; } else if (dx > 0.3 || dy > 0.3) { edge += y; edgeN++; }
  }
  let lap = 0, lap2 = 0, m = 0;
  for (let yy = 1; yy < h - 1; yy++) for (let x = 1; x < w - 1; x++) {
    const i = yy * w + x, v = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
    lap += v; lap2 += v * v; m++;
  }
  const sharp = lap2 / m - (lap / m) ** 2, bright = sum / n;
  const checks = {
    sharp: sharp >= 40 ? ["ok", "Sharp"] : sharp >= 15 ? ["warn", "Hold steady"] : ["bad", "Blurry — tap to focus"],
    light: bright < 55 ? ["bad", "Too dark"] : bright > 215 ? ["bad", "Too bright"] : ["ok", "Good"],
    glare: glare / n > 0.04 ? ["warn", "Reflections — tilt slightly"] : ["ok", "None"],
  };
  if (target === "lesion") {
    const contrast = edgeN && centreN ? edge / edgeN - centre / centreN : 0;
    checks.frame = skin / n < 0.25 ? ["bad", "Point at the skin"] : contrast > 10 ? ["ok", "Spot centred"] : ["warn", "Centre the spot in the circle"];
  } else {
    checks.frame = skin / n < 0.35 ? ["warn", "Fill the frame with skin"] : ["ok", "Good"];
  }
  return checks;
}

function coachTick() {
  const video = $("#camera-video");
  if (!video.videoWidth || cam.busy) return;
  const checks = frameMetrics(video, cam.target);
  let hint = null;
  for (const [k, [status, text]] of Object.entries(checks)) {
    const el = document.querySelector(`.coach-item[data-k="${k}"]`);
    el.className = `coach-item ${status}`;
    el.querySelector("b").textContent = text;
    if (!hint && status !== "ok") hint = text;
  }
  const allOk = !hint;
  cam.goodFrames = allOk ? cam.goodFrames + 1 : 0;
  $("#camera-hint").textContent = allOk ? (cam.goodFrames >= 3 ? "Perfect — hold still…" : "Looking good") : hint;
  $("#camera-hint").className = `camera-hint ${allOk ? "ok" : ""}`;
  if (allOk && cam.goodFrames >= 6 && $("#camera-auto").checked) capturePhoto();
}

async function openCamera(target) {
  cam.target = target; cam.goodFrames = 0; cam.busy = false;
  $("#camera-title").textContent = target === "lesion" ? "Photograph a spot" : "Photograph your face";
  document.querySelector(".camera-guide").classList.toggle("face", target !== "lesion");
  $("#camera-modal").hidden = false;
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: target === "lesion" ? "environment" : "user", width: { ideal: 1920 }, height: { ideal: 1440 } }, audio: false });
    const v = $("#camera-video");
    v.srcObject = cam.stream;
    await v.play();
    cam.timer = setInterval(coachTick, 200);
  } catch (e) {
    $("#camera-hint").textContent = window.isSecureContext
      ? `Camera unavailable (${e.name}). You can upload a photo instead.`
      : "The camera needs HTTPS (or localhost). You can upload a photo instead.";
  }
}

function closeCamera() {
  clearInterval(cam.timer);
  cam.stream?.getTracks().forEach((t) => t.stop());
  cam.stream = null;
  $("#camera-modal").hidden = true;
}

function capturePhoto() {
  const v = $("#camera-video");
  if (!v.videoWidth || cam.busy) return;
  cam.busy = true;
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob((blob) => {
    const file = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" });
    if (cam.target === "lesion") setFile(file);
    else {
      care.skinFile = file;
      if (care.skinUrl) URL.revokeObjectURL(care.skinUrl);
      care.skinUrl = showPreview($("#skin-preview"), $("#skin-drop"), file);
    }
    closeCamera();
  }, "image/jpeg", 0.92);
}

document.querySelectorAll(".camera-btn").forEach((b) => b.addEventListener("click", () => openCamera(b.dataset.target)));
$("#camera-close").addEventListener("click", closeCamera);
$("#camera-shoot").addEventListener("click", capturePhoto);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#camera-modal").hidden) closeCamera(); });

// =====================================================================
// 2. Body map
// =====================================================================
const map = { view: "front", selected: null, draft: null };
// Body parts: [shape, attrs, localization (front), localization (back)]
const BODY = [
  ["circle", { cx: 100, cy: 34, r: 22 }, "face", "scalp"],
  ["rect", { x: 90, y: 54, width: 20, height: 16, rx: 4 }, "neck", "neck"],
  ["rect", { x: 62, y: 68, width: 76, height: 80, rx: 18 }, "chest", "back"],
  ["rect", { x: 64, y: 140, width: 72, height: 82, rx: 12 }, "abdomen", "back"],
  ["rect", { x: 34, y: 74, width: 24, height: 150, rx: 12 }, "upper extremity", "upper extremity"],
  ["rect", { x: 142, y: 74, width: 24, height: 150, rx: 12 }, "upper extremity", "upper extremity"],
  ["ellipse", { cx: 46, cy: 238, rx: 13, ry: 16 }, "hand", "hand"],
  ["ellipse", { cx: 154, cy: 238, rx: 13, ry: 16 }, "hand", "hand"],
  ["rect", { x: 66, y: 214, width: 32, height: 182, rx: 14 }, "lower extremity", "lower extremity"],
  ["rect", { x: 102, y: 214, width: 32, height: 182, rx: 14 }, "lower extremity", "lower extremity"],
  ["ellipse", { cx: 80, cy: 408, rx: 17, ry: 10 }, "foot", "foot"],
  ["ellipse", { cx: 120, cy: 408, rx: 17, ry: 10 }, "foot", "foot"],
];

const spots = () => store.get(SPOTS_KEY, []);
const saveSpots = (s) => store.set(SPOTS_KEY, s);
const latest = (spot) => spot.checks.at(-1);

function drawBody() {
  const parts = BODY.map(([tag, a, front, back]) => {
    const attrs = Object.entries(a).map(([k, v]) => `${k}="${v}"`).join(" ");
    return `<${tag} ${attrs} class="part" data-loc="${esc(map.view === "front" ? front : back)}"/>`;
  }).join("");
  const pins = spots().filter((s) => s.view === map.view).map((s) => {
    const lvl = latest(s)?.level || "none";
    return `<g class="pin ${lvl} ${map.selected === s.id ? "sel" : ""}" data-id="${esc(s.id)}" tabindex="0" role="button" aria-label="${esc(s.name)}">
      <circle cx="${s.x}" cy="${s.y}" r="14" class="hit"/><circle cx="${s.x}" cy="${s.y}" r="6.5" class="dot"/></g>`;
  }).join("");
  const draft = map.draft && map.draft.view === map.view ? `<circle cx="${map.draft.x}" cy="${map.draft.y}" r="6.5" class="draft"/>` : "";
  $("#body-svg").innerHTML = `<svg viewBox="0 0 200 430" role="img" aria-label="Body map, ${map.view} view">
    <text x="100" y="428" text-anchor="middle" font-size="9" fill="var(--muted)">${map.view === "front" ? "FRONT" : "BACK"} · patient's left is on the ${map.view === "front" ? "right" : "left"}</text>
    ${parts}${pins}${draft}</svg>`;
  const svg = $("#body-svg svg");
  svg.querySelectorAll(".part").forEach((el) => el.addEventListener("click", (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    map.draft = { view: map.view, x: Math.round(p.x), y: Math.round(p.y), loc: el.dataset.loc };
    map.selected = null;
    drawBody(); renderSpotPanel();
  }));
  svg.querySelectorAll(".pin").forEach((el) => {
    const pick = () => { map.selected = el.dataset.id; map.draft = null; drawBody(); renderSpotPanel(); };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
  });
}

function renderSpotPanel() {
  const panel = $("#spot-panel");
  if (map.draft) {
    panel.innerHTML = `<form id="new-spot" class="card"><h3>New spot · ${esc(map.draft.loc)}</h3>
      <fieldset><label>Name<input id="spot-name" type="text" maxlength="60" placeholder="e.g. Mole on left shoulder" required></label></fieldset>
      <div class="actions"><button class="primary" type="submit" style="width:auto;padding:10px 18px">Save spot</button>
      <button class="ghost" type="button" id="spot-cancel">Cancel</button></div></form>`;
    $("#spot-name").focus();
    $("#spot-cancel").addEventListener("click", () => { map.draft = null; drawBody(); renderSpotPanel(); });
    $("#new-spot").addEventListener("submit", (e) => {
      e.preventDefault();
      const spot = { id: Math.random().toString(36).slice(2, 10), name: $("#spot-name").value.trim(), ...map.draft,
        created: new Date().toISOString(), checks: [] };
      saveSpots([...spots(), spot]);
      map.selected = spot.id; map.draft = null;
      drawBody(); renderSpotPanel();
    });
    return;
  }
  const spot = spots().find((s) => s.id === map.selected);
  if (!spot) {
    const all = spots();
    panel.innerHTML = `<div class="card"><h3>${all.length ? `${all.length} spot${all.length > 1 ? "s" : ""} mapped` : "No spots yet"}</h3>
      <p class="muted">Tap the body to pin a new spot, or select a pin to see its history.</p>
      ${all.length ? `<ul class="issues">${all.map((s) => `<li><a href="#bodymap" data-id="${esc(s.id)}" class="spot-link">${esc(s.name)}</a>
        <span class="muted"> · ${esc(s.loc)} · ${s.checks.length} check${s.checks.length === 1 ? "" : "s"}</span>
        ${latest(s) ? ` <span class="pill ${levelPill(latest(s).level)}">${esc(latest(s).level)}</span>` : ""}</li>`).join("")}</ul>` : ""}</div>`;
    panel.querySelectorAll(".spot-link").forEach((a) => a.addEventListener("click", (e) => {
      e.preventDefault(); const s = spots().find((x) => x.id === a.dataset.id);
      map.view = s.view; map.selected = s.id; syncTabs(); drawBody(); renderSpotPanel();
    }));
    return;
  }
  const checks = [...spot.checks].reverse();
  panel.innerHTML = `<div class="card">
    <div class="section-head"><h3>${esc(spot.name)}</h3><button type="button" class="ghost" id="spot-delete">Delete</button></div>
    <p class="muted">${esc(spot.loc)} · pinned ${new Date(spot.created).toLocaleDateString()}</p>
    <div class="actions" style="margin-top:0"><button type="button" class="primary" id="spot-check" style="width:auto;padding:10px 18px">Check this spot now</button>
      ${spot.checks.length >= 2 ? `<button type="button" class="ghost" id="spot-compare">Compare last two checks</button>` : ""}</div>
    <div id="spot-compare-out"></div>
    <h4 style="margin-top:16px">Timeline</h4>
    ${checks.length ? `<div class="timeline">${checks.map((c) => `<div class="tl-item"><img src="${esc(c.thumb)}" alt="">
      <div><span class="pill ${levelPill(c.level)}">${esc(c.level)}</span><div><b>${esc(c.top)}</b> · ${pct(c.p)}</div>
      <div class="muted">${new Date(c.ts).toLocaleDateString()} · malignancy ${pct(c.mal)}</div></div></div>`).join("")}</div>`
      : `<p class="muted">No checks yet. Tap "Check this spot now".</p>`}
  </div>`;
  $("#spot-delete").addEventListener("click", () => {
    if (!confirm(`Delete "${spot.name}" and its ${spot.checks.length} check(s) from this device?`)) return;
    saveSpots(spots().filter((s) => s.id !== spot.id)); map.selected = null; drawBody(); renderSpotPanel();
  });
  $("#spot-check").addEventListener("click", () => startSpotCheck(spot));
  $("#spot-compare")?.addEventListener("click", () => compareSpot(spot));
}

function startSpotCheck(spot) {
  state.spotId = spot.id;
  const sel = $("#localization");
  if ([...sel.options].some((o) => o.value === spot.loc)) sel.value = spot.loc;
  $("#spot-badge").hidden = false;
  $("#spot-badge").innerHTML = `📍 Checking <b>${esc(spot.name)}</b> <button type="button" class="link" id="spot-unlink">not this spot</button>`;
  $("#spot-unlink").addEventListener("click", () => { state.spotId = null; $("#spot-badge").hidden = true; });
  location.hash = "#check";
}

async function compareSpot(spot) {
  const [a, b] = spot.checks.slice(-2), out = $("#spot-compare-out");
  out.innerHTML = `<div class="spinner"></div>`;
  try {
    const fd = new FormData();
    fd.append("before", await blobFromDataUrl(a.image), "before.jpg");
    fd.append("after", await blobFromDataUrl(b.image), "after.jpg");
    const days = Math.max(1, Math.round((new Date(b.ts) - new Date(a.ts)) / 864e5));
    fd.append("days_between", String(days));
    const r = await postForm("/api/lesion/compare", fd);
    out.innerHTML = r.ok
      ? `<div class="change ${esc(r.change_level)}" style="margin-top:12px"><b>${esc(r.message)}</b>
          ${r.findings.length ? `<ul>${r.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
          <div class="muted" style="font-size:12px">${days} day(s) apart. ${esc(r.note)}</div></div>`
      : `<p class="muted">${esc(r.message)}</p>`;
  } catch (e) {
    out.innerHTML = `<p class="muted">Comparison failed: ${esc(e.message)}</p>`;
  }
}

function syncTabs() { document.querySelectorAll("#map-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.view === map.view)); }
document.querySelectorAll("#map-tabs button").forEach((b) => b.addEventListener("click", () => {
  map.view = b.dataset.view; map.draft = null; syncTabs(); drawBody(); renderSpotPanel();
}));

async function attachResultToSpot(result) {
  const all = spots(), spot = all.find((s) => s.id === state.spotId);
  if (!spot) return null;
  spot.checks.push({ ts: result.timestamp, level: result.triage.level, top: result.prediction.top.name,
    p: result.prediction.top.probability, mal: result.triage.malignancy_probability,
    thumb: await dataUrlFromUrl(state.previewUrl, 160, 0.75),
    image: await dataUrlFromUrl(state.previewUrl, 640, 0.85) });  // kept for change comparison
  spot.checks = spot.checks.slice(-12);
  saveSpots(all);
  return spot;
}

// =====================================================================
// 3. Dermatologist referrals
// =====================================================================
const referrals = () => store.get(REFERRALS_KEY, []);

function referralCard(result) {
  const urgent = result.triage.level === "high" || result.triage.level === "moderate";
  return `<div class="card referral" style="margin-top:16px">
    <h3>${urgent ? "Get a dermatologist's opinion" : "Want a professional to look at it?"}</h3>
    <p class="muted">Send this photo and your answers to a dermatologist for review. The photo is stored only for this
      case and you can delete it at any time.</p>
    <form id="ref-form">
      <fieldset>
        <label>How should the clinic contact you? <span class="muted">(optional)</span><input id="ref-contact" type="text" maxlength="200" placeholder="Phone or email"></label>
        <label>Anything the doctor should know? <span class="muted">(optional)</span><input id="ref-note" type="text" maxlength="1000" placeholder="e.g. it appeared 3 months ago and itches"></label>
      </fieldset>
      <label class="consent"><input id="ref-consent" type="checkbox" required> I agree to store this photo and share it with a dermatologist.</label>
      <button class="primary" type="submit">Send to a dermatologist</button>
    </form>
    <div id="ref-out"></div></div>`;
}

async function sendReferral(e) {
  e.preventDefault();
  const fd = new FormData();
  fd.append("image", state.file);
  fd.append("consent", "true");
  const age = $("#age").value, sex = $("#sex").value, loc = $("#localization").value;
  if (age) fd.append("age", age);
  if (sex) fd.append("sex", sex);
  if (loc) fd.append("localization", loc);
  if ($("#ref-contact").value) fd.append("contact", $("#ref-contact").value);
  if ($("#ref-note").value) fd.append("note", $("#ref-note").value);
  const btn = $("#ref-form button"); btn.disabled = true;
  try {
    const r = await postForm("/api/referrals", fd);
    store.set(REFERRALS_KEY, [{ id: r.case_id, token: r.access_token, ts: new Date().toISOString(), urgency: r.urgency,
      thumb: await dataUrlFromUrl(state.previewUrl, 160, 0.75) }, ...referrals()]);
    $("#ref-form").hidden = true;
    $("#ref-out").innerHTML = `<div class="change stable"><b>Sent.</b> Case <code>${esc(r.case_id)}</code>. The reply will appear under
      <a href="#history">History → My dermatologist referrals</a>. Your access code is saved on this device.</div>`;
  } catch (err) {
    $("#ref-out").innerHTML = `<p class="muted">Couldn't send: ${esc(err.message)}</p>`;
    btn.disabled = false;
  }
}

async function refFetch(ref, method = "GET") {
  const res = await fetch(`/api/referrals/${encodeURIComponent(ref.id)}`, { method, headers: { "X-Case-Token": ref.token } });
  if (res.status === 404) return { missing: true };
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function renderReferrals() {
  const list = referrals(), el = $("#referral-list");
  if (!list.length) { el.innerHTML = `<p class="muted">No referrals yet. After a lesion check you can send it to a dermatologist.</p>`; return; }
  el.innerHTML = list.map((r) => `<div class="card item" data-id="${esc(r.id)}"><img src="${esc(r.thumb)}" alt="">
    <div class="meta"><div><code>${esc(r.id)}</code> · <span class="pill ${levelPill(r.urgency)}">${esc(r.urgency)}</span></div>
    <div class="muted">Sent ${new Date(r.ts).toLocaleDateString()}</div><div class="ref-status muted">Checking status…</div>
    <button type="button" class="ghost ref-del" style="margin-top:6px">Delete case</button></div></div>`).join("");
  for (const r of list) {
    const card = el.querySelector(`[data-id="${CSS.escape(r.id)}"]`);
    card.querySelector(".ref-del").addEventListener("click", async () => {
      if (!confirm("Delete this case and its photo from the clinic's system?")) return;
      try { await refFetch(r, "DELETE"); } catch { /* already gone */ }
      store.set(REFERRALS_KEY, referrals().filter((x) => x.id !== r.id));
      renderReferrals();
    });
    try {
      const s = await refFetch(r);
      card.querySelector(".ref-status").innerHTML = s.missing ? "Case no longer exists."
        : s.response ? `<b style="color:var(--text)">Dermatologist's reply:</b> ${esc(s.response)}`
          : `Status: ${esc(s.status.replace("_", " "))}. Waiting for review.`;
    } catch (e) {
      card.querySelector(".ref-status").textContent = `Couldn't check status (${e.message}).`;
    }
  }
}

// =====================================================================
// Wiring
// =====================================================================
window.addEventListener("dermaai:result", async (e) => {
  const result = e.detail, actions = $("#result .actions");
  const spot = await attachResultToSpot(result);
  if (spot) {
    actions.insertAdjacentHTML("beforebegin", `<div class="change stable" style="margin-top:16px">📍 Saved to body-map spot
      <b>${esc(spot.name)}</b> (${spot.checks.length} check${spot.checks.length > 1 ? "s" : ""}). <a href="#bodymap" id="goto-spot">Open timeline</a></div>`);
    $("#goto-spot").addEventListener("click", () => { map.view = spot.view; map.selected = spot.id; syncTabs(); });
  }
  if (result.triage.level !== "retake") {
    actions.insertAdjacentHTML("afterend", referralCard(result));
    $("#ref-form").addEventListener("submit", sendReferral);
  }
});

window.addEventListener("dermaai:view", (e) => {
  if (e.detail === "bodymap") { drawBody(); renderSpotPanel(); }
  if (e.detail === "history") renderReferrals();
});
const initial = (location.hash || "").slice(1);
if (initial === "bodymap") { drawBody(); renderSpotPanel(); }
if (initial === "history") renderReferrals();
