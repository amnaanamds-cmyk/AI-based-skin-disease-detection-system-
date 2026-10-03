"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
const TOKEN_KEY = "dermaai.clinician.token";
const view = { status: "", selected: null };

function token() { try { return sessionStorage.getItem(TOKEN_KEY); } catch { return view.token; } }
function setToken(t) { view.token = t; try { t ? sessionStorage.setItem(TOKEN_KEY, t) : sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ } }

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), Authorization: `Bearer ${token()}` } });
  if (res.status === 401) { setToken(null); showLogin("Token rejected. Please sign in again."); throw new Error("unauthorized"); }
  const body = await res.json();
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
  return body;
}

function showLogin(msg = "") {
  $("#login").hidden = false; $("#dash").hidden = true; $("#logout").hidden = true;
  $("#login-error").textContent = msg;
}

async function loadQueue() {
  const q = await api(`/api/clinician/cases${view.status ? `?status=${view.status}` : ""}`);
  $("#login").hidden = true; $("#dash").hidden = false; $("#logout").hidden = false;
  const s = q.stats, u = s.open_by_urgency;
  $("#stats").innerHTML = [["Open cases", s.open], ["High priority", u.high || 0], ["Moderate", u.moderate || 0],
    ["Closed", s.by_status.closed || 0]].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const cases = view.status ? q.cases : q.cases.filter((c) => c.status !== "closed");
  $("#queue").innerHTML = cases.length ? cases.map((c) => `<button type="button" class="q-item ${esc(c.urgency)} ${view.selected === c.id ? "sel" : ""}" data-id="${esc(c.id)}">
      <div class="row1"><b>${esc(c.urgency.toUpperCase())}</b><span class="muted">${new Date(c.created_at).toLocaleString()}</span></div>
      <div>${esc(c.top)} · malignancy ${pct(c.malignancy)}</div>
      <div class="muted">${[c.age && `${Math.round(c.age)}y`, c.sex, c.localization].filter(Boolean).map(esc).join(" · ") || "no details"} · ${esc(c.status.replace("_", " "))}</div>
    </button>`).join("") : `<p class="muted">No cases here.</p>`;
  document.querySelectorAll(".q-item").forEach((b) => b.addEventListener("click", () => openCase(b.dataset.id)));
}

async function openCase(id) {
  view.selected = id;
  document.querySelectorAll(".q-item").forEach((b) => b.classList.toggle("sel", b.dataset.id === id));
  const c = await api(`/api/clinician/cases/${encodeURIComponent(id)}`), a = c.analysis, p = c.patient, t = a.triage;
  const abcde = a.abcde ? `Asymmetry ${a.abcde.asymmetry} · Border ${a.abcde.border_irregularity} · Colour ${a.abcde.color_variegation}
    (${a.abcde.colors_detected.map(esc).join(", ") || "–"})` : "Lesion not segmented";
  $("#detail").innerHTML = `<div class="card">
    <div class="triage ${esc(t.level)}" style="margin-bottom:12px"><div class="level">AI: ${esc(t.level)} priority</div>
      <div>${t.reasons.map(esc).join(" ")}</div></div>
    <div class="case-grid">
      <div><img src="${c.image}" alt="Submitted lesion photo"><p class="muted" style="font-size:12px">Case ${esc(c.id)} · submitted ${new Date(c.created_at).toLocaleString()}</p></div>
      <div>
        <h3>Patient</h3>
        <table class="delta-table"><tr><td>Age</td><td>${esc(p.age ?? "–")}</td></tr><tr><td>Sex</td><td>${esc(p.sex ?? "–")}</td></tr>
          <tr><td>Body site</td><td>${esc(p.localization ?? "–")}</td></tr><tr><td>Contact</td><td>${esc(p.contact ?? "–")}</td></tr></table>
        ${p.note ? `<p><b>Patient note:</b> ${esc(p.note)}</p>` : ""}
        <h3 style="margin-top:14px">AI estimates</h3>
        <div class="bars">${a.probabilities.slice(0, 4).map((x) => `<div class="bar"><span>${esc(x.name)}</span><span class="v">${pct(x.probability, 1)}</span>
          <div class="track"><div class="fill ${esc(x.malignancy)}" style="width:${(x.probability * 100).toFixed(1)}%"></div></div></div>`).join("")}</div>
        <p class="muted" style="font-size:13px">Uncertainty ${esc(a.uncertainty.level)} · Image quality ${a.quality.score}/100
          ${a.ood ? ` · ${a.ood.unfamiliar ? "⚠ unusual image for the model" : "typical image"}` : ""}<br>ABCDE: ${abcde}<br>
          Model ${esc(a.model.arch)} v${esc(a.model.version)}${a.model.demo_mode ? " <b>(DEMO — untrained)</b>" : ""}</p>
      </div>
    </div>
    <form id="review" style="margin-top:16px">
      <h3>Review</h3>
      <fieldset>
        <label>Clinical impression <span class="muted">(clinician only)</span><input id="impression" type="text" maxlength="500" value="${esc(c.clinical_impression ?? "")}"></label>
        <label>Message to patient<textarea id="response" maxlength="4000" placeholder="e.g. This looks like a benign mole. Re-check in 3 months, or sooner if it changes.">${esc(c.response ?? "")}</textarea></label>
        <div class="row">
          <label>Status<select id="status">${["new", "in_review", "closed"].map((s) => `<option value="${s}" ${s === c.status ? "selected" : ""}>${s.replace("_", " ")}</option>`).join("")}</select></label>
          <label>Reviewer<input id="reviewer" type="text" maxlength="120" value="${esc(c.reviewer ?? "")}"></label>
        </div>
      </fieldset>
      <div class="actions"><button class="primary" type="submit" style="width:auto;padding:10px 18px">Save review</button>
        <button type="button" class="ghost" id="fhir">Export FHIR R4</button><span id="saved" class="muted"></span></div>
    </form></div>`;
  $("#review").addEventListener("submit", async (e) => {
    e.preventDefault();
    await api(`/api/clinician/cases/${encodeURIComponent(id)}/review`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: $("#status").value, response: $("#response").value || null,
        clinical_impression: $("#impression").value || null, reviewer: $("#reviewer").value || null }) });
    $("#saved").textContent = "Saved ✓";
    await loadQueue();
  });
  $("#fhir").addEventListener("click", async () => {
    const bundle = await api(`/api/clinician/cases/${encodeURIComponent(id)}/fhir`);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(bundle, null, 2)], { type: "application/fhir+json" }));
    a.download = `dermaai-case-${id}.fhir.json`; a.click();
  });
}

$("#login").addEventListener("submit", async (e) => {
  e.preventDefault(); setToken($("#token").value.trim());
  try { await loadQueue(); } catch (err) { if (err.message !== "unauthorized") showLogin(err.message); }
});
$("#logout").addEventListener("click", (e) => { e.preventDefault(); setToken(null); showLogin(); });
document.querySelectorAll("#filter button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#filter button").forEach((x) => x.classList.toggle("on", x === b));
  view.status = b.dataset.status; loadQueue();
}));
if (token()) loadQueue().catch((err) => err.message !== "unauthorized" && showLogin(err.message)); else showLogin();
