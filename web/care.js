"use strict";
// Skin care analysis + routine, lesion change tracking, and the assistant chat.
// Relies on helpers defined in app.js: $, esc, pct, loadImage, state.
const SKIN_KEY = "dermaai.skin.v1";
const CHAT_KEY = "dermaai.chat.v1";
const LABELS = { acne: "Acne", redness: "Redness", pigmentation: "Dark spots / tone", aging: "Fine lines / ageing",
  texture: "Texture / pores", dryness: "Dryness", dullness: "Dullness", oiliness: "Oiliness" };
const care = { skinFile: null, skinUrl: null, before: null, after: null, lastSkin: null, chat: [] };

const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode or quota */ } },
};

function wireDrop(label, input, onFile) {
  label.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) => label.addEventListener(ev, (e) => { e.preventDefault(); label.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => label.addEventListener(ev, (e) => { e.preventDefault(); label.classList.remove("over"); }));
  label.addEventListener("drop", (e) => e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]));
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
}
function showPreview(img, label, file) {
  const url = URL.createObjectURL(file);
  img.src = url; img.hidden = false;
  const hint = label.querySelector(".hint"); if (hint) hint.hidden = true;
  return url;
}
async function postForm(url, fd) {
  const res = await fetch(url, { method: "POST", body: fd });
  const body = await res.json();
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  return body;
}
const loading = (el, text) => { el.innerHTML = `<div class="card"><div class="spinner"></div><p class="muted" style="text-align:center">${text}</p></div>`; };

// ---------------- options ----------------
window.addEventListener("dermaai:options", (e) => {
  const o = e.detail;
  $("#skin-type").innerHTML = o.skin_types.map((t) => `<option value="${esc(t)}">${esc(t[0].toUpperCase() + t.slice(1))}</option>`).join("");
  $("#skin-type").value = "normal";
  $("#concern-chips").innerHTML = o.concerns.map((c) =>
    `<label class="chip"><input type="checkbox" value="${esc(c)}">${esc(LABELS[c] || c)}</label>`).join("");
});

// ---------------- skin care ----------------
wireDrop($("#skin-drop"), $("#skin-file"), (f) => {
  if (!f.type.startsWith("image/")) return alert("Please choose an image file.");
  care.skinFile = f;
  if (care.skinUrl) URL.revokeObjectURL(care.skinUrl);
  care.skinUrl = showPreview($("#skin-preview"), $("#skin-drop"), f);
});

$("#skin-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const concerns = [...document.querySelectorAll("#concern-chips input:checked")].map((i) => i.value);
  const profile = { skin_type: $("#skin-type").value, concerns, sensitive: $("#sensitive").checked, pregnant: $("#pregnant").checked,
    sun_exposure: $("#sun").value, budget: $("#budget").value };
  const age = $("#skin-age").value;
  if (age) profile.age = Number(age);
  const out = $("#skin-result");
  loading(out, care.skinFile ? "Analysing your skin…" : "Building your routine…");
  try {
    let data;
    if (care.skinFile) {
      const fd = new FormData();
      fd.append("image", care.skinFile);
      Object.entries(profile).forEach(([k, v]) => fd.append(k, Array.isArray(v) ? v.join(",") : String(v)));
      data = await postForm("/api/skin/analyze", fd);
    } else {
      const res = await fetch("/api/skin/routine", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(profile) });
      const routine = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(routine.detail));
      data = { analysis: null, routine };
    }
    const prev = store.get(SKIN_KEY, []).at(-1);
    if (data.analysis?.ok) {
      store.set(SKIN_KEY, [...store.get(SKIN_KEY, []), { ts: new Date().toISOString(), health: data.analysis.skin_health_score,
        scores: data.analysis.scores }].slice(-60));
    }
    care.lastSkin = data;
    renderSkin(data, prev);
  } catch (err) {
    out.innerHTML = `<div class="card"><h3>Something went wrong</h3><p class="muted">${esc(err.message)}</p></div>`;
  }
});

function ring(score) {
  const r = 34, c = 2 * Math.PI * r, color = score >= 70 ? "var(--low)" : score >= 45 ? "var(--mod)" : "var(--high)";
  return `<svg width="88" height="88" viewBox="0 0 88 88" role="img" aria-label="Skin health score ${score} of 100">
    <circle cx="44" cy="44" r="${r}" fill="none" stroke="var(--surface-2)" stroke-width="8"/>
    <circle cx="44" cy="44" r="${r}" fill="none" stroke="${color}" stroke-width="8" stroke-linecap="round"
      stroke-dasharray="${(c * score) / 100} ${c}" transform="rotate(-90 44 44)"/>
    <text x="44" y="50" text-anchor="middle" font-size="22" font-weight="800" fill="var(--text)">${score}</text></svg>`;
}

function renderSkin({ analysis: a, routine: r }, prev) {
  let html = "";
  if (a && !a.ok) html += `<div class="change possible"><b>Couldn't analyse the photo.</b> ${esc(a.message)} Your routine below is based on your answers.</div>`;
  if (a?.ok) {
    const delta = (code, v) => {
      if (!prev?.scores || prev.scores[code] == null) return "";
      const d = v - prev.scores[code];
      return d === 0 ? "" : ` <span class="muted">(${d > 0 ? "▲" : "▼"} ${Math.abs(d)} since last)</span>`;
    };
    html += `<div class="result-grid">
      <div class="card">
        <h3>Skin analysis</h3>
        <div class="score-ring">${ring(a.skin_health_score)}<div><b>Skin health score</b><div class="big">out of 100 · photo suggests ${esc(a.skin_type_hint)} skin</div>
          ${prev ? `<div class="big">Previous: ${prev.health}</div>` : ""}</div></div>
        ${a.concerns.map((c) => `<div class="meter"><div class="head"><span>${esc(c.name)}<span class="lvl ${c.level}">${c.level}</span>${delta(c.code, c.score)}</span><b>${c.score}</b></div>
          <div class="track"><div class="fill ${c.level}" style="width:${c.score}%"></div></div></div>`).join("")}
        <p class="disclaimer">${esc(a.note)}</p>
      </div>
      <div class="card"><h3>What was measured</h3><img src="${a.visualization}" alt="Photo with detected redness (red tint), dark spots (blue tint) and skin area outline" style="width:100%;border-radius:10px">
        <p class="muted" style="font-size:12px">Red tint: inflamed spots · Blue tint: darker patches · Cyan line: analysed skin area</p></div>
    </div>`;
  }
  const step = (s) => `<li><b>${esc(s.title)}</b><div class="how">${esc(s.how)}</div>
    ${s.products.map((p) => `<div class="product"><div class="pname">${esc(p.name)} <span class="tag">${esc(p.price_tier)}</span>${p.fragrance_free ? '<span class="tag">fragrance-free</span>' : ""}</div>
      <div class="why">Key ingredients: ${p.key_ingredients.map(esc).join(", ")}${p.why.length ? ` · ${p.why.map(esc).join(" · ")}` : ""}</div></div>`).join("")}</li>`;
  html += `<div class="card" style="margin-top:16px">
      <h3>Your personalised routine</h3>
      <p class="muted">Focus: ${r.concerns.length ? r.concerns.map((c) => `${esc(LABELS[c.code] || c.code)} <span class="tag">${esc(c.source)}</span>`).join(" ") : "healthy-skin maintenance"}</p>
      <div class="routine"><div><h4>☀️ Morning</h4><ol>${r.am.map(step).join("")}</ol></div><div><h4>🌙 Evening</h4><ol>${r.pm.map(step).join("")}</ol></div></div>
      <h4 style="margin-top:14px">Tips</h4><ul class="tips">${r.tips.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>
      <p class="disclaimer">${esc(r.disclaimer)}</p>
    </div>
    <div class="card" style="margin-top:16px"><h3>Skin health score over time</h3><div id="skin-chart"></div></div>
    <div class="actions"><a class="ghost btn" href="#assistant">Ask the assistant about this routine</a>
      <button type="button" class="ghost" onclick="window.print()">Print / save</button></div>`;
  $("#skin-result").innerHTML = html;
  drawTrend($("#skin-chart"), store.get(SKIN_KEY, []));
}

function drawTrend(el, points) {
  if (points.length < 2) {
    el.innerHTML = `<p class="muted">Analyse a new photo every 2-4 weeks, in the same light, to see your progress here.</p>`;
    return;
  }
  const W = 600, H = 150, P = { l: 30, r: 12, t: 12, b: 24 };
  const x = (i) => P.l + (i * (W - P.l - P.r)) / (points.length - 1);
  const y = (v) => P.t + ((100 - v) * (H - P.t - P.b)) / 100;
  const grid = [0, 50, 100].map((v) => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)" stroke-width="1"/>
    <text x="${P.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v}</text>`).join("");
  const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.health).toFixed(1)}`).join("");
  const fmt = (ts) => new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const dots = points.map((p, i) => `<g class="pt" data-i="${i}"><circle cx="${x(i)}" cy="${y(p.health)}" r="4.5" fill="var(--brand)" stroke="var(--surface)" stroke-width="2"/>
    <circle cx="${x(i)}" cy="${y(p.health)}" r="14" fill="transparent"/></g>`).join("");
  const last = points.at(-1);
  el.innerHTML = `<div class="chart-wrap"><svg viewBox="0 0 ${W} ${H}" class="spark" style="height:auto" role="img" aria-label="Skin health score over ${points.length} checks">
    ${grid}<path d="${path}" fill="none" stroke="var(--brand)" stroke-width="2" stroke-linejoin="round"/>${dots}
    <text x="${P.l}" y="${H - 6}" font-size="11" fill="var(--muted)">${fmt(points[0].ts)}</text>
    <text x="${W - P.r}" y="${H - 6}" text-anchor="end" font-size="11" fill="var(--muted)">${fmt(last.ts)} · ${last.health}</text>
    </svg><div class="chart-tip" hidden></div></div>
    <details><summary class="muted" style="font-size:13px;cursor:pointer">Show as table</summary>
    <table class="delta-table">${points.map((p) => `<tr><td>${new Date(p.ts).toLocaleString()}</td><td>${p.health}</td></tr>`).join("")}</table></details>`;
  const tip = el.querySelector(".chart-tip"), svg = el.querySelector("svg");
  el.querySelectorAll(".pt").forEach((g) => {
    g.addEventListener("pointerenter", () => {
      const i = +g.dataset.i, p = points[i], box = svg.getBoundingClientRect(), s = box.width / W;
      tip.textContent = `${fmt(p.ts)}: ${p.health}/100`;
      tip.style.left = `${x(i) * s}px`; tip.style.top = `${y(p.health) * s}px`; tip.hidden = false;
    });
    g.addEventListener("pointerleave", () => (tip.hidden = true));
  });
}

// ---------------- lesion tracking ----------------
function trackReady() { $("#track-submit").disabled = !(care.before && care.after); }
wireDrop($("#before-file").parentElement, $("#before-file"), (f) => { care.before = f; showPreview($("#before-preview"), $("#before-file").parentElement, f); trackReady(); });
wireDrop($("#after-file").parentElement, $("#after-file"), (f) => { care.after = f; showPreview($("#after-preview"), $("#after-file").parentElement, f); trackReady(); });

$("#track-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("before", care.before); fd.append("after", care.after);
  if ($("#days").value) fd.append("days_between", $("#days").value);
  const out = $("#track-result");
  loading(out, "Comparing photos…");
  try {
    const r = await postForm("/api/lesion/compare", fd);
    if (!r.ok) { out.innerHTML = `<div class="card"><h3>Couldn't compare</h3><p class="muted">${esc(r.message)}</p></div>`; return; }
    const d = r.deltas, sign = (v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(0)}%`;
    const title = { significant: "Significant change", possible: "Possible change", stable: "Looks stable" }[r.change_level];
    out.innerHTML = `<div class="change ${esc(r.change_level)}"><div class="level" style="font-size:12px;font-weight:700;text-transform:uppercase">${esc(title)}</div>
        <h2 style="margin:4px 0">${esc(r.message)}</h2>${r.findings.length ? `<ul>${r.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}</div>
      <div class="result-grid">
        <div class="card"><h3>Outlines</h3><div class="compare">
          <figure style="margin:0"><canvas id="cv-before" width="300" height="300"></canvas><figcaption>Before</figcaption></figure>
          <figure style="margin:0"><canvas id="cv-after" width="300" height="300"></canvas><figcaption>After</figcaption></figure></div></div>
        <div class="card"><h3>Measured change</h3><table class="delta-table">
          <tr><td>Area</td><td>${sign(d.area_change)}</td></tr>
          <tr><td>Diameter</td><td>${sign(d.diameter_change)}</td></tr>
          <tr><td>Asymmetry</td><td>${d.asymmetry_change >= 0 ? "+" : ""}${d.asymmetry_change.toFixed(2)}</td></tr>
          <tr><td>Border irregularity</td><td>${d.border_change >= 0 ? "+" : ""}${d.border_change.toFixed(2)}</td></tr>
          <tr><td>New colours</td><td>${d.new_colors.length ? d.new_colors.map(esc).join(", ") : "none"}</td></tr>
          ${r.monthly_area_growth != null ? `<tr><td>Growth per month</td><td>${sign(r.monthly_area_growth)}</td></tr>` : ""}
        </table><p class="disclaimer">${esc(r.note)}</p></div>
      </div>
      <div class="actions"><a class="ghost btn" href="#assistant">Ask the assistant</a></div>`;
    await drawOutline($("#cv-before"), URL.createObjectURL(care.before), r.outlines.before);
    await drawOutline($("#cv-after"), URL.createObjectURL(care.after), r.outlines.after);
    care.lastTrack = r;
  } catch (err) {
    out.innerHTML = `<div class="card"><h3>Comparison failed</h3><p class="muted">${esc(err.message)}</p></div>`;
  }
});

async function drawOutline(c, src, outline) {
  const img = await loadImage(src), ctx = c.getContext("2d");
  const s = Math.min(c.width / img.width, c.height / img.height), w = img.width * s, h = img.height * s, x = (c.width - w) / 2, y = (c.height - h) / 2;
  ctx.drawImage(img, x, y, w, h);
  ctx.beginPath();
  outline.forEach(([px, py], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x + px * w, y + py * h));
  ctx.closePath(); ctx.lineWidth = 2.5; ctx.strokeStyle = "#22d3ee"; ctx.setLineDash([6, 4]); ctx.stroke();
}

// ---------------- assistant ----------------
const SUGGESTIONS = ["How do I check a mole with ABCDE?", "Explain my last result", "What should I use for dark spots?",
  "Is retinol safe during pregnancy?", "How much sunscreen should I use?"];

function chatContext() {
  const ctx = {};
  const r = state.result;
  if (r) Object.assign(ctx, { lesion_triage: r.triage.level, lesion_top_prediction: `${r.prediction.top.name} (${pct(r.prediction.top.probability)})`,
    lesion_malignancy_probability: pct(r.triage.malignancy_probability), model_demo_mode: r.model.demo_mode ? "yes, results not meaningful" : "no" });
  const s = care.lastSkin;
  if (s?.analysis?.ok) Object.assign(ctx, { skin_health_score: s.analysis.skin_health_score,
    skin_concerns: s.analysis.concerns.filter((c) => c.score >= 30).map((c) => `${c.name} ${c.score}`).join(", ") });
  if (s?.routine) Object.assign(ctx, { routine_actives: `AM ${s.routine.actives.am || "none"}, PM ${s.routine.actives.pm || "none"}` });
  if (care.lastTrack) ctx.lesion_change = `${care.lastTrack.change_level}: ${care.lastTrack.findings.join(" ")}`;
  return Object.keys(ctx).length ? ctx : null;
}

function renderChat() {
  const log = $("#chat-log");
  if (!care.chat.length) {
    log.innerHTML = `<div class="msg bot">Hi! I'm the DermaAI assistant. Ask me about moles, acne, pigmentation, sun protection or your routine.
I can't diagnose, but I'll tell you when something needs a doctor.</div>`;
  } else {
    log.innerHTML = care.chat.map((m) => `<div class="msg ${m.role === "user" ? "user" : "bot"} ${esc(m.level || "")}">${esc(m.content)}${
      m.sources?.length ? `<div class="src">${m.engine === "claude" ? "Answered by Claude AI" : "From DermaAI knowledge base"} · ${m.sources.map((s) => esc(s.title)).join(", ")}</div>` : ""}</div>`).join("");
  }
  log.scrollTop = log.scrollHeight;
  $("#chat-suggest").innerHTML = SUGGESTIONS.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
  $("#chat-suggest").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
}

async function send(text) {
  text = text.trim();
  if (!text) return;
  // The server accepts a bounded history; the recent turns are what matter for context.
  const history = care.chat.slice(-20).map(({ role, content }) => ({ role, content: content.slice(0, 4000) }));
  care.chat.push({ role: "user", content: text });
  renderChat();
  $("#chat-log").insertAdjacentHTML("beforeend", `<div class="msg bot" id="typing">…</div>`);
  try {
    const res = await fetch("/api/assistant/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history, context: chatContext() }) });
    const r = await res.json();
    if (!res.ok) throw new Error(typeof r.detail === "string" ? r.detail : "Request failed");
    care.chat.push({ role: "assistant", content: r.reply, level: r.escalation?.level, sources: r.sources, engine: r.engine });
  } catch (err) {
    care.chat.push({ role: "assistant", content: `Sorry, I couldn't answer that (${err.message}).` });
  }
  store.set(CHAT_KEY, care.chat.slice(-40));
  renderChat();
}

$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#chat-text").value; $("#chat-text").value = ""; send(t); });
care.chat = store.get(CHAT_KEY, []);
window.addEventListener("dermaai:view", (e) => { if (e.detail === "assistant") renderChat(); });
if ((location.hash || "").slice(1) === "assistant") renderChat();
