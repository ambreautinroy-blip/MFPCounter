// Duct Counter front-end. Coordinates of marks are full-resolution field pixels;
// field images are served at half resolution (IMG_SCALE), the overview mosaic in its own pixels.
const IMG_SCALE = 0.5;
const FOV_R = 1450, FOV_C = [2016, 1520];   // approximate field-of-view disc, full res

const $ = (s) => document.querySelector(s);
const api = async (url, opt) => { const r = await fetch(url, opt); if (!r.ok) throw new Error(await r.text()); return r.json(); };
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const uid = () => Math.random().toString(16).slice(2, 10);

const S = {
  summary: [], cut: null, view: "overview", fi: 0, field: null, img: null, mosaic: null, mosaicPts: null,
  cls: "normal", selected: null, tf: { k: 1, x: 0, y: 0 }, jobs: null, lastJobStatus: null,
};
const layers = () => ({ manual: $("#l-manual").checked, model: $("#l-model").checked, same: $("#l-same").checked, cuts: $("#l-cuts").checked,
                        target: $("#l-target").checked, heat: $("#l-heat").checked });

// training target / model heatmap overlays (quarter resolution PNGs), loaded on demand per field
const overlays = {};
function overlay(kind) {
  const f = S.cut.fields[S.fi].field, key = `${kind}/${S.cut.id}/${f}`;
  if (!overlays[key]) {
    overlays[key] = "loading";
    loadImage(`/img/${kind}/${S.cut.id}/${f}?t=${Date.now()}`).then((im) => { overlays[key] = im; draw(); })
      .catch(() => { overlays[key] = null; });
  }
  return overlays[key] instanceof Image ? overlays[key] : null;
}

// ------------------------------------------------------------------ tree

function renderTree() {
  const el = $("#tree"); el.innerHTML = "";
  let mouse = null, section = null;
  for (const c of S.summary) {
    if (c.mfp !== mouse) { mouse = c.mfp; section = null; el.insertAdjacentHTML("beforeend", `<div class="tree-mouse">MFP ${c.mfp}</div>`); }
    if (c.section !== section) { section = c.section; el.insertAdjacentHTML("beforeend", `<div class="tree-section">Section ${c.section}</div>`); }
    const d = document.createElement("div");
    d.className = "tree-cut" + (S.cut && S.cut.id === c.id ? " on" : "");
    d.innerHTML = `<span>Cut ${c.cut} <span class="muted">· ${c.n_fields} fields</span></span>
      <span class="n" title="manual / model"><b>${c.manual}</b> / ${c.model ?? "–"}</span>`;
    d.onclick = () => openCut(c.id);
    el.appendChild(d);
  }
}

// ------------------------------------------------------------------ cut / field loading

async function openCut(id, fi = 0, view = null) {
  S.edit = null;
  S.cut = await api(`/api/cut/${id}`);
  S.fi = fi; S.selected = null;
  S.mosaic = await loadImage(`/img/mosaic/${id}?v=${S.cut.mosaic_v}`);
  S.mosaicPts = await api(`/api/mosaic_points/${id}`);
  location.hash = id;
  renderTree(); renderStrip(); renderHead();
  setView(view || (S.view === "summary" ? "overview" : S.view), true);
  if (S.view === "field") await openField(fi); else { fitView(); renderCounts(); }
}

let fieldReq = 0;
async function openField(fi, keepView = false) {
  const req = ++fieldReq, cut = S.cut;
  S.fi = fi; S.selected = null;
  const f = cut.fields[fi];
  const [data, img] = await Promise.all([api(`/api/field/${cut.id}/${f.field}`), loadImage(`/img/field/${cut.id}/${f.field}?s=2`)]);
  if (req !== fieldReq || cut !== S.cut) return;   // a newer field was requested meanwhile
  data.cutId = cut.id; data.fieldNo = f.field;
  S.field = data; S.img = img;
  $("#reviewed").checked = data.reviewed;
  if (!keepView) fitView();
  renderStrip(); renderHead(); renderCounts(); draw();
  // prefetch neighbours
  for (const j of [fi - 1, fi + 1]) if (S.cut.fields[j]) new Image().src = `/img/field/${S.cut.id}/${S.cut.fields[j].field}?s=2`;
}

function loadImage(src) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
}

function renderHead() {
  if (S.view === "summary") {
    $("#title").textContent = "All sections";
    $("#subtitle").textContent = `${S.summary.length} cuts · manual vs model counts`;
    return;
  }
  const c = S.cut; if (!c) return;
  const f = c.fields[S.fi];
  $("#title").textContent = `T47D MFP ${c.mfp} · Section ${c.section} · Cut ${c.cut}`;
  $("#subtitle").textContent = S.view === "field"
    ? `${f.name} — field ${S.fi + 1} of ${c.fields.length}`
    : `${c.fields.length} stitched fields · ${c.reviewed}/${c.n_fields} reviewed` + (c.model_name ? ` · predictions: ${c.model_name}` : "");
  $("#hint").textContent = S.view === "field"
    ? "Click: add · click mark: select · Del / right-click: remove · drag: pan · wheel: zoom · ←/→: field · ↑/↓: cut"
    : S.edit ? "Alignment: drag a field to move it · Q / E rotate 0.5° (Shift: 5°) · drag empty space to pan · Esc cancel"
    : "Overview of the stitched cut · click a location to open that field · ↑/↓: cut";
}

function renderStrip() {
  const el = $("#strip"); el.innerHTML = "";
  S.cut.fields.forEach((f, i) => {
    const d = document.createElement("div");
    d.className = "thumb" + (S.view === "field" && i === S.fi ? " on" : "") + (f.reviewed ? " rev" : "");
    d.innerHTML = `<img loading="lazy" src="/img/field/${S.cut.id}/${f.field}?s=8">
      <div class="cap"><span class="f">F${f.field}</span><span><b>${f.manual}</b> / <span class="m">${f.model ?? "–"}</span></span></div>`;
    d.onclick = () => { setView("field"); openField(i); };
    el.appendChild(d);
  });
}

function renderCounts() {
  const c = S.cut; if (!c) return;
  const sec = S.summary.filter((x) => x.mfp === c.mfp && x.section === c.section);
  const mean = (k) => sec.some((x) => x[k] == null) ? "–" : (sec.reduce((a, x) => a + x[k], 0) / sec.length).toFixed(1);
  const f = c.fields[S.fi];
  const at = (n, a) => n == null ? "–" : `${n}${a ? ` <span class="sub">(${a} atyp.)</span>` : ""}`;
  $("#counts").innerHTML = `
    <tr><th></th><th>Manual</th><th>Model</th></tr>
    ${S.view === "field" ? `<tr><td>Field ${f.field}</td><td>${at(f.manual, f.manual_atypical)}</td><td>${at(f.model, f.model_atypical)}</td></tr>` : ""}
    <tr><td>Cut ${c.cut}</td><td class="big">${at(c.manual, c.manual_atypical)}</td><td class="big">${at(c.model, c.model_atypical)}</td></tr>
    <tr><td>Section ${c.section} <span class="sub">mean of ${sec.length} cuts</span></td><td>${mean("manual")}</td><td>${mean("model")}</td></tr>`;
}

function renderSummary() {
  const rows = [];
  const fmt = (d) => d == null ? "" : `<span class="${d > 0 ? "pos" : d < 0 ? "neg" : ""}">${d > 0 ? "+" : ""}${d}</span>`;
  const groups = {};
  for (const c of S.summary) (groups[`${c.mfp}.${c.section}`] ??= []).push(c);
  for (const [k, cs] of Object.entries(groups)) {
    for (const c of cs) {
      rows.push(`<tr class="cut" data-id="${c.id}"><td>MFP ${c.mfp}</td><td>Section ${c.section}</td><td>Cut ${c.cut}</td>
        <td>${c.n_fields}</td><td>${c.reviewed}/${c.n_fields}</td><td>${c.manual}</td><td>${c.manual_atypical || ""}</td>
        <td>${c.model ?? "–"}</td><td>${c.model_atypical || ""}</td><td>${c.model == null ? "" : fmt(c.model - c.manual)}</td></tr>`);
    }
    const mean = (key) => cs.some((c) => c[key] == null) ? null : cs.reduce((a, c) => a + c[key], 0) / cs.length;
    const mm = mean("manual"), md = mean("model");
    rows.push(`<tr class="sec"><td>MFP ${cs[0].mfp}</td><td>Section ${cs[0].section}</td><td>mean of ${cs.length} cuts</td>
      <td></td><td></td><td>${mm.toFixed(1)}</td><td></td><td>${md == null ? "–" : md.toFixed(1)}</td><td></td>
      <td>${md == null ? "" : fmt(+(md - mm).toFixed(1))}</td></tr>`);
  }
  $("#summary").innerHTML = `<table><thead><tr><th>Mouse</th><th>Section</th><th>Cut</th><th>Fields</th><th>Reviewed</th>
    <th>Manual</th><th>Atyp.</th><th>Model</th><th>Atyp.</th><th>Model − manual</th></tr></thead><tbody>${rows.join("")}</tbody></table>
    <p class="note">Counts are unique ducts per cut: overlapping fields are stitched and a duct is counted once, in the first
    field that shows it. For Ambre's mice, model counts come from the model that was not trained on that mouse.</p>`;
  document.querySelectorAll("#summary tr.cut").forEach((r) => (r.onclick = () => { setView("overview", true); openCut(r.dataset.id, 0, "overview"); }));
}

function setView(v, silent = false) {
  S.view = v;
  if (S.edit && v !== "overview") S.edit = null;
  alignButtons();
  document.querySelectorAll("#view-switch button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("#summary").hidden = v !== "summary";
  $("#viewer").hidden = $("#strip").hidden = v === "summary";
  $("#field-tools").classList.toggle("disabled", v !== "field");
  if (v === "summary") { renderSummary(); renderHead(); return; }
  $("#viewer").classList.toggle("overview", v === "overview");
  resize();  // the canvas has no size while hidden
  if (silent || !S.cut) return;
  if (v === "field") openField(S.fi); else { renderStrip(); renderHead(); renderCounts(); fitView(); }
}

// ------------------------------------------------------------------ view transform

const canvas = $("#canvas"), ctx = canvas.getContext("2d");

function imgSize() {
  return S.view === "field" ? [S.img.width / IMG_SCALE, S.img.height / IMG_SCALE] : [S.mosaic.width, S.mosaic.height];
}

function fitView() {
  if (!S.cut) return;
  if (S.view === "field" && !S.img) return;
  const r = canvas.getBoundingClientRect();
  const [w, h] = imgSize();
  const k = Math.min(r.width / w, r.height / h) * 0.97;
  S.tf = { k, x: (r.width - w * k) / 2, y: (r.height - h * k) / 2 };
  draw();
}

const toScreen = (x, y) => [x * S.tf.k + S.tf.x, y * S.tf.k + S.tf.y];
const toImage = (sx, sy) => [(sx - S.tf.x) / S.tf.k, (sy - S.tf.y) / S.tf.k];

function resize() {
  const r = canvas.getBoundingClientRect(), d = devicePixelRatio || 1;
  canvas.width = r.width * d; canvas.height = r.height * d;
  ctx.setTransform(d, 0, 0, d, 0, 0);
  draw();
}

// ------------------------------------------------------------------ drawing

function ring(x, y, r, color, width = 2, dash = null, alpha = 1) {
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
}

function dot(x, y, r, color) {
  ctx.beginPath(); ctx.arc(x, y, r + 1.5, 0, 2 * Math.PI); ctx.fillStyle = "#fff"; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
}

function label(x, y, t, color) {
  ctx.font = "11px -apple-system, Helvetica, sans-serif";
  const w = ctx.measureText(t).width + 6;
  ctx.fillStyle = "rgba(255,255,255,.85)"; ctx.fillRect(x, y - 7, w, 14);
  ctx.fillStyle = color; ctx.fillText(t, x + 3, y + 4);
}

function draw() {
  const r = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  if (!S.cut || (S.view === "field" && (!S.img || !S.field)) || (S.view === "overview" && !S.mosaic)) return;
  const L = layers(), C = { normal: css("--normal"), atypical: css("--atypical"), model: css("--model"), same: css("--same"), cuts: css("--cuts"), accent: css("--select") };
  const zr = Math.max(7, Math.min(18, 140 * S.tf.k));   // marker radius follows zoom within limits

  if (S.view === "overview" && S.edit) { drawAlignment(); return; }
  if (S.view === "overview") {
    const [w, h] = imgSize();
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(S.mosaic, S.tf.x, S.tf.y, w * S.tf.k, h * S.tf.k);
    S.cut.F.forEach((F, i) => {     // field numbers at field centres
      const [sx, sy] = toScreen(F[0][0] * FOV_C[0] + F[0][1] * FOV_C[1] + F[0][2], F[1][0] * FOV_C[0] + F[1][1] * FOV_C[1] + F[1][2]);
      label(sx - 10, sy, `F${S.cut.fields[i].field}`, "#333");
    });
    const rr = Math.max(4, Math.min(9, 6 * S.tf.k));
    if (L.model) for (const p of S.mosaicPts.model) { const [x, y] = toScreen(p.x, p.y); ring(x, y, rr + 4, C.model, 1.5); }
    if (L.manual) for (const p of S.mosaicPts.manual) { const [x, y] = toScreen(p.x, p.y); dot(x, y, rr * 0.6, C[p.cls]); }
    return;
  }

  const [w, h] = imgSize();
  ctx.drawImage(S.img, S.tf.x, S.tf.y, w * S.tf.k, h * S.tf.k);
  for (const kind of ["target", "heatmap"]) {
    if (!(kind === "target" ? L.target : L.heat)) continue;
    const im = overlay(kind);
    if (im) ctx.drawImage(im, S.tf.x, S.tf.y, w * S.tf.k, h * S.tf.k);
  }
  const F = S.field;
  if (L.cuts) for (const p of F.other_cuts) {
    const [x, y] = toScreen(p.x, p.y); ring(x, y, zr + 6, C.cuts, 1.5, [2, 3]);
    if (S.tf.k > 0.25) label(x + zr + 6, y - zr, `cut ${p.cut.split(".")[2]}`, C.cuts);
  }
  if (L.same) for (const p of F.same_cut) {
    const [x, y] = toScreen(p.x, p.y); ring(x, y, zr, C.same, 2, [5, 4]);
    label(x + zr + 2, y - zr, `counted in F${p.field}`, "#555");
  }
  if (L.model) for (const p of F.model) {
    const [x, y] = toScreen(p.x, p.y);
    if (p.counted) ring(x, y, zr + 3, p.cls === "atypical" ? C.atypical : C.model, 2);
    else ring(x, y, zr + 3, C.model, 1, [3, 3], 0.6);
  }
  if (L.manual) for (const p of F.points) {
    const [x, y] = toScreen(p.x, p.y);
    ring(x, y, zr - 2, "#fff", 4); ring(x, y, zr - 2, C[p.cls], 2.5); dot(x, y, 2.5, C[p.cls]);
    if (p.id === S.selected) ring(x, y, zr + 7, C.accent, 2.5);
  }
}

// ------------------------------------------------------------------ manual alignment (overview)
// Each field is drawn from its thumbnail (1/8 resolution) with its own transform F (full-res field
// pixels -> mosaic pixels). Drag moves the selected field, Q/E rotate it around its centre.

const THUMB = 8;

async function startAlignEdit() {
  const thumbs = await Promise.all(S.cut.fields.map((f) => loadImage(`/img/field/${S.cut.id}/${f.field}?s=${THUMB}`)));
  S.edit = { F: S.cut.F.map((m) => m.map((r) => r.slice())), sel: null, thumbs, dirty: false };
  alignButtons(); draw();
}

function stopAlignEdit() { S.edit = null; alignButtons(); draw(); }

function alignButtons() {
  const on = !!S.edit;
  $("#align-edit").hidden = on || S.view !== "overview";
  $("#align-save").hidden = $("#align-cancel").hidden = !on;
  renderHead();
}

function drawAlignment() {
  const d = devicePixelRatio || 1, E = S.edit, k = S.tf.k;
  const order = [...E.F.keys()].reverse();          // field 1 on top, like the mosaic
  if (E.sel != null) { order.splice(order.indexOf(E.sel), 1); order.push(E.sel); }
  for (const i of order) {
    const F = E.F[i], im = E.thumbs[i];
    ctx.save();
    ctx.setTransform(d * k * F[0][0] * THUMB, d * k * F[1][0] * THUMB, d * k * F[0][1] * THUMB, d * k * F[1][1] * THUMB,
                     d * (k * F[0][2] + S.tf.x), d * (k * F[1][2] + S.tf.y));
    ctx.beginPath(); ctx.arc(FOV_C[0] / THUMB, FOV_C[1] / THUMB, FOV_R / THUMB, 0, 2 * Math.PI); ctx.clip();
    ctx.globalAlpha = i === E.sel ? 0.75 : 1;
    ctx.drawImage(im, 0, 0);
    ctx.restore();
  }
  E.F.forEach((F, i) => {
    const [cx, cy] = toScreen(...applyF(F, FOV_C));
    const r = FOV_R * Math.hypot(F[0][0], F[1][0]) * k;
    ring(cx, cy, r, i === E.sel ? css("--accent") : "rgba(255,255,255,.6)", i === E.sel ? 2.5 : 1, i === E.sel ? null : [4, 4]);
    label(cx - 10, cy, `F${S.cut.fields[i].field}`, "#333");
  });
}

const applyF = (F, [x, y]) => [F[0][0] * x + F[0][1] * y + F[0][2], F[1][0] * x + F[1][1] * y + F[1][2]];

function fieldAt(mx, my, F) {
  for (let i = 0; i < F.length; i++) {
    const [cx, cy] = applyF(F[i], FOV_C);
    if (Math.hypot(mx - cx, my - cy) < FOV_R * Math.hypot(F[i][0][0], F[i][1][0])) return i;
  }
  return null;
}

function rotateSel(deg) {
  const E = S.edit; if (!E || E.sel == null) return;
  const F = E.F[E.sel], [cx, cy] = applyF(F, FOV_C), a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  const R = [[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy]];
  E.F[E.sel] = [
    [R[0][0] * F[0][0] + R[0][1] * F[1][0], R[0][0] * F[0][1] + R[0][1] * F[1][1], R[0][0] * F[0][2] + R[0][1] * F[1][2] + R[0][2]],
    [R[1][0] * F[0][0] + R[1][1] * F[1][0], R[1][0] * F[0][1] + R[1][1] * F[1][1], R[1][0] * F[0][2] + R[1][1] * F[1][2] + R[1][2]],
    [0, 0, 1]];
  E.dirty = true; draw();
}

async function saveAlignment() {
  try {
    await api(`/api/alignment/${S.cut.id}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ F: S.edit.F }) });
    S.lastJobStatus = "running";
    stopAlignEdit(); pollJobs();
  } catch (e) { $("#job-log").textContent = "Could not save alignment: " + e.message; }
}

$("#align-edit").onclick = startAlignEdit;
$("#align-cancel").onclick = stopAlignEdit;
$("#align-save").onclick = saveAlignment;

// ------------------------------------------------------------------ editing

// Saves are per field: the target and the points are captured at edit time, so navigating
// away before the debounce fires still writes to the right field.
const pending = new Map();
function save() {
  const F = S.field, key = `${F.cutId}/${F.fieldNo}`;
  clearTimeout(pending.get(key));
  pending.set(key, setTimeout(async () => {
    pending.delete(key);
    const s = await api(`/api/field/${key}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: F.points, reviewed: F.reviewed }),
    });
    const i = S.summary.findIndex((x) => x.id === s.id); if (i >= 0) S.summary[i] = s;
    for (const k of Object.keys(overlays)) if (k.startsWith(`target/${s.id}/`)) delete overlays[k];
    renderTree(); draw();
    if (!S.cut || s.id !== S.cut.id) return;
    Object.assign(S.cut, { ...s, F: S.cut.F, mosaic_size: S.cut.mosaic_size, others: S.cut.others, aligned: S.cut.aligned });
    S.mosaicPts = await api(`/api/mosaic_points/${S.cut.id}`);
    renderStrip(); renderCounts(); renderHead();
  }, 250));
}

function hit(sx, sy) {
  let best = null, bd = 14;
  for (const p of S.field.points) {
    const [x, y] = toScreen(p.x, p.y), d = Math.hypot(x - sx, y - sy);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

function insideFov(x, y) { return Math.hypot(x - FOV_C[0], y - FOV_C[1]) < FOV_R; }

function clickField(sx, sy) {
  const p = hit(sx, sy);
  if (p) { S.selected = p.id; draw(); return; }
  const [x, y] = toImage(sx, sy);
  const [w, h] = imgSize();
  if (x < 0 || y < 0 || x > w || y > h) return;
  const n = { id: uid(), x: Math.round(x), y: Math.round(y), cls: S.cls, src: "manual" };
  S.field.points.push(n); S.selected = n.id;
  draw(); save();
}

function removePoint(id) {
  S.field.points = S.field.points.filter((p) => p.id !== id);
  if (S.selected === id) S.selected = null;
  draw(); save();
}

function clickOverview(sx, sy) {
  const [mx, my] = toImage(sx, sy);
  // first field whose field of view contains the clicked point (same rule as the counts)
  for (let i = 0; i < S.cut.F.length; i++) {
    const F = S.cut.F[i], det = F[0][0] * F[1][1] - F[0][1] * F[1][0];
    const dx = mx - F[0][2], dy = my - F[1][2];
    const x = (F[1][1] * dx - F[0][1] * dy) / det, y = (-F[1][0] * dx + F[0][0] * dy) / det;
    if (insideFov(x, y)) {
      setView("field", true);
      openField(i).then(() => {
        const r = canvas.getBoundingClientRect(), k = Math.min(r.width, r.height) / 1400;
        S.tf = { k, x: r.width / 2 - x * k, y: r.height / 2 - y * k }; draw();
      });
      return;
    }
  }
}

// ------------------------------------------------------------------ input

let drag = null;
canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  drag = { x: e.clientX, y: e.clientY, tx: S.tf.x, ty: S.tf.y, moved: false };
  if (S.edit) {  // alignment editor: grab the field under the cursor (or pan if none)
    const r = canvas.getBoundingClientRect();
    const i = fieldAt(...toImage(e.clientX - r.left, e.clientY - r.top), S.edit.F);
    S.edit.sel = i;
    if (i != null) drag.F0 = S.edit.F[i].map((row) => row.slice());
    draw();
  }
});
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.hypot(dx, dy) > 4) drag.moved = true;
  if (!drag.moved) return;
  if (S.edit && drag.F0) {
    const F = S.edit.F[S.edit.sel];
    F[0][2] = drag.F0[0][2] + dx / S.tf.k; F[1][2] = drag.F0[1][2] + dy / S.tf.k;
    S.edit.dirty = true;
  } else { S.tf.x = drag.tx + dx; S.tf.y = drag.ty + dy; }
  draw();
});
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.moved || !S.cut || S.edit) return;
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  if (S.view === "field") clickField(sx, sy); else clickOverview(sx, sy);
});
canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (S.view !== "field") return;
  const r = canvas.getBoundingClientRect(), p = hit(e.clientX - r.left, e.clientY - r.top);
  if (p) removePoint(p.id);
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  const f = Math.exp(-e.deltaY * 0.0015);
  S.tf.x = sx - (sx - S.tf.x) * f; S.tf.y = sy - (sy - S.tf.y) * f; S.tf.k *= f;
  draw();
}, { passive: false });

// Up/down: previous/next cut (across sections and mice). In the overview it stays in the overview;
// in the field view it lands on the field that shows the same tissue when the two cuts are reliably
// aligned, else on the same field number.
async function stepCut(dir) {
  const i = S.summary.findIndex((c) => c.id === S.cut.id) + dir;
  if (i < 0 || i >= S.summary.length) return;
  const target = S.summary[i].id;
  if (S.view === "overview") return openCut(target, 0, "overview");
  const same = S.summary[i].mfp === S.cut.mfp && S.summary[i].section === S.cut.section;
  const r = same ? await api(`/api/corresponding/${S.cut.id}/${S.cut.fields[S.fi].field}/${target}`)
                 : { index: 0 };
  await openCut(target, r.index, "field");
}

function setCls(c) {
  S.cls = c;
  document.querySelectorAll("#cls-switch button").forEach((b) => b.classList.toggle("on", b.dataset.cls === c));
  if (S.view === "field" && S.selected) {
    const p = S.field.points.find((q) => q.id === S.selected);
    if (p && p.cls !== c) { p.cls = c; draw(); save(); }
  }
}

window.addEventListener("keydown", (e) => {
  if ((e.target.tagName === "INPUT" && e.target.type !== "checkbox") || !S.cut) return;
  if (e.target.type === "checkbox" && e.key.startsWith("Arrow")) e.preventDefault();
  if (S.edit) {  // alignment editor keys
    if (e.key === "q" || e.key === "Q") rotateSel(e.shiftKey ? -5 : -0.5);
    else if (e.key === "e" || e.key === "E") rotateSel(e.shiftKey ? 5 : 0.5);
    else if (e.key === "Escape") stopAlignEdit();
    return;
  }
  if (e.key === "1") setCls("normal");
  else if (e.key === "2") setCls("atypical");
  else if ((e.key === "Delete" || e.key === "Backspace") && S.selected && S.view === "field") { e.preventDefault(); removePoint(S.selected); }
  else if (e.key === "ArrowRight" && S.view === "field" && S.fi < S.cut.fields.length - 1) openField(S.fi + 1);
  else if (e.key === "ArrowLeft" && S.view === "field" && S.fi > 0) openField(S.fi - 1);
  else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && S.view !== "summary") { e.preventDefault(); stepCut(e.key === "ArrowDown" ? 1 : -1); }
  else if (e.key === "m" || e.key === "M") { $("#l-model").checked = !$("#l-model").checked; draw(); }
  else if (e.key === "t" || e.key === "T") { $("#l-target").checked = !$("#l-target").checked; draw(); }
  else if (e.key === "h" || e.key === "H") { $("#l-heat").checked = !$("#l-heat").checked; draw(); }
  else if (e.key === "f" || e.key === "F") fitView();
  else if (e.key === "Escape") { S.selected = null; draw(); }
});

document.querySelectorAll("#view-switch button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
document.querySelectorAll("#cls-switch button").forEach((b) => (b.onclick = () => setCls(b.dataset.cls)));
["#l-manual", "#l-model", "#l-same", "#l-cuts", "#l-target", "#l-heat"].forEach((s) => ($(s).onchange = draw));
$("#reviewed").onchange = (e) => { S.field.reviewed = e.target.checked; save(); };
$("#clear-field").onclick = () => {
  if (!S.field.points.length) return;
  S.field.points = []; draw(); save();
};
$("#accept-model").onclick = () => {
  const pts = S.field.model.filter((p) => p.counted).map((p) => ({ id: uid(), x: p.x, y: p.y, cls: p.cls, src: "model" }));
  S.field.points = pts; draw(); save();
};

// ------------------------------------------------------------------ jobs

async function pollJobs() {
  try {
    const j = await api("/api/jobs");
    $("#model-label").textContent = j.model ? `Model: ${j.model}` : "No model yet";
    const last = j.jobs[0];
    const pill = $("#job-pill");
    if (last) {
      pill.hidden = false;
      pill.className = "pill " + last.status;
      pill.textContent = `${{ train: "Training", infer: "Prediction", realign: "Re-alignment" }[last.kind] || last.kind}: ${last.status}`;
      $("#job-log").textContent = last.log.join("\n");
      const running = last.status === "running";
      $("#run-train").disabled = $("#run-infer").disabled = running;
      if (S.lastJobStatus === "running" && !running) await refresh();
      S.lastJobStatus = last.status;
    }
  } catch (e) { /* server restarting */ }
  setTimeout(pollJobs, 3000);
}

async function startJob(kind) {
  try { await api(`/api/jobs/${kind}`, { method: "POST" }); S.lastJobStatus = "running"; pollJobs(); }
  catch (e) { alert(e.message); }
}
$("#run-train").onclick = () => startJob("train");
$("#run-infer").onclick = () => startJob("infer");

async function refresh() {
  S.summary = await api("/api/summary");
  renderTree();
  if (S.cut) await openCut(S.cut.id, S.fi, S.view);
}

// ------------------------------------------------------------------ start

window.addEventListener("resize", resize);
window.addEventListener("hashchange", () => {
  const id = location.hash.slice(1);
  if (S.cut && id !== S.cut.id && S.summary.some((c) => c.id === id)) openCut(id, 0);
});
(async () => {
  resize();
  S.summary = await api("/api/summary");
  renderTree();
  const id = location.hash.slice(1);
  await openCut(S.summary.some((c) => c.id === id) ? id : S.summary[0].id);
  pollJobs();
})();
