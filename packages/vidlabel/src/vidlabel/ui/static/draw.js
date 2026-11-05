"use strict";
/* Drawing tools for the viewer: select and edit, box, span, brush, eraser and polygon.
   Everything is recorded in the author's draft (see app.js); nothing reaches the dataset until it is committed.
   Drawing happens on the whole frame, so coordinates are always the video's own pixels. */

const TOOLS = [["select", "Select", "V"], ["box", "Box", "B"], ["span", "Span", "S"], ["brush", "Brush", "M"], ["eraser", "Eraser", "E"], ["polygon", "Polygon", "P"]];
const HANDLES = { nw: [0, 0], n: [.5, 0], ne: [1, 0], e: [1, .5], se: [1, 1], s: [.5, 1], sw: [0, 1], w: [0, .5] };
const MASK_TOOLS = ["brush", "eraser", "polygon"];

function drawInit() {
  V.tool = "select"; V.label = (S.info.labels[0] || {}).name || ""; V.track = "none";
  V.work = null; V.drag = null; V.poly = []; V.cursor = null; V.spanDrag = null; V.stroke = null;
  V.brush = Math.max(3, Math.round(Math.min(V.size[0], V.size[1]) / 40));
}
const canDraw = () => !V.view;
const editableBox = r => r && r.kind === "box" && r.pending !== "deleted" && r.x0 != null;

/* ---------- toolbar ---------- */
function toolbarEl() { V.el.tools = h("div", { class: "tools row tight" }); paintToolbar(); return V.el.tools; }
function paintToolbar() {
  if (!V || !V.el.tools) return;
  const on = canDraw(), labels = S.info.labels, tracks = V.tl ? V.tl.tracks : [];
  if (!labels.some(l => l.name === V.label)) V.label = (labels[0] || {}).name || "";
  const labelSel = h("select", { "aria-label": "Label for new drawings", title: "Label for new drawings", onchange: e => { V.label = e.target.value; retint(); draw(); } },
    labels.map(l => h("option", { value: l.name, selected: l.name === V.label }, l.name)));
  const trackSel = h("select", { "aria-label": "Track for new boxes and masks", title: "Track for new boxes and masks", onchange: e => { V.track = e.target.value; } },
    h("option", { value: "none", selected: V.track === "none" }, "No track"), h("option", { value: "new", selected: V.track === "new" }, "New track"),
    tracks.map(t => h("option", { value: String(t.track_id), selected: V.track === String(t.track_id) }, `Track ${t.track_id} (${t.labels.join(", ")})`)));
  V.el.tools.replaceChildren(
    TOOLS.map(([k, name, key]) => h("button", { class: "small" + (V.tool === k ? " on" : ""), "aria-pressed": V.tool === k, disabled: !on && k !== "select", title: `${name} (${key})`, onclick: () => setTool(k) }, name)),
    on ? (labels.length ? [labelSel, trackSel] : h("a", { href: "#/labels" }, "Add a label to start drawing")) : h("span", { class: "muted" }, "Choose “Whole frame” to draw. Rows drawn there reach a view with “Copy labels in”."));
}
function setTool(k) {
  if (!canDraw() && k !== "select") return;
  V.tool = k; V.poly = []; V.drag = null; V.stroke = null;
  if (MASK_TOOLS.includes(k)) ensureWork();
  V.el.ov.style.cursor = k === "select" ? "default" : "crosshair";
  paintToolbar(); drawSide(); drawOverlay();
}
function needLabel() {
  if (!canDraw()) return false;
  if (!V.label) { toast("Add a label first (Labels page), then draw.", true); return false; }
  return true;
}
function nextTrackId() { return Math.max(0, ...((V.tl && V.tl.tracks) || []).map(t => t.track_id)) + 1; }
function trackValue() { return V.track === "none" ? null : V.track === "new" ? nextTrackId() : +V.track; }
async function afterEdit() { await show(V.n); await loadTimeline(); paintToolbar(); }

/* ---------- pointer events on the overlay ---------- */
function bindDrawing(ov) {
  ov.addEventListener("pointerdown", ev => { if (ev.button !== 0) return; ov.setPointerCapture(ev.pointerId); onDown(ev); });
  ov.addEventListener("pointermove", onMove);
  ov.addEventListener("pointerup", onUp);
  ov.addEventListener("pointercancel", () => { V.drag = null; V.stroke = null; drawOverlay(); });
  ov.addEventListener("dblclick", () => { if (V.tool === "polygon") closePoly(); });
  ov.addEventListener("pointerleave", () => { V.cursor = null; drawOverlay(); });
}
function pt(ev) { const c = V.el.ov, r = c.getBoundingClientRect(); return [(ev.clientX - r.left) * c.width / r.width, (ev.clientY - r.top) * c.height / r.height]; }
const clampPt = p => [Math.max(0, Math.min(V.size[0], p[0])), Math.max(0, Math.min(V.size[1], p[1]))];
const pxScale = () => { const c = V.el.ov, r = c.getBoundingClientRect(); return c.width / Math.max(1, r.width); };

function onDown(ev) {
  const p = clampPt(pt(ev));
  if (V.tool === "select") return selectDown(p);
  if (V.tool === "span") return;
  if (!needLabel()) return;
  if (V.tool === "box") V.drag = { kind: "new", a: p, rect: [p[0], p[1], p[0], p[1]] };
  else if (V.tool === "brush" || V.tool === "eraser") { ensureWork(); pushSnap(); V.stroke = { erase: V.tool === "eraser", last: p }; strokeTo(p, p); }
  else if (V.tool === "polygon") polyClick(p);
}
function onMove(ev) {
  const p = clampPt(pt(ev));
  V.cursor = p;
  if (V.drag) {
    const d = V.drag;
    if (d.kind === "new") d.rect = [d.a[0], d.a[1], p[0], p[1]];
    else if (d.kind === "move") { const dx = p[0] - d.start[0], dy = p[1] - d.start[1], [x0, y0, x1, y1] = d.orig;
      const w = x1 - x0, hh = y1 - y0, nx = Math.max(0, Math.min(V.size[0] - w, x0 + dx)), ny = Math.max(0, Math.min(V.size[1] - hh, y0 + dy)); d.rect = [nx, ny, nx + w, ny + hh]; }
    else if (d.kind === "resize") { const r = d.orig.slice(); if (d.handle.includes("w")) r[0] = p[0]; if (d.handle.includes("e")) r[2] = p[0]; if (d.handle.includes("n")) r[1] = p[1]; if (d.handle.includes("s")) r[3] = p[1]; d.rect = r; }
  } else if (V.stroke) { strokeTo(V.stroke.last, p); V.stroke.last = p; return; }
  drawOverlay();
}
async function onUp(ev) {
  if (V.stroke) { V.stroke = null; drawSide(); return; }
  const d = V.drag; if (!d) return;
  V.drag = null;
  const [a, b, c, e] = d.rect, r = [Math.min(a, c), Math.min(b, e), Math.max(a, c), Math.max(b, e)].map(x => Math.round(x * 10) / 10);
  if (d.kind === "new") {
    if (r[2] - r[0] < 3 || r[3] - r[1] < 3) { drawOverlay(); return; }
    const row = { kind: "box", label: V.label, start_frame: V.n, box: r }, tv = trackValue();
    if (tv != null) row.track_id = tv;
    const res = await draftOp("/api/draft/add", { video_id: V.vid, row });
    if (res) { V.sel = res.id; if (V.track === "new") V.track = String(tv); await afterEdit(); }
  } else {
    const moved = r.some((x, i) => Math.abs(x - d.orig[i]) > 0.5);
    if (moved && r[2] - r[0] >= 2 && r[3] - r[1] >= 2) { if (await draftOp("/api/draft/update", { video_id: V.vid, ids: [d.row.annotation_id], values: { box: r } })) await afterEdit(); }
  }
  drawOverlay();
}

/* ---------- select, move and resize ---------- */
function handleAt(row, p) {
  const tol = 9 * pxScale(), [x0, y0, x1, y1] = [row.x0, row.y0, row.x1, row.y1];
  for (const [name, [fx, fy]] of Object.entries(HANDLES)) if (Math.abs(p[0] - (x0 + (x1 - x0) * fx)) <= tol && Math.abs(p[1] - (y0 + (y1 - y0) * fy)) <= tol) return name;
  return null;
}
function selectDown(p) {
  const cur = V.rows.find(r => r.annotation_id === V.sel);
  if (editableBox(cur)) {
    const hd = handleAt(cur, p);
    if (hd) { V.drag = { kind: "resize", handle: hd, row: cur, orig: [cur.x0, cur.y0, cur.x1, cur.y1], rect: [cur.x0, cur.y0, cur.x1, cur.y1] }; return; }
  }
  const hits = V.rows.filter(a => a.kind !== "marker" && a.x0 != null && p[0] >= a.x0 && p[0] <= a.x1 && p[1] >= a.y0 && p[1] <= a.y1)
    .sort((a, b) => (a.x1 - a.x0) * (a.y1 - a.y0) - (b.x1 - b.x0) * (b.y1 - b.y0));
  V.sel = hits.length ? hits[0].annotation_id : null;
  const row = hits[0];
  if (editableBox(row)) V.drag = { kind: "move", row, start: p, orig: [row.x0, row.y0, row.x1, row.y1], rect: [row.x0, row.y0, row.x1, row.y1] };
  draw(); drawSide(); drawOverlay();
}

/* ---------- masks: a working layer the size of the frame ---------- */
function ensureWork() {
  if (V.work) return;
  const mk = () => { const c = document.createElement("canvas"); c.width = V.size[0]; c.height = V.size[1]; return c; };
  V.work = { cv: mk(), tint: mk(), target: null, label: null, snaps: [] };
  drawSide();
}
function pushSnap() { const w = V.work; w.snaps.push(w.cv.getContext("2d").getImageData(0, 0, w.cv.width, w.cv.height)); if (w.snaps.length > 8) w.snaps.shift(); }
function undoStroke() {
  const w = V.work; if (!w || !w.snaps.length) return false;
  w.cv.getContext("2d").putImageData(w.snaps.pop(), 0, 0); retint(); draw(); drawOverlay(); return true;
}
function retint() {
  const w = V && V.work; if (!w) return;
  const x = w.tint.getContext("2d"); x.globalCompositeOperation = "source-over"; x.clearRect(0, 0, w.tint.width, w.tint.height);
  x.drawImage(w.cv, 0, 0); x.globalCompositeOperation = "source-in"; x.fillStyle = labelColor(w.label || V.label); x.fillRect(0, 0, w.tint.width, w.tint.height); x.globalCompositeOperation = "source-over";
}
let _raf = 0;
function strokeTo(a, b) {
  const x = V.work.cv.getContext("2d");
  x.globalCompositeOperation = V.stroke.erase ? "destination-out" : "source-over";
  x.strokeStyle = x.fillStyle = "#fff"; x.lineWidth = V.brush * 2; x.lineCap = x.lineJoin = "round";
  x.beginPath(); x.moveTo(a[0], a[1]); x.lineTo(b[0], b[1]); x.stroke();
  x.beginPath(); x.arc(b[0], b[1], V.brush, 0, 2 * Math.PI); x.fill();
  x.globalCompositeOperation = "source-over";
  if (!_raf) _raf = requestAnimationFrame(() => { _raf = 0; if (V && V.work) { retint(); draw(); drawOverlay(); } });
}
function polyClick(p) {
  if (V.poly.length >= 3 && Math.hypot(p[0] - V.poly[0][0], p[1] - V.poly[0][1]) < 10 * pxScale()) return closePoly();
  V.poly.push(p); drawOverlay();
}
function closePoly() {
  if (V.poly.length < 3) { V.poly = []; drawOverlay(); return; }
  ensureWork(); pushSnap();
  const x = V.work.cv.getContext("2d"); x.fillStyle = "#fff"; x.beginPath(); V.poly.forEach((q, i) => (i ? x.lineTo(q[0], q[1]) : x.moveTo(q[0], q[1]))); x.closePath(); x.fill();
  V.poly = []; retint(); draw(); drawOverlay(); drawSide();
}
function packMask(cv) {
  const W = cv.width, H = cv.height, d = cv.getContext("2d").getImageData(0, 0, W, H).data, bytes = new Uint8Array((W * H + 7) >> 3);
  let any = false;
  for (let i = 0, j = 3; i < W * H; i++, j += 4) if (d[j] >= 128) { bytes[i >> 3] |= 0x80 >> (i & 7); any = true; }
  if (!any) return null;
  let bin = ""; for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  return { width: W, height: H, data: btoa(bin) };
}
async function saveMask() {
  const w = V.work, bits = packMask(w.cv);
  if (!bits) { toast("The mask is empty. Paint something first, or discard it.", true); return; }
  let res;
  if (w.target) res = await draftOp("/api/draft/update", { video_id: V.vid, ids: [w.target], values: { mask_bits: bits } });
  else {
    const row = { kind: "mask", label: V.label, start_frame: V.n, mask_bits: bits }, tv = trackValue();
    if (tv != null) row.track_id = tv;
    res = await draftOp("/api/draft/add", { video_id: V.vid, row });
    if (res) { V.sel = res.id; if (V.track === "new") V.track = String(tv); }
  }
  if (res) { V.work = null; if (!MASK_TOOLS.includes(V.tool)) V.tool = "select"; await afterEdit(); }
}
async function editMask(row) {
  const b = await attempt(() => GET(`/api/mask-bits/${encodeURIComponent(V.vid)}/${encodeURIComponent(row.annotation_id)}`, { frame: V.n }));
  if (!b) return;
  V.work = null; ensureWork();
  const img = new ImageData(b.width, b.height), raw = atob(b.data);
  for (let i = 0; i < b.width * b.height; i++) { const o = i * 4; img.data[o] = img.data[o + 1] = img.data[o + 2] = 255; img.data[o + 3] = (raw.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1 ? 255 : 0; }
  V.work.cv.getContext("2d").putImageData(img, 0, 0); V.work.target = row.annotation_id; V.work.label = row.label;
  retint(); setTool("brush"); draw();
}
function discardMask() { V.work = null; V.poly = []; draw(); drawSide(); drawOverlay(); }

/* ---------- spans on the timeline ---------- */
function spanStart(e) {
  if (!needLabel()) return;
  const a = tlFrame(e); V.spanDrag = { a, b: a }; drawTimeline();
  const mv = ev => { V.spanDrag.b = tlFrame(ev); drawTimeline(); };
  const up = async () => {
    removeEventListener("pointermove", mv); removeEventListener("pointerup", up);
    const s = V.spanDrag; V.spanDrag = null; if (!s) return;
    const lo = Math.min(s.a, s.b), hi = Math.max(s.a, s.b);
    if (await draftOp("/api/draft/add", { video_id: V.vid, row: { kind: "marker", label: V.label, start_frame: lo, end_frame: hi } })) { toast(lo === hi ? `Labelled frame ${lo}` : `Labelled frames ${lo} to ${hi}`); await afterEdit(); }
  };
  addEventListener("pointermove", mv); addEventListener("pointerup", up);
}

/* ---------- the overlay: what is being drawn right now ---------- */
function drawOverlay() {
  if (!V || !V.el.ov) return;
  const ov = V.el.ov, x = ov.getContext("2d"), lw = Math.max(2, ov.width / 360), s = pxScale();
  x.clearRect(0, 0, ov.width, ov.height);
  const pink = "#ff4fa6";
  if (V.drag) { const [a, b, c, e] = V.drag.rect; x.strokeStyle = pink; x.lineWidth = lw; x.setLineDash([lw * 3, lw * 2]); x.strokeRect(Math.min(a, c), Math.min(b, e), Math.abs(c - a), Math.abs(e - b)); x.setLineDash([]); }
  const sel = V.rows.find(r => r.annotation_id === V.sel);
  if (V.tool === "select" && editableBox(sel)) {
    const [x0, y0, x1, y1] = V.drag && V.drag.row === sel ? V.drag.rect : [sel.x0, sel.y0, sel.x1, sel.y1], hs = 5 * s;
    x.fillStyle = "#fff"; x.strokeStyle = "#000"; x.lineWidth = 1 * s;
    for (const [fx, fy] of Object.values(HANDLES)) { const px = x0 + (x1 - x0) * fx, py = y0 + (y1 - y0) * fy; x.fillRect(px - hs, py - hs, hs * 2, hs * 2); x.strokeRect(px - hs, py - hs, hs * 2, hs * 2); }
  }
  if (V.poly.length) {
    x.strokeStyle = pink; x.fillStyle = pink; x.lineWidth = lw;
    x.beginPath(); V.poly.forEach((q, i) => (i ? x.lineTo(q[0], q[1]) : x.moveTo(q[0], q[1]))); if (V.cursor) x.lineTo(V.cursor[0], V.cursor[1]); x.stroke();
    V.poly.forEach((q, i) => { x.beginPath(); x.arc(q[0], q[1], (i ? 3 : 6) * s, 0, 2 * Math.PI); x.fill(); });
  }
  if ((V.tool === "brush" || V.tool === "eraser") && V.cursor) { x.strokeStyle = "#fff"; x.lineWidth = 1.5 * s; x.beginPath(); x.arc(V.cursor[0], V.cursor[1], V.brush, 0, 2 * Math.PI); x.stroke(); x.strokeStyle = "#000"; x.lineWidth = .75 * s; x.stroke(); }
}

/* ---------- side panel: the mask session and the selected row ---------- */
function drawPanel() {
  const out = [];
  if (V.work) {
    const w = V.work;
    out.push(h("div", { class: "notice" }, h("b", {}, w.target ? "Editing a mask" : "New mask"), h("div", { class: "muted" }, w.target ? `Label ${w.label}. Brush adds, eraser removes, polygon adds a shape.` : `Label ${V.label}. Brush and polygon add, eraser removes.`),
      h("label", { class: "row tight", style: "margin:6px 0" }, "Brush size", h("input", { type: "range", min: 1, max: Math.max(20, Math.round(Math.min(...V.size) / 6)), value: V.brush, oninput: e => { V.brush = +e.target.value; drawOverlay(); } })),
      h("div", { class: "row tight" }, h("button", { class: "primary small", onclick: saveMask }, w.target ? "Apply to the mask" : "Save mask"), h("button", { class: "small", onclick: () => undoStroke() }, "Undo stroke"),
        h("button", { class: "small", onclick: discardMask }, "Discard"))));
  } else if (V.tool === "box") out.push(h("p", { class: "muted" }, "Drag on the frame to draw a box. It is recorded at this frame; step forward with →, and draw again to follow the same track."));
  else if (V.tool === "span") out.push(h("p", { class: "muted" }, "Drag on the timeline to label a stretch of frames. A click labels one frame."));
  else if (V.tool === "polygon") out.push(h("p", { class: "muted" }, "Click to place corners; click the first point, double-click or press Enter to close the shape. Backspace removes the last corner."));
  return out;
}
function selectionPanel() {
  const r = V.rows.find(x => x.annotation_id === V.sel);
  if (!r || r.pending === "deleted") return null;
  const id = r.annotation_id, vid = V.vid, labels = S.info.labels;
  const labelSel = h("select", { "aria-label": "Label", onchange: async e => { if (await draftOp("/api/draft/update", { video_id: vid, ids: [id], values: { label: e.target.value } })) afterEdit(); } },
    labels.map(l => h("option", { value: l.name, selected: l.name === r.label }, l.name)));
  const body = [h("div", { class: "row" }, "Label", labelSel)];
  if (r.kind !== "marker") {
    const tr = h("input", { type: "number", min: 0, style: "width:90px", value: r.track_id ?? "", placeholder: "none", "aria-label": "Track number",
      onchange: async e => { if (await draftOp("/api/draft/update", { video_id: vid, ids: [id], values: { track_id: e.target.value === "" ? null : +e.target.value } })) afterEdit(); } });
    body.push(h("div", { class: "row", style: "margin-top:6px" }, "Track", tr));
  }
  if (r.kind === "marker") {
    const a = h("input", { type: "number", min: 0, max: V.frames - 1, value: r.start_frame, style: "width:84px" }), b = h("input", { type: "number", min: 0, max: V.frames - 1, value: r.end_frame, style: "width:84px" });
    body.push(h("div", { class: "row", style: "margin-top:6px" }, "Frames", a, "to", b, h("button", { class: "small", onclick: async () => { if (await draftOp("/api/draft/update", { video_id: vid, ids: [id], values: { start_frame: +a.value, end_frame: +b.value } })) afterEdit(); } }, "Apply")));
  }
  if (r.kind === "mask") body.push(h("div", { style: "margin-top:8px" }, h("button", { class: "small", onclick: () => editMask(r) }, "Edit mask shape")));
  if (r.kind === "box") body.push(h("p", { class: "muted" }, "With the Select tool, drag the box to move it or its handles to resize."));
  return h("div", { style: "margin-top:14px;border-top:1.5px solid var(--ink);padding-top:10px" }, h("h2", {}, "Selected"), body);
}

/* ---------- keys ---------- */
function drawKey(ev) {
  if (!V || ev.altKey) return false;
  const k = ev.key, low = k.toLowerCase(), mod = ev.ctrlKey || ev.metaKey;
  if (mod && low === "z" && !ev.shiftKey && V.work && V.work.snaps.length) { undoStroke(); return true; }
  if (mod) return false;
  const tool = TOOLS.find(t => t[2].toLowerCase() === low);
  if (tool) { setTool(tool[0]); return true; }
  if (k === "Enter") { if (V.poly.length) closePoly(); else if (V.work) saveMask(); else return false; return true; }
  if (k === "Escape") { if (V.poly.length || V.drag) { V.poly = []; V.drag = null; drawOverlay(); return true; } return false; }
  if (k === "Backspace" && V.poly.length) { V.poly.pop(); drawOverlay(); return true; }
  if ((k === "Delete" || k === "Backspace") && V.sel) { const id = V.sel; draftOp("/api/draft/delete", { video_id: V.vid, ids: [id] }).then(r => r && afterEdit()); return true; }
  if (k === "[" || k === "]") { V.brush = Math.max(1, Math.round(V.brush * (k === "]" ? 1.25 : 0.8))); drawOverlay(); drawSide(); return true; }
  return false;
}
