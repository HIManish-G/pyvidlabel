"use strict";
/* The viewer: one source of truth for "which frame are we on".

   V.n is that frame. Everything on screen (the frame box, the time, the timeline playhead, the picture and what is drawn on it)
   is painted from V.n by one function, paintAll(), in one animation frame. Moving only ever means calling goto(n).

   The picture can arrive a moment after V.n changes, because the frame has to be decoded. Until it does, the page keeps showing
   the last picture together with that picture's own annotations (V.shown is the frame actually on the canvas) and marks the stage
   as loading, so a picture and its annotations never belong to different frames. Drawing is refused while they differ.

   Frames are fetched by a small scheduler (the nearest first, then the ones ahead in the direction of travel), kept decoded in an
   LRU cache, and requests the page no longer wants are cancelled here and dropped by the server. */

let V = null;
const CHUNK = 128;           // frames of annotations fetched per request

function stopViewer() {
  if (!V) return;
  const v = V; V = null;
  cancelAnimationFrame(v.raf); cancelAnimationFrame(v.playRaf); clearTimeout(v.tlFetch); clearTimeout(v.hashTimer); clearTimeout(v.sideTimer);
  document.removeEventListener("keydown", v.onkey);
  v.ro && v.ro.disconnect();
  for (const c of v.fs.pending.values()) c.abort();
  for (const b of v.fs.map.values()) b.close();
}

/* ---------- frames ---------- */
function keepFrames(v) {
  const px = Math.max(1, v.size[0] * v.size[1]);
  return Math.max(24, Math.min(240, Math.floor(400e6 / (px * 4))));
}
function storeFrame(v, n, bmp) {
  const fs = v.fs;
  fs.map.set(n, bmp);
  const limit = keepFrames(v);
  for (const [k, b] of fs.map) {
    if (fs.map.size <= limit) break;
    if (k === v.n || b === v.shownBmp) continue;
    fs.map.delete(k); b.close();
  }
}
function touchFrame(v, n) { const b = v.fs.map.get(n); if (b) { v.fs.map.delete(n); v.fs.map.set(n, b); } }

async function fetchFrame(v, n) {
  const fs = v.fs, ctl = new AbortController();
  fs.pending.set(n, ctl);
  try {
    const q = new URLSearchParams({ cid: fs.cid, seq: fs.seq }); if (v.view) q.set("view_id", v.view);
    const r = await fetch(`/api/frame/${encodeURIComponent(v.vid)}/${n}?${q}`, { signal: ctl.signal });
    if (r.status === 204) return;                      // the server saw a newer request first; pump() asks again if still wanted
    if (!r.ok) { fs.bad.add(n); return; }
    const bmp = await createImageBitmap(await r.blob());
    if (V !== v) { bmp.close(); return; }
    storeFrame(v, n, bmp);
  } catch (e) { if (e.name !== "AbortError") fs.bad.add(n); }
  finally { fs.pending.delete(n); if (V === v) { requestPaint("stage"); pump(); } }
}
/* the frames worth having, nearest first. While the current frame is missing it is the only one asked for, so the decoder is
   never busy with a neighbour when the picture the person is looking at is waiting. */
function wanted(v) {
  const out = [v.n];
  if (!v.fs.map.has(v.n)) return out;
  const d = v.dir || 1, ahead = v.playing ? 40 : 10;
  for (let i = 1; i <= ahead; i++) { const a = v.n + d * i; if (a >= 0 && a < v.frames) out.push(a); if (i <= 2) { const b = v.n - d * i; if (b >= 0 && b < v.frames) out.push(b); } }
  return out;
}
function pump() {
  const v = V; if (!v || !v.found) return;
  const want = wanted(v), set = new Set(want);
  for (const [n, ctl] of v.fs.pending) if (!set.has(n)) ctl.abort();
  for (const n of want) {
    if (v.fs.pending.size >= 4) break;
    if (!v.fs.map.has(n) && !v.fs.pending.has(n) && !v.fs.bad.has(n)) fetchFrame(v, n);
  }
}

/* ---------- annotations ---------- */
const chunkOf = n => Math.floor(n / CHUNK);
function rowsAt(v, n) {
  const c = v.ann.chunks.get(chunkOf(n));
  return Array.isArray(c) ? c.filter(r => r.start_frame <= n && n <= r.end_frame) : null;
}
function loadChunk(v, ci) {
  if (ci < 0 || ci * CHUNK >= v.frames) return Promise.resolve();
  if (v.ann.chunks.has(ci)) return v.ann.waiting.get(ci) || Promise.resolve();      // already loaded, or loading: wait for that same request
  const gen = v.ann.gen;
  v.ann.chunks.set(ci, "loading");
  const a = ci * CHUNK, b = Math.min(v.frames - 1, a + CHUNK - 1);
  const p = GET(`/api/annotations-range/${encodeURIComponent(v.vid)}`, { start: a, end: b, view_id: v.view }).then(d => {
    if (V !== v || v.ann.gen !== gen) return;
    // a row covering several chunks is returned by each; filtering by frame in rowsAt() keeps one copy per frame
    v.ann.chunks.set(ci, d.rows); requestPaint("stage");
  }).catch(e => { if (V === v && v.ann.gen === gen) { v.ann.chunks.delete(ci); toast(e.message, true); } }).finally(() => { if (v.ann.waiting.get(ci) === p) v.ann.waiting.delete(ci); });
  v.ann.waiting.set(ci, p);
  return p;
}
function ensureAnn(n) {
  const v = V, ci = chunkOf(n), p = loadChunk(v, ci), at = n % CHUNK;
  if (at > CHUNK - 24) loadChunk(v, ci + 1); else if (at < 24) loadChunk(v, ci - 1);
  return p;
}
/* after any edit: forget what was fetched and fetch the current stretch again */
async function refreshAnn() {
  const v = V; v.ann.gen++; v.ann.chunks.clear(); v.ann.waiting.clear(); v.sideKey = null;
  await ensureAnn(v.n);
  if (V === v) requestPaint("stage");
}
async function show(n) { if (n !== V.n) goto(n); await refreshAnn(); await loadTimelineSoon(); }
function loadTimelineSoon() { return loadTimeline(); }

const maskKey = (v, r) => `${r.annotation_id}:${r.mask_rev}:${v.view}`;
function maskFor(v, r, frame) {
  const key = maskKey(v, r);
  let e = v.masks.get(key);
  if (e) return e.ready ? e : null;
  e = { ready: false }; v.masks.set(key, e);
  fetch(`/api/mask-png/${encodeURIComponent(v.vid)}/${encodeURIComponent(r.annotation_id)}?` + new URLSearchParams({ frame, rev: r.mask_rev, view_id: v.view }), { headers: { "X-Vidlabel-Author": encodeURIComponent(S.author || "anonymous") } })
    .then(async res => {
      if (res.status === 204 || !res.ok) { e.empty = true; e.ready = true; return; }
      e.x = +res.headers.get("X-Mask-X"); e.y = +res.headers.get("X-Mask-Y"); e.bmp = await createImageBitmap(await res.blob()); e.tint = new Map(); e.ready = true;
    }).catch(() => { e.empty = true; e.ready = true; }).finally(() => { if (V === v) requestPaint("stage"); });
  while (v.masks.size > 300) v.masks.delete(v.masks.keys().next().value);
  return null;
}
function tinted(e, color) {
  let c = e.tint.get(color);
  if (!c) {
    c = document.createElement("canvas"); c.width = e.bmp.width; c.height = e.bmp.height;
    const x = c.getContext("2d"); x.drawImage(e.bmp, 0, 0); x.globalCompositeOperation = "source-in"; x.fillStyle = color; x.fillRect(0, 0, c.width, c.height);
    e.tint.set(color, c);
  }
  return c;
}

/* ---------- moving ---------- */
function goto(n, opts = {}) {
  const v = V; if (!v) return;
  n = Math.max(0, Math.min(v.frames - 1, Math.round(n)));
  if (n === v.n && !opts.force) return;
  if (n !== v.n) { v.dir = n > v.n ? 1 : -1; v.n = n; v.fs.seq++; }
  ensureAnn(n); pump(); followPlayhead();
  requestPaint("all");
  clearTimeout(v.hashTimer);
  v.hashTimer = setTimeout(() => { if (V === v) history.replaceState(null, "", `#/video/${encodeURIComponent(v.vid)}?f=${v.n}` + (v.view ? "&view=" + encodeURIComponent(v.view) : "")); }, 250);
}
const seek = n => goto(n);

function togglePlay(on) {
  const v = V; if (!v) return;
  const play = on === undefined ? !v.playing : on;
  if (play === v.playing) return;
  v.playing = play; v.clock = null; cancelAnimationFrame(v.playRaf);
  v.el.playBtn.textContent = play ? "Pause" : "Play"; v.el.playBtn.setAttribute("aria-pressed", play);
  if (play) { if (v.n >= v.frames - 1) goto(0); v.dir = 1; v.playRaf = requestAnimationFrame(tick); } else drawSideSoon(true);
  requestPaint("all");
}
/* playback follows a clock, not a timer: the frame that should be showing is worked out from the time that has passed, and it
   only moves on when that frame is already decoded, so the picture never races ahead of itself and a slow decode just waits */
function tick(ts) {
  const v = V; if (!v || !v.playing) return;
  v.playRaf = requestAnimationFrame(tick);
  if (!v.clock) v.clock = { t: ts, n: v.n };
  const want = v.clock.n + Math.floor((ts - v.clock.t) / 1000 * v.fps * v.rate);
  if (want <= v.n) return;
  const next = v.n + 1;
  if (next >= v.frames) { if (v.loop) { goto(0); v.clock = null; } else togglePlay(false); return; }
  if (!v.fs.map.has(next) && v.found) { v.clock = { t: ts, n: v.n }; v.buffering = true; requestPaint("stage"); pump(); return; }
  v.buffering = false;
  if (want - next > 2) v.clock = { t: ts, n: next };            // we fell behind: slow down instead of skipping
  goto(next);
}

/* ---------- painting: everything from V.n, once per animation frame ---------- */
function requestPaint(what) {
  const v = V; if (!v) return;
  v.dirty = v.dirty === "all" || what === "all" ? "all" : (v.dirty && v.dirty !== what ? "all" : what);
  if (!v.raf) v.raf = requestAnimationFrame(() => { const w = v.dirty; v.raf = 0; v.dirty = null; if (V === v) paintAll(w); });
}
const draw = () => requestPaint("stage");
const drawTimeline = () => requestPaint("timeline");
function paintAll(what) {
  const v = V;
  if (what !== "timeline") paintStage();
  paintTimeline();
  paintTransport();
  v.painted = { n: v.n, shown: v.shown, playhead: v.n, rows: (v.rows || []).length };
  v.el.root.dataset.frame = v.n; v.el.root.dataset.shown = v.shown;
  const tl = v.el.tlc; if (tl) { tl.setAttribute("aria-valuenow", v.n); tl.setAttribute("aria-valuetext", `frame ${v.n} of ${v.frames - 1}, ${timecode(v.n, v.fps, true)}`); }
}
function paintTransport() {
  const v = V, e = v.el, last = v.frames - 1;
  if (document.activeElement !== e.frameIn) e.frameIn.value = v.n;
  e.of.textContent = `/ ${last}  ·  ${timecode(v.n, v.fps, true)}`;
  e.prev.disabled = v.n <= 0; e.next.disabled = v.n >= last; e.first.disabled = v.n <= 0; e.endBtn.disabled = v.n >= last;
}
function paintStage() {
  const v = V, c = v.el.canvas, x = c.getContext("2d");
  const bmp = v.fs.map.get(v.n);
  if (bmp) { touchFrame(v, v.n); v.shownBmp = bmp; v.shown = v.n; if (v.size[0] !== bmp.width || v.size[1] !== bmp.height) { v.size = [bmp.width, bmp.height]; sizeStage(); } }
  else if (!v.found && v.shown !== v.n) v.shown = v.n;
  const lag = v.shown !== v.n && v.found;
  v.el.stage.classList.toggle("lag", lag);
  v.el.hudState.textContent = v.buffering ? "buffering" : lag ? "loading frame " + v.n : v.fs.bad.has(v.n) ? "frame could not be read" : "";
  const [w, hh] = v.size;
  if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; }
  if (v.el.ov.width !== w || v.el.ov.height !== hh) { v.el.ov.width = w; v.el.ov.height = hh; }
  if (v.shownBmp && v.found) x.drawImage(v.shownBmp, 0, 0, w, hh);
  else { x.fillStyle = "#2a3140"; x.fillRect(0, 0, w, hh); x.fillStyle = "#8a94a6"; x.font = `${Math.max(12, w / 40)}px sans-serif`; x.fillText(v.found ? "loading…" : "video file not found", 12, 24); }
  const rows = v.shown == null ? null : rowsAt(v, v.shown);
  v.rows = rows || [];
  if (rows && v.sel && !rows.some(r => r.annotation_id === v.sel) && !(v.tl && v.tl.markers.some(m => m.annotation_id === v.sel))) v.sel = null;
  const lw = Math.max(2, w / 360);
  for (const r of v.rows) {
    const col = labelColor(r.label), deleted = r.pending === "deleted", sel = r.annotation_id === v.sel;
    if (r.kind === "mask" && v.masks_on && !(v.work && v.work.target === r.annotation_id)) {
      const e = maskFor(v, r, v.shown);
      if (e && !e.empty) { x.globalAlpha = deleted ? .15 : sel ? Math.min(1, v.maskAlpha + .25) : v.maskAlpha; x.drawImage(tinted(e, col), e.x, e.y); x.globalAlpha = 1; }
    }
    if ((r.kind === "box" || r.kind === "mask") && v.boxes_on && r.x0 != null) {
      x.lineWidth = sel ? lw * 1.8 : lw; x.strokeStyle = sel ? "#fff" : col;
      x.setLineDash(r.pending && r.pending !== "deleted" ? [lw * 3, lw * 2] : deleted ? [lw, lw * 3] : []);
      x.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); x.setLineDash([]);
      const text = r.label + (r.track_id != null ? " #" + r.track_id : "");
      x.font = `${Math.max(11, w / 55)}px sans-serif`; const tw = x.measureText(text).width + 6, th = Math.max(11, w / 55) + 4;
      x.fillStyle = col; x.fillRect(r.x0, Math.max(0, r.y0 - th), tw, th); x.fillStyle = "#fff"; x.fillText(text, r.x0 + 3, Math.max(th - 4, r.y0 - 4));
    }
  }
  drawOverlay();
  const key = `${v.shown}:${v.ann.gen}:${rows ? 1 : 0}:${v.rows.map(r => r.annotation_id + (r.pending || "")).join(",")}`;
  if (key !== v.sideKey) { v.sideKey = key; drawSideSoon(); }
}
function drawSideSoon(now) {
  const v = V; clearTimeout(v.sideTimer);
  if (now || !v.playing) { drawSide(); return; }
  v.sideTimer = setTimeout(() => V === v && drawSide(), 250);     // while playing, the list is only refreshed a few times a second
}

/* ---------- the stage: fitting, zooming and panning the picture ---------- */
function sizeStage() {
  const v = V, st = v.el.stage, [w, hh] = v.size;
  const s = Math.min(st.clientWidth / w, st.clientHeight / hh);
  if (!(s > 0) || !isFinite(s)) return;
  v.fit = { w: Math.floor(w * s), h: Math.floor(hh * s), l: Math.floor((st.clientWidth - w * s) / 2), t: Math.floor((st.clientHeight - hh * s) / 2) };
  const L = v.el.layers.style;
  L.width = v.fit.w + "px"; L.height = v.fit.h + "px"; L.left = v.fit.l + "px"; L.top = v.fit.t + "px";
  applyZoom();
}
function applyZoom() {
  const v = V, z = v.zoom, st = v.el.stage;
  if (z <= 1.0001) { v.pan = [0, 0]; v.zoom = 1; }
  else {   // keep a good part of the picture inside the stage
    const W = v.fit.w * z, H = v.fit.h * z, m = 60;
    v.pan[0] = Math.max(m - v.fit.l - W, Math.min(st.clientWidth - m - v.fit.l, v.pan[0]));
    v.pan[1] = Math.max(m - v.fit.t - H, Math.min(st.clientHeight - m - v.fit.t, v.pan[1]));
  }
  v.el.layers.style.transform = `translate(${v.pan[0]}px, ${v.pan[1]}px) scale(${v.zoom})`;
  v.el.layers.classList.toggle("crisp", v.zoom > 2.5);
  v.el.hudZoom.textContent = Math.round(v.zoom * v.fit.w / v.size[0] * 100) + "%";
  v.el.fitBtn.hidden = v.zoom <= 1.0001;
  requestPaint("stage");
}
function zoomStage(factor, cx, cy) {
  const v = V, st = v.el.stage.getBoundingClientRect();
  const z2 = Math.max(1, Math.min(40, v.zoom * factor));
  if (cx == null) { cx = st.width / 2; cy = st.height / 2; } else { cx -= st.left; cy -= st.top; }
  v.pan = [cx - v.fit.l - (cx - v.fit.l - v.pan[0]) * z2 / v.zoom, cy - v.fit.t - (cy - v.fit.t - v.pan[1]) * z2 / v.zoom];
  v.zoom = z2; applyZoom();
}
function bindStage(stage) {
  stage.addEventListener("wheel", ev => { ev.preventDefault(); zoomStage(Math.exp(-ev.deltaY * (ev.deltaMode ? .05 : .0018)), ev.clientX, ev.clientY); }, { passive: false });
  stage.addEventListener("pointerdown", ev => {
    if (ev.button !== 1 && !(ev.button === 0 && V.spaceDown)) return;
    ev.preventDefault(); stage.setPointerCapture(ev.pointerId);
    const o = { x: ev.clientX, y: ev.clientY, p: [...V.pan] }; stage.style.cursor = "grabbing";
    const mv = e => { V.pan = [o.p[0] + e.clientX - o.x, o.p[1] + e.clientY - o.y]; applyZoom(); };
    const up = () => { stage.removeEventListener("pointermove", mv); stage.removeEventListener("pointerup", up); stage.style.cursor = ""; };
    stage.addEventListener("pointermove", mv); stage.addEventListener("pointerup", up);
  });
  stage.addEventListener("pointermove", ev => {
    if (!V.fit) return;
    const r = V.el.ov.getBoundingClientRect(), px = Math.floor((ev.clientX - r.left) * V.size[0] / r.width), py = Math.floor((ev.clientY - r.top) * V.size[1] / r.height);
    V.el.hudPos.textContent = px >= 0 && py >= 0 && px < V.size[0] && py < V.size[1] ? `x ${px}  y ${py}` : "";
  });
  stage.addEventListener("pointerleave", () => { if (V) V.el.hudPos.textContent = ""; });
  stage.addEventListener("contextmenu", ev => ev.preventDefault());
}

/* ---------- the page ---------- */
async function pageViewer(main, parts, q, alive) {
  const vid = parts[1];
  const meta = await GET("/api/videos/" + encodeURIComponent(vid));
  const views = (await GET("/api/views")).views.filter(v => v.video_id === vid);
  if (!alive()) return;
  const m = meta.metadata, fps = m.fps.num / m.fps.den;
  const n0 = Math.max(0, Math.min(+q.f || 0, m.frame_count - 1));
  V = { vid, views, n: n0, shown: null, shownBmp: null, frames: m.frame_count, view: q.view || "", playing: false, rate: 1, loop: false, dir: 1, rows: [], sel: null,
        boxes_on: true, masks_on: true, maskAlpha: .45, size: [m.width, m.height], fps, found: meta.file_found, zoom: 1, pan: [0, 0], fit: null,
        fs: { map: new Map(), pending: new Map(), bad: new Set(), cid: Math.random().toString(36).slice(2), seq: 0 },
        ann: { chunks: new Map(), waiting: new Map(), gen: 0 }, masks: new Map(), tl: null };
  const canvas = h("canvas", { width: m.width, height: m.height }), ov = h("canvas", { class: "ov", width: m.width, height: m.height, "aria-label": "Frame: draw and edit annotations here" });
  const layers = h("div", { class: "layers" }, canvas, ov);
  const hudState = h("span", { class: "hud-state" }), hudPos = h("span"), hudZoom = h("span");
  const stage = h("div", { class: "stage" }, layers, h("div", { class: "hud mono" }, hudState, hudPos, hudZoom));
  const tlc = h("canvas", { class: "tlc", height: 120, tabindex: 0, role: "slider", "aria-orientation": "horizontal", "aria-valuemin": 0, "aria-valuemax": m.frame_count - 1, "aria-valuenow": n0, "aria-label": "Timeline: the current frame. Arrow keys step one frame, Shift 10; scroll to zoom." });
  const side = h("div", { class: "side" });
  const frameIn = h("input", { type: "number", min: 0, max: m.frame_count - 1, class: "framein", "aria-label": "Current frame", value: n0,
    onchange: e => { goto(+e.target.value); e.target.blur(); } });
  const mkb = (txt, title, fn, extra = {}) => h("button", { class: "small", title, "aria-label": title, onclick: fn, ...extra }, txt);
  const els = {
    first: mkb("⏮", "First frame (Home)", () => goto(0)), prev: mkb("◀", "Previous frame (←)", () => goto(V.n - 1)),
    playBtn: h("button", { class: "small play", onclick: () => togglePlay(), "aria-pressed": false, title: "Play or pause (Space)" }, "Play"),
    next: mkb("▶", "Next frame (→)", () => goto(V.n + 1)), endBtn: mkb("⏭", "Last frame (End)", () => goto(V.frames - 1)),
    of: h("span", { class: "mono" }), fitBtn: mkb("Reset zoom", "Show the whole picture (0)", () => { V.zoom = 1; applyZoom(); }, { hidden: true }),
  };
  V.el = { canvas, ov, layers, stage, tlc, side, frameIn, hudState, hudPos, hudZoom, ...els };
  drawInit(); tlInit(V);
  const rate = h("select", { "aria-label": "Playback speed", onchange: e => { V.rate = +e.target.value; V.clock = null; } },
    [.25, .5, 1, 2].map(r => h("option", { value: r, selected: r === 1 }, r + "×")));
  const viewSel = h("select", { "aria-label": "View", onchange: e => { V.view = e.target.value; V.masks.clear(); V.ann.gen++; V.ann.chunks.clear(); V.ann.waiting.clear(); for (const b of V.fs.map.values()) b.close(); V.fs.map.clear(); V.fs.bad.clear(); V.shownBmp = null; V.shown = null;
      V.work = null; V.poly = []; paintToolbar(); loadTimeline(); goto(V.n, { force: true }); } },
    h("option", { value: "" }, "Whole frame"), views.map(v => h("option", { value: v.view_id, selected: v.view_id === V.view }, "View: " + v.view_id)));
  const tools = toolbarEl();
  const root = h("div", { class: "viewer" },
    h("div", { class: "bar-top row" }, h("h1", { class: "sr" }, "Viewer: " + vid), h("b", {}, vid), h("span", { class: "muted" }, `${m.width}×${m.height} · ${fmt(m.frame_count)} frames · ${fps.toFixed(2)} fps`), viewSel,
      h("label", { class: "row tight" }, h("input", { type: "checkbox", checked: true, onchange: e => { V.boxes_on = e.target.checked; draw(); } }), "boxes"),
      h("label", { class: "row tight" }, h("input", { type: "checkbox", checked: true, onchange: e => { V.masks_on = e.target.checked; draw(); } }), "masks"),
      h("span", { class: "spacer" }), h("a", { href: "#/review/" + encodeURIComponent(vid) }, "Review tracks →"),
      h("div", { class: "toolrow" }, tools)),
    stage, side,
    h("div", { class: "transport" },
      h("div", { class: "row tight controls" }, els.first, els.prev, els.playBtn, els.next, els.endBtn, h("span", { class: "sep" }), frameIn, els.of,
        rate, h("button", { class: "small", "aria-pressed": false, title: "Start again from the first frame when the end is reached", onclick: e => { V.loop = !V.loop; e.target.setAttribute("aria-pressed", V.loop); e.target.classList.toggle("on", V.loop); } }, "Loop"),
        h("span", { class: "spacer" }), els.fitBtn,
        mkb("−", "Zoom the timeline out", () => zoomTimeline(1.6)), mkb("+", "Zoom the timeline in", () => zoomTimeline(1 / 1.6)), mkb("Fit", "Fit: show the whole timeline", () => fitTimeline())),
      tlc));
  V.el.root = root;
  main.className = "full";
  main.replaceChildren(root);
  if (!meta.file_found) side.append(h("div", { class: "notice warn keep" }, "The video file was not found on this computer, so frames are shown as a grey placeholder. The annotations are still shown and can be reviewed."));
  tlBind(tlc); bindStage(stage); bindDrawing(ov);
  V.onkey = viewerKey;
  document.addEventListener("keydown", V.onkey);
  V.ro = new ResizeObserver(() => { if (V) { sizeStage(); requestPaint("all"); } });
  V.ro.observe(stage); V.ro.observe(tlc);
  const v = V;
  sizeStage();
  await Promise.all([loadTimeline(), ensureAnn(v.n)]);
  if (V !== v) return;
  pump();
  requestPaint("all");
  if (!v.found) await new Promise(r => requestAnimationFrame(r));
}
function viewerKey(ev) {
  const v = V; if (!v) return;
  if (/input|textarea|select/i.test(ev.target.tagName) || (ev.key === " " && ev.target.tagName === "BUTTON")) return;
  if (drawKey(ev)) { ev.preventDefault(); return; }
  const k = ev.key, mod = ev.ctrlKey || ev.metaKey, step = ev.shiftKey ? 10 : 1;
  if (mod && k.toLowerCase() === "z") draftOp(ev.shiftKey ? "/api/draft/redo" : "/api/draft/undo").then(() => show(V.n));
  else if (mod && k.toLowerCase() === "y") draftOp("/api/draft/redo").then(() => show(V.n));
  else if (mod) return;
  else if (k === "ArrowRight") goto(v.n + step); else if (k === "ArrowLeft") goto(v.n - step);
  else if (k === " ") togglePlay(); else if (k === "Home") goto(0); else if (k === "End") goto(v.frames - 1);
  else if (k === "." || k === ",") jumpEdge(k === "." ? 1 : -1);
  else if (k.toLowerCase() === "c") { if (!ev.repeat) copyBoxForward(); }
  else if (k === "0") { v.zoom = 1; applyZoom(); }
  else if (k === "=" || k === "+") zoomStage(1.4); else if (k === "-") zoomStage(1 / 1.4);
  else return;
  ev.preventDefault();
}

/* ---------- shortcuts ---------- */
/* the frames where something starts or ends (spans and tracks), so . and , move between them */
function jumpEdge(dir) {
  const v = V; if (!v.tl) return;
  const edges = new Set();
  for (const m of v.tl.markers) { if (m.pending !== "deleted") { edges.add(m.start_frame); edges.add(m.end_frame); } }
  for (const t of v.tl.tracks) { edges.add(t.first_frame); edges.add(t.last_frame); }
  const pick = [...edges].filter(f => dir > 0 ? f > v.n : f < v.n).sort((a, b) => a - b);
  const f = dir > 0 ? pick[0] : pick[pick.length - 1];
  if (f == null) { toast(dir > 0 ? "No later annotation starts or ends." : "No earlier annotation starts or ends.", false); return; }
  goto(f);
}
/* C: copy the selected box to the next frame, keeping its label and track, and move there; press again to follow an object along */
async function copyBoxForward() {
  const v = V;
  if (v.copying) return;                                      // one copy at a time: the next one needs this one's result
  v.copying = true;
  try { await copyBoxOnce(); } finally { v.copying = false; }
}
async function copyBoxOnce() {
  const v = V, r = v.rows.find(x => x.annotation_id === v.sel);
  if (!canDraw() || v.shown !== v.n || !editableBox(r)) { toast("Select a box on this frame, then press C to copy it forward.", true); return; }
  if (v.n >= v.frames - 1) { toast("This is the last frame.", true); return; }
  await ensureAnn(v.n + 1);
  const there = rowsAt(v, v.n + 1);
  if (!there) { toast("The next frame's annotations have not loaded yet. Try again.", true); return; }
  if (r.track_id != null && there.some(x => x.kind === "box" && x.track_id === r.track_id && x.pending !== "deleted")) { toast(`Track ${r.track_id} already has a box on frame ${v.n + 1}. Step forward and adjust it instead.`, true); return; }
  const row = { kind: "box", label: r.label, start_frame: v.n + 1, box: [r.x0, r.y0, r.x1, r.y1], ...vtag() };
  if (r.track_id != null) row.track_id = r.track_id;
  const res = await draftOp("/api/draft/add", { video_id: v.vid, row });
  if (!res || V !== v) return;
  v.sel = res.id; goto(v.n + 1); await afterEdit();
}

/* ---------- side panel ---------- */
function drawSide() {
  if (!V) return;
  const side = V.el.side, keep = side.querySelector(".notice.keep");
  side.replaceChildren(keep || "", drawPanel(), h("h2", {}, `On frame ${V.shown ?? V.n} (${V.rows.length})`));
  if (!V.rows.length) side.append(h("div", { class: "empty" }, "Nothing annotated on this frame. Pick a tool above to draw."));
  for (const r of V.rows) {
    const id = r.annotation_id, vid = V.vid;
    side.append(h("div", { class: "annrow" + (id === V.sel ? " sel" : "") + (r.pending ? " pend" : "") + (r.pending === "deleted" ? " deleted" : ""), onclick: () => { V.sel = id; draw(); drawSide(); } },
      h("div", { class: "row" }, dot(r.label), h("b", { class: "lab" }, r.label), tag(r.kind, "tag"), r.track_id != null ? h("span", { class: "muted" }, "track " + r.track_id) : null,
        h("span", { class: "spacer" }), r.pending ? tag(r.pending, "pending") : null),
      h("div", { class: "row tight muted" }, tag(r.status), r.origin, r.confidence != null ? "· " + pct(r.confidence) : "", r.kind === "marker" ? `· frames ${r.start_frame}–${r.end_frame}` : ""),
      r.pending === "deleted" ? null : h("div", { class: "row tight", style: "margin-top:4px" },
        h("button", { class: "small ok", onclick: async e => { e.stopPropagation(); await setStatus(vid, [id], "approved"); show(V.n); } }, "Approve"),
        h("button", { class: "small bad", onclick: async e => { e.stopPropagation(); await setStatus(vid, [id], "rejected"); show(V.n); } }, "Reject"),
        h("button", { class: "small", title: "Delete this row (still undoable until you commit)", onclick: async e => { e.stopPropagation(); await draftOp("/api/draft/delete", { video_id: vid, ids: [id] }); show(V.n); } }, "Delete"))));
  }
  const sp = selectionPanel(); if (sp) side.append(sp);
  side.append(h("p", { class: "muted", style: "margin-top:12px" }, h("span", { class: "kbd" }, "←"), " ", h("span", { class: "kbd" }, "→"), " step · ",
    h("span", { class: "kbd" }, "Shift"), " 10 frames · ", h("span", { class: "kbd" }, "Space"), " play · scroll to zoom, middle-drag to pan · ", h("span", { class: "kbd" }, ", ."), " previous/next annotation edge · ", h("span", { class: "kbd" }, "C"), " copy box forward · ", h("span", { class: "kbd" }, "Ctrl+Z"), " undo · ", h("span", { class: "kbd" }, "V B S M E P"), " tools"));
}
