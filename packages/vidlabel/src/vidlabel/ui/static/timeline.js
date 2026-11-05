"use strict";
/* The timeline of the viewer: a ruler, how many annotations each stretch has, spans, tracks and a minimap.

   One rule keeps it exact: frame f occupies the stretch [f, f+1) of the horizontal axis, and the visible window is [t0, t1).
   Every pixel <-> frame conversion goes through tlX() and frameAtX(), for drawing and for the mouse alike, and the playhead is
   always painted from V.n, the single current frame, in the same animation frame as the picture. */

const TL = { RULER: 24, DENS: 34, SPAN_ROW: 14, TRACK_ROW: 5, MINI: 18, GAP: 5, MAX_SPAN_ROWS: 4, MAX_TRACK_ROWS: 6, MIN_SPAN: 8 };

function tlInit(v) {
  v.tv = { t0: 0, t1: Math.max(1, v.frames) };
  v.hover = null; v.tlDrag = null; v.tlWin = null; v.tlAll = null; v.tlFetch = 0;
}
const tlW = () => V.el.tlc.clientWidth || 1;
const tlSpan = () => V.tv.t1 - V.tv.t0;
const tlX = f => (f - V.tv.t0) / tlSpan() * tlW();
function frameAtX(x) { return Math.max(0, Math.min(V.frames - 1, Math.floor(V.tv.t0 + x / tlW() * tlSpan()))); }
function tlFrame(ev) { return frameAtX(ev.clientX - V.el.tlc.getBoundingClientRect().left); }
const tlZoomed = () => tlSpan() < V.frames - 0.5;

function setWindow(t0, t1) {
  const v = V, minSpan = Math.min(v.frames, TL.MIN_SPAN);
  let span = Math.max(minSpan, Math.min(v.frames, t1 - t0));
  t0 = Math.max(0, Math.min(v.frames - span, t0));
  v.tv = { t0, t1: t0 + span };
  requestPaint("timeline"); tlFetchSoon();
}
function zoomTimeline(factor, cx) {
  const W = tlW(), span = tlSpan(), at = V.tv.t0 + (cx == null ? W / 2 : cx) / W * span, ns = span * factor;
  const frac = cx == null ? .5 : cx / W;
  setWindow(at - frac * ns, at - frac * ns + ns);
}
function panTimeline(frames) { setWindow(V.tv.t0 + frames, V.tv.t1 + frames); }
function fitTimeline() { setWindow(0, V.frames); }
/* keep the playhead inside the window; while playing, turn the page when it reaches the right edge */
function followPlayhead() {
  if (!tlZoomed() || V.tlDrag) return;
  const f = V.n + .5, span = tlSpan();
  if (f < V.tv.t0 || f > V.tv.t1) setWindow(V.playing ? f - span * .08 : f - span / 2, V.playing ? f + span * .92 : f + span / 2);
}

/* ---------- data ---------- */
async function loadTimeline() {
  const v = V, base = { view_id: v.view };
  v.tlAll = await GET("/api/timeline/" + encodeURIComponent(v.vid), { ...base, bins: 400 });
  if (V !== v) return;
  v.tl = v.tlAll; v.tlWin = null;
  tlPack(v);
  paintToolbar(); requestPaint("timeline");
  if (tlZoomed()) tlFetchSoon(0);
}
function tlFetchSoon(ms = 140) {
  const v = V; clearTimeout(v.tlFetch);
  if (!tlZoomed()) { v.tlWin = null; return; }
  v.tlFetch = setTimeout(async () => {
    if (V !== v) return;
    const span = tlSpan(), a = Math.max(0, Math.floor(v.tv.t0 - span * .5)), b = Math.min(v.frames, Math.ceil(v.tv.t1 + span * .5));
    const bins = Math.min(2000, Math.ceil((b - a) / span * tlW() / 3));
    try {
      const win = await GET("/api/timeline/" + encodeURIComponent(v.vid), { view_id: v.view, start: a, end: b, bins });
      if (V === v) { v.tlWin = win; requestPaint("timeline"); }
    } catch (e) { /* the overview is still shown */ }
  }, ms);
}
/* put each span and each track on the first row where it does not overlap another, so they stay readable */
function tlPack(v) {
  const pack = (items, lo, hi, cap) => {
    const ends = [];
    for (const it of [...items].sort((a, b) => lo(a) - lo(b))) {
      let r = ends.findIndex(e => e < lo(it));
      if (r < 0) r = ends.length < cap ? ends.length : cap - 1;
      ends[r] = Math.max(ends[r] ?? -1, hi(it)); it._row = r;
    }
    return Math.min(ends.length, cap);
  };
  v.spanRows = pack(v.tl.markers, m => m.start_frame, m => m.end_frame, TL.MAX_SPAN_ROWS);
  v.trackRows = pack(v.tl.tracks, t => t.first_frame, t => t.last_frame, TL.MAX_TRACK_ROWS);
}
function tlLayout(v) {
  let y = 0;
  const L = { ruler: { y: 0, h: TL.RULER } }; y += TL.RULER + TL.GAP;
  L.dens = { y, h: TL.DENS }; y += TL.DENS + TL.GAP;
  const sr = Math.max(1, v.spanRows || 0);
  L.spans = { y, h: sr * TL.SPAN_ROW }; y += L.spans.h + TL.GAP;
  const tr = v.trackRows || 0;
  L.tracks = { y, h: tr * TL.TRACK_ROW }; if (tr) y += L.tracks.h + TL.GAP;
  L.mini = { y, h: TL.MINI }; L.H = y + TL.MINI + 2;
  return L;
}

/* ---------- painting ---------- */
function tlTheme() {
  const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim();
  return { ink: g("--ink"), ink2: g("--ink-2"), rule: g("--rule"), wash: g("--wash"), accent: g("--accent"), pending: g("--pending"), ok: g("--ok"), bad: g("--bad"), sheet: g("--sheet"), font: g("--font") };
}
function timecode(f, fps, fine) {
  const s = f / fps, m = Math.floor(s / 60), r = s - m * 60;
  return fine ? `${m}:${r.toFixed(2).padStart(5, "0")}` : `${m}:${String(Math.floor(r)).padStart(2, "0")}`;
}
function rulerSteps(v, W) {
  const ppf = W / tlSpan(), minFrames = 78 / ppf, fps = v.fps || 25;
  if (minFrames >= fps * .9) {
    const s = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200].find(x => x * fps >= minFrames) || 14400;
    return { step: Math.max(1, Math.round(s * fps)), time: true };
  }
  return { step: [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000].find(x => x >= minFrames) || 2000, time: false };
}
function paintTimeline() {
  const v = V, c = v.el.tlc, W = c.clientWidth;
  if (!W || !v.tl) return;
  const th = v.theme || (v.theme = tlTheme()), L = tlLayout(v), dpr = devicePixelRatio || 1;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(L.H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(L.H * dpr); c.style.height = L.H + "px"; }
  const x = c.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, L.H);
  const { t0, t1 } = v.tv, n = v.frames, ppf = W / tlSpan(), fps = v.fps || 25;
  x.font = `600 10.5px ${th.font}`; x.textBaseline = "alphabetic";

  /* ruler */
  x.fillStyle = th.ink2; x.strokeStyle = th.ink2; x.lineWidth = 1; x.textAlign = "left";
  const { step, time } = rulerSteps(v, W);
  x.beginPath();
  for (let f = Math.ceil(t0 / step) * step; f <= t1; f += step) {
    const px = Math.round(tlX(f)) + .5;
    x.moveTo(px, L.ruler.h - 8); x.lineTo(px, L.ruler.h);
    x.fillText(time ? timecode(f, fps) : String(f), px + 3, L.ruler.h - 10);
    const sub = step / 5;
    if (sub >= 1 && sub * ppf >= 6) for (let k = 1; k < 5; k++) { const sx = Math.round(tlX(f + sub * k)) + .5; x.moveTo(sx, L.ruler.h - 4); x.lineTo(sx, L.ruler.h); }
  }
  x.stroke();
  x.fillStyle = th.rule; x.fillRect(0, L.ruler.h, W, 1);

  /* annotation counts (the zoomed window when it covers what is visible, else the whole-video overview) */
  const src = v.tlWin && v.tlWin.start <= t0 && v.tlWin.end >= t1 ? v.tlWin : v.tlAll;
  const stat = [["approved", th.ok], ["unreviewed", th.accent], ["rejected", th.bad]];
  let max = 1;
  src.density.forEach((d, i) => { const a = src.start + i * src.bin_size; if (a + src.bin_size >= t0 && a <= t1) max = Math.max(max, d); });
  x.fillStyle = th.wash; x.fillRect(Math.max(0, tlX(0)), L.dens.y, Math.min(W, tlX(n)) - Math.max(0, tlX(0)), L.dens.h);
  src.density.forEach((d, i) => {
    if (!d) return;
    const a = src.start + i * src.bin_size, x0 = tlX(a), x1 = tlX(a + src.bin_size);
    if (x1 < 0 || x0 > W) return;
    let yy = L.dens.y + L.dens.h;
    for (const [st, col] of stat) {
      const cnt = (src.by_status[st] || [])[i] || 0; if (!cnt) continue;
      const bh = Math.max(1, Math.round(cnt / max * (L.dens.h - 2)));
      x.fillStyle = col; x.globalAlpha = .75; x.fillRect(x0, yy - bh, Math.max(1, x1 - x0 - (x1 - x0 > 3 ? 1 : 0)), bh); yy -= bh;
    }
  });
  x.globalAlpha = 1;

  /* spans (the one being dragged is drawn where the mouse has put it) */
  const drag = v.tlDrag && v.tlDrag.kind === "marker" ? v.tlDrag : null;
  x.textAlign = "left";
  for (const m of v.tl.markers) {
    const live = drag && drag.id === m.annotation_id, a = live ? drag.a : m.start_frame, b = live ? drag.b : m.end_frame;
    const x0 = tlX(a), x1 = tlX(b + 1);
    if (x1 < 0 || x0 > W) continue;
    const y0 = L.spans.y + m._row * TL.SPAN_ROW, w = Math.max(3, x1 - x0), del = m.pending === "deleted";
    x.globalAlpha = del ? .3 : 1; x.fillStyle = labelColor(m.label); x.fillRect(x0, y0, w, TL.SPAN_ROW - 2);
    x.globalAlpha = 1;
    if (m.pending && !del) { x.strokeStyle = th.pending; x.lineWidth = 1.5; x.setLineDash([3, 2]); x.strokeRect(x0 + .5, y0 + .5, w - 1, TL.SPAN_ROW - 3); x.setLineDash([]); }
    if (v.sel === m.annotation_id) { x.strokeStyle = th.ink; x.lineWidth = 2; x.strokeRect(x0 + 1, y0 + 1, w - 2, TL.SPAN_ROW - 4); }
    if (w > 46) { x.save(); x.beginPath(); x.rect(x0, y0, w, TL.SPAN_ROW); x.clip(); x.fillStyle = "#fff"; x.fillText(m.label, Math.max(x0, 0) + 4, y0 + TL.SPAN_ROW - 4); x.restore(); }
  }

  /* tracks: where each one first and last appears */
  for (const t of v.tl.tracks) {
    const x0 = tlX(t.first_frame), x1 = tlX(t.last_frame + 1);
    if (x1 < 0 || x0 > W) continue;
    x.fillStyle = labelColor(t.labels[0] || ""); x.globalAlpha = t.pending ? 1 : .8;
    x.fillRect(x0, L.tracks.y + t._row * TL.TRACK_ROW, Math.max(2, x1 - x0), TL.TRACK_ROW - 1.5);
  }
  x.globalAlpha = 1;

  /* a span being drawn with the span tool */
  const sd = v.spanDrag;
  if (sd) { const lo = Math.min(sd.a, sd.b), hi = Math.max(sd.a, sd.b); x.fillStyle = th.pending; x.globalAlpha = .3; x.fillRect(tlX(lo), L.dens.y, Math.max(3, tlX(hi + 1) - tlX(lo)), L.spans.y + L.spans.h - L.dens.y); x.globalAlpha = 1; }

  /* hover guide */
  if (v.hover != null && !v.tlDrag) {
    const hx = Math.round(tlX(v.hover + .5)) + .5; x.fillStyle = th.ink2; x.globalAlpha = .45; x.fillRect(hx, L.ruler.h, 1, L.spans.y + L.spans.h - L.ruler.h); x.globalAlpha = 1;
  }

  /* the playhead: the whole column of the current frame, its centre line and a head in the ruler */
  const cx = tlX(v.n + .5), colW = Math.max(0, ppf);
  if (cx >= -6 && cx <= W + 6) {
    if (colW >= 3) { x.fillStyle = th.ink; x.globalAlpha = .12; x.fillRect(tlX(v.n), L.ruler.h, colW, L.mini.y - L.ruler.h - TL.GAP); x.globalAlpha = 1; }
    const px = Math.round(cx) + .5;
    x.fillStyle = th.ink; x.fillRect(px - .5, 0, 1.5, L.mini.y - TL.GAP);
    x.beginPath(); x.moveTo(px - 5, 0); x.lineTo(px + 5.5, 0); x.lineTo(px + .25, 8); x.closePath(); x.fill();
  }

  /* hover read-out */
  if (v.hover != null && !v.tlDrag) {
    const txt = `${v.hover}  ·  ${timecode(v.hover, fps, true)}`, tw = x.measureText(txt).width + 10, hx = tlX(v.hover + .5);
    const bx = Math.max(2, Math.min(W - tw - 2, hx - tw / 2));
    x.fillStyle = th.ink; x.fillRect(bx, L.ruler.h + 2, tw, 16); x.fillStyle = th.sheet; x.textAlign = "left"; x.fillText(txt, bx + 5, L.ruler.h + 13.5);
  }

  /* minimap: the whole video, with the window and the playhead */
  const mini = L.mini, all = v.tlAll, mx = f => f / n * W;
  x.fillStyle = th.wash; x.fillRect(0, mini.y, W, mini.h);
  const mmax = Math.max(1, ...all.density);
  x.fillStyle = th.accent; x.globalAlpha = .6;
  all.density.forEach((d, i) => { if (d) { const bh = Math.max(1, Math.round(d / mmax * (mini.h - 2))); x.fillRect(mx(i * all.bin_size), mini.y + mini.h - bh, Math.max(1, mx(all.bin_size) - .5), bh); } });
  x.globalAlpha = 1;
  if (tlZoomed()) {
    x.fillStyle = th.ink; x.globalAlpha = .12; x.fillRect(mx(t0), mini.y, mx(t1) - mx(t0), mini.h); x.globalAlpha = 1;
    x.strokeStyle = th.ink; x.lineWidth = 1.5; x.strokeRect(mx(t0) + .75, mini.y + .75, Math.max(3, mx(t1) - mx(t0)) - 1.5, mini.h - 1.5);
  }
  x.fillStyle = th.ink; x.fillRect(Math.round(mx(v.n + .5)) - .5, mini.y, 1.5, mini.h);
}

/* ---------- the mouse ---------- */
function markerAt(x, y, L) {
  if (y < L.spans.y || y >= L.spans.y + L.spans.h) return null;
  const row = Math.floor((y - L.spans.y) / TL.SPAN_ROW), f = V.tv.t0 + x / tlW() * tlSpan();
  let best = null;
  for (const m of V.tl.markers) {
    if (m._row !== row || m.pending === "deleted") continue;
    const x0 = tlX(m.start_frame), x1 = tlX(m.end_frame + 1);
    if (x >= x0 - 4 && x <= x1 + 4 && (!best || m.start_frame > best.start_frame)) { best = m; best._edge = x - x0 <= 6 ? "l" : x1 - x <= 6 ? "r" : "move"; }
  }
  return best && (best._f = f, best);
}
function tlBind(c) {
  const pos = ev => { const r = c.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  c.addEventListener("pointerdown", ev => {
    if (ev.button !== 0 || !V.tl) return;
    const [x, y] = pos(ev), L = tlLayout(V);
    if (y >= L.mini.y) { c.setPointerCapture(ev.pointerId); V.tlDrag = { kind: "mini", grab: tlZoomed() && x >= V.tv.t0 / V.frames * tlW() && x <= V.tv.t1 / V.frames * tlW() ? x / tlW() * V.frames - V.tv.t0 : tlSpan() / 2 }; return tlMini(x); }
    if (V.tool === "span" && canDraw() && y >= L.dens.y && y < L.mini.y) return spanStart(ev);
    const m = markerAt(x, y, L);
    if (m) {
      c.setPointerCapture(ev.pointerId);
      V.sel = m.annotation_id; drawSide();
      V.tlDrag = { kind: "marker", id: m.annotation_id, m, edge: V.tool === "select" && canDraw() ? m._edge : "none", x0: x, f0: Math.floor(m._f), a0: m.start_frame, b0: m.end_frame, a: m.start_frame, b: m.end_frame, moved: false };
      goto(Math.max(m.start_frame, Math.min(m.end_frame, Math.floor(m._f)))); requestPaint("timeline"); return;
    }
    c.setPointerCapture(ev.pointerId);
    V.tlDrag = { kind: "scrub", x }; goto(frameAtX(x)); tlScrubLoop();
  });
  c.addEventListener("pointermove", ev => {
    const [x, y] = pos(ev), d = V.tlDrag;
    if (!d) {
      const L = tlLayout(V), m = markerAt(x, y, L), f = frameAtX(x);
      c.style.cursor = y >= L.mini.y ? "grab" : m && V.tool === "select" && canDraw() ? (m._edge === "move" ? "grab" : "ew-resize") : "col-resize";
      const nh = y >= L.mini.y ? null : f;
      if (V.hover !== nh) { V.hover = nh; requestPaint("timeline"); }
      return;
    }
    if (d.kind === "scrub") { d.x = x; goto(frameAtX(x)); }
    else if (d.kind === "mini") tlMini(x);
    else if (d.kind === "marker") {
      const df = Math.round((x - d.x0) / tlW() * tlSpan());
      if (!d.moved && Math.abs(x - d.x0) < 3) return;
      if (d.edge === "none") return;
      d.moved = true;
      if (d.edge === "l") d.a = Math.max(0, Math.min(d.b0, d.a0 + df)); else if (d.edge === "r") d.b = Math.min(V.frames - 1, Math.max(d.a0, d.b0 + df));
      else { const len = d.b0 - d.a0, a = Math.max(0, Math.min(V.frames - 1 - len, d.a0 + df)); d.a = a; d.b = a + len; }
      requestPaint("timeline");
    }
  });
  const end = async () => {
    const d = V.tlDrag; V.tlDrag = null;
    if (!d) return;
    if (d.kind === "marker" && d.moved && (d.a !== d.a0 || d.b !== d.b0)) {
      if (await draftOp("/api/draft/update", { video_id: V.vid, ids: [d.id], values: { start_frame: d.a, end_frame: d.b } })) await afterEdit();
    }
    requestPaint("timeline");
  };
  c.addEventListener("pointerup", end); c.addEventListener("pointercancel", end);
  c.addEventListener("pointerleave", () => { if (V.hover != null) { V.hover = null; requestPaint("timeline"); } });
  c.addEventListener("wheel", ev => {
    ev.preventDefault();
    const r = c.getBoundingClientRect(), dx = Math.abs(ev.deltaX) > Math.abs(ev.deltaY) ? ev.deltaX : (ev.shiftKey ? ev.deltaY : 0);
    if (dx) panTimeline(dx / r.width * tlSpan());
    else zoomTimeline(Math.exp(ev.deltaY * (ev.deltaMode ? 0.05 : 0.0016)), ev.clientX - r.left);
  }, { passive: false });
  c.addEventListener("dblclick", ev => { if (pos(ev)[1] >= tlLayout(V).mini.y) fitTimeline(); });
}
function tlMini(x) {
  const d = V.tlDrag, W = tlW(), f = x / W * V.frames;
  setWindow(f - d.grab, f - d.grab + tlSpan());
}
/* while scrubbing past the end of a zoomed timeline, keep turning the page */
function tlScrubLoop() {
  const d = V && V.tlDrag;
  if (!d || d.kind !== "scrub") return;
  const W = tlW();
  if (tlZoomed() && (d.x < 0 || d.x > W)) {
    const over = d.x < 0 ? d.x : d.x - W;
    panTimeline(Math.sign(over) * Math.min(Math.abs(over) / 40, 6) * tlSpan() / 120);
    goto(frameAtX(Math.max(0, Math.min(W, d.x))));
  }
  requestAnimationFrame(tlScrubLoop);
}
