"use strict";
/* vidlabel web interface. Plain JavaScript, no build step, nothing loaded from the internet.
   Pages render into <main>. Every change goes into the author's draft; "Changes" commits it to the dataset. */

const S = { author: "", session: null, info: null, draft: null, jobs: [], page: null };
const $ = (sel, root = document) => root.querySelector(sel);
/* replaceChildren that accepts nested arrays, null and plain text, like h() does (the built-in turns an array into "[object HTMLDivElement],..." text) */
const _replaceChildren = Element.prototype.replaceChildren;
Element.prototype.replaceChildren = function (...kids) {
  return _replaceChildren.apply(this, kids.flat(Infinity).filter(k => k != null && k !== false).map(k => (k instanceof Node ? k : document.createTextNode(String(k)))));
};

/* ---------- small helpers ---------- */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "value") e.value = v;
    else if (v === true) e.setAttribute(k, "");
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
const fmt = n => (n == null ? "–" : Number(n).toLocaleString());
const pct = x => (x == null ? "–" : (x * 100).toFixed(0) + "%");
function labelColor(name) {
  let x = 0; for (const c of String(name)) x = (x * 31 + c.charCodeAt(0)) >>> 0;
  return `hsl(${x % 360} 70% 52%)`;
}
function tag(text, cls) { return h("span", { class: "tag " + (cls || text) }, text); }
function dot(label) { return h("span", { class: "dot", style: `background:${labelColor(label)}` }); }

function toast(msg, bad) {
  const t = h("div", { class: "toast" + (bad ? " bad" : "") }, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), bad ? 7000 : 3000);
}

async function api(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-Requested-With": "vidlabel", "X-Vidlabel-Author": encodeURIComponent(S.author || "anonymous") },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) {
    const err = new Error((data && data.error) || r.statusText || "request failed");
    err.status = r.status; err.data = data;
    throw err;
  }
  return data;
}
const GET = (p, q) => api("GET", q ? p + "?" + new URLSearchParams(Object.entries(q).filter(([, v]) => v != null && v !== "")).toString() : p);
const POST = (p, b) => api("POST", p, b || {});

/* run something, show errors as a message */
async function attempt(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg); return r; }
  catch (e) { toast(e.message, true); return undefined; }
}

function modal(title, body, buttons) {
  const back = h("div", { class: "modal-back", onclick: e => { if (e.target === back) close(); } });
  const close = () => back.remove();
  back.append(h("div", { class: "modal", role: "dialog" }, h("h2", {}, title), body,
    h("div", { class: "row", style: "margin-top:14px;justify-content:flex-end" }, (buttons || []).map(b =>
      h("button", { class: b.primary ? "primary" : "", onclick: async () => { if ((await b.run?.(close)) !== false && !b.stay) close(); } }, b.text)))));
  $("#modal-root").append(back);
  return close;
}
function confirmBox(title, text) {
  return new Promise(res => modal(title, h("p", {}, text), [{ text: "Cancel", run: () => res(false) }, { text: "OK", primary: true, run: () => res(true) }]));
}

/* ---------- state refresh ---------- */
async function refreshDraft() {
  if (!S.session || !S.session.has_dataset) return;
  S.draft = await GET("/api/draft");
  renderTape();
  if (!$("#drawer").hidden) renderDrawer();
}
async function refreshInfo() {
  if (!S.session.has_dataset) return;
  S.info = await GET("/api/dataset");
}

/* ---------- draft actions used by every page ---------- */
async function draftOp(path, body, okMsg) {
  const r = await attempt(() => POST(path, body), okMsg);
  if (r === undefined) return false;
  await refreshDraft();
  return r;
}
const setStatus = (video_id, ids, status) => draftOp("/api/draft/update", { video_id, ids, values: { status } });
const setTrackStatus = (video_id, track_id, status, view_id) => draftOp("/api/draft/update-track", { video_id, track_id, view_id, values: { status } });

/* ---------- the draft tape along the bottom ---------- */
function renderTape() {
  const st = S.draft, s = st.summary, n = s.added + s.modified + s.deleted;
  const bar = $("#tape-bar"), cs = getComputedStyle(document.documentElement);
  bar.replaceChildren(...[["added", "--ok"], ["modified", "--warn"], ["deleted", "--bad"]].filter(([k]) => s[k]).map(([k, c]) => h("i", { style: `width:${s[k] / n * 100}%;background:${cs.getPropertyValue(c)}` })));
  $("#tape-text").textContent = n ? `${s.added} added, ${s.modified} changed, ${s.deleted} deleted` : "No pending changes";
  const badge = $("#changes-count"); badge.textContent = n; badge.hidden = !n;
  $("#undo-btn").disabled = !st.can_undo; $("#redo-btn").disabled = !st.can_redo; $("#commit-btn").disabled = n === 0;
}

/* ---------- the Changes drawer ---------- */
function renderDrawer() {
  const d = $("#drawer"), st = S.draft;
  d.replaceChildren();
  d.append(h("div", { class: "row" }, h("h2", { style: "flex:1;margin:0" }, "Your changes"), h("button", { class: "small", onclick: () => setDrawer(false) }, "Close")));
  d.append(h("p", { class: "muted" }, `Nothing below is in the dataset yet. It is saved as a draft for "${S.author}" and written only when you commit.`));
  const s = st.summary, n = s.added + s.modified + s.deleted;
  d.append(h("div", { class: "row" }, tag(`${s.added} added`, "ok"), tag(`${s.modified} changed`, "warn"), tag(`${s.deleted} deleted`, "bad"),
    s.videos.length ? h("span", { class: "muted" }, "in " + s.videos.join(", ")) : null));
  d.append(h("div", { class: "row", style: "margin:10px 0" },
    h("button", { disabled: !st.can_undo, onclick: () => draftOp("/api/draft/undo") }, "Undo"),
    h("button", { disabled: !st.can_redo, onclick: () => draftOp("/api/draft/redo") }, "Redo"),
    h("button", { class: "bad", disabled: !st.ops.length, onclick: async () => { if (await confirmBox("Discard all changes?", "This throws away your draft. The dataset is not touched.")) { await draftOp("/api/draft/discard", {}); render(); } } }, "Discard all")));
  if (st.ops.length) {
    d.append(h("ul", { class: "ops" }, st.ops.map(o => h("li", { class: o.undone ? "undone" : "" },
      h("span", { class: "t" }, o.label), h("span", { class: "muted" }, o.video_id),
      h("button", { class: "small", title: "Remove this edit and anything built on it", onclick: async () => { await draftOp("/api/draft/discard", { index: o.index }); render(); } }, "×")))));
  } else d.append(h("div", { class: "empty" }, "No pending changes."));
  const msg = h("textarea", { rows: 2, placeholder: "Describe what you changed (required), e.g. “rejected the blurry tracks in clip 3”" });
  const note = h("div");
  const commit = async (reapply) => {
    if (!msg.value.trim()) { toast("Write a short message first.", true); msg.focus(); return; }
    note.replaceChildren(h("span", { class: "muted" }, "Checking and writing…"));
    try {
      const r = await POST("/api/draft/commit", { message: msg.value.trim(), reapply });
      toast(`Committed #${r.seq}: ${fmt(r.summary.added)} added, ${fmt(r.summary.modified)} changed, ${fmt(r.summary.deleted)} deleted.`);
      setDrawer(false); await refreshDraft(); await refreshInfo(); render();
    } catch (e) {
      if (e.status === 409 && e.data && e.data.conflict) {
        note.replaceChildren(h("div", { class: "notice warn" }, `The dataset changed after you started editing (${e.data.videos.join(", ")}). `,
          "You can apply your edits on top of the new version; edits to rows that no longer exist are dropped.",
          h("div", { style: "margin-top:6px" }, h("button", { onclick: () => commit(true) }, "Apply my edits on top"))));
      } else note.replaceChildren(h("div", { class: "notice bad" }, e.message,
        (e.data && e.data.issues || []).slice(0, 8).map(i => h("div", { class: "mono" }, `${i.code}: ${i.message}`))));
    }
  };
  d.append(h("h3", {}, "Commit"), msg, h("div", { class: "row", style: "margin-top:6px" },
    h("button", { class: "primary", disabled: n === 0, onclick: () => commit(false) }, "Commit to dataset")), note);
  d.append(h("p", { class: "muted" }, "A commit is all or nothing, is checked against the validator first, and can be undone later from History."));
}
function setDrawer(open) { $("#drawer").hidden = !open; document.body.classList.toggle("drawer-open", open); if (open) refreshDraft().then(renderDrawer); }
function toggleDrawer() { setDrawer($("#drawer").hidden); }

/* ---------- jobs ---------- */
let jobTimer = null;
async function pollJobs() {
  try { S.jobs = (await GET("/api/jobs")).jobs; } catch (e) { return; }
  const active = S.jobs.filter(j => j.status === "queued" || j.status === "running");
  const btn = $("#jobs-btn");
  btn.hidden = S.jobs.length === 0;
  btn.textContent = active.length ? `⏳ ${active[0].title}${active[0].progress != null ? " " + pct(active[0].progress) : ""}` : "Jobs";
  if (!$("#modal-root .jobs-modal")) { /* nothing */ }
  const live = $("#modal-root .jobs-live"); if (live) renderJobsList(live);
  clearTimeout(jobTimer);
  if (active.length) jobTimer = setTimeout(pollJobs, 700);
}
function renderJobsList(root) {
  root.replaceChildren(...(S.jobs.length ? S.jobs : []).map(j => h("div", { class: "card" },
    h("div", { class: "row" }, h("b", { style: "flex:1" }, j.title), tag(j.status, j.status === "done" ? "ok" : j.status === "failed" ? "bad" : "warn"),
      (j.status === "running" || j.status === "queued") ? h("button", { class: "small", onclick: () => attempt(() => POST(`/api/jobs/${j.id}/cancel`)).then(pollJobs) }, "Cancel") : null),
    h("div", { class: "bar" + (j.progress == null && j.status === "running" ? " indet" : "") }, h("i", { style: `width:${(j.progress ?? (j.status === "done" ? 1 : 0)) * 100}%` })),
    h("div", { class: "muted" }, (j.messages || []).slice(-1)[0] || ""),
    j.error ? h("div", { class: "notice bad" }, j.error) : null)));
  if (!S.jobs.length) root.append(h("div", { class: "empty" }, "No jobs yet."));
}
function showJobs() {
  const live = h("div", { class: "jobs-live" });
  renderJobsList(live);
  const wrap = h("div", { class: "jobs-modal" }, live);
  const close = modal("Jobs", wrap, [{ text: "Close" }]);
  live.classList.add("jobs-live");
}
/* start a job, wait for it, return its result (or undefined after showing the error) */
async function runJob(kind, body, onProgress) {
  let job;
  try { job = await POST("/api/jobs/" + kind, body); } catch (e) { toast(e.message, true); return undefined; }
  pollJobs();
  for (;;) {
    await new Promise(r => setTimeout(r, 400));
    job = await GET("/api/jobs/" + job.id);
    onProgress && onProgress(job);
    if (job.status === "done") return job.result;
    if (job.status === "failed") { toast(job.error || "the job failed", true); return undefined; }
    if (job.status === "cancelled") { toast("Cancelled."); return undefined; }
  }
}
function progressBox() {
  const fill = h("i", { style: "width:0" }), bar = h("div", { class: "bar indet" }, fill), msg = h("div", { class: "muted" });
  const box = h("div", {}, bar, msg);
  box.update = j => { bar.classList.toggle("indet", j.progress == null); fill.style.width = ((j.progress ?? 0) * 100) + "%"; msg.textContent = (j.messages || []).slice(-1)[0] || ""; };
  return box;
}

/* ---------- folder picker ---------- */
function pickFolder(start, then, files) {
  const list = h("div", { class: "picker" }), cur = h("input", { type: "text", value: start || "" });
  const go = async (p) => {
    const r = await attempt(() => GET("/api/fs", { path: p }));
    if (!r) return;
    cur.value = r.path;
    list.replaceChildren(
      r.parent != null ? h("div", { onclick: () => go(r.parent) }, "↑  ..") : null,
      r.entries.filter(e => e.dir || files).map(e => h("div", { onclick: () => e.dir ? go(e.path) : (cur.value = e.path) }, (e.dir ? "📁 " : "🎞 ") + e.name + (e.is_dataset ? "  (dataset)" : ""))));
  };
  go(start || "");
  modal(files ? "Choose a video file" : "Choose a folder", h("div", {}, cur, h("div", { style: "height:8px" }), list),
    [{ text: "Cancel" }, { text: files ? "Use this file" : "Use this folder", primary: true, run: () => then(cur.value) }]);
}

/* ---------- router ---------- */
const NAV = [["overview", "Overview"], ["browse", "Videos"], ["labels", "Labels"], ["views", "Views"], ["review", "Review"], ["validation", "Validation"], ["releases", "Releases"], ["history", "History"]];
function parseHash() {
  const [path, query] = (location.hash.slice(2) || "overview").split("?");
  return { parts: path.split("/").map(decodeURIComponent), q: Object.fromEntries(new URLSearchParams(query || "")) };
}
function go(path, q) { location.hash = "#/" + path + (q ? "?" + new URLSearchParams(q) : ""); }
let renderToken = 0;
async function render() {
  if (!S.session) return;
  const { parts, q } = parseHash();
  const main = $("#main");
  const mine = ++renderToken;
  stopViewer();
  main.className = "";
  $("#nav").replaceChildren(...NAV.map(([k, t]) => h("a", { href: "#/" + k, class: (parts[0] === k || (k === "browse" && parts[0] === "video")) ? "on" : "" }, t)));
  if (!S.session.has_dataset) return pageSetup(main);
  try {
    const pages = { overview: pageOverview, browse: pageBrowse, video: pageViewer, labels: pageLabels, views: pageViews, review: pageReview, validation: pageValidation, releases: pageReleases, history: pageHistory };
    await (pages[parts[0]] || pageOverview)(main, parts, q, () => mine === renderToken);
  } catch (e) { main.replaceChildren(h("div", { class: "page" }, h("div", { class: "notice bad" }, e.message))); }
}

/* ---------- setup ---------- */
function pageSetup(main) {
  const path = h("input", { type: "text", value: S.session.path });
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Create a dataset"),
    h("p", { class: "muted" }, "This folder is not a vidlabel dataset yet. A dataset is a folder of small files; your videos stay where they are."),
    h("div", { class: "card" }, h("label", {}, "Dataset folder"), path,
      h("div", { class: "row", style: "margin-top:10px" },
        S.session.can_browse_files ? h("button", { onclick: () => pickFolder(path.value, p => { path.value = p; }) }, "Browse…") : null,
        h("button", { class: "primary", onclick: async () => { if (await attempt(() => POST("/api/init", { path: path.value }))) { S.session = await GET("/api/session"); await boot2(); } } }, "Create dataset")))));
}

/* ---------- overview ---------- */
async function pageOverview(main, parts, q, alive) {
  await refreshInfo();
  const i = S.info, st = i.stats;
  const list = (obj, kind) => Object.entries(obj || {}).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
    h("tr", {}, h("td", {}, kind === "status" ? tag(k) : k), h("td", { class: "num" }, fmt(v))));
  const score = h("div", { class: "score" });
  const total = st.annotations;
  main.replaceChildren(h("div", { class: "page" },
    h("h1", {}, i.path.split(/[\\/]/).filter(Boolean).pop() || "Dataset"), h("div", { class: "muted mono" }, i.path),
    h("div", { class: "figures" }, [["videos", st.videos], ["annotations", st.annotations], ["labels", st.labels_in_vocab], ["views", st.views], ["releases", i.releases.length]].map(([l, n]) => h("div", {}, h("b", {}, fmt(n)), h("span", {}, l)))),
    h("h2", {}, "Every video on one ruler"),
    total || st.videos ? [score, h("div", { class: "score-key" }, tag("approved", "approved"), tag("unreviewed", "unreviewed"), tag("rejected", "rejected"), h("span", {}, "Band length is video length; height is how many annotations start there. Click a band to open the video at that frame."))]
      : h("div", { class: "empty" }, "Nothing here yet. Add a video on the Videos page, then annotate it with the library or import rows with add_annotations."),
    h("div", { class: "grid cols2", style: "margin-top:34px" },
      h("div", { class: "card" }, h("h2", {}, "Review status"), h("table", {}, h("tbody", {}, list(st.by_status, "status")))),
      h("div", { class: "card" }, h("h2", {}, "Kind and origin"), h("table", {}, h("tbody", {}, list(st.by_kind), list(st.by_origin))))),
    h("div", { class: "card" }, h("h2", {}, "Labels"), i.labels.length ? h("table", {}, h("tbody", {}, i.labels.map(l => h("tr", {}, h("td", {}, dot(l.name), l.name), h("td", { class: "num" }, fmt(st.by_label[l.name] || 0)))))) : h("div", { class: "empty" }, "No labels yet."),
      h("p", {}, h("a", { href: "#/labels" }, "Manage labels")))));
  if (st.video_details.length) drawScore(score, st.video_details);
}
function videoTable(rows) {
  if (!rows.length) return h("div", { class: "empty" }, "No videos yet. Add one with ", h("span", { class: "mono" }, "vidlabel add-video"), ".");
  return h("table", {}, h("thead", {}, h("tr", {}, ["Video", "Size", "fps", "Frames", "Codec", "Annotations", ""].map((t, k) => h("th", { class: k > 0 && k < 6 ? "num" : "" }, t)))),
    h("tbody", {}, rows.map(v => h("tr", {}, h("td", {}, h("a", { href: "#/video/" + encodeURIComponent(v.video_id) }, v.video_id)),
      h("td", { class: "num" }, `${v.width}×${v.height}`), h("td", { class: "num" }, +Number(v.fps).toFixed(3)), h("td", { class: "num" }, fmt(v.frame_count)), h("td", { class: "num" }, v.codec || "–"),
      h("td", { class: "num" }, fmt(v.annotations)), h("td", { class: "right" }, h("a", { href: "#/review/" + encodeURIComponent(v.video_id) }, "Review"))))));
}
/* ---------- the score: every video as a band, coloured by review status ---------- */
const STATUS_COLORS = () => { const c = getComputedStyle(document.documentElement); return { approved: c.getPropertyValue("--ok").trim(), unreviewed: c.getPropertyValue("--warn").trim(), rejected: c.getPropertyValue("--bad").trim(), ink: c.getPropertyValue("--ink").trim(), rule: c.getPropertyValue("--rule").trim() }; };
async function drawScore(root, videos) {
  const maxFrames = Math.max(1, ...videos.map(v => v.frame_count));
  const rows = videos.slice(0, 40).map(v => {
    const cv = h("canvas", { height: 38 }), n = h("div", { class: "n" }, fmt(v.annotations));
    return { v, cv, el: h("div", { class: "score-row" }, h("a", { class: "name", href: "#/video/" + encodeURIComponent(v.video_id), title: v.video_id }, v.video_id), cv, n) };
  });
  root.replaceChildren(...rows.map(r => r.el));
  if (videos.length > 40) root.append(h("div", { class: "muted", style: "padding:8px 0" }, `Showing the first 40 of ${videos.length} videos.`));
  const col = STATUS_COLORS();
  for (const { v, cv } of rows) {
    let tl; try { tl = await GET("/api/timeline/" + encodeURIComponent(v.video_id)); } catch (e) { continue; }
    const paint = () => {
      const full = cv.parentElement.clientWidth - 150 - 90 - 28, w = Math.max(40, Math.round(full * v.frame_count / maxFrames)), dpr = devicePixelRatio || 1, H = 38;
      cv.style.width = w + "px"; cv.width = w * dpr; cv.height = H * dpr;
      const x = cv.getContext("2d"); x.scale(dpr, dpr);
      x.fillStyle = col.rule; x.fillRect(0, H - 1, w, 1);
      const bw = Math.max(1, (tl.bin_size / Math.max(1, tl.frame_count)) * w), mx = Math.max(1, ...tl.density);
      tl.density.forEach((d, i) => {
        if (!d) return;
        let y = H - 2;
        for (const st of ["approved", "unreviewed", "rejected"]) {
          const hh = Math.round(((tl.by_status[st] || [])[i] || 0) / mx * (H - 8));
          if (hh) { x.fillStyle = col[st]; x.fillRect(Math.floor(i * bw), y - hh, Math.ceil(bw) - (bw > 3 ? 1 : 0), hh); y -= hh; }
        }
      });
      x.fillStyle = col.ink;
      for (let f = 0; f <= v.frame_count; f += Math.max(1, 10 ** Math.floor(Math.log10(v.frame_count)))) x.fillRect(Math.round(f / v.frame_count * (w - 1)), H - 6, 1, 6);
    };
    paint();
    cv.addEventListener("click", e => { const r = cv.getBoundingClientRect(); go("video/" + encodeURIComponent(v.video_id), { f: Math.round((e.clientX - r.left) / r.width * (v.frame_count - 1)) }); });
    cv.title = "Click to open the video at this frame";
  }
}

/* ---------- videos: add, health, where the files are ---------- */
function checkResult(r) {
  return h("div", {}, h("div", { class: "row" }, tag(r.ok ? "matches the dataset" : "does not match", r.ok ? "ok" : "bad"), h("span", { class: "mono muted" }, r.file)),
    r.problems.map(p => h("div", { class: "notice bad" }, p)), r.notes.map(n => h("div", { class: "notice" }, n)),
    r.keyframes ? h("p", { class: "muted" }, `Keyframes every ${r.keyframes.median} frames (most ${r.keyframes.max}); average frames decoded per random read: ${Number(r.keyframes.mean_cost).toFixed(1)}.`) : null,
    r.seek ? h("p", { class: r.seek.exact ? "" : "notice warn" }, r.seek.exact ? `Random access is exact at ${r.seek.positions} test positions.` : `Random access returned different pictures at ${r.seek.mismatches.length} positions; read in order instead, or index the video.`) : null);
}
function checkDialog(v) {
  const hash = h("input", { type: "checkbox" }), seek = h("input", { type: "checkbox" }), out = h("div"), path = h("input", { type: "text", placeholder: "leave empty to use the recorded location" });
  const run = async (btn) => {
    btn.disabled = true; const pb = progressBox(); out.replaceChildren(pb);
    const r = await runJob("check-video", { video_id: v.video_id, hash: hash.checked, seek: seek.checked, video_path: path.value.trim() || null }, pb.update);
    btn.disabled = false; out.replaceChildren(r ? checkResult(r) : "");
  };
  const body = h("div", {}, h("p", { class: "muted" }, "Compares the file with what the dataset recorded: size, frame count, timing."),
    h("label", { class: "row tight" }, hash, "Also compare the content fingerprint (reads the whole file)"),
    h("label", { class: "row tight" }, seek, "Also test random access (decodes the video once)"), h("div", { style: "height:8px" }), path, h("div", { style: "height:10px" }),
    h("button", { class: "primary", onclick: e => run(e.target) }, "Run check"), h("div", { style: "height:10px" }), out);
  modal("Check " + v.video_id, body, [{ text: "Close" }]);
}
async function pageBrowse(main) {
  await refreshInfo();
  const { videos } = await GET("/api/videos");
  const { roots } = await GET("/api/roots");
  const path = h("input", { type: "text", placeholder: "path to a video file" }), vid = h("input", { type: "text", placeholder: "name (default: the file name)" });
  const hash = h("input", { type: "checkbox", checked: true }), index = h("input", { type: "checkbox", checked: true }), pbox = h("div");
  const add = async (btn) => {
    if (!path.value.trim()) { toast("Choose a video file first.", true); return; }
    btn.disabled = true; const pb = progressBox(); pbox.replaceChildren(pb);
    const r = await runJob("add-video", { path: path.value.trim(), video_id: vid.value.trim() || null, hash: hash.checked, index: index.checked }, pb.update);
    btn.disabled = false; pbox.replaceChildren();
    if (r) { S.flash = [h("div", { class: "notice" }, `Added ${r.video_id}: ${r.width}×${r.height}, ${fmt(r.frame_count)} frames.`), r.notes.map(n => h("div", { class: "notice warn" }, n))]; render(); }
  };
  const rootsBox = h("div");
  const paintRoots = () => rootsBox.replaceChildren(roots.length ? roots.map(r => h("div", { class: "row" }, h("span", { class: "mono" }, r), h("button", { class: "small", onclick: async () => { roots.splice(roots.indexOf(r), 1); await saveRoots(); } }, "Remove"))) : h("span", { class: "muted" }, "None. Videos are looked up where the dataset recorded them."));
  const saveRoots = async () => { if (await attempt(() => POST("/api/roots", { roots }))) { paintRoots(); render(); } };
  paintRoots();
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Videos"), S.flash || [],
    h("p", { class: "muted" }, "The dataset stores facts about each video, never the pixels. Frames are read from the file on this computer."),
    h("div", { class: "card" }, h("h2", {}, "Add a video"),
      h("div", { class: "row", style: "margin-bottom:8px" }, path, S.session.can_browse_files ? h("button", { onclick: () => pickFolder(path.value.trim().replace(/^["']|["']$/g, ""), p => { path.value = p; if (!vid.value) vid.value = ""; }, true) }, "Browse") : null),
      vid, h("div", { class: "row", style: "margin:8px 0" }, h("label", { class: "row tight" }, hash, "Fingerprint the file"), h("label", { class: "row tight" }, index, "Read keyframes and timestamps")),
      h("button", { class: "primary", onclick: e => add(e.target) }, "Add video"), pbox),
    h("div", { class: "card" }, h("h2", {}, `${videos.length} video${videos.length === 1 ? "" : "s"}`),
      videos.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Video", "Size", "fps", "Frames", "File", "Index", "Fingerprint", ""].map(t => h("th", {}, t)))),
        h("tbody", {}, videos.map(v => h("tr", {}, h("td", {}, h("a", { href: "#/video/" + encodeURIComponent(v.video_id) }, v.video_id)), h("td", {}, `${v.width}×${v.height}`),
          h("td", {}, +Number(v.fps).toFixed(3) + (v.is_vfr ? " variable" : "")), h("td", {}, fmt(v.frame_count)), h("td", {}, v.file_found ? tag("found", "ok") : tag("not found", "warn")),
          h("td", {}, v.has_index ? tag("yes", "ok") : tag("no", "warn")), h("td", {}, v.has_hash ? tag("yes", "ok") : tag("no", "warn")),
          h("td", { class: "right" }, h("div", { class: "row tight", style: "justify-content:flex-end" }, h("button", { class: "small", onclick: () => checkDialog(v) }, "Check"),
            h("button", { class: "small", onclick: async () => { const pb = progressBox(); const close = modal("Index " + v.video_id, pb, []); const r = await runJob("index", { video_ids: [v.video_id], force: true }, pb.update); close(); if (r) { const x = r.results[0]; toast(x.ok ? `Indexed ${v.video_id}: keyframe every ${x.keyframe_median} frames.` : x.error, !x.ok); render(); } } }, "Index"))))))) : h("div", { class: "empty" }, "No videos yet. Add one above.")),
    h("div", { class: "card quiet" }, h("h2", {}, "Where your videos are"),
      h("p", { class: "muted" }, "If files moved, add the folder they are in. It is searched by file name, for this session only."), rootsBox,
      S.session.can_browse_files ? h("button", { style: "margin-top:8px", onclick: () => pickFolder("", async p => { roots.push(p); await saveRoots(); }) }, "Add a folder") : null),
    cacheCard()));
  S.flash = null;
}
function cacheCard() {
  const fmtSel = h("select", {}, ["jpg", "png"].map(f => h("option", { value: f }, f))), q = h("input", { type: "number", value: 95, min: 1, max: 100, style: "width:80px" }),
    frames = h("select", {}, [["annotated", "only frames with annotations"], ["all", "every frame"]].map(([v, t]) => h("option", { value: v }, t))), out = h("div");
  return h("details", { class: "card quiet" }, h("summary", { style: "cursor:pointer;font-weight:700" }, "Frame cache"),
    h("p", { class: "muted" }, "Optional. Saves decoded frames as pictures so training does not decode video again. It can be rebuilt at any time."),
    h("div", { class: "row" }, "Save", frames, "as", fmtSel, "quality", q, h("button", { onclick: async e => {
      e.target.disabled = true; const pb = progressBox(); out.replaceChildren(pb);
      const r = await runJob("cache", { format: fmtSel.value, quality: +q.value, frames: frames.value, release: S.info.releases.length ? "latest" : "none" }, pb.update);
      e.target.disabled = false; out.replaceChildren(r ? h("div", { class: "notice" }, `Wrote ${fmt(r.written)} frames (${r.megabytes} MB) in ${r.seconds} s, skipped ${fmt(r.skipped)} that were already there. Folder: ${r.out}`) : "");
    } }, "Build cache")), out);
}

/* ---------- labels ---------- */
async function pageLabels(main) {
  const { labels } = await GET("/api/labels");
  const input = h("input", { type: "text", placeholder: "new labels, separated by commas" });
  const add = async () => { if (!input.value.trim()) return; if (await attempt(() => POST("/api/labels", { names: [input.value] }), "Labels added")) { await refreshInfo(); render(); } };
  input.addEventListener("keydown", e => { if (e.key === "Enter") add(); });
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Labels"),
    h("p", { class: "muted" }, "Every annotation uses a name from this list, which keeps spelling consistent. Labels can be added here; renaming or removing one is not supported yet."),
    h("div", { class: "card" }, labels.length ? h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Label"), h("th", { class: "num" }, "Annotations"))),
      h("tbody", {}, labels.map(l => h("tr", {}, h("td", {}, dot(l.name), l.name), h("td", { class: "num" }, fmt(l.count)))))) : h("div", { class: "empty" }, "No labels yet. Add the first one below."),
      h("div", { class: "row", style: "margin-top:12px" }, h("div", { style: "flex:1;max-width:420px" }, input), h("button", { class: "primary", onclick: add }, "Add labels")))));
}

/* ---------- views ---------- */
function regionText(v) { return v.region ? `${v.region.x},${v.region.y}  ${v.region.width}×${v.region.height}` : "whole frame"; }
function framesText(v) { return (v.frame_ranges || (v.frame_range ? [v.frame_range] : [])).map(r => r.join("–")).join(", ") || "all"; }
function deriveDialog(v) {
  const labels = h("select", { multiple: true, size: Math.min(6, Math.max(2, S.info.labels.length)) }, S.info.labels.map(l => h("option", { value: l.name }, l.name)));
  const out = h("div");
  const call = async (dry) => {
    const sel = [...labels.selectedOptions].map(o => o.value);
    const r = await attempt(() => POST("/api/views/derive", { video_id: v.video_id, view_id: v.view_id, labels: sel.length ? sel : null, dry_run: dry }));
    if (!r) return;
    const facts = Object.entries(r).filter(([k, x]) => !["ids", "dry_run", "copied"].includes(k) && typeof x !== "object").map(([k, x]) => `${k.replace(/_/g, " ")}: ${x}`);
    out.replaceChildren(h("div", { class: "notice" }, dry ? `${fmt(r.copied)} rows would be copied into this view.` : `Copied ${fmt(r.copied)} rows into this view.`), facts.length ? h("p", { class: "muted" }, facts.join("; ")) : null);
    if (!dry) await refreshInfo();
  };
  modal(`Copy whole-frame annotations into ${v.view_id}`, h("div", {}, h("p", { class: "muted" }, "Copies rows that lie inside this view's region and time windows. The originals stay. Running it again copies only what is new. Pick labels to limit it (none picked: all)."),
    labels, h("div", { class: "row", style: "margin-top:10px" }, h("button", { onclick: () => call(true) }, "Preview"), h("button", { class: "primary", onclick: () => call(false) }, "Copy rows")), out), [{ text: "Close" }]);
}
async function pageViews(main) {
  await refreshInfo();
  const { views } = await GET("/api/views");
  const videos = S.info.videos;
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Views"),
    h("p", { class: "muted" }, "A view is a named crop and set of time windows on one video, such as a camera region or the winter frames. No new video is made; annotations stay in the full frame's coordinates."),
    h("div", { class: "card" }, views.length ? h("table", {}, h("thead", {}, h("tr", {}, ["View", "Video", "Region", "Resized to", "Frames", "Labels", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, views.map(v => h("tr", {}, h("td", {}, h("a", { href: `#/video/${encodeURIComponent(v.video_id)}?view=${encodeURIComponent(v.view_id)}` }, v.view_id)), h("td", {}, v.video_id), h("td", { class: "mono" }, regionText(v)),
        h("td", { class: "mono" }, v.output_size ? v.output_size.join("×") : "–"), h("td", { class: "mono" }, framesText(v)), h("td", {}, (v.allowed_labels || ["any"]).join(", ")),
        h("td", { class: "right" }, h("button", { class: "small", onclick: () => deriveDialog(v) }, "Copy labels in…")))))) : h("div", { class: "empty" }, videos.length ? "No views yet. Create one below." : "Add a video first, then define views on it."),
    ), videos.length ? viewForm(videos) : null));
}
function viewForm(videos) {
  const f = { vid: h("select", {}, videos.map(v => h("option", { value: v }, v))), id: h("input", { type: "text", placeholder: "view name, e.g. left_camera" }),
    x: h("input", { type: "number", min: 0, style: "width:84px" }), y: h("input", { type: "number", min: 0, style: "width:84px" }), w: h("input", { type: "number", min: 1, style: "width:84px" }), hh: h("input", { type: "number", min: 1, style: "width:84px" }),
    ow: h("input", { type: "number", min: 1, style: "width:84px" }), oh: h("input", { type: "number", min: 1, style: "width:84px" }), frames: h("input", { type: "text", placeholder: "e.g. 0-499, 1200-1500 (empty: all frames)" }) };
  const wrap = h("div", { class: "crop-wrap" }), cv = h("canvas"), allowed = h("select", { multiple: true, size: 4 }, S.info.labels.map(l => h("option", { value: l.name }, l.name)));
  let img = null, drag = null;
  const rect = () => { const x = +f.x.value, y = +f.y.value, w = +f.w.value, hh = +f.hh.value; return w > 0 && hh > 0 ? { x, y, w, hh } : null; };
  const paint = () => {
    const x = cv.getContext("2d"); x.clearRect(0, 0, cv.width, cv.height);
    if (img) x.drawImage(img, 0, 0); else { x.fillStyle = "#2a3140"; x.fillRect(0, 0, cv.width, cv.height); }
    const r = rect(); if (!r) return;
    x.fillStyle = "#0009"; x.beginPath(); x.rect(0, 0, cv.width, cv.height); x.rect(r.x, r.y, r.w, r.hh); x.fill("evenodd");
    x.strokeStyle = "#ff4fa6"; x.lineWidth = Math.max(2, cv.width / 300); x.strokeRect(r.x, r.y, r.w, r.hh);
  };
  const load = async () => {
    const m = (await GET("/api/videos/" + encodeURIComponent(f.vid.value))).metadata;
    img = null; cv.width = m.rotation % 180 ? m.height : m.width; cv.height = m.rotation % 180 ? m.width : m.height; paint();
    const im = new Image(); im.onload = () => { img = im; cv.width = im.naturalWidth; cv.height = im.naturalHeight; paint(); }; im.src = `/api/frame/${encodeURIComponent(f.vid.value)}/0`;
  };
  const pos = e => { const b = cv.getBoundingClientRect(); return [Math.round((e.clientX - b.left) * cv.width / b.width), Math.round((e.clientY - b.top) * cv.height / b.height)]; };
  cv.addEventListener("pointerdown", e => { drag = pos(e); cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => { if (!drag) return; const [px, py] = pos(e); f.x.value = Math.max(0, Math.min(drag[0], px)); f.y.value = Math.max(0, Math.min(drag[1], py)); f.w.value = Math.abs(px - drag[0]); f.hh.value = Math.abs(py - drag[1]); paint(); });
  cv.addEventListener("pointerup", () => { drag = null; });
  for (const k of ["x", "y", "w", "hh"]) f[k].addEventListener("input", paint);
  f.vid.addEventListener("change", load);
  wrap.append(cv); load();
  const create = async () => {
    const spec = { view_id: f.id.value.trim(), video_id: f.vid.value };
    const r = rect(); if (r) spec.region = { x: r.x, y: r.y, width: r.w, height: r.hh };
    if (f.ow.value && f.oh.value) spec.output_size = [+f.ow.value, +f.oh.value];
    const fr = f.frames.value.split(",").map(t => t.trim()).filter(Boolean).map(t => t.split("-").map(Number));
    if (fr.length === 1) spec.frame_range = fr[0]; else if (fr.length) spec.frame_ranges = fr;
    const lab = [...allowed.selectedOptions].map(o => o.value); if (lab.length) spec.allowed_labels = lab;
    if (!spec.view_id) { toast("Give the view a name.", true); return; }
    if (await attempt(() => POST("/api/views", spec), "View created")) { await refreshInfo(); render(); }
  };
  return h("div", { class: "card" }, h("h2", {}, "Create a view"),
    h("div", { class: "row", style: "margin-bottom:10px" }, f.vid, h("div", { style: "flex:1;min-width:200px" }, f.id)),
    h("p", { class: "muted" }, "Drag on the first frame to choose the region, or type the numbers (pixels, as the video is displayed)."), wrap,
    h("div", { class: "row", style: "margin:10px 0" }, "x", f.x, "y", f.y, "width", f.w, "height", f.hh),
    h("div", { class: "row", style: "margin-bottom:10px" }, "Resize the crop to", f.ow, "×", f.oh, h("span", { class: "muted" }, "(optional)")),
    f.frames, h("p", { class: "muted", style: "margin:10px 0 4px" }, "Limit the labels allowed in this view (optional; hold Ctrl or Cmd to pick several):"), allowed,
    h("div", { style: "margin-top:12px" }, h("button", { class: "primary", onclick: create }, "Create view")));
}


/* ---------- viewer ---------- */
let V = null;
function stopViewer() {
  if (V) { clearInterval(V.timer); document.removeEventListener("keydown", V.onkey); V = null; }
}
async function pageViewer(main, parts, q, alive) {
  const vid = parts[1];
  const meta = await GET("/api/videos/" + encodeURIComponent(vid));
  const views = (await GET("/api/views")).views.filter(v => v.video_id === vid);
  if (!alive()) return;
  const m = meta.metadata, fps = m.fps.num / m.fps.den;
  V = { vid, n: Math.max(0, Math.min(+q.f || 0, m.frame_count - 1)), frames: m.frame_count, view: q.view || "", playing: false, rows: [], sel: null,
        boxes: true, masks: true, img: null, size: [m.width, m.height], fps, found: meta.file_found, req: 0, tl: null, timer: null, maskImgs: new Map() };
  const canvas = h("canvas", { width: m.width, height: m.height });
  const tlc = h("canvas", { height: 64 });
  const slider = h("input", { type: "range", min: 0, max: m.frame_count - 1, value: V.n, style: "flex:1" });
  const info = h("span", { class: "mono" });
  const side = h("div", { class: "side" });
  const playBtn = h("button", { onclick: () => togglePlay() }, "▶ Play");
  V.el = { canvas, tlc, slider, info, side, playBtn };
  const viewSel = h("select", { onchange: e => { V.view = e.target.value; V.maskImgs.clear(); loadTimeline().then(() => show(V.n)); } },
    h("option", { value: "" }, "Whole frame"), views.map(v => h("option", { value: v.view_id, selected: v.view_id === V.view }, "View: " + v.view_id)));
  main.className = "full";
  main.replaceChildren(h("div", { class: "viewer" },
    h("div", { class: "bar-top row" }, h("b", {}, vid), h("span", { class: "muted" }, `${m.width}×${m.height} · ${fmt(m.frame_count)} frames · ${fps.toFixed(2)} fps`), viewSel,
      h("label", { class: "row tight" }, h("input", { type: "checkbox", checked: true, onchange: e => { V.boxes = e.target.checked; draw(); } }), "boxes"),
      h("label", { class: "row tight" }, h("input", { type: "checkbox", checked: true, onchange: e => { V.masks = e.target.checked; draw(); } }), "masks"),
      h("span", { class: "spacer" }), h("a", { href: "#/review/" + encodeURIComponent(vid) }, "Review tracks →")),
    h("div", { class: "stage" }, canvas),
    side,
    h("div", { class: "transport" }, tlc, h("div", { class: "row", style: "margin-top:6px" }, playBtn,
      h("button", { onclick: () => seek(V.n - 1) }, "◀"), h("button", { onclick: () => seek(V.n + 1) }, "▶"), slider, info))));
  if (!meta.file_found) side.append(h("div", { class: "notice warn" }, "The video file was not found on this computer, so frames are shown as a grey placeholder. The annotations are still shown and can be reviewed."));
  slider.addEventListener("input", () => seek(+slider.value));
  tlc.addEventListener("pointerdown", e => { const f = tlFrame(e); seek(f); const mv = ev => seek(tlFrame(ev)); const up = () => { removeEventListener("pointermove", mv); removeEventListener("pointerup", up); }; addEventListener("pointermove", mv); addEventListener("pointerup", up); });
  canvas.addEventListener("click", ev => pickAt(ev));
  V.onkey = ev => {
    if (/input|textarea|select/i.test(ev.target.tagName)) return;
    const k = ev.key;
    if (k === "ArrowRight") seek(V.n + (ev.shiftKey ? 10 : 1)); else if (k === "ArrowLeft") seek(V.n - (ev.shiftKey ? 10 : 1));
    else if (k === " ") togglePlay(); else if (k === "Home") seek(0); else if (k === "End") seek(V.frames - 1);
    else if ((ev.ctrlKey || ev.metaKey) && k.toLowerCase() === "z") draftOp(ev.shiftKey ? "/api/draft/redo" : "/api/draft/undo").then(() => show(V.n));
    else if ((ev.ctrlKey || ev.metaKey) && k.toLowerCase() === "y") draftOp("/api/draft/redo").then(() => show(V.n));
    else return;
    ev.preventDefault();
  };
  document.addEventListener("keydown", V.onkey);
  new ResizeObserver(() => { if (V) { drawTimeline(); fitCanvas(); } }).observe(tlc);
  new ResizeObserver(() => { if (V) fitCanvas(); }).observe(canvas.parentElement);
  await loadTimeline();
  await show(V.n);
}
function tlFrame(ev) { const r = V.el.tlc.getBoundingClientRect(); return Math.round(Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * (V.frames - 1)); }
function togglePlay() {
  if (!V) return;
  V.playing = !V.playing; V.el.playBtn.textContent = V.playing ? "⏸ Pause" : "▶ Play";
  clearInterval(V.timer);
  if (V.playing) V.timer = setInterval(() => { if (!V.busy) { if (V.n >= V.frames - 1) return togglePlay(); seek(V.n + 1); } }, Math.max(40, 1000 / Math.min(V.fps, 12)));
}
function seek(n) { if (!V) return; n = Math.max(0, Math.min(V.frames - 1, n)); if (n !== V.n || !V.img) show(n); }
async function show(n) {
  const v = V, token = ++v.req;
  v.n = n; v.busy = true;
  v.el.slider.value = n;
  const q = v.view ? "?view_id=" + encodeURIComponent(v.view) : "";
  const imgP = v.found ? new Promise(res => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = `/api/frame/${encodeURIComponent(v.vid)}/${n}${q}`; }) : Promise.resolve(null);
  const dataP = GET(`/api/frame-data/${encodeURIComponent(v.vid)}/${n}`, { view_id: v.view }).catch(e => { toast(e.message, true); return { rows: [] }; });
  const [img, data] = await Promise.all([imgP, dataP]);
  if (V !== v || token !== v.req) return;
  v.busy = false; v.img = img || null; v.rows = data.rows;
  if (img) v.size = [img.naturalWidth, img.naturalHeight];
  await loadMasks();
  if (V !== v || token !== v.req) return;
  if (v.sel && !v.rows.some(r => r.annotation_id === v.sel)) v.sel = null;
  draw(); drawTimeline(); drawSide();
  history.replaceState(null, "", `#/video/${encodeURIComponent(v.vid)}?f=${n}` + (v.view ? "&view=" + encodeURIComponent(v.view) : ""));
}
function loadMasks() {
  return Promise.all(V.rows.filter(r => r.mask_png).map(r => new Promise(res => {
    const key = r.mask_png.length + ":" + r.mask_png.slice(-40);
    if (V.maskImgs.has(r.annotation_id + key)) { r._img = V.maskImgs.get(r.annotation_id + key); return res(); }
    const im = new Image(); im.onload = () => { V.maskImgs.set(r.annotation_id + key, im); r._img = im; res(); }; im.onerror = () => res(); im.src = r.mask_png;
  })));
}
function draw() {
  const v = V, c = v.el.canvas, ctx = c.getContext("2d");
  const [w, hh] = v.size;
  if (c.width !== w || c.height !== hh) { c.width = w; c.height = hh; }
  if (v.img) ctx.drawImage(v.img, 0, 0, w, hh);
  else { ctx.fillStyle = "#2a3140"; ctx.fillRect(0, 0, w, hh); ctx.fillStyle = "#8a94a6"; ctx.font = `${Math.max(12, w / 40)}px sans-serif`; ctx.fillText(v.found ? "frame could not be read" : "video file not found", 12, 24); }
  fitCanvas();
  const lw = Math.max(2, w / 360);
  for (const r of v.rows) {
    const col = labelColor(r.label), deleted = r.pending === "deleted", sel = r.annotation_id === v.sel;
    if (r.kind === "mask" && v.masks && r._img) {
      const off = document.createElement("canvas"); off.width = r._img.width; off.height = r._img.height;
      const oc = off.getContext("2d"); oc.drawImage(r._img, 0, 0); oc.globalCompositeOperation = "source-in"; oc.fillStyle = col; oc.fillRect(0, 0, off.width, off.height);
      ctx.globalAlpha = deleted ? 0.15 : sel ? 0.7 : 0.45; ctx.drawImage(off, r.mask_x, r.mask_y); ctx.globalAlpha = 1;
    }
    if ((r.kind === "box" || r.kind === "mask") && v.boxes && r.x0 != null) {
      ctx.lineWidth = sel ? lw * 1.8 : lw; ctx.strokeStyle = sel ? "#fff" : col;
      ctx.setLineDash(r.pending && r.pending !== "deleted" ? [lw * 3, lw * 2] : deleted ? [lw, lw * 3] : []);
      ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0); ctx.setLineDash([]);
      const text = r.label + (r.track_id != null ? " #" + r.track_id : "");
      ctx.font = `${Math.max(11, w / 55)}px sans-serif`; const tw = ctx.measureText(text).width + 6, th = Math.max(11, w / 55) + 4;
      ctx.fillStyle = col; ctx.fillRect(r.x0, Math.max(0, r.y0 - th), tw, th); ctx.fillStyle = "#fff"; ctx.fillText(text, r.x0 + 3, Math.max(th - 4, r.y0 - 4));
    }
  }
  V.el.info.textContent = `frame ${V.n} / ${V.frames - 1}  ·  ${(V.n / V.fps).toFixed(2)} s`;
}
function fitCanvas() {
  const c = V.el.canvas, st = c.parentElement, s = Math.min(st.clientWidth / c.width, st.clientHeight / c.height);
  if (s > 0 && isFinite(s)) { c.style.width = Math.floor(c.width * s) + "px"; c.style.height = Math.floor(c.height * s) + "px"; }
}
function pickAt(ev) {
  const c = V.el.canvas, r = c.getBoundingClientRect(), x = (ev.clientX - r.left) * c.width / r.width, y = (ev.clientY - r.top) * c.height / r.height;
  const hits = V.rows.filter(a => a.x0 != null && x >= a.x0 && x <= a.x1 && y >= a.y0 && y <= a.y1).sort((a, b) => (a.x1 - a.x0) * (a.y1 - a.y0) - (b.x1 - b.x0) * (b.y1 - b.y0));
  V.sel = hits.length ? hits[0].annotation_id : null; draw(); drawSide();
}
function drawSide() {
  const side = V.el.side, keep = side.querySelector(".notice");
  side.replaceChildren(keep || "", h("h2", {}, `On this frame (${V.rows.length})`));
  if (!V.rows.length) side.append(h("div", { class: "empty" }, "Nothing annotated on this frame."));
  for (const r of V.rows) {
    const id = r.annotation_id, vid = V.vid;
    side.append(h("div", { class: "annrow" + (id === V.sel ? " sel" : "") + (r.pending ? " pend" : "") + (r.pending === "deleted" ? " deleted" : ""), onclick: () => { V.sel = id; draw(); drawSide(); } },
      h("div", { class: "row" }, dot(r.label), h("b", { class: "lab" }, r.label), tag(r.kind, "tag"), r.track_id != null ? h("span", { class: "muted" }, "track " + r.track_id) : null,
        h("span", { class: "spacer" }), r.pending ? tag(r.pending, "pending") : null),
      h("div", { class: "row tight muted" }, tag(r.status), r.origin, r.confidence != null ? "· " + pct(r.confidence) : "", r.kind === "marker" ? `· frames ${r.start_frame}–${r.end_frame}` : ""),
      r.pending === "deleted" ? null : h("div", { class: "row tight", style: "margin-top:4px" },
        h("button", { class: "small ok", onclick: async e => { e.stopPropagation(); await setStatus(vid, [id], "approved"); show(V.n); loadTimeline(); } }, "Approve"),
        h("button", { class: "small bad", onclick: async e => { e.stopPropagation(); await setStatus(vid, [id], "rejected"); show(V.n); loadTimeline(); } }, "Reject"),
        h("button", { class: "small", title: "Delete this row (still undoable until you commit)", onclick: async e => { e.stopPropagation(); await draftOp("/api/draft/delete", { video_id: vid, ids: [id] }); show(V.n); loadTimeline(); } }, "Delete"))));
  }
  side.append(h("p", { class: "muted", style: "margin-top:12px" }, h("span", { class: "kbd" }, "←"), " ", h("span", { class: "kbd" }, "→"), " step · ",
    h("span", { class: "kbd" }, "Shift"), " 10 frames · ", h("span", { class: "kbd" }, "Space"), " play · ", h("span", { class: "kbd" }, "Ctrl+Z"), " undo"));
}
async function loadTimeline() {
  V.tl = await GET("/api/timeline/" + encodeURIComponent(V.vid), { view_id: V.view });
  drawTimeline();
}
function drawTimeline() {
  if (!V || !V.tl) return;
  const c = V.el.tlc, dpr = devicePixelRatio || 1, w = c.clientWidth, hh = 64;
  if (!w) return;
  c.width = w * dpr; c.height = hh * dpr;
  const ctx = c.getContext("2d"); ctx.scale(dpr, dpr);
  const cs = getComputedStyle(document.documentElement), accent = cs.getPropertyValue("--accent").trim(), pend = cs.getPropertyValue("--pending").trim();
  const n = V.frames, x = f => (f / Math.max(1, n - 1)) * (w - 1), tl = V.tl;
  const max = Math.max(1, ...tl.density), bw = (tl.bin_size / Math.max(1, n - 1)) * (w - 1);
  ctx.fillStyle = accent; ctx.globalAlpha = .55;
  tl.density.forEach((d, i) => { const bh = Math.ceil((d / max) * 34); ctx.fillRect(x(i * tl.bin_size), 40 - bh, Math.max(1, bw - 0.5), bh); });
  ctx.globalAlpha = 1;
  for (const mk of tl.markers) {
    ctx.fillStyle = mk.pending === "deleted" ? "#999" : labelColor(mk.label);
    ctx.fillRect(x(mk.start_frame), 44, Math.max(2, x(mk.end_frame) - x(mk.start_frame)), 8);
    if (mk.pending) { ctx.strokeStyle = pend; ctx.lineWidth = 1.5; ctx.strokeRect(x(mk.start_frame), 44, Math.max(2, x(mk.end_frame) - x(mk.start_frame)), 8); }
  }
  ctx.fillStyle = cs.getPropertyValue("--ink").trim(); ctx.fillRect(Math.round(x(V.n)) - 1, 0, 2, hh);
}

/* ---------- review ---------- */
async function pageReview(main, parts, q, alive) {
  await refreshInfo();
  const vids = S.info.videos;
  if (!vids.length) { main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Review"), h("div", { class: "card empty" }, "No videos yet."))); return; }
  const vid = vids.includes(parts[1]) ? parts[1] : vids[0];
  const viewIds = S.info.views.filter(Boolean);
  const view = q.view || "";
  const { tracks } = await GET("/api/tracks", { video_id: vid, view_id: view });
  if (!alive()) return;
  const st = { status: q.status || "", label: q.label || "", sort: q.sort || "id", sel: new Set() };
  const meta = (await GET("/api/videos/" + encodeURIComponent(vid))).metadata;
  const holder = h("div");
  const nav = (extra) => go("review/" + encodeURIComponent(vid), { view, ...st_q(), ...extra });
  const st_q = () => ({ status: st.status, label: st.label, sort: st.sort });
  const labels = [...new Set(tracks.flatMap(t => t.labels))].sort();
  const paint = () => {
    let rows = tracks.filter(t => (!st.label || t.labels.includes(st.label)) && (!st.status || (t.statuses[st.status] || 0) > 0));
    rows.sort((a, b) => st.sort === "conf" ? (a.mean_confidence ?? 2) - (b.mean_confidence ?? 2) : st.sort === "len" ? b.n_rows - a.n_rows : a.track_id - b.track_id);
    const act = async (ids, status) => {
      for (const t of ids) if (!(await setTrackStatus(vid, t, status, view || null))) return;
      toast(`${ids.length} track${ids.length > 1 ? "s" : ""} marked ${status} (in your draft)`);
      st.sel.clear(); await reload();
    };
    const all = h("input", { type: "checkbox", onchange: e => { rows.forEach(t => e.target.checked ? st.sel.add(t.track_id) : st.sel.delete(t.track_id)); paint(); } });
    holder.replaceChildren(
      h("div", { class: "row", style: "margin-bottom:8px" }, h("span", { class: "muted" }, `${rows.length} of ${tracks.length} tracks`), h("span", { class: "spacer" }),
        h("span", { class: "muted" }, `${st.sel.size} selected`),
        h("button", { class: "ok", disabled: !st.sel.size, onclick: () => act([...st.sel], "approved") }, "Approve selected"),
        h("button", { class: "bad", disabled: !st.sel.size, onclick: () => act([...st.sel], "rejected") }, "Reject selected"),
        h("button", { disabled: !st.sel.size, onclick: () => act([...st.sel], "unreviewed") }, "Reset selected")),
      rows.length ? h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, all), h("th", {}, ""), h("th", {}, "Track"), h("th", {}, "Label"), h("th", {}, "Frames"), h("th", { class: "num" }, "Rows"),
        h("th", { class: "num" }, "Confidence"), h("th", {}, "Status"), h("th", {}, "Origin"), h("th", {}, ""))),
        h("tbody", {}, rows.map(t => {
          const mid = Math.round((t.first_frame + t.last_frame) / 2);
          return h("tr", {}, h("td", {}, h("input", { type: "checkbox", checked: st.sel.has(t.track_id), onchange: e => { e.target.checked ? st.sel.add(t.track_id) : st.sel.delete(t.track_id); paint(); } })),
            h("td", {}, h("img", { class: "thumb", loading: "lazy", alt: "", src: `/api/frame/${encodeURIComponent(vid)}/${mid}?max_side=160` + (view ? "&view_id=" + encodeURIComponent(view) : ""), onerror: e => { e.target.style.visibility = "hidden"; } })),
            h("td", {}, h("a", { href: `#/video/${encodeURIComponent(vid)}?f=${t.first_frame}` + (view ? "&view=" + encodeURIComponent(view) : "") }, "#" + t.track_id), t.pending ? [" ", tag("changed", "pending")] : null),
            h("td", {}, t.labels.map(l => h("span", {}, dot(l), l, " "))), h("td", { class: "mono" }, `${t.first_frame}–${t.last_frame}`), h("td", { class: "num" }, fmt(t.n_rows)),
            h("td", { class: "num" }, pct(t.mean_confidence)), h("td", {}, Object.entries(t.statuses).map(([s, n]) => h("span", {}, tag(`${s} ${n}`, s), " "))), h("td", {}, t.origins.join(", ")),
            h("td", { class: "right" }, h("div", { class: "row tight", style: "justify-content:flex-end" },
              h("button", { class: "small ok", onclick: () => act([t.track_id], "approved") }, "Approve"), h("button", { class: "small bad", onclick: () => act([t.track_id], "rejected") }, "Reject"),
              h("button", { class: "small", title: "Delete the whole track (undoable until you commit)", onclick: async () => { if (await draftOp("/api/draft/delete-track", { video_id: vid, track_id: t.track_id, view_id: view || null })) reload(); } }, "Delete"))));
        }))) : h("div", { class: "empty" }, tracks.length ? "No tracks match these filters." : "This video has no tracked objects" + (view ? " in this view." : ".")));
  };
  const reload = async () => { const r = await GET("/api/tracks", { video_id: vid, view_id: view }); tracks.splice(0, tracks.length, ...r.tracks); paint(); };
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Review tracks"),
    h("p", { class: "muted" }, "Approve or reject whole tracks. Nothing is written to the dataset until you commit from ", h("b", {}, "Changes"), "."),
    h("div", { class: "card" }, h("div", { class: "row", style: "margin-bottom:10px" },
      h("select", { onchange: e => go("review/" + encodeURIComponent(e.target.value)) }, vids.map(v => h("option", { value: v, selected: v === vid }, v))),
      viewIds.length ? h("select", { onchange: e => go("review/" + encodeURIComponent(vid), { view: e.target.value }) }, h("option", { value: "" }, "Whole frame"), viewIds.map(v => h("option", { value: v, selected: v === view }, "View: " + v))) : null,
      h("select", { onchange: e => { st.status = e.target.value; paint(); } }, h("option", { value: "" }, "Any status"), ["unreviewed", "approved", "rejected"].map(s => h("option", { value: s, selected: s === st.status }, s))),
      h("select", { onchange: e => { st.label = e.target.value; paint(); } }, h("option", { value: "" }, "Any label"), labels.map(l => h("option", { value: l, selected: l === st.label }, l))),
      h("select", { onchange: e => { st.sort = e.target.value; paint(); } }, [["id", "Sort: track id"], ["conf", "Sort: lowest confidence first"], ["len", "Sort: longest first"]].map(([v, t]) => h("option", { value: v, selected: v === st.sort }, t)))),
      holder)));
  paint();
}

/* ---------- validation ---------- */
function issueTable(issues) {
  if (!issues.length) return h("div", { class: "notice" }, "No problems found.");
  return h("table", {}, h("thead", {}, h("tr", {}, ["Level", "Code", "Message", "Count"].map(t => h("th", {}, t)))),
    h("tbody", {}, issues.map(i => h("tr", {}, h("td", {}, tag(i.level, i.level)), h("td", { class: "mono" }, i.code), h("td", {}, i.message), h("td", { class: "num" }, i.count ?? "")))));
}
async function pageValidation(main) {
  const box = h("div");
  const paintResult = r => box.replaceChildren(h("div", { class: "row" }, tag(r.ok ? "valid" : "has errors", r.ok ? "ok" : "bad"), h("span", { class: "muted" }, r.deep ? "full check, masks decoded" : "quick check")), issueTable(r.issues));
  paintResult(await GET("/api/validate"));
  const deepBox = h("div");
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Validation"),
    h("p", { class: "muted" }, "The same checks as ", h("span", { class: "mono" }, "vidlabel validate"), ". Releases can only be built from a valid dataset."),
    h("div", { class: "card" }, box),
    h("div", { class: "card" }, h("h2", {}, "Full check"), h("p", { class: "muted" }, "Also decodes every mask and compares it with its box and the frame size. Slow on large datasets; it runs in the background."),
      h("button", { onclick: async e => { e.target.disabled = true; const pb = progressBox(); deepBox.replaceChildren(pb); const r = await runJob("validate", { deep: true }, pb.update); deepBox.replaceChildren(); e.target.disabled = false; if (r) paintResult(r); } }, "Run full check"), deepBox)));
}

/* ---------- releases ---------- */
async function pageReleases(main) {
  await refreshInfo();
  const { releases } = await GET("/api/releases");
  const stat = { unreviewed: true, approved: true, rejected: false };
  const out = h("div", {}, S.flash || []), pbox = h("div");
  S.flash = null;
  const build = async (btn) => {
    btn.disabled = true; const pb = progressBox(); pbox.replaceChildren(pb); out.replaceChildren();
    const r = await runJob("build", { statuses: Object.keys(stat).filter(k => stat[k]) }, pb.update);
    pbox.replaceChildren(); btn.disabled = false;
    if (r) { S.flash = [h("div", { class: "notice" }, `Built ${r.release_id}: ${fmt(r.n_rows)} rows.`), r.notes.map(n => h("div", { class: "notice warn" }, n))]; await refreshInfo(); render(); }
  };
  const sample = sampleCard(releases);
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "Releases"),
    h("p", { class: "muted" }, "A release is a frozen, checked copy of the annotations for training. Its id is a fingerprint of its files, so the same data always gives the same id."),
    h("div", { class: "card" }, releases.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Release", "Created", "Rows", "Videos", "Statuses", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, releases.map(r => h("tr", {}, h("td", { class: "mono" }, r.release_id), h("td", {}, (r.created_at || "").replace("T", " ").slice(0, 19)), h("td", { class: "num" }, fmt(r.n_rows)),
        h("td", { class: "num" }, fmt(r.n_videos)), h("td", {}, ((r.extra || {}).statuses || []).join(", ")), h("td", {}, r.is_latest ? tag("latest", "ok") : null)))))
      : h("div", { class: "empty" }, "No releases yet.")),
    h("div", { class: "card" }, h("h2", {}, "Build a release"),
      h("div", { class: "row" }, "Include rows that are:", Object.keys(stat).map(k => h("label", { class: "row tight" }, h("input", { type: "checkbox", checked: stat[k], onchange: e => { stat[k] = e.target.checked; } }), k))),
      h("p", { class: "muted" }, "Rows you have only drafted (not committed) are not included. Including rejected rows builds a release that is not made ‘latest’."),
      h("button", { class: "primary", onclick: e => build(e.target) }, "Build release"), pbox, out),
    sample));
}
function sampleCard(releases) {
  const out = h("div"), frame = h("input", { type: "number", value: 0, min: 0, style: "width:90px" });
  const vsel = h("select", {}, (S.info.videos).map(v => h("option", { value: v }, v)));
  return h("div", { class: "card" }, h("h2", {}, "Inspect a training sample"),
    h("p", { class: "muted" }, "What a training loader would return for one frame: the targets, not the pixels."),
    h("div", { class: "row" }, vsel, "frame", frame, h("button", { onclick: async () => {
      const r = await attempt(() => GET("/api/sample", { video_id: vsel.value, frame: frame.value, release: releases.length ? "latest" : "none", kind: ["box", "mask"] }));
      if (!r) return;
      out.replaceChildren(h("div", { class: "muted" }, `image size ${r.image_size.join("×")} · classes: ${r.class_names.join(", ")} · ${r.in_samples ? "this frame is in the training set" : "this frame is not in the training set (no annotations on it)"}`),
        h("table", {}, h("tbody", {}, Object.entries(r.targets).map(([k, t]) => h("tr", {}, h("td", { class: "mono" }, k), h("td", { class: "mono" }, "shape " + JSON.stringify(t.shape)), h("td", { class: "mono" }, t.values ? JSON.stringify(t.values) : "(large)"))))));
    } }, "Inspect")), out);
}

/* ---------- history ---------- */
async function pageHistory(main) {
  const { commits } = await GET("/api/history");
  main.replaceChildren(h("div", { class: "page" }, h("h1", {}, "History"),
    h("p", { class: "muted" }, "Every commit made through this interface. Reverting makes a new commit that puts things back, so nothing is lost. The last 50 are kept."),
    h("div", { class: "card" }, commits.length ? h("table", {}, h("thead", {}, h("tr", {}, ["#", "When", "Who", "Message", "Added", "Changed", "Deleted", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, commits.map(c => h("tr", {}, h("td", {}, c.seq), h("td", {}, (c.time || "").replace("T", " ").slice(0, 16)), h("td", {}, c.author),
        h("td", {}, c.kind === "revert" ? tag("revert", "warn") : null, " ", c.message, (c.skipped || []).length ? h("div", { class: "muted" }, `${c.skipped.length} row(s) were left alone because they changed since`) : null),
        h("td", { class: "num" }, fmt(c.summary.added)), h("td", { class: "num" }, fmt(c.summary.modified)), h("td", { class: "num" }, fmt(c.summary.deleted)),
        h("td", {}, h("button", { class: "small", onclick: async () => { if (!(await confirmBox("Revert commit #" + c.seq + "?", "This adds a new commit that undoes it. Rows changed again since then are left as they are."))) return; if (await attempt(() => POST(`/api/history/${c.seq}/revert`), "Reverted")) { await refreshInfo(); render(); } } }, "Revert")))))) : h("div", { class: "empty" }, "No commits yet."))));
}

/* ---------- start ---------- */
async function boot2() {
  await refreshInfo().catch(() => {});
  await refreshDraft().catch(() => {});
  await pollJobs();
  render();
}
async function boot() {
  S.author = localStorage.getItem("vl_author") || "";
  $("#changes-btn").addEventListener("click", toggleDrawer);
  $("#commit-btn").addEventListener("click", () => setDrawer(true));
  $("#undo-btn").addEventListener("click", () => draftOp("/api/draft/undo").then(() => render()));
  $("#redo-btn").addEventListener("click", () => draftOp("/api/draft/redo").then(() => render()));
  $("#jobs-btn").addEventListener("click", showJobs);
  $("#author-btn").addEventListener("click", () => askAuthor(true));
  S.session = await GET("/api/session");
  window.addEventListener("hashchange", render);
  if (!S.author) { await askAuthor(false); }
  $("#author-btn").textContent = "👤 " + S.author;
  await boot2();
}
function askAuthor(cancelable) {
  return new Promise(res => {
    const inp = h("input", { type: "text", value: S.author, placeholder: "your name", maxlength: 64 });
    const save = () => { const v = inp.value.trim(); if (!v) { toast("Please enter a name.", true); return false; } S.author = v; try { localStorage.setItem("vl_author", v); } catch (e) { /* private mode */ } $("#author-btn").textContent = "👤 " + v; res(); if (S.session && S.session.has_dataset) refreshDraft().then(render); };
    modal("Who is editing?", h("div", {}, h("p", { class: "muted" }, "Your name is saved with the changes you commit, and each person keeps their own draft. There are no accounts or passwords."), inp),
      cancelable ? [{ text: "Cancel", run: () => res() }, { text: "Save", primary: true, run: save }] : [{ text: "Continue", primary: true, run: save }]);
    setTimeout(() => inp.focus(), 50);
  });
}
boot().catch(e => { document.body.append(h("pre", { style: "padding:20px" }, "Could not start: " + e.message)); });
