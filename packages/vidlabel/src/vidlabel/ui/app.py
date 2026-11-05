"""The web server of `vidlabel ui`: a small Starlette application. Every route is a thin wrapper over the library, so nothing can be
written that the command line could not write, and every rule of the validator applies.

Safety, because a local server that edits files can otherwise be driven by any web page open in your browser:
- it listens on 127.0.0.1 unless told otherwise;
- every request needs the random token (cookie `vl_token`, set when you open the printed URL, or header `X-Vidlabel-Token`);
- the `Host` and `Origin` headers must name this server (this blocks DNS rebinding and cross-site requests);
- every request that changes something must carry the header `X-Requested-With: vidlabel`.

Annotation ids are 63-bit numbers, which JavaScript cannot hold exactly, so the interface sends and receives them as strings.
"""
from __future__ import annotations

import base64
import json
import math
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import numpy as np
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import __version__
from .. import masks as _masks
from ..dataset import Dataset
from ..validate import ValidationError
from . import draft as D
from . import ops
from .jobs import JobManager

STATIC = Path(__file__).parent / "static"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".mts", ".wmv", ".flv"}
_ID_KEYS = {"annotation_id", "ids", "added_ids", "deleted_ids", "sample_ids"}
MAX_PAGE = 2000


# ------------------------------------------------------------------ state
@dataclass
class UIState:
    path: Path
    token: str
    host: str = "127.0.0.1"
    port: int = 8765
    fs_roots: list[Path] | None = None          # None: the whole disk (only allowed when listening on this computer only)
    ds: Dataset | None = None
    jobs: JobManager = field(default_factory=JobManager)
    write_lock: threading.RLock = field(default_factory=threading.RLock)
    readers: dict = field(default_factory=dict)
    readers_lock: threading.Lock = field(default_factory=threading.Lock)
    video_roots: list[Path] = field(default_factory=list)   # folders searched for videos that moved (this session only)
    frame_cache: dict = field(default_factory=dict)          # (video, view, frame, max_side) -> JPEG bytes, least recently used first
    frame_cache_bytes: int = 0
    latest_seq: dict = field(default_factory=dict)           # page id -> highest request number seen (older ones are dropped)

    @property
    def local_only(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    def allowed_hosts(self) -> set[str]:
        names = {"127.0.0.1", "localhost", "[::1]"} if self.local_only else {self.host}
        return {f"{n}:{self.port}" for n in names} | ({n for n in names} if self.port in (80, 443) else set())

    def dataset(self) -> Dataset:
        if self.ds is None:
            raise Conflict409("no dataset is open yet: create one first")
        return self.ds

    def exports_dir(self) -> Path:
        return self.dataset().path / D.STATE_DIR / "exports"


class Conflict409(Exception):
    """A request that cannot be done in the current state (HTTP 409)."""


# ------------------------------------------------------------------ routes table
ROUTES: list[dict] = []


def route(method: str, path: str, *, covers: tuple[str, ...] = (), write: bool = False, raw: bool = False):
    """Register a handler `fn(state, ctx)`. `covers` names the CLI commands and `Dataset` methods the route stands for (a test checks
    that every command and public method is covered or excluded on purpose). `write` serialises the handler with other writers."""
    def deco(fn: Callable):
        ROUTES.append({"method": method, "path": path, "fn": fn, "covers": covers, "write": write, "raw": raw})
        return fn
    return deco


@dataclass
class Ctx:
    q: Any
    body: dict
    params: dict
    author: str

    def get(self, name: str, default: Any = None) -> Any:
        return self.body.get(name, default)

    def need(self, name: str) -> Any:
        if self.body.get(name) is None:
            raise ValueError(f"'{name}' is required")
        return self.body[name]

    def qlist(self, name: str) -> list[str] | None:
        v = [x for x in self.q.getlist(name)]
        return v or None


# ------------------------------------------------------------------ json helpers
def _num(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def out(obj: Any) -> Any:
    """Make a value safe to send as JSON: big ids become strings, bytes and NaN are removed."""
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if k in _ID_KEYS:
                res[k] = [str(i) for i in v] if isinstance(v, (list, tuple)) else (str(v) if isinstance(v, int) else v)
            elif isinstance(v, (bytes, bytearray)):
                continue
            else:
                res[k] = out(v)
        return res
    if isinstance(obj, (list, tuple)):
        return [out(v) for v in obj]
    if isinstance(obj, set):
        return sorted(out(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _num(float(obj))
    if isinstance(obj, np.ndarray):
        return out(obj.tolist())
    return _num(obj)


def to_id(v: Any) -> Any:
    """An id from the page: a numeric string or number becomes an int; a pending row's `p3` stays as it is."""
    if isinstance(v, str) and D.is_temp_id(v):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"not an annotation id: {v!r}") from None


def row_out(r: dict) -> dict:
    r = dict(r)
    r["has_mask"] = r.get("mask") is not None
    r.pop("mask", None)
    if "_pending" in r:
        r["pending"] = r.pop("_pending")
    return out(r)


def _frames(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    a, _, b = text.partition("-")
    try:
        lo, hi = int(a), int(b or a)
    except ValueError:
        raise ValueError(f"frames must look like '10-50', got '{text}'") from None
    return (lo, hi)


def _filters(ctx: Ctx) -> dict:
    f: dict[str, Any] = {}
    for name in ("kind", "label", "origin", "status"):
        if (v := ctx.qlist(name)) is not None:
            f[name] = v
    if (v := ctx.qlist("view_id")) is not None:
        f["view_id"] = [None if x == "" else x for x in v]
    if (v := ctx.qlist("track_id")) is not None:
        f["track_id"] = [int(x) for x in v]
    if fr := _frames(ctx.q.get("frames")):
        f["frames"] = fr
    return f


def bits_to_mask(obj: Any, size: tuple[int, int] | None = None) -> np.ndarray:
    """A mask from the page: `{"width", "height", "data"}` where data is base64 of the bits, row by row, 8 to a byte (most significant first)."""
    try:
        w, h = int(obj["width"]), int(obj["height"])
        raw = base64.b64decode(obj["data"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("a mask must be {width, height, data} with data as base64 of packed bits") from None
    if w <= 0 or h <= 0 or w * h > 1 << 28:
        raise ValueError("the mask size is not valid")
    if size is not None and (w, h) != tuple(size):
        raise ValueError(f"the mask is {w}×{h} but the video frame is {size[0]}×{size[1]}")
    if len(raw) != (w * h + 7) // 8:
        raise ValueError("the mask data has the wrong length")
    return np.unpackbits(np.frombuffer(raw, np.uint8))[:w * h].reshape(h, w).astype(bool)


def mask_to_bits(mask: np.ndarray) -> dict:
    h, w = mask.shape
    return {"width": w, "height": h, "data": base64.b64encode(np.packbits(mask.reshape(-1)).tobytes()).decode("ascii")}


def _with_mask(state: UIState, video_id: str, data: dict, key: str) -> dict:
    """Replace `mask_bits` in a request's row or values by a boolean `mask` array of the right size."""
    data = dict(data)
    if "mask_bits" in data:
        data["mask"] = bits_to_mask(data.pop("mask_bits"), state.dataset().video(video_id).display_size)
    return data


def _draft(state: UIState, ctx: Ctx) -> D.Draft:
    return D.Draft.load(state.dataset(), ctx.author)


# ------------------------------------------------------------------ session and overview
@route("GET", "/api/session")
def session(state: UIState, ctx: Ctx):
    return {"version": __version__, "path": str(state.path), "has_dataset": state.ds is not None, "local_only": state.local_only,
            "can_browse_files": state.local_only or state.fs_roots is not None, "author": ctx.author,
            "busy": state.jobs.busy()}


@route("POST", "/api/init", covers=("init", "Dataset.create"), write=True)
def init_dataset(state: UIState, ctx: Ctx):
    folder = _fs_check(state, Path(ctx.need("path")))
    state.ds = Dataset.create(folder, exist_ok=False)
    state.path = folder
    return {"path": str(folder)}


@route("GET", "/api/dataset", covers=("info", "Dataset.stats", "Dataset.manifest", "Dataset.vocab", "Dataset.video_ids", "Dataset.view_ids",
                                      "Dataset.open"))
def dataset_info(state: UIState, ctx: Ctx):
    ds = state.dataset()
    d = D.Draft.load(ds, ctx.author)
    return out({"path": str(ds.path), "spec_version": ds.manifest.spec_version, "library_version": __version__,
                "stats": ds.stats(), "labels": [l.model_dump(exclude_none=True) for l in ds.vocab.labels],
                "videos": ds.video_ids(), "views": ds.view_ids(), "releases": _releases(ds), "draft": d.state()["summary"]})


@route("GET", "/api/validate", covers=("validate", "Dataset.validate"))
def validate_now(state: UIState, ctx: Ctx):
    return ops.validate(state.dataset(), deep=False)


@route("GET", "/api/labels", covers=("Dataset.vocab",))
def labels(state: UIState, ctx: Ctx):
    ds = state.dataset()
    counts = ds.stats()["by_label"]
    return {"labels": [{**l.model_dump(exclude_none=True), "count": counts.get(l.name, 0)} for l in ds.vocab.labels]}


@route("POST", "/api/labels", covers=("add-labels", "Dataset.add_labels"), write=True)
def add_labels(state: UIState, ctx: Ctx):
    names = [n.strip() for chunk in ctx.need("names") for n in str(chunk).split(",") if n.strip()]
    if not names:
        raise ValueError("give at least one label name")
    state.dataset().add_labels(names)
    return {"added": names}


# ------------------------------------------------------------------ videos and views
@route("GET", "/api/videos", covers=("Dataset.video", "Dataset.video_ids"))
def videos(state: UIState, ctx: Ctx):
    ds = state.dataset()
    rows = []
    for v in ds.video_ids():
        m = ds.video(v)
        idx = ds.frame_index(v)
        rows.append({"video_id": v, "file_found": ds.resolve_video_path(v, state.video_roots) is not None, "width": m.width, "height": m.height,
                     "frame_count": m.frame_count, "fps": m.fps.value,
                     "codec": m.codec, "has_index": idx is not None, "has_hash": bool(m.content_hash),
                     "sources": [s.uri for s in m.sources], "is_vfr": m.is_vfr})
    return out({"videos": rows})


@route("GET", "/api/videos/{video_id}", covers=("Dataset.video", "Dataset.frame_index", "Dataset.resolve_video_path"))
def video_detail(state: UIState, ctx: Ctx):
    ds = state.dataset()
    vid = ctx.params["video_id"]
    m = ds.video(vid)
    idx = ds.frame_index(vid)
    found = ds.resolve_video_path(vid, state.video_roots)
    return out({"metadata": json.loads(m.model_dump_json(exclude_none=True)), "frame_index": idx.interval_stats() if idx else None,
                "file_found": bool(found), "file": str(found) if found else None,
                "views": [v for v in ds.view_ids() if ds.view(v).video_id == vid],
                "annotations": ds.annotations(vid, columns=["annotation_id"]).num_rows})


@route("POST", "/api/videos/{video_id}/locate", covers=("Dataset.resolve_video_path",))
def locate_video(state: UIState, ctx: Ctx):
    """Where is this video on this computer? Searches the given folders by file name."""
    ds = state.dataset()
    roots = [_fs_check(state, Path(r)) for r in ctx.get("roots", [])] + state.video_roots
    p = ds.resolve_video_path(ctx.params["video_id"], roots)
    return {"found": bool(p), "file": str(p) if p else None}


def _roots(state: UIState, ctx: Ctx) -> list[str]:
    return [str(_fs_check(state, Path(r))) for r in ctx.get("roots", [])] + [str(r) for r in state.video_roots]


@route("GET", "/api/roots")
def get_roots(state: UIState, ctx: Ctx):
    return {"roots": [str(r) for r in state.video_roots]}


@route("POST", "/api/roots", write=True)
def set_roots(state: UIState, ctx: Ctx):
    """Folders to search for videos that are not where the dataset recorded them. Kept for this session only."""
    state.video_roots = [_fs_check(state, Path(r)) for r in ctx.get("roots", [])]
    with state.readers_lock:
        for reader, lock in state.readers.values():
            with lock:
                reader.close()
        state.readers.clear()
    return {"roots": [str(r) for r in state.video_roots]}


@route("GET", "/api/views", covers=("Dataset.view", "Dataset.view_ids"))
def views(state: UIState, ctx: Ctx):
    ds = state.dataset()
    return out({"views": [json.loads(ds.view(v).model_dump_json(exclude_none=True)) for v in ds.view_ids()]})


@route("POST", "/api/views", covers=("add-view", "Dataset.add_view"), write=True)
def add_view(state: UIState, ctx: Ctx):
    v = state.dataset().add_view(dict(ctx.body))
    return json.loads(v.model_dump_json(exclude_none=True))


@route("POST", "/api/views/derive", covers=("derive-view", "Dataset.derive_view_rows"), write=True)
def derive_view(state: UIState, ctx: Ctx):
    return out(state.dataset().derive_view_rows(ctx.need("video_id"), ctx.need("view_id"), labels=ctx.get("labels") or None,
                                                origins=ctx.get("origins") or None, dry_run=bool(ctx.get("dry_run", False))))


# ------------------------------------------------------------------ annotations (with the author's draft applied)
def _rows(state: UIState, ctx: Ctx, video_id: str, filters: dict, use_draft: bool = True) -> list[dict]:
    ds = state.dataset()
    d = D.Draft.load(ds, ctx.author) if use_draft else None
    if d is not None and any(o["video_id"] == video_id for o in d.ops[:d.cursor]):
        return d.rows(video_id, **filters)
    return ds.annotations(video_id, **filters).to_pylist()


@route("GET", "/api/annotations", covers=("Dataset.annotations",))
def annotations(state: UIState, ctx: Ctx):
    ds = state.dataset()
    f = _filters(ctx)
    limit = min(int(ctx.q.get("limit", 200)), MAX_PAGE)
    offset = max(int(ctx.q.get("offset", 0)), 0)
    vid = ctx.q.get("video_id")
    use_draft = ctx.q.get("draft", "1") != "0"
    if vid and use_draft:
        d = D.Draft.load(ds, ctx.author)
        if any(o["video_id"] == vid for o in d.ops[:d.cursor]):
            rows = d.rows(vid, **f)
            return {"total": len(rows), "rows": [row_out(r) for r in rows[offset:offset + limit]]}
    t = ds.annotations(vid or None, **{k: v for k, v in f.items()})
    return {"total": t.num_rows, "rows": [row_out(r) for r in t.slice(offset, limit).to_pylist()]}


@route("GET", "/api/tracks", covers=("tracks", "Dataset.tracks"))
def tracks(state: UIState, ctx: Ctx):
    ds = state.dataset()
    vid = ctx.q.get("video_id") or ""
    view = ctx.q.get("view_id") or None
    d = D.Draft.load(ds, ctx.author)
    if any(o["video_id"] == vid for o in d.ops[:d.cursor]):
        return out({"tracks": _tracks_from_rows(d.rows(vid, view_id=[view], include_deleted=False))})
    return out({"tracks": ds.tracks(vid, view_id=view)})


def _tracks_from_rows(rows: list[dict]) -> list[dict]:
    by: dict[int, list[dict]] = {}
    for r in rows:
        if r.get("track_id") is not None:
            by.setdefault(r["track_id"], []).append(r)
    res = []
    for t, rs in sorted(by.items()):
        conf = [r["confidence"] for r in rs if r.get("confidence") is not None]
        st: dict[str, int] = {}
        for r in rs:
            st[r["status"]] = st.get(r["status"], 0) + 1
        res.append({"track_id": t, "labels": sorted({r["label"] for r in rs}), "n_rows": len(rs),
                    "first_frame": min(r["start_frame"] for r in rs), "last_frame": max(r["end_frame"] for r in rs),
                    "mean_confidence": sum(conf) / len(conf) if conf else None, "statuses": st,
                    "origins": sorted({r["origin"] for r in rs}), "pending": any(r.get("_pending") for r in rs)})
    return res


@route("GET", "/api/timeline/{video_id}", covers=("Dataset.annotations",))
def timeline(state: UIState, ctx: Ctx):
    """Everything the timeline needs in one call: markers, track spans, and how many annotations each stretch of frames has.
    `start`, `end` (exclusive) and `bins` show just a window of the video, so zooming in gets finer counts."""
    ds = state.dataset()
    vid = ctx.params["video_id"]
    view = ctx.q.get("view_id") or None
    n = ds.video(vid).frame_count
    start = max(0, min(int(ctx.q.get("start", 0)), max(n - 1, 0)))
    end = max(start + 1, min(int(ctx.q.get("end", n)), max(n, 1)))
    bins = max(1, min(int(ctx.q.get("bins", 200)), 2000))
    rows = _rows(state, ctx, vid, {"view_id": [view]})
    size = max(1, math.ceil((end - start) / bins))
    nb = math.ceil((end - start) / size)
    density = [0] * nb
    by_status = {st: [0] * nb for st in ("approved", "unreviewed", "rejected")}
    for r in rows:
        if r.get("_pending") == "deleted" or r["kind"] == "marker":
            continue
        a = r["start_frame"]
        b = r.get("end_frame")
        b = a if b is None or b < a else b
        if b < start or a >= end:
            continue
        for i in range((max(a, start) - start) // size, (min(b, end - 1) - start) // size + 1):
            density[i] += 1
            by_status.setdefault(r["status"], [0] * nb)[i] += 1
    markers = [row_out(r) for r in rows if r["kind"] == "marker"]
    return out({"frame_count": n, "start": start, "end": end, "bin_size": size, "density": density, "by_status": by_status,
                "markers": markers, "tracks": _tracks_from_rows([r for r in rows if r.get("_pending") != "deleted"])})


def _png(rgba: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("could not encode the mask image")
    return buf.tobytes()


def _mask_overlay(blob: bytes, view) -> tuple[bytes, int, int] | None:
    """The mask as a white PNG whose alpha is the mask, cropped to its pixels, and the position of its top-left corner, in the
    pixels of the picture being shown (the whole frame, or the view's crop)."""
    if view is None:
        crop, y0, x0 = _masks.decode_crop(blob)
    else:
        from ..frames import crop_mask_to_view
        full = crop_mask_to_view(_masks.decode(blob), view)
        ys, xs = np.nonzero(full)
        if not len(ys):
            return None
        y0, x0 = int(ys.min()), int(xs.min())
        crop = full[y0:int(ys.max()) + 1, x0:int(xs.max()) + 1]
    if crop.size == 0 or not crop.any():
        return None
    rgba = np.zeros((*crop.shape, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = crop.astype(np.uint8) * 255
    return _png(rgba), x0, y0


@route("GET", "/api/annotations-range/{video_id}", covers=("Dataset.annotations",))
def annotations_range(state: UIState, ctx: Ctx):
    """The annotations on frames `start`..`end` (inclusive) in one call, in the pixels of the picture shown, without mask pixels:
    a mask carries `mask_rev` instead and its image comes from `/api/mask-png`. The player asks for a window around the current
    frame so stepping and playing need no request per frame."""
    import zlib
    ds = state.dataset()
    vid = ctx.params["video_id"]
    start, end = int(ctx.q.get("start", 0)), int(ctx.q.get("end", 0))
    if end < start or end - start > 1200:
        raise ValueError("ask for at most 1200 frames at a time")
    view_id = ctx.q.get("view_id") or None
    view = ds.view(view_id) if view_id else None
    res = []
    for r in _rows(state, ctx, vid, {"frames": (start, end), "view_id": [view_id]}):
        o = row_out({k: v for k, v in r.items() if k != "mask"})
        if r["kind"] != "marker" and r.get("x0") is not None and view is not None:
            o["x0"], o["y0"], o["x1"], o["y1"] = view.to_view_box((r["x0"], r["y0"], r["x1"], r["y1"]))
        if r["kind"] == "mask":
            o["mask_rev"] = "%08x" % (zlib.crc32(bytes(r["mask"])) if r.get("mask") is not None else 0)
        res.append(o)
    return {"start": start, "end": end, "view_id": view_id, "rows": res}


@route("GET", "/api/mask-png/{video_id}/{ann_id}", covers=("preview",), raw=True)
def mask_png(state: UIState, ctx: Ctx):
    """One mask as a white PNG whose alpha is the mask (cropped to its pixels). Its corner is in `X-Mask-X` and `X-Mask-Y`. The
    picture only changes when the mask does (`rev`), so it is cached for a long time."""
    ds = state.dataset()
    vid, aid, n = ctx.params["video_id"], to_id(ctx.params["ann_id"]), int(ctx.q.get("frame", 0))
    view_id = ctx.q.get("view_id") or None
    view = ds.view(view_id) if view_id else None
    for r in _rows(state, ctx, vid, {"frames": (n, n), "view_id": [view_id]}):
        if r["annotation_id"] == aid and r.get("mask") is not None:
            m = _mask_overlay(bytes(r["mask"]), view)
            if m is None:
                return Response(status_code=204)
            cache = "private, max-age=31536000, immutable" if ctx.q.get("rev") else "no-store"
            return Response(m[0], media_type="image/png", headers={"X-Mask-X": str(m[1]), "X-Mask-Y": str(m[2]), "Cache-Control": cache})
    raise KeyError(f"no mask {ctx.params['ann_id']} on frame {n} of video '{vid}'")


@route("GET", "/api/mask-bits/{video_id}/{ann_id}")
def mask_bits(state: UIState, ctx: Ctx):
    """One mask in full-frame size, to edit it in the page. Needs `frame` (any frame the row covers) to find the row quickly."""
    vid, aid, n = ctx.params["video_id"], to_id(ctx.params["ann_id"]), int(ctx.q.get("frame", 0))
    for r in _rows(state, ctx, vid, {"frames": (n, n)}):
        if r["annotation_id"] == aid and r.get("mask") is not None:
            return mask_to_bits(_masks.decode(bytes(r["mask"])))
    raise KeyError(f"no mask {ctx.params['ann_id']} on frame {n} of video '{vid}'")


@route("GET", "/api/frame-data/{video_id}/{frame}", covers=("preview",))
def frame_data(state: UIState, ctx: Ctx):
    """The annotations on one frame (with the draft applied), already in the pixels of the picture shown, masks as small images."""
    ds = state.dataset()
    vid, n = ctx.params["video_id"], int(ctx.params["frame"])
    view_id = ctx.q.get("view_id") or None
    view = ds.view(view_id) if view_id else None
    rows = _rows(state, ctx, vid, {"frames": (n, n), "view_id": [view_id]})
    res = []
    for r in rows:
        o = row_out(r)
        if r["kind"] != "marker" and r.get("x0") is not None and view is not None:
            o["x0"], o["y0"], o["x1"], o["y1"] = view.to_view_box((r["x0"], r["y0"], r["x1"], r["y1"]))
        if r["kind"] == "mask" and r.get("mask") is not None:
            m = _mask_overlay(bytes(r["mask"]), view)
            if m:
                o["mask_png"], o["mask_x"], o["mask_y"] = "data:image/png;base64," + base64.b64encode(m[0]).decode(), m[1], m[2]
        res.append(o)
    return {"frame": n, "view_id": view_id, "rows": res}


# ------------------------------------------------------------------ frames
def _reader(state: UIState, video_id: str, view_id: str | None):
    from ..frames import FrameReader
    key = (video_id, view_id)
    with state.readers_lock:
        entry = state.readers.pop(key, None)
        if entry is None:
            ds = state.dataset()
            meta = ds.video(video_id)
            path = ds.resolve_video_path(video_id, state.video_roots)
            if path is None:
                raise FileNotFoundError(f"cannot find the video file for '{video_id}'. Use 'Locate the video' on the Videos page.")
            entry = (FrameReader(path, meta, ds.view(view_id) if view_id else None, index=ds.frame_index(video_id)), threading.Lock())
        state.readers[key] = entry                      # most recently used goes last
        while len(state.readers) > 4:
            old = next(iter(state.readers))
            reader, lock = state.readers.pop(old)
            with lock:
                reader.close()
    return entry


FRAME_CACHE_LIMIT = 96 * 1024 * 1024


def _superseded(state: UIState, ctx: Ctx) -> bool:
    """True when the page has since asked for something newer (`cid` is the page, `seq` counts its requests). Such a request is
    not worth decoding: the picture it would produce has already been replaced."""
    cid, seq = ctx.q.get("cid"), ctx.q.get("seq")
    if not cid or seq is None:
        return False
    with state.readers_lock:
        return int(seq) < state.latest_seq.get(cid, -1)


def _note_seq(state: UIState, ctx: Ctx) -> None:
    cid, seq = ctx.q.get("cid"), ctx.q.get("seq")
    if cid and seq is not None:
        with state.readers_lock:
            state.latest_seq[cid] = max(state.latest_seq.get(cid, -1), int(seq))
            while len(state.latest_seq) > 64:
                state.latest_seq.pop(next(iter(state.latest_seq)))


@route("GET", "/api/frame/{video_id}/{frame}", covers=("preview",), raw=True)
def frame_image(state: UIState, ctx: Ctx):
    """One frame as a JPEG. Repeated frames come from a small memory cache. A request that the page has already replaced with a
    newer one (`cid`, `seq`) is answered 204 without decoding, so scrubbing never queues up behind old frames."""
    import cv2
    vid, n = ctx.params["video_id"], int(ctx.params["frame"])
    ds = state.dataset()
    if not 0 <= n < ds.video(vid).frame_count:
        raise KeyError(f"video '{vid}' has no frame {n}")
    view_id = ctx.q.get("view_id") or None
    max_side = int(ctx.q.get("max_side", 0))
    _note_seq(state, ctx)
    key = (vid, view_id, n, max_side)
    headers = {"Cache-Control": "private, max-age=300"}
    with state.readers_lock:
        hit = state.frame_cache.pop(key, None)
        if hit is not None:
            state.frame_cache[key] = hit
    if hit is not None:
        return Response(hit, media_type="image/jpeg", headers=headers)
    if _superseded(state, ctx):
        return Response(status_code=204)
    reader, lock = _reader(state, vid, view_id)
    with lock:
        if _superseded(state, ctx):
            return Response(status_code=204)
        rgb = reader.read(n)
    if max_side and max(rgb.shape[:2]) > max_side:
        scale = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode the frame")
    data = buf.tobytes()
    with state.readers_lock:
        if key not in state.frame_cache:
            state.frame_cache[key] = data
            state.frame_cache_bytes += len(data)
        while state.frame_cache_bytes > FRAME_CACHE_LIMIT and len(state.frame_cache) > 1:
            state.frame_cache_bytes -= len(state.frame_cache.pop(next(iter(state.frame_cache))))
    return Response(data, media_type="image/jpeg", headers=headers)


# ------------------------------------------------------------------ the draft
@route("GET", "/api/draft")
def draft_state(state: UIState, ctx: Ctx):
    return out(_draft(state, ctx).state())


def _author(ctx: Ctx) -> str:
    return ctx.author


@route("POST", "/api/draft/add", write=True, covers=("Dataset.add_annotations",))
def draft_add(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    vid = ctx.need("video_id")
    row = _with_mask(state, vid, dict(ctx.need("row")), "row")
    tmp = d.add(vid, row)
    return {"id": tmp, "draft": out(d.state())}


@route("POST", "/api/draft/update", write=True, covers=("Dataset.update_annotations",))
def draft_update(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    vid = ctx.need("video_id")
    d.update(vid, [to_id(i) for i in ctx.need("ids")], **_with_mask(state, vid, dict(ctx.need("values")), "values"))
    return out(d.state())


@route("POST", "/api/draft/delete", write=True, covers=("Dataset.delete_annotations",))
def draft_delete(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    d.delete(ctx.need("video_id"), [to_id(i) for i in ctx.need("ids")])
    return out(d.state())


@route("POST", "/api/draft/update-track", write=True, covers=("review", "Dataset.update_track"))
def draft_update_track(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    d.update_track(ctx.need("video_id"), int(ctx.need("track_id")), view_id=ctx.get("view_id") or None, **dict(ctx.need("values")))
    return out(d.state())


@route("POST", "/api/draft/delete-track", write=True, covers=("Dataset.delete_track",))
def draft_delete_track(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    d.delete_track(ctx.need("video_id"), int(ctx.need("track_id")), view_id=ctx.get("view_id") or None)
    return out(d.state())


@route("POST", "/api/draft/undo", write=True)
def draft_undo(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    d.undo()
    return out(d.state())


@route("POST", "/api/draft/redo", write=True)
def draft_redo(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    d.redo()
    return out(d.state())


@route("POST", "/api/draft/discard", write=True)
def draft_discard(state: UIState, ctx: Ctx):
    d = _draft(state, ctx)
    if ctx.get("index") is None:
        d.clear()
    else:
        d.discard(int(ctx.get("index")))
    return out(d.state())


@route("POST", "/api/draft/check", write=True)
def draft_check(state: UIState, ctx: Ctx):
    return out(_draft(state, ctx).check(reapply=bool(ctx.get("reapply", False))))


@route("POST", "/api/draft/commit", write=True, covers=("Dataset.add_annotations", "Dataset.update_annotations", "Dataset.delete_annotations"))
def draft_commit(state: UIState, ctx: Ctx):
    return out(_draft(state, ctx).commit(ctx.get("message", ""), reapply=bool(ctx.get("reapply", False))))


@route("GET", "/api/history")
def get_history(state: UIState, ctx: Ctx):
    return out({"commits": D.history(state.dataset())})


@route("POST", "/api/history/{seq}/revert", write=True)
def revert_commit(state: UIState, ctx: Ctx):
    return out(D.revert(state.dataset(), int(ctx.params["seq"]), ctx.author))


# ------------------------------------------------------------------ releases and samples
def _releases(ds: Dataset) -> list[dict]:
    root = ds.path / "compiled"
    if not root.exists():
        return []
    latest = (root / "LATEST").read_text(encoding="utf-8").strip() if (root / "LATEST").exists() else None
    res = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "release.json").exists():
            r = json.loads((p / "release.json").read_text(encoding="utf-8"))
            r["is_latest"] = p.name == latest
            r["bytes"] = (p / "annotations.parquet").stat().st_size if (p / "annotations.parquet").exists() else None
            res.append(r)
    return sorted(res, key=lambda r: r.get("created_at", ""), reverse=True)


@route("GET", "/api/releases", covers=("build",))
def releases(state: UIState, ctx: Ctx):
    return out({"releases": _releases(state.dataset())})


@route("GET", "/api/sample", covers=("FrameSamples.targets", "FrameSamples.describe"))
def sample_inspector(state: UIState, ctx: Ctx):
    """What one training sample would contain for a frame (the targets, not the pixels)."""
    from ..samples import FrameSamples
    ds = state.dataset()
    vid, n = ctx.q.get("video_id"), int(ctx.q.get("frame", 0))
    meta = ds.video(vid)
    release = ctx.q.get("release") or "latest"
    s = FrameSamples(ds, release=None if release == "none" else release, view_id=ctx.q.get("view_id") or None,
                     kinds=tuple(ctx.qlist("kind") or ("box", "mask")), with_masks=False, box_format=ctx.q.get("box_format", "xyxy"),
                     normalized=ctx.q.get("normalized") == "1", label_offset=int(ctx.q.get("label_offset", 0)))
    view = s.view
    size = tuple(view.output_size) if view is not None and view.output_size else (
        (view.region.width, view.region.height) if view is not None and view.region else meta.display_size)
    t = s.targets(vid, n, (int(size[0]), int(size[1])))
    return out({"describe": s.describe(), "class_names": s.class_names, "image_size": list(size),
                "targets": {k: {"shape": list(np.shape(v)), "values": np.asarray(v).tolist() if np.size(v) <= 200 else None}
                            for k, v in t.items()}, "in_samples": (vid, n) in set(s.index)})


# ------------------------------------------------------------------ jobs
def _job(state: UIState, kind: str, title: str, fn: Callable, writes: bool = False):
    def run(ctx):
        if writes:
            with state.write_lock:
                return fn(ctx)
        return fn(ctx)
    return state.jobs.submit(kind, title, run).as_dict()


@route("POST", "/api/jobs/validate", covers=("validate",))
def job_validate(state: UIState, ctx: Ctx):
    ds, deep = state.dataset(), bool(ctx.get("deep", False))
    return out(_job(state, "validate", "Validate" + (" (deep)" if deep else ""), lambda c: ops.validate(ds, deep=deep, ctx=c)))


@route("POST", "/api/jobs/build", covers=("build", "Dataset.build"))
def job_build(state: UIState, ctx: Ctx):
    ds = state.dataset()
    kw = {k: ctx.get(k) for k in ("statuses", "origins", "kinds", "video_ids", "view_ids")}
    return out(_job(state, "build", "Build a release", lambda c: ops.build(ds, ctx=c, **kw), writes=True))


@route("POST", "/api/jobs/add-video", covers=("add-video", "Dataset.add_video"))
def job_add_video(state: UIState, ctx: Ctx):
    ds = state.dataset()
    file = _fs_check(state, Path(ctx.need("path")))
    kw = {"video_id": ctx.get("video_id") or None, "hash": bool(ctx.get("hash", True)), "index": bool(ctx.get("index", True))}
    return out(_job(state, "add-video", f"Add {file.name}", lambda c: ops.add_video(ds, file, ctx=c, **kw), writes=True))


@route("POST", "/api/jobs/index", covers=("index", "Dataset.set_frame_index"))
def job_index(state: UIState, ctx: Ctx):
    ds = state.dataset()
    roots = _roots(state, ctx)
    kw = {"video_ids": ctx.get("video_ids") or None, "force": bool(ctx.get("force", False)), "verify": bool(ctx.get("verify", True)),
          "roots": roots}
    return out(_job(state, "index", "Index videos", lambda c: ops.index_videos(ds, ctx=c, **kw), writes=True))


@route("POST", "/api/jobs/check-video", covers=("check-video",))
def job_check_video(state: UIState, ctx: Ctx):
    ds = state.dataset()
    vid = ctx.need("video_id")
    roots = _roots(state, ctx)
    vp = str(_fs_check(state, Path(ctx.get("video_path")))) if ctx.get("video_path") else None
    kw = {"video_path": vp, "roots": roots, "hash": bool(ctx.get("hash", False)), "seek": bool(ctx.get("seek", False))}
    return out(_job(state, "check-video", f"Check {vid}", lambda c: ops.check_video(ds, vid, ctx=c, **kw)))


@route("POST", "/api/jobs/cache", covers=("cache",))
def job_cache(state: UIState, ctx: Ctx):
    ds = state.dataset()
    dest = _fs_check(state, Path(ctx.get("out") or (ds.path / D.STATE_DIR / "cache")))
    kw = {k: ctx.get(k) for k in ("release", "view_id", "frames", "format", "quality") if ctx.get(k) is not None}
    kw["overwrite"] = bool(ctx.get("overwrite", False))
    kw["roots"] = _roots(state, ctx)
    return out(_job(state, "cache", "Build the frame cache", lambda c: ops.build_cache(ds, dest, ctx=c, **kw)))


@route("POST", "/api/jobs/preview", covers=("preview",))
def job_preview(state: UIState, ctx: Ctx):
    ds = state.dataset()
    vid = ctx.need("video_id")
    frames = None
    if ctx.get("frames"):
        from ..preview import parse_frames
        frames = parse_frames(str(ctx.get("frames")), ds.video(vid).frame_count)
    tag = os.urandom(4).hex()
    dest = state.exports_dir() / tag
    kw = {"frames": frames, "view_id": ctx.get("view_id") or None, "mp4": bool(ctx.get("mp4", False)),
          "roots": _roots(state, ctx)}

    def run(c):
        res = ops.export_preview(ds, vid, dest, ctx=c, **kw)
        res["export"] = tag
        return res
    return out(_job(state, "preview", f"Export previews of {vid}", run))


@route("GET", "/api/jobs", covers=())
def list_jobs(state: UIState, ctx: Ctx):
    return out({"jobs": state.jobs.list()})


@route("GET", "/api/jobs/{job_id}")
def get_job(state: UIState, ctx: Ctx):
    return out(state.jobs.get(ctx.params["job_id"]).as_dict())


@route("POST", "/api/jobs/{job_id}/cancel")
def cancel_job(state: UIState, ctx: Ctx):
    return out(state.jobs.cancel(ctx.params["job_id"]).as_dict())


@route("GET", "/api/exports/{tag}/{name}", raw=True)
def download_export(state: UIState, ctx: Ctx):
    tag, name = ctx.params["tag"], ctx.params["name"]
    if not tag.isalnum() or "/" in name or "\\" in name or name.startswith("."):
        raise KeyError("no such export")
    p = state.exports_dir() / tag / name
    if not p.is_file():
        raise KeyError("no such export")
    return FileResponse(p)


@route("GET", "/api/schema", covers=("schema",))
def schema(state: UIState, ctx: Ctx):
    from vidlabel_spec import json_schema
    return json_schema()


# ------------------------------------------------------------------ a folder picker (this computer's disk)
def _fs_check(state: UIState, p: Path) -> Path:
    """Paths the page may name must be inside the allowed folders; on a network-visible server there must be some."""
    text = str(p).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":     # a path pasted with quotes ("Copy as path" on Windows)
        text = text[1:-1].strip()
    p = Path(os.path.expanduser(text)).resolve()
    if state.fs_roots is None:
        if not state.local_only:
            raise PermissionError("this server is shared on the network, so file access is off. Start it with --fs-root FOLDER to allow one")
        return p
    for root in state.fs_roots:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"{p} is outside the folders this server may use: {', '.join(str(r) for r in state.fs_roots)}")


@route("GET", "/api/fs", covers=())
def list_folder(state: UIState, ctx: Ctx):
    raw = ctx.q.get("path") or ""
    if not raw:
        if state.fs_roots:
            return {"path": "", "parent": None, "entries": [{"name": str(r), "path": str(r), "dir": True} for r in state.fs_roots]}
        if os.name == "nt":
            import string
            drives = [f"{c}:\\" for c in string.ascii_uppercase if os.path.exists(f"{c}:\\")]
            return {"path": "", "parent": None, "entries": [{"name": d, "path": d, "dir": True} for d in drives]}
        raw = str(Path.home())
    p = _fs_check(state, Path(raw))
    if p.is_file():                                  # a file was given: show the folder it is in
        p = p.parent
    if not p.is_dir():
        raise FileNotFoundError(f"there is no folder at {p}")
    entries = []
    for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
        if c.name.startswith("."):
            continue
        try:
            is_dir = c.is_dir()
        except OSError:
            continue
        if is_dir or c.suffix.lower() in VIDEO_EXT:
            entries.append({"name": c.name, "path": str(c), "dir": is_dir, "video": not is_dir,
                            "is_dataset": is_dir and (c / "manifest.json").exists()})
    parent = None
    try:
        parent = str(_fs_check(state, p.parent)) if p.parent != p else None
    except PermissionError:
        parent = None
    return {"path": str(p), "parent": parent, "entries": entries}


# ------------------------------------------------------------------ the application
def _json_error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def _handle(exc: BaseException) -> JSONResponse:
    if isinstance(exc, D.Conflict):
        return _json_error(409, str(exc), conflict=True, videos=exc.videos)
    if isinstance(exc, Conflict409):
        return _json_error(409, str(exc))
    if isinstance(exc, ValidationError):
        return _json_error(422, str(exc), issues=[{"code": i.code, "message": i.message} for i in exc.issues])
    if isinstance(exc, PermissionError):
        return _json_error(403, str(exc))
    if isinstance(exc, (KeyError, FileNotFoundError)):
        return _json_error(404, str(exc.args[0]) if exc.args else str(exc))
    if isinstance(exc, FileExistsError):
        return _json_error(409, str(exc))
    if isinstance(exc, (ValueError, D.DraftError, TypeError)):
        return _json_error(400, str(exc))
    if isinstance(exc, ImportError):
        return _json_error(501, str(exc))
    return _json_error(500, f"{type(exc).__name__}: {exc}")


def create_app(path: str | Path, *, token: str | None = None, host: str = "127.0.0.1", port: int = 8765,
               fs_roots: list[str | Path] | None = None) -> Starlette:
    """Build the application for the dataset at `path` (a folder that is not yet a dataset starts the interface in setup mode)."""
    path = Path(path)
    state = UIState(path=path, token=token or secrets.token_urlsafe(24), host=host, port=port,
                    fs_roots=[Path(r).resolve() for r in fs_roots] if fs_roots else None)
    if (path / "manifest.json").exists():
        state.ds = Dataset.open(path)
    if not state.local_only and state.fs_roots is None:
        pass                                    # allowed: file access simply stays off (see _fs_check)

    def wrap(spec: dict):
        fn = spec["fn"]

        async def endpoint(request: Request):
            body: dict = {}
            if request.method in ("POST", "PUT", "DELETE"):
                raw = await request.body()
                if raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        return _json_error(400, "the request body is not valid JSON")
                    if not isinstance(body, dict):
                        return _json_error(400, "the request body must be a JSON object")
            author = unquote(request.headers.get("x-vidlabel-author", "")).strip() or "anonymous"
            ctx = Ctx(q=request.query_params, body=body, params=dict(request.path_params), author=author[:64])

            def call():
                if spec["write"]:
                    with state.write_lock:
                        return fn(state, ctx)
                return fn(state, ctx)
            try:
                res = await run_in_threadpool(call)
            except BaseException as e:               # becomes a clear message on the page, with the right status code
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                if os.environ.get("VIDLABEL_DEBUG") == "1":
                    import traceback
                    traceback.print_exc()
                return _handle(e)
            return res if isinstance(res, Response) else JSONResponse(res)
        return endpoint

    async def index(request: Request):
        given = request.query_params.get("token")
        if given and secrets.compare_digest(given, state.token):
            resp = RedirectResponse("/")
            resp.set_cookie("vl_token", state.token, httponly=True, samesite="strict", path="/")
            return resp
        if not _authorised(request, state):
            return PlainTextResponse("Open the address that `vidlabel ui` printed (it ends with ?token=...).", status_code=401)
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    routes = [Route("/", index, methods=["GET"])]
    for spec in ROUTES:
        routes.append(Route(spec["path"], wrap(spec), methods=[spec["method"]]))
    routes.append(Mount("/static", app=StaticFiles(directory=STATIC), name="static"))

    app = Starlette(routes=routes)
    app.state.ui = state
    app.add_middleware(_Guard, state=state)
    return app


def _authorised(request: Request, state: UIState) -> bool:
    given = request.cookies.get("vl_token") or request.headers.get("x-vidlabel-token") or ""
    return bool(given) and secrets.compare_digest(given, state.token)


class _Guard:
    """Host, Origin, token and CSRF-header checks for every request (see the module docstring)."""

    def __init__(self, app, state: UIState):
        self.app, self.state = app, state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        req = Request(scope)
        problem = self.check(req)
        if problem:
            await PlainTextResponse(problem[1], status_code=problem[0])(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def check(self, req: Request) -> tuple[int, str] | None:
        host = req.headers.get("host", "")
        if self.state.local_only:
            if host not in self.state.allowed_hosts():          # blocks DNS rebinding
                return 421, f"unexpected Host '{host}'"
            allowed = self.state.allowed_hosts()
        else:
            allowed = {host}                                     # shared: any address works, the token is what protects
        origin = req.headers.get("origin")
        if origin and urlparse(origin).netloc not in allowed:
            return 403, "cross-site request refused"
        path = req.url.path
        if path == "/" or path.startswith("/static/"):
            return None
        if not _authorised(req, self.state):
            return 401, "missing or wrong access token: open the address that `vidlabel ui` printed"
        if req.method in ("POST", "PUT", "DELETE") and req.headers.get("x-requested-with") != "vidlabel":
            return 403, "missing X-Requested-With header"
        return None
