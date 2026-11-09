"""What an AI agent may do with a dataset, as plain functions (the MCP server in `vidlabel.mcp_server` is a thin layer over this).

The rules are the point:

- An agent can **read** (overview, annotations, tracks, a frame picture with the annotations drawn on it, validation).
- An agent can **propose**: add boxes, spans and polygon masks. They go into the agent's own *draft* (the same draft a person's
  web-interface session uses, under the author `agent:<name>`), always as `origin="model"`, `status="unreviewed"`, with an optional
  confidence and a short note. Every row is checked by the library before it is recorded, exactly like a row drawn by hand.
- An agent **cannot** approve or reject rows, change or delete rows that are already in the dataset, change `origin` or `author`, or
  commit. Nothing reaches the dataset until a person takes the proposals into their own draft (web interface, *Proposals*) and commits.
- A batch is all or nothing: if one row is refused, none of the batch is recorded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import Dataset
from .ui import draft as D

MAX_ROWS_PER_CALL = 500
MAX_PENDING_ROWS = 20000          # one agent cannot pile up more than this in its draft
MAX_LIST = 500


class AgentError(ValueError):
    """A mistake the agent can read and fix; the message says what was wrong."""


def agent_author(name: str | None) -> str:
    """`agent:<name>`. The name is reduced to lower-case letters, digits, `-` and `_`; if anything had to be changed, a short hash of the
    original is added, so two different names never share a draft (also on case-insensitive disks)."""
    import hashlib
    raw = (name or "").strip() or "agent"
    kept = "".join(c for c in raw.lower() if c in "abcdefghijklmnopqrstuvwxyz0123456789-_")[:32]
    if kept != raw:
        kept = (kept + "-" if kept else "") + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
    return "agent:" + kept


def _id_out(i: Any) -> Any:
    return str(i) if isinstance(i, int) else i


def _row_out(r: dict) -> dict:
    o = {k: v for k, v in r.items() if k not in ("mask", "mask_codec", "mask_b64") and v is not None}
    o["annotation_id"] = _id_out(r["annotation_id"])
    o["has_mask"] = r.get("mask") is not None
    if "_pending" in o:
        o["pending"] = o.pop("_pending")
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in o.items()}


def _at(i):
    return "" if i is None else f"item {i}: "


def _int(v: Any, what: str, i: int | None, lo: int = 0, hi: int = 2**53) -> int:
    """A whole number (`3` or `3.0`; not true/false, text, 1.5 or something huge)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
        raise AgentError(f"{_at(i)}{what} must be a whole number, got {v!r}")
    v = int(v)
    if not lo <= v <= hi:
        raise AgentError(f"{_at(i)}{what} {v} is out of range ({lo} to {hi})")
    return v


def _number(v: Any, what: str, i: int | None) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise AgentError(f"{_at(i)}{what} must be a finite number, got {v!r}")
    return float(v)


def _label(v: Any, i: int) -> str:
    if not isinstance(v, str) or not v.strip():
        raise AgentError(f"item {i}: label must be a text, got {v!r}. Use one of the labels from `overview`")
    return v


class AgentSession:
    def __init__(self, path: str | Path, name: str | None = None):
        self.ds = Dataset.open(path)
        self.author = agent_author(name)

    # ------------------------------------------------------------------ reading
    def overview(self) -> dict:
        ds = self.ds
        d = D.Draft.load(ds, self.author)
        return {"path": str(ds.path), "spec_version": ds.manifest.spec_version, "stats": ds.stats(),
                "labels": [l.model_dump(exclude_none=True) for l in ds.vocab.labels],
                "videos": [{"video_id": v, "frames": ds.video(v).frame_count, "width": ds.video(v).display_size[0],
                            "height": ds.video(v).display_size[1], "fps": _fps(ds.video(v))} for v in ds.video_ids()],
                "views": ds.view_ids(), "your_author_name": self.author, "your_pending_changes": d.summary(),
                "rules": "You can read and propose. Proposals are added as origin=model, status=unreviewed, into your own draft; "
                         "a person reviews and commits them. You cannot approve, reject, delete existing rows or commit."}

    def list_annotations(self, video_id: str, *, kind: str | None = None, label: str | None = None, status: str | None = None,
                         origin: str | None = None, track_id: int | None = None, frame_start: int | None = None,
                         frame_end: int | None = None, view_id: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        self._video(video_id)
        if label is not None and label not in {l.name for l in self.ds.vocab.labels}:
            raise AgentError(f"no label '{label}'. Labels: {[l.name for l in self.ds.vocab.labels]}")
        if view_id is not None and view_id not in self.ds.view_ids():
            raise AgentError(f"no view '{view_id}'. Views: {self.ds.view_ids()}")
        if track_id is not None:
            track_id = _int(track_id, "track_id", None)
        limit, offset = _int(limit, "limit", None, -10**9, 10**9), _int(offset, "offset", None, -10**9, 10**12)
        filters: dict[str, Any] = {"view_id": [view_id]}
        for k, v in (("kind", kind), ("label", label), ("status", status), ("origin", origin), ("track_id", track_id)):
            if v is not None:
                filters[k] = v
        if frame_start is not None or frame_end is not None:
            lo = 0 if frame_start is None else _int(frame_start, "frame_start", None, -10**9, 2**53)
            hi = self.ds.video(video_id).frame_count if frame_end is None else _int(frame_end, "frame_end", None, -10**9, 2**53)
            filters["frames"] = (lo, hi)
        try:
            rows = D.Draft.load(self.ds, self.author).rows(video_id, include_deleted=False, **filters)
        except (D.DraftError, ValueError) as e:
            raise AgentError(str(e)) from None
        limit, offset = max(1, min(limit, MAX_LIST)), max(0, offset)
        part = rows[offset:offset + limit]
        return {"video_id": video_id, "total": len(rows), "offset": offset, "returned": len(part), "rows": [_row_out(r) for r in part]}

    def list_tracks(self, video_id: str, view_id: str | None = None) -> dict:
        self._video(video_id)
        if view_id is not None and view_id not in self.ds.view_ids():
            raise AgentError(f"no view '{view_id}'. Views: {self.ds.view_ids()}")
        return {"video_id": video_id, "tracks": self.ds.tracks(video_id, view_id=view_id)}

    def validate(self, deep: bool = False) -> dict:
        from .ui import ops
        return ops.validate(self.ds, deep=bool(deep))

    def frame_image(self, video_id: str, frame: int, *, annotations: bool = True, max_side: int = 768) -> bytes:
        """A JPEG of one frame, with the annotations on it (including your own pending proposals) drawn on top."""
        import cv2
        from .frames import FrameReader
        from .preview import draw_annotations
        meta = self._video(video_id)
        frame = _int(frame, "frame", None, -10**9, 2**53)
        max_side = _int(max_side, "max_side", None, -10**9, 10**9)
        if not 0 <= frame < meta.frame_count:
            raise AgentError(f"video '{video_id}' has frames 0 to {meta.frame_count - 1}; there is no frame {frame}")
        path = self.ds.resolve_video_path(video_id, [])
        if path is None or not Path(path).exists():
            raise AgentError(f"the video file for '{video_id}' was not found on this computer, so no picture can be made")
        reader = FrameReader(path, meta, None, index=self.ds.frame_index(video_id))
        try:
            rgb = reader.read(frame)
        finally:
            reader.close()
        if annotations:
            rows = D.Draft.load(self.ds, self.author).rows(video_id, include_deleted=False, frames=(frame, frame), view_id=[None])
            rgb = draw_annotations(rgb, rows, frame)
        max_side = max(16, min(max_side, 2048))
        if max(rgb.shape[:2]) > max_side:
            s = max_side / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise AgentError("could not encode the picture")
        return buf.tobytes()

    # ------------------------------------------------------------------ proposing
    def propose_boxes(self, video_id: str, boxes: Sequence[dict], note: str | None = None) -> dict:
        """Each item: `frame`, `label`, `box` = [x0, y0, x1, y1] in the video's own pixels, optional `track_id`, `confidence` (0 to 1)."""
        W, H = self._video(video_id).display_size
        rows = []
        for i, b in enumerate(self._items(boxes, "boxes")):
            self._need(b, ("frame", "label", "box"), i)
            self._only(b, ("frame", "label", "box", "track_id", "confidence"), i)
            bx = b["box"]
            if not (isinstance(bx, (list, tuple)) and len(bx) == 4):
                raise AgentError(f"item {i}: box must be [x0, y0, x1, y1]")
            bx = [_number(v, "every box value", i) for v in bx]
            if bx[0] < 0 or bx[1] < 0 or bx[2] > W or bx[3] > H:                       # an agent that guesses coordinates is caught here
                raise AgentError(f"item {i}: box {list(bx)} is outside the frame (0 to {W} wide, 0 to {H} high). Boxes use the video's own pixels")
            r = {"kind": "box", "label": _label(b["label"], i), "start_frame": self._frame(video_id, b["frame"], i), "box": tuple(bx)}
            rows.append(self._with_common(r, b, i))
        return self._record(video_id, rows, note)

    def propose_spans(self, video_id: str, spans: Sequence[dict], note: str | None = None) -> dict:
        """Label stretches of frames. Each item: `start_frame`, `end_frame` (inclusive), `label`, optional `confidence`."""
        rows = []
        for i, s in enumerate(self._items(spans, "spans")):
            self._need(s, ("start_frame", "end_frame", "label"), i)
            self._only(s, ("start_frame", "end_frame", "label", "confidence"), i)
            r = {"kind": "marker", "label": _label(s["label"], i), "start_frame": self._frame(video_id, s["start_frame"], i),
                 "end_frame": self._frame(video_id, s["end_frame"], i)}
            rows.append(self._with_common(r, s, i))
        return self._record(video_id, rows, note)

    def propose_masks(self, video_id: str, masks: Sequence[dict], note: str | None = None) -> dict:
        """Each item: `frame`, `label`, `polygon` = [[x, y], ...] (3 or more corners, the video's own pixels), optional `track_id`, `confidence`."""
        meta = self._video(video_id)
        W, H = meta.display_size
        rows = []
        for i, m in enumerate(self._items(masks, "masks")):
            self._need(m, ("frame", "label", "polygon"), i)
            self._only(m, ("frame", "label", "polygon", "track_id", "confidence"), i)
            pts = self._polygon(m["polygon"], W, H, i)
            r = {"kind": "mask", "label": _label(m["label"], i), "start_frame": self._frame(video_id, m["frame"], i), "mask": _fill_polygon(pts, W, H)}
            rows.append(self._with_common(r, m, i))
        return self._record(video_id, rows, note)

    def retract(self, ids: Sequence[Any]) -> dict:
        """Take back proposals of yours that are still only in your draft (their ids look like `p3`). Rows already in the dataset
        are not yours to remove."""
        d = D.Draft.load(self.ds, self.author)
        want = {str(i) for i in ids}
        mine = {o["tmp"]: j for j, o in enumerate(d.ops) if o["op"] == "add"}
        unknown = sorted(want - set(mine))
        if unknown:
            raise AgentError(f"not pending proposals of yours: {unknown[:5]}. Only ids like 'p3' from your own draft can be retracted")
        removed = 0
        for j in sorted((mine[t] for t in want), reverse=True):
            removed += d.discard(j)
        return {"retracted": removed, "pending": d.summary()}

    def draft_status(self) -> dict:
        d = D.Draft.load(self.ds, self.author)
        st = d.state()
        return {"author": self.author, "summary": st["summary"], "can_undo": st["can_undo"],
                "edits": [{"video_id": o["video_id"], "what": o["label"], **({"note": o["note"]} if "note" in o else {})}
                          for o in st["ops"] if not o["undone"]][:200],
                "next_step": "Ask a person to open Proposals in the vidlabel web interface (`vidlabel ui PATH`), take these into their "
                             "draft, review them and commit. You cannot commit."}

    # ------------------------------------------------------------------ internals
    def _video(self, video_id: str):
        try:
            return self.ds.video(video_id)
        except KeyError:
            raise AgentError(f"no video '{video_id}'. Videos: {self.ds.video_ids()[:20]}") from None

    @staticmethod
    def _items(items: Any, what: str) -> list:
        if not isinstance(items, (list, tuple)) or not items:
            raise AgentError(f"`{what}` must be a non-empty list")
        if len(items) > MAX_ROWS_PER_CALL:
            raise AgentError(f"at most {MAX_ROWS_PER_CALL} {what} per call; send the rest in another call")
        if not all(isinstance(x, dict) for x in items):
            raise AgentError(f"every item of `{what}` must be an object")
        return list(items)

    @staticmethod
    def _need(item: dict, keys: Sequence[str], i: int) -> None:
        missing = [k for k in keys if item.get(k) is None]
        if missing:
            raise AgentError(f"item {i}: missing {missing}")

    @staticmethod
    def _only(item: dict, allowed: Sequence[str], i: int) -> None:
        extra = sorted(set(item) - set(allowed))
        if extra:
            raise AgentError(f"item {i}: not allowed here: {extra}. Allowed: {list(allowed)} (origin, status and author are set for you)")

    def _frame(self, video_id: str, v: Any, i: int) -> int:
        n = self.ds.video(video_id).frame_count
        f = _int(v, "frame", i, 0, 2**53)
        if f >= n:
            raise AgentError(f"item {i}: frame {f} does not exist; video '{video_id}' has frames 0 to {n - 1}")
        return f

    @staticmethod
    def _with_common(r: dict, src: dict, i: int) -> dict:
        if src.get("track_id") is not None:
            r["track_id"] = _int(src["track_id"], "track_id", i, 0, 2**53)
        if src.get("confidence") is not None:
            c = _number(src["confidence"], "confidence", i)
            if not 0 <= c <= 1:
                raise AgentError(f"item {i}: confidence must be between 0 and 1, got {c}")
            r["confidence"] = c
        r["origin"] = "model"                          # fixed: an agent's rows are never "human"
        r["status"] = "unreviewed"                     # fixed: only a person approves or rejects
        return r

    @staticmethod
    def _polygon(poly: Any, W: int, H: int, i: int) -> np.ndarray:
        try:
            pts = np.asarray(poly, dtype=float)
        except (TypeError, ValueError):
            raise AgentError(f"item {i}: polygon must be a list of [x, y] pairs") from None
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise AgentError(f"item {i}: polygon needs at least 3 corners, each [x, y]")
        if (pts[:, 0] < 0).any() or (pts[:, 1] < 0).any() or (pts[:, 0] > W).any() or (pts[:, 1] > H).any():
            raise AgentError(f"item {i}: polygon corners must lie inside the frame (0 to {W} wide, 0 to {H} high)")
        x, y = pts[:, 0], pts[:, 1]
        if abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2 < 1:
            raise AgentError(f"item {i}: the polygon has no area (corners on one line or point, or the outline crosses itself and cancels out)")
        return pts

    def _record(self, video_id: str, rows: list[dict], note: str | None) -> dict:
        """All or nothing: put every row in the draft, or none of them."""
        self._video(video_id)
        if note is not None and not isinstance(note, str):
            raise AgentError("note must be text")
        note = (note.strip()[:300] or None) if note else None
        d = D.Draft.load(self.ds, self.author)
        pending = d.summary()["added"]
        if pending + len(rows) > MAX_PENDING_ROWS:
            raise AgentError(f"your draft already holds {pending} proposals; the limit is {MAX_PENDING_ROWS}. A person needs to take them in, or retract some")
        snap = (list(d.ops), d.cursor, dict(d.base), d.next_tmp)
        ids = []
        try:
            for i, r in enumerate(rows):
                try:
                    ids.append(d.add(video_id, r, note=note))
                except (D.DraftError, ValueError, TypeError) as e:
                    raise AgentError(f"item {i}: {e}. Nothing from this call was recorded.") from None
        except AgentError:
            d.ops, d.cursor, d.base, d.next_tmp = snap
            d.save()
            raise
        return {"recorded": len(ids), "ids": ids, "pending": d.summary(),
                "next_step": "These are in your draft only. A person reviews them under Proposals in the web interface."}


def _fps(meta) -> float | None:
    f = getattr(meta, "fps", None)
    if f is None:
        return None
    try:
        return round(f.num / f.den, 4) if f.den else None
    except AttributeError:
        return None


def _fill_polygon(pts: np.ndarray, W: int, H: int) -> np.ndarray:
    import cv2
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if not m.any():
        raise AgentError("the polygon covers no pixels (is it too small, or a line?)")
    return m.astype(bool)
