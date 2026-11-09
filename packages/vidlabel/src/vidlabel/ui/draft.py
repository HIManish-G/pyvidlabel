"""Pending edits (a *draft*), committing them, and the history of commits, for the web interface.

No web code lives here, so all of it is tested without a browser. The model is git's:

- Edits go into a **draft**, a list of operations kept on disk (`.vidlabel-ui/drafts/<author>.json` inside the dataset folder; the
  spec ignores that folder). Nothing in the dataset changes until **commit**.
- `undo`/`redo` move a cursor through that list. `rows()` shows a video's annotations *with the draft applied*, each row marked
  `_pending` = added / modified / deleted, so a screen can draw the result before it is committed.
- **commit** checks that the videos have not changed since the draft began (otherwise `Conflict`), tries the whole draft on a copy,
  applies it with the library (`delete_annotations`, `update_annotations`, `add_annotations`), restores every video it touched if
  anything fails, and records the commit in `.vidlabel-ui/history/`.
- **revert** builds the opposite changes of a past commit and commits them as a new commit.

Annotation ids are 63-bit integers. Rows added in a draft have temporary ids `"p1"`, `"p2"`, ... until they are committed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from vidlabel_spec import COLUMN_NAMES, Kind, Status

from .. import masks as _masks
from .._arrow import ARROW_SCHEMA, empty_table
from ..dataset import _IMMUTABLE, Dataset, _enum_value, _origin_value, _rows_to_table
from ..validate import ValidationError, check_annotation_table, errors

STATE_DIR = ".vidlabel-ui"
HISTORY_LIMIT = 50
_UPDATABLE = (set(COLUMN_NAMES) - _IMMUTABLE) | {"box", "mask"}
_FILTERS = {"kind", "label", "origin", "status", "view_id", "track_id", "frames"}


class DraftError(ValueError):
    """The edit cannot be recorded or the draft cannot be applied (the message says why)."""


class Conflict(DraftError):
    """Videos changed on disk after the draft was started; `videos` lists them."""
    def __init__(self, videos: Sequence[str]):
        self.videos = list(videos)
        super().__init__("these videos changed since your first edit (someone used the command line or committed): "
                         + ", ".join(self.videos))


# ---------------------------------------------------------------- helpers
def _safe_name(author: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", (author or "").strip())[:64].strip(".") or "anonymous"


def is_temp_id(i: Any) -> bool:
    return isinstance(i, str) and bool(re.fullmatch(r"p\d+", i))


def _fingerprint(ds: Dataset, video_id: str) -> str:
    p = ds._ann_path(video_id)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def _json_row(row: Mapping[str, Any]) -> dict:
    """A row as JSON-safe values (a mask as base64 of its encoded bytes)."""
    out = {}
    for k, v in row.items():
        if k == "mask" and v is not None and not isinstance(v, (bytes, bytearray)):
            v = _masks.encode(v)
        if isinstance(v, (bytes, bytearray)):
            out["mask_b64"] = base64.b64encode(bytes(v)).decode("ascii")
        elif k == "box":
            out["x0"], out["y0"], out["x1"], out["y1"] = [float(x) for x in v]
        elif k == "mask_codec" and row.get("mask") is None and "mask_b64" not in row:
            continue
        else:
            out[k] = v
    return out


def _library_row(row: Mapping[str, Any]) -> dict:
    """The inverse of `_json_row`: what `Dataset.add_annotations` takes."""
    r = {k: v for k, v in row.items() if k not in ("mask_b64", "_pending")}
    if row.get("mask_b64"):
        r["mask"] = base64.b64decode(row["mask_b64"])
        r.setdefault("mask_codec", _masks.CODEC)
    return r


def _library_values(values: Mapping[str, Any]) -> dict:
    """Update values as `Dataset.update_annotations` takes them (a mask as bytes, not base64)."""
    v = {k: x for k, x in values.items() if k != "mask_b64"}
    if values.get("mask_b64"):
        v["mask"] = base64.b64decode(values["mask_b64"])
        for k in ("mask_codec", "x0", "y0", "x1", "y1"):          # the library sets these from the new mask
            v.pop(k, None)
    return v


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write(path, json.dumps(obj, indent=1).encode("utf-8"))


def _clean_values(values: Mapping[str, Any]) -> dict:
    """Check and normalise values for an update; geometry given as `box` becomes x0..y1."""
    if not values:
        raise DraftError("give at least one value to change, e.g. status='approved'")
    bad = set(values) - _UPDATABLE
    if bad:
        raise DraftError(f"cannot change {sorted(bad)}")
    out: dict[str, Any] = {}
    try:
        for k, v in values.items():
            if k == "box":
                if len(v) != 4:
                    raise DraftError("box must be (x0, y0, x1, y1)")
                out.update(zip(("x0", "y0", "x1", "y1"), (float(x) for x in v)))
            elif k == "mask":
                if v is None:
                    raise DraftError("a mask cannot be cleared; delete the row instead")
                blob = bytes(v) if isinstance(v, (bytes, bytearray)) else _masks.encode(v)
                box = _masks.bbox(blob)
                if box is None:
                    raise DraftError("the mask is empty; delete the row instead")
                out["mask_b64"] = base64.b64encode(blob).decode("ascii")
                out["mask_codec"] = _masks.CODEC
                out.update(zip(("x0", "y0", "x1", "y1"), (float(x) for x in box)))     # the box follows the mask
            elif k == "status":
                out[k] = _enum_value(v, Status, k)
            elif k == "kind":
                out[k] = _enum_value(v, Kind, k)
            elif k == "origin":
                out[k] = _origin_value(v)
            elif k == "extra" and v is not None and not isinstance(v, str):
                out[k] = json.dumps(v, sort_keys=True)
            else:
                out[k] = v
    except ValueError as e:
        raise DraftError(str(e)) from None
    return out


def _matches(row: Mapping[str, Any], f: Mapping[str, Any]) -> bool:
    for name in ("kind", "label", "origin", "status"):
        want = f.get(name)
        if want is not None and row.get(name) not in ([want] if isinstance(want, str) else list(want)):
            return False
    if "view_id" in f and f["view_id"] is not None:
        want = [f["view_id"]] if isinstance(f["view_id"], str) else list(f["view_id"])
        if row.get("view_id") not in want:
            return False
    if f.get("track_id") is not None:
        want = [f["track_id"]] if isinstance(f["track_id"], int) else list(f["track_id"])
        if row.get("track_id") not in want:
            return False
    if f.get("frames") is not None:
        a, b = f["frames"]
        if row["end_frame"] < a or row["start_frame"] > b:
            return False
    return True


class _VideoState:
    """What a draft does to one video: new rows by temporary id, changed values by id, removed ids."""
    def __init__(self) -> None:
        self.adds: dict[str, dict] = {}
        self.upd: dict[int, dict] = {}
        self.deleted: set[int] = set()

    def apply(self, op: Mapping[str, Any]) -> None:
        kind = op["op"]
        if kind == "add":
            self.adds[op["tmp"]] = dict(op["row"])
        elif kind == "update":
            for i in op["ids"]:
                if is_temp_id(i):
                    if i in self.adds:
                        self.adds[i].update(op["values"])
                else:
                    self.upd.setdefault(int(i), {}).update(op["values"])
        elif kind == "delete":
            for i in op["ids"]:
                if is_temp_id(i):
                    self.adds.pop(i, None)
                else:
                    self.deleted.add(int(i))
                    self.upd.pop(int(i), None)


# ---------------------------------------------------------------- the draft
class Draft:
    """One author's pending edits to a dataset. Create with `Draft.load(ds, author)`; every method that records an edit saves the draft."""

    def __init__(self, ds: Dataset, author: str):
        self.ds = ds
        self.author = (author or "").strip() or "anonymous"
        self.path = ds.path / STATE_DIR / "drafts" / f"{_safe_name(self.author)}.json"
        self.ops: list[dict] = []
        self.cursor = 0
        self.base: dict[str, str] = {}
        self.next_tmp = 1

    @classmethod
    def load(cls, ds: Dataset, author: str) -> "Draft":
        d = cls(ds, author)
        if d.path.exists():
            s = json.loads(d.path.read_text(encoding="utf-8"))
            d.ops, d.cursor, d.base, d.next_tmp = s["ops"], s["cursor"], s["base"], s["next_tmp"]
        return d

    def save(self) -> None:
        if not self.ops and self.path.exists():
            self.path.unlink()
            return
        if self.ops:
            _write_json(self.path, {"author": self.author, "ops": self.ops, "cursor": self.cursor, "base": self.base,
                                    "next_tmp": self.next_tmp})

    # ---- recording ----
    def _record(self, op: dict) -> dict:
        vid = op["video_id"]
        self.ds.video(vid)                                           # KeyError for an unknown video
        del self.ops[self.cursor:]                                   # a new edit ends the redo history
        if vid not in self.base:
            self.base[vid] = _fingerprint(self.ds, vid)
        self.ops.append(op)
        self.cursor = len(self.ops)
        self.save()
        return op

    def _state(self, video_id: str) -> _VideoState:
        st = _VideoState()
        for op in self.ops[:self.cursor]:
            if op["video_id"] == video_id:
                st.apply(op)
        return st

    def _current_ids(self, video_id: str) -> set:
        """Ids that exist now in this video as the draft leaves it (committed rows not deleted, plus pending rows)."""
        st = self._state(video_id)
        base = set(self.ds.annotations(video_id, columns=["annotation_id"]).column("annotation_id").to_pylist())
        return (base - st.deleted) | set(st.adds)

    def add(self, video_id: str, row: Mapping[str, Any], *, coords: str = "source", note: str | None = None) -> str:
        """Record a new annotation; returns its temporary id. The row is checked now against the video, labels and views.
        `note` is a short free-text reason kept with the edit (shown to whoever reviews it); it is not stored in the dataset."""
        r = _json_row(row)
        if coords == "view":
            raise DraftError("draw in full-frame coordinates (the interface converts view pixels before recording)")
        r.setdefault("origin", "human")
        if "author" not in r:
            r["author"] = self.author
        lib = _library_row(r)
        try:
            tbl = _rows_to_table([lib], video_id, set())
            meta = self.ds.video(video_id)
            bad = errors(check_annotation_table(tbl, frame_count=meta.frame_count or None, size=meta.display_size,
                                                vocab_names={l.name for l in self.ds.vocab.labels}, views=self.ds._views_map(),
                                                video_id=video_id))
        except ValueError as e:
            raise DraftError(str(e)) from None
        if bad:
            raise DraftError("; ".join(i.message for i in bad))
        if r["kind"] != "marker" and r.get("track_id") is not None:
            f = r["start_frame"]
            for o in self.rows(video_id, include_deleted=False, frames=(f, f)):
                if (o["kind"] != "marker" and o.get("track_id") == r["track_id"] and o["start_frame"] == f and (o["kind"] == "mask") == (r["kind"] == "mask")
                        and o.get("view_id") == r.get("view_id")):
                    raise DraftError(f"track {r['track_id']} already has a {o['kind']} on frame {f}. A track can have only one per frame: "
                                     f"choose another track, or edit the one that is there.")
        row_json = {k: v for k, v in r.items() if k != "box"}
        stored = tbl.to_pylist()[0]
        for k in ("x0", "y0", "x1", "y1"):
            if stored[k] is not None:
                row_json[k] = stored[k]
        tmp = f"p{self.next_tmp}"
        self.next_tmp += 1
        op = {"op": "add", "video_id": video_id, "tmp": tmp, "row": row_json, "label": f"add {r['kind']} '{r['label']}' at frame {r['start_frame']}"}
        if note:
            op["note"] = str(note)[:300]
        self._record(op)
        return tmp

    def _check_ids(self, video_id: str, ids: Iterable[Any]) -> list:
        ids = [i if is_temp_id(i) else int(i) for i in ids]
        have = self._current_ids(video_id)
        missing = [i for i in ids if i not in have]
        if missing:
            raise DraftError(f"no such annotation in video '{video_id}': {missing[:5]}")
        return ids

    def update(self, video_id: str, ids: Iterable[Any], **values: Any) -> dict:
        """Record a change of columns (status, label, box, ...) on committed or pending rows."""
        vals = _clean_values(values)
        if "label" in vals and vals["label"] not in {l.name for l in self.ds.vocab.labels}:
            raise DraftError(f"label '{vals['label']}' is not in the vocabulary")
        ids = self._check_ids(video_id, ids)
        if "track_id" in vals:
            self._check_track_move(video_id, {i: vals["track_id"] for i in ids})
        what = ", ".join(f"{k} → {v}" for k, v in vals.items() if k not in ("x0", "y0", "x1", "y1", "mask_codec", "mask_b64")) or "geometry"
        if "mask_b64" in vals:
            what = "mask shape" + (", " + what if what != "geometry" else "")
        return self._record({"op": "update", "video_id": video_id, "ids": ids, "values": vals,
                             "label": f"change {what} ({len(ids)} row{'s' if len(ids) != 1 else ''})"})

    def _check_track_move(self, video_id: str, moved: dict) -> None:
        """Refuse a change of track number that would leave one track with two boxes (or two masks) on a frame. The commit would refuse
        it too (`duplicate_track_in_frame`); saying so here keeps the draft always committable."""
        if all(t is None for t in moved.values()):
            return
        rows = [r for r in self.rows(video_id, include_deleted=False) if r["kind"] != "marker"]
        key = lambda r, t: (r.get("view_id"), t, r["start_frame"], r["kind"] == "mask")
        count: dict = {}
        for r in rows:
            t = moved.get(r["annotation_id"], r.get("track_id"))
            if t is not None:
                count[key(r, t)] = count.get(key(r, t), 0) + 1
        for r in rows:
            if r["annotation_id"] in moved and moved[r["annotation_id"]] is not None:
                k = key(r, moved[r["annotation_id"]])
                if count[k] > 1:
                    raise DraftError(f"track {k[1]} would have two {'masks' if k[3] else 'boxes'} on frame {k[2]}. A track can have only one per frame.")

    def delete(self, video_id: str, ids: Iterable[Any]) -> dict:
        ids = self._check_ids(video_id, ids)
        return self._record({"op": "delete", "video_id": video_id, "ids": ids,
                             "label": f"delete {len(ids)} row{'s' if len(ids) != 1 else ''}"})

    def _track_ids(self, video_id: str, track_id: int, view_id: str | None) -> list:
        rows = self.rows(video_id, track_id=int(track_id), view_id=[view_id], include_deleted=False)
        if not rows:
            raise DraftError(f"video '{video_id}' has no rows for track {track_id}" + (f" in view '{view_id}'" if view_id else ""))
        return [r["annotation_id"] for r in rows]

    def update_track(self, video_id: str, track_id: int, *, view_id: str | None = None, **values: Any) -> dict:
        vals = _clean_values(values)
        ids = self._track_ids(video_id, track_id, view_id)
        what = ", ".join(f"{k} → {v}" for k, v in vals.items())
        return self._record({"op": "update", "video_id": video_id, "ids": ids, "values": vals,
                             "label": f"track {track_id}: {what} ({len(ids)} rows)"})

    def delete_track(self, video_id: str, track_id: int, *, view_id: str | None = None) -> dict:
        ids = self._track_ids(video_id, track_id, view_id)
        return self._record({"op": "delete", "video_id": video_id, "ids": ids, "label": f"delete track {track_id} ({len(ids)} rows)"})

    # ---- undo, redo, discard ----
    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.ops)

    def undo(self) -> None:
        if not self.can_undo:
            raise DraftError("nothing to undo")
        self.cursor -= 1
        self.save()

    def redo(self) -> None:
        if not self.can_redo:
            raise DraftError("nothing to redo")
        self.cursor += 1
        self.save()

    def clear(self) -> None:
        self.ops, self.cursor, self.base = [], 0, {}
        self.save()

    def discard(self, index: int) -> int:
        """Drop one edit, and the later edits that used rows it created. Returns how many edits were removed."""
        if not 0 <= index < len(self.ops):
            raise DraftError(f"no edit number {index}")
        gone = {index}
        tmps = {self.ops[index]["tmp"]} if self.ops[index]["op"] == "add" else set()
        for j in range(index + 1, len(self.ops)):
            if tmps & {i for i in self.ops[j].get("ids", []) if is_temp_id(i)}:
                gone.add(j)
                if self.ops[j]["op"] == "add":
                    tmps.add(self.ops[j]["tmp"])
        self.cursor -= sum(1 for j in gone if j < self.cursor)
        self.ops = [o for j, o in enumerate(self.ops) if j not in gone]
        if not self.ops:
            self.base = {}
        self.save()
        return len(gone)

    # ---- looking at the result ----
    def rows(self, video_id: str, *, include_deleted: bool = True, **filters: Any) -> list[dict]:
        """The video's annotations with the draft applied, as dicts. Each has `_pending` set to added, modified or deleted when the
        draft affects it. Accepts the filters of `Dataset.annotations` (`kind`, `label`, `origin`, `status`, `view_id`, `track_id`,
        `frames`). Masks are left as bytes."""
        unknown = set(filters) - _FILTERS
        if unknown:
            raise DraftError(f"unknown filter {sorted(unknown)}")
        st = self._state(video_id)
        # a change can move a row into or out of a filter, so with changes pending only the frame range is used while reading
        read = {"frames": filters["frames"]} if st.upd and filters.get("frames") else ({} if st.upd else dict(filters))
        out: list[dict] = []
        for r in self.ds.annotations(video_id, **read).to_pylist():
            i = r["annotation_id"]
            if i in st.deleted:
                if include_deleted:
                    r["_pending"] = "deleted"
                    out.append(r)
                continue
            if i in st.upd:
                r.update(st.upd[i])
                if "mask_b64" in r:
                    r["mask"] = base64.b64decode(r.pop("mask_b64"))
                r["_pending"] = "modified"
            out.append(r)
        for tmp, raw in st.adds.items():
            r = {c: None for c in COLUMN_NAMES}
            r.update({k: v for k, v in raw.items() if k != "mask_b64"})
            r.update(annotation_id=tmp, video_id=video_id, _pending="added")
            if r.get("end_frame") is None:
                r["end_frame"] = r["start_frame"]
            if r.get("status") is None:
                r["status"] = "unreviewed"
            if raw.get("mask_b64"):
                r["mask"] = base64.b64decode(raw["mask_b64"])
            out.append(r)
        if st.upd or st.adds or filters:
            out = [r for r in out if _matches(r, filters)]
        out.sort(key=lambda r: (r["start_frame"], str(r["annotation_id"])))
        return out

    def summary(self) -> dict:
        """Counts for a Changes panel: rows added, rows modified, rows deleted, and the videos touched."""
        added = modified = deleted = 0
        videos = []
        for vid in sorted({o["video_id"] for o in self.ops[:self.cursor]}):
            st = self._state(vid)
            added += len(st.adds)
            modified += len(st.upd)
            deleted += len(st.deleted)
            videos.append(vid)
        return {"added": added, "modified": modified, "deleted": deleted, "videos": videos}

    def state(self) -> dict:
        return {"author": self.author, "can_undo": self.can_undo, "can_redo": self.can_redo, "cursor": self.cursor,
                "ops": [{"index": i, "label": o["label"], "video_id": o["video_id"], "kind": o["op"], "undone": i >= self.cursor,
                         **({"note": o["note"]} if o.get("note") else {})}
                        for i, o in enumerate(self.ops)],
                "summary": self.summary()}

    # ---- turning the draft into library calls ----
    def net(self) -> dict:
        """What committing would do, per video: `adds` (rows for `add_annotations`), `updates` (`(ids, values)` groups) and `deletes`."""
        out: dict[str, dict] = {}
        for vid in sorted({o["video_id"] for o in self.ops[:self.cursor]}):
            st = self._state(vid)
            groups: dict[str, tuple[list[int], dict]] = {}
            for i, v in sorted(st.upd.items()):
                groups.setdefault(json.dumps(v, sort_keys=True), ([], _library_values(v)))[0].append(i)
            net = {"adds": [_library_row(r) for r in st.adds.values()], "updates": list(groups.values()),
                   "deletes": sorted(st.deleted)}
            if net["adds"] or net["updates"] or net["deletes"]:
                out[vid] = net
        return out

    def conflicts(self) -> list[str]:
        return [v for v in self.net() if v in self.base and _fingerprint(self.ds, v) != self.base[v]]

    def check(self, *, reapply: bool = False) -> dict:
        """Try the draft on a copy of the affected videos. Raises `Conflict`, `ValidationError` or `DraftError`; changes nothing."""
        net, skipped = self._net_for_commit(reapply)
        with tempfile.TemporaryDirectory() as tmp:
            scratch = _scratch_copy(self.ds, Path(tmp), net)
            _apply_net(scratch, net)
        return {"ok": True, "skipped": skipped, **_count(net)}

    def _net_for_commit(self, reapply: bool) -> tuple[dict, list[str]]:
        net = self.net()
        if not net:
            raise DraftError("nothing to commit")
        bad = self.conflicts()
        if bad and not reapply:
            raise Conflict(bad)
        return (_drop_missing(self.ds, net) if bad else (net, []))

    def commit(self, message: str, *, reapply: bool = False) -> dict:
        """Apply the draft. All or nothing: if anything fails, every touched video is put back and the draft is kept."""
        message = (message or "").strip()
        if not message:
            raise DraftError("write a commit message")
        net, skipped = self._net_for_commit(reapply)
        with tempfile.TemporaryDirectory() as tmp:               # find problems before touching the dataset
            _apply_net(_scratch_copy(self.ds, Path(tmp), net), dict(net))
        rec = commit_net(self.ds, net, message=message, author=self.author, skipped=skipped)
        self.clear()
        return rec


# ---------------------------------------------------------------- applying changes
def _count(net: Mapping[str, Mapping]) -> dict:
    return {"added": sum(len(n["adds"]) for n in net.values()),
            "modified": sum(len(ids) for n in net.values() for ids, _ in n["updates"]),
            "deleted": sum(len(n["deletes"]) for n in net.values()), "videos": sorted(net)}


def _scratch_copy(ds: Dataset, root: Path, net: Mapping[str, Any]) -> Dataset:
    for name in ("manifest.json", "vocab.json"):
        shutil.copy2(ds.path / name, root / name)
    if (ds.path / "views").exists():
        shutil.copytree(ds.path / "views", root / "views")
    for vid in net:
        shutil.copytree(ds._vdir(vid), root / "videos" / vid, ignore=shutil.ignore_patterns("frame_index.json"))
    return Dataset.open(root)


def _apply_net(ds: Dataset, net: Mapping[str, Mapping]) -> dict[str, list[int]]:
    """Deletes first, then changes, then additions (so a track id freed by a delete can be reused). Returns the ids that were added."""
    added: dict[str, list[int]] = {}
    for vid, n in net.items():
        if n["deletes"]:
            ds.delete_annotations(vid, n["deletes"])
        for ids, values in n["updates"]:
            ds.update_annotations(vid, ids, **values)
        added[vid] = ds.add_annotations(n["adds"], vid) if n["adds"] else []
    return added


def _drop_missing(ds: Dataset, net: Mapping[str, Mapping]) -> tuple[dict, list[str]]:
    """Remove changes that refer to rows that are no longer there (and additions whose id is taken); say what was skipped."""
    out: dict[str, dict] = {}
    skipped: list[str] = []
    for vid, n in net.items():
        have = set(ds._existing(vid).column("annotation_id").to_pylist())
        dels = [i for i in n["deletes"] if i in have]
        skipped += [f"{vid}: row {i} is already gone (delete skipped)" for i in n["deletes"] if i not in have]
        ups = []
        for ids, values in n["updates"]:
            keep = [i for i in ids if i in have]
            skipped += [f"{vid}: row {i} is already gone (change skipped)" for i in ids if i not in have]
            if keep:
                ups.append((keep, values))
        adds = [r for r in n["adds"] if r.get("annotation_id") not in have]
        skipped += [f"{vid}: row {r['annotation_id']} already exists (not added again)" for r in n["adds"] if r.get("annotation_id") in have]
        if dels or ups or adds:
            out[vid] = {"adds": adds, "updates": ups, "deletes": dels}
    return out, skipped


def commit_net(ds: Dataset, net: Mapping[str, Mapping], *, message: str, author: str, skipped: Sequence[str] = (),
               kind: str = "commit", reverts: int | None = None) -> dict:
    """Apply `net` to the dataset all-or-nothing and record it in the history. Used by `Draft.commit` and `revert`."""
    if not net:
        raise DraftError("nothing to commit")
    backups: dict[str, bytes | None] = {}
    for vid in net:
        p = ds._ann_path(vid)
        backups[vid] = p.read_bytes() if p.exists() else None
    before = _before_image(ds, net)
    try:
        added = _apply_net(ds, net)
    except BaseException:
        for vid, data in backups.items():
            p = ds._ann_path(vid)
            if data is None:
                p.unlink(missing_ok=True)
            else:
                _atomic_write(p, data)
        raise
    seq = _next_seq(ds)
    counts = _count(net)
    rec = {"seq": seq, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "author": author, "message": message, "kind": kind,
           "reverts": reverts, "summary": counts, "skipped": list(skipped),
           "videos": {vid: {"added_ids": added[vid], "deleted_ids": n["deletes"],
                            "updates": [{"ids": ids, "fields": sorted(values)} for ids, values in n["updates"]]}
                      for vid, n in net.items()}}
    hist = ds.path / STATE_DIR / "history"
    hist.mkdir(parents=True, exist_ok=True)
    if before.num_rows:
        pq.write_table(before, hist / f"{seq:06d}.before.parquet")
    _write_json(hist / f"{seq:06d}.json", rec)
    _prune(hist)
    return rec


def _before_image(ds: Dataset, net: Mapping[str, Mapping]) -> pa.Table:
    parts = []
    for vid, n in net.items():
        ids = set(n["deletes"]) | {i for g, _ in n["updates"] for i in g}
        if not ids:
            continue
        t = ds._existing(vid)
        mask = pc.is_in(t.column("annotation_id"), value_set=pa.array(sorted(ids), type=pa.int64()))
        parts.append(t.filter(mask))
    return pa.concat_tables(parts) if parts else empty_table()


def _next_seq(ds: Dataset) -> int:
    hist = ds.path / STATE_DIR / "history"
    nums = [int(p.stem) for p in hist.glob("??????.json")] if hist.exists() else []
    return max(nums, default=0) + 1


def _prune(hist: Path, keep: int = HISTORY_LIMIT) -> None:
    files = sorted(hist.glob("??????.json"))
    for f in files[:-keep]:
        f.unlink(missing_ok=True)
        (hist / f"{f.stem}.before.parquet").unlink(missing_ok=True)


# ---------------------------------------------------------------- history and revert
# ---------------------------------------------------------------- an agent's proposals (drafts of authors named `agent:...`)
def _other_drafts(ds: Dataset, me: str) -> list["Draft"]:
    folder = ds.path / STATE_DIR / "drafts"
    out = []
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            author = json.loads(f.read_text(encoding="utf-8")).get("author") or f.stem
        except (OSError, ValueError):
            continue
        if not author.startswith("agent:") or _safe_name(author) == _safe_name(me):   # people's drafts are theirs alone
            continue
        d = Draft.load(ds, author)
        if d.ops[:d.cursor]:
            out.append(d)
    return out


def proposals(ds: Dataset, me: str, *, preview: int = 200) -> list[dict]:
    """What agents (authors named `agent:...`) have proposed, to be taken into your draft. A person's draft is never touched by someone
    else. Only added rows can be taken over."""
    res = []
    for d in _other_drafts(ds, me):
        active = d.ops[:d.cursor]
        adds = [o for o in active if o["op"] == "add"]
        rows = [{"video_id": o["video_id"], "kind": o["row"]["kind"], "label": o["row"]["label"], "start_frame": o["row"]["start_frame"],
                 "end_frame": o["row"].get("end_frame", o["row"]["start_frame"]), "track_id": o["row"].get("track_id"),
                 "confidence": o["row"].get("confidence"), "origin": o["row"].get("origin"), "note": o.get("note")} for o in adds[:preview]]
        by_video: dict[str, int] = {}
        for o in adds:
            by_video[o["video_id"]] = by_video.get(o["video_id"], 0) + 1
        res.append({"author": d.author, "is_agent": d.author.startswith("agent:"), "adds": len(adds), "other_edits": len(active) - len(adds),
                    "by_video": by_video, "rows": rows, "updated": d.path.stat().st_mtime if d.path.exists() else None})
    return res


def _row_to_add(row: Mapping[str, Any]) -> dict:
    """A stored draft row in the form `Draft.add` takes."""
    r = {k: v for k, v in row.items() if k not in ("mask_b64", "x0", "y0", "x1", "y1")}
    if row.get("mask_b64"):
        r["mask"] = base64.b64decode(row["mask_b64"])           # the library works the box out from the mask
    elif row.get("x0") is not None and row.get("kind") != "marker":
        r["box"] = [row["x0"], row["y0"], row["x1"], row["y1"]]
    return r


def _remove_ops(d: "Draft", indexes: set[int]) -> None:
    d.cursor -= sum(1 for j in indexes if j < d.cursor)
    d.ops = [o for j, o in enumerate(d.ops) if j not in indexes]
    if not d.ops:
        d.base = {}
    d.save()


def adopt(ds: Dataset, me: str, author: str, video_id: str | None = None) -> dict:
    """Take an agent's added rows into your own draft (optionally only one video's). Each is checked again against the dataset as
    your draft leaves it, exactly like a row you drew, and the whole take-over is all or nothing. Taken rows leave the other draft.
    You still review them (they show as pending) and commit; nothing reaches the dataset here."""
    other = next((d for d in _other_drafts(ds, me) if d.author == author), None)
    if other is None:
        raise DraftError(f"{author} has nothing waiting")
    picks = [(j, o) for j, o in enumerate(other.ops[:other.cursor]) if o["op"] == "add" and (video_id is None or o["video_id"] == video_id)]
    if not picks:
        raise DraftError(f"{author} has no added rows" + (f" in video '{video_id}'" if video_id else "") + " to take in")
    mine = Draft.load(ds, me)
    snap = (list(mine.ops), mine.cursor, dict(mine.base), mine.next_tmp)
    try:
        for n, (j, o) in enumerate(picks, 1):
            try:
                mine.add(o["video_id"], _row_to_add(o["row"]), note=o.get("note"))
            except (DraftError, ValueError, KeyError) as e:
                raise DraftError(f"row {n} of {len(picks)} ({o['label']}) cannot be taken in: {e} Nothing was taken in.") from None
    except DraftError:
        mine.ops, mine.cursor, mine.base, mine.next_tmp = snap
        mine.save()
        raise
    _remove_ops(other, {j for j, _ in picks})
    return {"taken": len(picks), "left_with_author": len([o for o in other.ops[:other.cursor]])}


def discard_proposals(ds: Dataset, me: str, author: str, video_id: str | None = None) -> dict:
    """Throw away an agent's pending proposals (optionally one video's). Nothing in the dataset changes."""
    other = next((d for d in _other_drafts(ds, me) if d.author == author), None)
    if other is None:
        raise DraftError(f"{author} has nothing waiting")
    gone = {j for j, o in enumerate(other.ops) if video_id is None or o["video_id"] == video_id}
    if not gone:
        raise DraftError(f"{author} has nothing in video '{video_id}'")
    _remove_ops(other, gone)
    return {"discarded": len(gone)}



def history(ds: Dataset, limit: int = HISTORY_LIMIT) -> list[dict]:
    """Past commits, newest first (at most the last 50 are kept)."""
    hist = ds.path / STATE_DIR / "history"
    files = sorted(hist.glob("??????.json"), reverse=True)[:limit] if hist.exists() else []
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def get_commit(ds: Dataset, seq: int) -> dict:
    p = ds.path / STATE_DIR / "history" / f"{int(seq):06d}.json"
    if not p.exists():
        raise DraftError(f"no commit number {seq} (only the last {HISTORY_LIMIT} are kept)")
    return json.loads(p.read_text(encoding="utf-8"))


def revert(ds: Dataset, seq: int, author: str = "anonymous") -> dict:
    """Undo a past commit by committing its opposite: rows it added are deleted, rows it deleted are put back with their ids, and
    values it changed are set back. Rows that were changed again since are left alone and listed in `skipped`."""
    rec = get_commit(ds, seq)
    bp = ds.path / STATE_DIR / "history" / f"{int(seq):06d}.before.parquet"
    before = {r["annotation_id"]: r for r in pq.read_table(bp).to_pylist()} if bp.exists() else {}
    net: dict[str, dict] = {}
    skipped: list[str] = []
    for vid, ch in rec["videos"].items():
        have = set(ds._existing(vid).column("annotation_id").to_pylist())
        deletes = [i for i in ch["added_ids"] if i in have]
        skipped += [f"{vid}: row {i} was already removed" for i in ch["added_ids"] if i not in have]
        adds = []
        for i in ch["deleted_ids"]:
            if i in have:
                skipped += [f"{vid}: row {i} exists again (not restored)"]
            else:
                adds.append({k: v for k, v in before[i].items()})
        groups: dict[str, tuple[list[int], dict]] = {}
        for u in ch["updates"]:
            for i in u["ids"]:
                if i not in have:
                    skipped.append(f"{vid}: row {i} is gone (value not restored)")
                    continue
                old = {f: before[i][f] for f in u["fields"]}
                groups.setdefault(json.dumps(old, sort_keys=True, default=str), ([], old))[0].append(i)
        if deletes or adds or groups:
            net[vid] = {"adds": adds, "updates": list(groups.values()), "deletes": deletes}
    if not net:
        raise DraftError("nothing left to revert: every row of this commit has been changed or removed since")
    with tempfile.TemporaryDirectory() as tmp:
        _apply_net(_scratch_copy(ds, Path(tmp), net), net)
    return commit_net(ds, net, message=f"Revert #{seq}: {rec['message']}", author=author, skipped=skipped, kind="revert", reverts=seq)
