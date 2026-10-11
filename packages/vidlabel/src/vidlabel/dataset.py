"""A vidlabel dataset on disk: manifest.json, vocab.json, videos/<id>/{metadata.json, annotations.parquet}, views/."""
from __future__ import annotations

import json
import os
import secrets
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from vidlabel_spec import (COLUMN_NAMES, SPEC_VERSION, Kind, Label, Manifest, Origin, Status, VideoMetadata, View, Vocab)

from . import masks as _masks
from ._arrow import ARROW_SCHEMA, empty_table, write_parquet
from .validate import Issue, ValidationError, check_annotation_table, errors

_ROW_FIELDS = set(COLUMN_NAMES) | {"box"}
_IMMUTABLE = {"annotation_id", "video_id", "mask", "mask_codec"}


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _enum_value(v, enum_cls, field: str) -> str:
    try:
        return enum_cls(v.value if hasattr(v, "value") else v).value
    except ValueError:
        raise ValueError(f"{field}={v!r}; must be one of {[e.value for e in enum_cls]}") from None


class Dataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "manifest.json").exists():
            raise FileNotFoundError(f"{self.path} is not a vidlabel dataset (no manifest.json). Use Dataset.create().")

    # ---- construction ----
    @classmethod
    def create(cls, path: str | Path, exist_ok: bool = False) -> "Dataset":
        p = Path(path)
        if (p / "manifest.json").exists():
            if not exist_ok:
                raise FileExistsError(f"{p} already contains a dataset")
            return cls(p)
        p.mkdir(parents=True, exist_ok=True)
        (p / "videos").mkdir(exist_ok=True)
        (p / "views").mkdir(exist_ok=True)
        _write_json(p / "vocab.json", Vocab().model_dump_json(indent=2))
        _write_json(p / "manifest.json", Manifest().model_dump_json(indent=2))
        return cls(p)

    @classmethod
    def open(cls, path: str | Path) -> "Dataset":
        return cls(path)

    def __repr__(self) -> str:
        return f"Dataset({str(self.path)!r}, videos={len(self.manifest.videos)})"

    # ---- manifest / vocab ----
    @property
    def manifest(self) -> Manifest:
        return Manifest.model_validate_json((self.path / "manifest.json").read_text())

    def _save_manifest(self, m: Manifest) -> None:
        _write_json(self.path / "manifest.json", m.model_dump_json(indent=2))

    @property
    def vocab(self) -> Vocab:
        return Vocab.model_validate_json((self.path / "vocab.json").read_text())

    def add_labels(self, labels: Iterable[str | Label | Mapping]) -> list[str]:
        """Add labels to the shared vocabulary (existing names are left alone). Returns the names actually added."""
        vocab = self.vocab
        have = {l.name for l in vocab.labels}
        added = []
        for l in labels:
            lab = Label(name=l) if isinstance(l, str) else (l if isinstance(l, Label) else Label(**l))
            if lab.name not in have:
                vocab.labels.append(lab)
                have.add(lab.name)
                added.append(lab.name)
        if added:
            vocab = Vocab.model_validate(vocab.model_dump())  # re-run checks (parents etc.)
            _write_json(self.path / "vocab.json", vocab.model_dump_json(indent=2))
        return added

    # ---- videos ----
    def video_ids(self) -> list[str]:
        return list(self.manifest.videos)

    def _vdir(self, video_id: str) -> Path:
        return self.path / "videos" / video_id

    def add_video(self, meta: VideoMetadata | Mapping[str, Any], exist_ok: bool = False) -> VideoMetadata:
        meta = meta if isinstance(meta, VideoMetadata) else VideoMetadata.model_validate(meta)
        m = self.manifest
        if meta.video_id in m.videos and not exist_ok:
            raise FileExistsError(f"video '{meta.video_id}' already exists")
        _write_json(self._vdir(meta.video_id) / "metadata.json", meta.model_dump_json(indent=2, exclude_none=True))
        if meta.video_id not in m.videos:
            m.videos = sorted([*m.videos, meta.video_id])
            self._save_manifest(m)
        return meta

    def video(self, video_id: str) -> VideoMetadata:
        p = self._vdir(video_id) / "metadata.json"
        if not p.exists():
            raise KeyError(f"unknown video '{video_id}'")
        return VideoMetadata.model_validate_json(p.read_text())

    def resolve_video_path(self, video_id: str, roots: Sequence[str | Path] = ()) -> Path | None:
        """Find the video file: try each recorded source, then search `roots` for the same file name."""
        meta = self.video(video_id)
        names = []
        for s in meta.sources:
            uri = s.uri[7:] if s.uri.startswith("file://") else s.uri
            if "://" not in uri:
                if Path(uri).exists():
                    return Path(uri)
                names.append(Path(uri).name)
        for root in roots:
            for n in names:
                for hit in Path(root).rglob(n):
                    return hit
        return None

    # ---- views ----
    def add_view(self, view: View | Mapping[str, Any], exist_ok: bool = False) -> View:
        view = view if isinstance(view, View) else View.model_validate(view)
        self.video(view.video_id)  # must exist
        m = self.manifest
        if view.view_id in m.views and not exist_ok:
            raise FileExistsError(f"view '{view.view_id}' already exists")
        meta = self.video(view.video_id)
        if view.region:
            w, h = meta.display_size
            if view.region.x + view.region.width > w or view.region.y + view.region.height > h:
                raise ValueError(f"view region extends outside the {w}x{h} video")
        _write_json(self.path / "views" / f"{view.view_id}.json", view.model_dump_json(indent=2, exclude_none=True))
        if view.view_id not in m.views:
            m.views = sorted([*m.views, view.view_id])
            self._save_manifest(m)
        return view

    def view(self, view_id: str) -> View:
        p = self.path / "views" / f"{view_id}.json"
        if not p.exists():
            raise KeyError(f"unknown view '{view_id}'")
        return View.model_validate_json(p.read_text())

    def view_ids(self) -> list[str]:
        return list(self.manifest.views)

    # ---- annotations: writing ----
    def _ann_path(self, video_id: str) -> Path:
        return self._vdir(video_id) / "annotations.parquet"

    def _existing(self, video_id: str) -> pa.Table:
        p = self._ann_path(video_id)
        return pq.read_table(p) if p.exists() else empty_table()

    def add_annotations(self, rows: Iterable[Mapping[str, Any]], video_id: str | None = None, *,
                        create_labels: bool = False, validate: bool = True) -> list[int]:
        """Append annotations. Rows are dicts using the column names (plus `box=(x0,y0,x1,y1)` shorthand).

        A mask row may pass `mask=<bool ndarray HxW>`; it is encoded and its box filled in automatically.
        Returns the new annotation ids. Raises ValidationError (writing nothing) if any row is invalid.
        """
        groups: dict[str, list[Mapping]] = {}
        for r in rows:
            vid = r.get("video_id", video_id)
            if vid is None:
                raise ValueError("row has no video_id and none was given")
            if video_id is not None and r.get("video_id", video_id) != video_id:
                raise ValueError(f"row video_id {r['video_id']!r} != video_id argument {video_id!r}")
            groups.setdefault(vid, []).append(r)
        for vid in groups:
            self.video(vid)  # KeyError if unknown
        if create_labels:
            self.add_labels(sorted({r["label"] for g in groups.values() for r in g if "label" in r}))
        vocab_names = {l.name for l in self.vocab.labels}
        view_ids = set(self.view_ids())

        prepared = []
        for vid, group in groups.items():
            existing = self._existing(vid)
            used = set(existing.column("annotation_id").to_pylist())
            new = _rows_to_table(group, vid, used)
            if validate:
                meta = self.video(vid)
                bad = errors(check_annotation_table(new, frame_count=meta.frame_count or None, size=meta.display_size,
                                                    vocab_names=vocab_names, view_ids=view_ids))
                if bad:
                    raise ValidationError(bad)
            prepared.append((vid, existing, new))
        out_ids: list[int] = []
        for vid, existing, new in prepared:  # write only after every group validated
            table = pa.concat_tables([existing, new]).sort_by([("start_frame", "ascending"), ("annotation_id", "ascending")])
            write_parquet(table, self._ann_path(vid))
            out_ids.extend(new.column("annotation_id").to_pylist())
        return out_ids

    def delete_annotations(self, video_id: str, annotation_ids: Iterable[int]) -> int:
        t = self._existing(video_id)
        keep = pc.invert(pc.is_in(t.column("annotation_id"), value_set=pa.array(list(annotation_ids), type=pa.int64())))
        out = t.filter(keep)
        n = t.num_rows - out.num_rows
        if n:
            write_parquet(out, self._ann_path(video_id))
        return n

    def update_annotations(self, video_id: str, annotation_ids: Iterable[int], **values: Any) -> int:
        """Set columns (e.g. status='approved', label='x') on the given annotations. Returns rows changed."""
        bad = set(values) - (set(COLUMN_NAMES) - _IMMUTABLE)
        if bad:
            raise ValueError(f"cannot update {sorted(bad)}")
        t = self._existing(video_id)
        sel = pc.is_in(t.column("annotation_id"), value_set=pa.array(list(annotation_ids), type=pa.int64()))
        n = int(pc.sum(sel.cast(pa.int8())).as_py() or 0)
        if not n:
            return 0
        for k, v in values.items():
            if k == "extra" and not isinstance(v, str) and v is not None:
                v = json.dumps(v, sort_keys=True)
            elif k == "kind":
                v = _enum_value(v, Kind, k)
            elif k == "origin":
                v = _enum_value(v, Origin, k)
            elif k == "status":
                v = _enum_value(v, Status, k)
            idx = t.schema.get_field_index(k)
            new_col = pc.if_else(sel, pa.scalar(v, type=t.schema.field(k).type), t.column(k))
            t = t.set_column(idx, t.schema.field(k), new_col)
        meta = self.video(video_id)
        bad_issues = errors(check_annotation_table(t, frame_count=meta.frame_count or None, size=meta.display_size,
                                                   vocab_names={l.name for l in self.vocab.labels},
                                                   view_ids=set(self.view_ids())))
        if bad_issues:
            raise ValidationError(bad_issues)
        write_parquet(t.sort_by([("start_frame", "ascending"), ("annotation_id", "ascending")]), self._ann_path(video_id))
        return n

    # ---- annotations: reading ----
    def annotations(self, video_id: str | None = None, *, kind: str | Sequence[str] | None = None,
                    label: str | Sequence[str] | None = None, origin: str | Sequence[str] | None = None,
                    status: str | Sequence[str] | None = None, frames: tuple[int, int] | None = None,
                    columns: Sequence[str] | None = None) -> pa.Table:
        """Query annotations as a pyarrow Table. `frames=(a, b)` keeps annotations overlapping [a, b] inclusive."""
        expr = None

        def add(e):
            nonlocal expr
            expr = e if expr is None else expr & e

        for name, val in (("kind", kind), ("label", label), ("origin", origin), ("status", status)):
            if val is None:
                continue
            vals = [val] if isinstance(val, str) else list(val)
            add(pads.field(name).isin(vals))
        if frames is not None:
            add((pads.field("start_frame") <= frames[1]) & (pads.field("end_frame") >= frames[0]))
        cols = list(columns) if columns else None
        tables = []
        for vid in ([video_id] if video_id else self.manifest.videos):
            p = self._ann_path(vid)
            if p.exists():
                tables.append(pq.read_table(p, columns=cols, filters=expr))
        return pa.concat_tables(tables) if tables else empty_table(cols)

    def stats(self) -> dict[str, Any]:
        t = self.annotations(columns=["kind", "label", "origin", "status", "video_id"])
        c = lambda n: dict(Counter(t.column(n).to_pylist()))
        return {"videos": len(self.manifest.videos), "views": len(self.manifest.views), "annotations": t.num_rows,
                "labels_in_vocab": len(self.vocab.labels), "by_kind": c("kind"), "by_label": c("label"),
                "by_origin": c("origin"), "by_status": c("status")}

    # ---- convenience ----
    def validate(self, deep: bool = False) -> list[Issue]:
        from .validate import validate_dataset
        return validate_dataset(self.path, deep=deep)

    def build(self, **kwargs):
        from .build import build as _build
        return _build(self, **kwargs)


def _rows_to_table(rows: Sequence[Mapping[str, Any]], video_id: str, used_ids: set[int]) -> pa.Table:
    cols: dict[str, list] = {n: [] for n in COLUMN_NAMES}
    for i, raw in enumerate(rows):
        unknown = set(raw) - _ROW_FIELDS
        if unknown:
            raise ValueError(f"row {i}: unknown field(s) {sorted(unknown)}; put custom data inside 'extra'")
        r = dict(raw)
        for req in ("kind", "label", "origin", "start_frame"):
            if r.get(req) is None:
                raise ValueError(f"row {i}: '{req}' is required")
        r["video_id"] = video_id
        r["kind"] = _enum_value(r["kind"], Kind, "kind")
        r["origin"] = _enum_value(r["origin"], Origin, "origin")
        r["status"] = _enum_value(r.get("status", "unreviewed"), Status, "status")
        r.setdefault("end_frame", r["start_frame"])
        if r["end_frame"] is None:
            r["end_frame"] = r["start_frame"]
        box = r.pop("box", None)
        if box is not None:
            if len(box) != 4:
                raise ValueError(f"row {i}: box must be (x0, y0, x1, y1)")
            r["x0"], r["y0"], r["x1"], r["y1"] = box
        m = r.get("mask")
        if m is not None and not isinstance(m, (bytes, bytearray, memoryview)):
            blob = _masks.encode(m)
            r["mask"], r["mask_codec"] = blob, _masks.CODEC
            bb = _masks.bbox(blob)
            if bb is not None and r.get("x0") is None:
                r["x0"], r["y0"], r["x1"], r["y1"] = bb
        elif m is not None:
            r["mask"] = bytes(m)
            r.setdefault("mask_codec", _masks.CODEC)
            if r["mask_codec"] == _masks.CODEC and r.get("x0") is None:
                bb = _masks.bbox(r["mask"])
                if bb is not None:
                    r["x0"], r["y0"], r["x1"], r["y1"] = bb
        ex = r.get("extra")
        if ex is not None and not isinstance(ex, str):
            r["extra"] = json.dumps(ex, sort_keys=True)
        aid = r.get("annotation_id")
        if aid is None:
            aid = secrets.randbits(63)
            while aid in used_ids:
                aid = secrets.randbits(63)
        elif aid in used_ids:
            raise ValueError(f"row {i}: annotation_id {aid} already exists")
        used_ids.add(aid)
        r["annotation_id"] = aid
        for n in COLUMN_NAMES:
            cols[n].append(r.get(n))
    try:
        return pa.Table.from_pydict(cols, schema=ARROW_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
        raise ValueError(f"could not convert rows to the annotation schema: {e}") from None
