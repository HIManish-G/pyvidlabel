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
from vidlabel_spec import (COLUMN_NAMES, SPEC_VERSION, FrameIndex, Kind, Label, Manifest, Origin, Status, VideoMetadata, View, Vocab,
                           is_valid_origin)

from . import masks as _masks
from ._arrow import ARROW_SCHEMA, empty_table, write_parquet
from .validate import Issue, ValidationError, check_annotation_table, check_tracks, errors

_ROW_FIELDS = set(COLUMN_NAMES) | {"box"}
_IMMUTABLE = {"annotation_id", "video_id", "mask", "mask_codec"}


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _enum_value(v, enum_cls, field: str) -> str:
    try:
        return enum_cls(v.value if hasattr(v, "value") else v).value
    except ValueError:
        raise ValueError(f"{field}={v!r}; must be one of {[e.value for e in enum_cls]}") from None


def _origin_value(v) -> str:
    if not is_valid_origin(v):
        raise ValueError(f"origin={v!r}; must be one of {[o.value for o in Origin]} or a custom value like 'x-crowd' "
                         "(lowercase letters, digits, '.', '_' or '-' after 'x-')")
    return v.value if hasattr(v, "value") else v


def _check_index_fits(meta: VideoMetadata, index: FrameIndex) -> None:
    if meta.frame_count and index.frame_count != meta.frame_count:
        raise ValueError(f"the frame index has {index.frame_count} frames but video '{meta.video_id}' has {meta.frame_count}")
    if meta.content_hash and index.content_hash and meta.content_hash != index.content_hash:
        raise ValueError(f"the frame index was made from a different file than video '{meta.video_id}' (hash differs)")


class Dataset:
    """A vidlabel dataset: a folder of plain files (see docs/SPEC.md). Make one with `Dataset.create(path)`, open one with
    `Dataset.open(path)`. There is no server and no lock: one writer per video at a time."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "manifest.json").exists():
            raise FileNotFoundError(f"{self.path} is not a vidlabel dataset (no manifest.json). Make one with `vidlabel init` or Dataset.create().")

    # ---- construction ----
    @classmethod
    def create(cls, path: str | Path, exist_ok: bool = False) -> "Dataset":
        """Make an empty dataset folder. Raises FileExistsError if one is already there, unless `exist_ok=True`."""
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
        """Open an existing dataset. Raises FileNotFoundError if the folder is not a vidlabel dataset."""
        return cls(path)

    def __repr__(self) -> str:
        return f"Dataset({str(self.path)!r}, videos={len(self.manifest.videos)})"

    # ---- manifest / vocab ----
    @property
    def manifest(self) -> Manifest:
        """The parsed manifest.json: spec version, video ids and view ids."""
        return Manifest.model_validate_json((self.path / "manifest.json").read_text(encoding="utf-8"))

    def _save_manifest(self, m: Manifest) -> None:
        _write_json(self.path / "manifest.json", m.model_dump_json(indent=2))

    @property
    def vocab(self) -> Vocab:
        """The parsed vocab.json: the one shared label list."""
        return Vocab.model_validate_json((self.path / "vocab.json").read_text(encoding="utf-8"))

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
        """Ids of all registered videos, sorted."""
        return list(self.manifest.videos)

    def _vdir(self, video_id: str) -> Path:
        from vidlabel_spec._models import _check_id
        try:
            _check_id(str(video_id))
        except ValueError:
            raise KeyError(f"no video '{video_id}' in this dataset") from None
        return self.path / "videos" / video_id

    def add_video(self, meta: VideoMetadata | Mapping[str, Any], exist_ok: bool = False, *,
                  frame_index: FrameIndex | None = None) -> VideoMetadata:
        """Register a video from a VideoMetadata (or a dict of its fields), e.g. from `probe_video`. The video file itself
        is never copied. Raises FileExistsError if the id exists, unless `exist_ok=True` (which replaces the metadata).

        `frame_index` (from `probe_frame_index`) is stored next to the metadata; see `set_frame_index`."""
        meta = meta if isinstance(meta, VideoMetadata) else VideoMetadata.model_validate(meta)
        m = self.manifest
        if meta.video_id in m.videos and not exist_ok:
            raise FileExistsError(f"video '{meta.video_id}' already exists")
        if frame_index is not None:
            _check_index_fits(meta, frame_index)
        _write_json(self._vdir(meta.video_id) / "metadata.json", meta.model_dump_json(indent=2, exclude_none=True))
        if frame_index is not None:
            _write_json(self._vdir(meta.video_id) / "frame_index.json", frame_index.model_dump_json(exclude_none=True))
        if meta.video_id not in m.videos:
            m.videos = sorted([*m.videos, meta.video_id])
            self._save_manifest(m)
        return meta

    def frame_index(self, video_id: str) -> FrameIndex | None:
        """The stored frame index of a video (timestamps and keyframes), or None if it has none."""
        self.video(video_id)  # unknown ids raise KeyError
        p = self._vdir(video_id) / "frame_index.json"
        return FrameIndex.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None

    def set_frame_index(self, video_id: str, index: FrameIndex) -> None:
        """Store (or replace) the frame index of a registered video. Raises ValueError if it has a different number of
        frames than the metadata says, or was made from a different file (hash mismatch)."""
        _check_index_fits(self.video(video_id), index)
        _write_json(self._vdir(video_id) / "frame_index.json", index.model_dump_json(exclude_none=True))

    def video(self, video_id: str) -> VideoMetadata:
        """Metadata of one registered video. Raises KeyError for an unknown id."""
        p = self._vdir(video_id) / "metadata.json"
        if not p.exists():
            raise KeyError(f"unknown video '{video_id}'")
        return VideoMetadata.model_validate_json(p.read_text(encoding="utf-8"))

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
        """Define a named crop and/or frame range of a registered video (see docs/views.md). The region must fit inside the
        video, `frame_range` inside its length, and `allowed_labels` must already be in the vocabulary."""
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
        if view.ranges and meta.frame_count and view.ranges[-1][1] >= meta.frame_count:
            raise ValueError(f"view frame_range ends beyond the video ({meta.frame_count} frames)")
        if view.allowed_labels is not None:
            unknown = sorted(set(view.allowed_labels) - {l.name for l in self.vocab.labels})
            if unknown:
                raise ValueError(f"allowed_labels not in vocab: {unknown} (add them with add_labels first)")
        _write_json(self.path / "views" / f"{view.view_id}.json", view.model_dump_json(indent=2, exclude_none=True))
        if view.view_id not in m.views:
            m.views = sorted([*m.views, view.view_id])
            self._save_manifest(m)
        return view

    def view(self, view_id: str) -> View:
        """One view definition. Raises KeyError for an unknown id."""
        from vidlabel_spec._models import _check_id
        try:
            _check_id(str(view_id))
        except ValueError:
            raise KeyError(f"no view '{view_id}' in this dataset") from None
        p = self.path / "views" / f"{view_id}.json"
        if not p.exists():
            raise KeyError(f"unknown view '{view_id}'")
        return View.model_validate_json(p.read_text(encoding="utf-8"))

    def view_ids(self) -> list[str]:
        """Ids of all views, sorted."""
        return list(self.manifest.views)

    def _views_map(self) -> dict[str, View]:
        return {v: self.view(v) for v in self.manifest.views}

    # ---- annotations: writing ----
    def _ann_path(self, video_id: str) -> Path:
        return self._vdir(video_id) / "annotations.parquet"

    def _existing(self, video_id: str) -> pa.Table:
        p = self._ann_path(video_id)
        return pq.read_table(p) if p.exists() else empty_table()

    def add_annotations(self, rows: Iterable[Mapping[str, Any]], video_id: str | None = None, *,
                        create_labels: bool = False, validate: bool = True, coords: str = "source") -> list[int]:
        """Append annotations. Rows are dicts using the column names (plus `box=(x0,y0,x1,y1)` shorthand).

        A mask row may pass `mask=<bool ndarray HxW>`; it is encoded and its box filled in automatically.
        Returns the new annotation ids. Raises ValidationError (writing nothing) if any row is invalid.

        coords="source" (default): boxes/masks are in full source-video pixels.
        coords="view": boxes/masks are in the pixels of the row's `view_id` (the crop as the pipeline saw it) and are
        converted to source coordinates here (offset, plus scaling if the view has `output_size`). Masks must be
        boolean arrays the size of the view region or of its `output_size`.
        """
        if coords not in ("source", "view"):
            raise ValueError("coords must be 'source' or 'view'")
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
        views_map = self._views_map()

        prepared = []
        for vid, group in groups.items():
            existing = self._existing(vid)
            used = set(existing.column("annotation_id").to_pylist())
            if coords == "view":
                group = self._rows_view_to_source(group, vid, views_map)
            W, H = self.video(vid).display_size
            for i, r in enumerate(group):
                m = r.get("mask")
                if m is not None and hasattr(m, "shape") and tuple(m.shape) != (H, W):
                    raise ValueError(f"row {i}: the mask is {m.shape[-1]}x{m.shape[0]} but video '{vid}' is {W}x{H}; a mask must "
                                     f"cover the whole frame (or pass coords='view' for a mask drawn in a view's pixels)")
            new = _rows_to_table(group, vid, used)
            if validate:
                meta = self.video(vid)
                bad = errors(check_annotation_table(new, frame_count=meta.frame_count or None, size=meta.display_size,
                                                    vocab_names=vocab_names, views=views_map, video_id=vid))
                new_tracks = pc.drop_null(new.column("track_id"))
                if len(new_tracks):                                    # tracks must stay consistent with earlier calls too
                    same = existing.filter(pc.is_in(existing.column("track_id"), value_set=new_tracks.combine_chunks()))
                    bad += errors(check_tracks(pa.concat_tables([same, new])))
                if bad:
                    raise ValidationError(bad)
            prepared.append((vid, existing, new))
        out_ids: list[int] = []
        for vid, existing, new in prepared:  # write only after every group validated
            table = pa.concat_tables([existing, new]).sort_by([("start_frame", "ascending"), ("annotation_id", "ascending")])
            write_parquet(table, self._ann_path(vid))
            out_ids.extend(new.column("annotation_id").to_pylist())
        return out_ids

    def _rows_view_to_source(self, rows, vid: str, views_map: dict[str, View]) -> list[dict]:
        out = []
        W, H = self.video(vid).display_size
        for i, raw in enumerate(rows):
            r = dict(raw)
            box = r.pop("box", None)
            corner = [r.pop(k, None) for k in ("x0", "y0", "x1", "y1")]
            if box is None and any(c is not None for c in corner):
                if any(c is None for c in corner):
                    raise ValueError(f"row {i}: give all of x0, y0, x1, y1 or use box=(x0, y0, x1, y1)")
                box = tuple(corner)
            mask = r.get("mask")
            if box is None and mask is None:
                out.append(r)  # nothing geometric to convert (e.g. a marker)
                continue
            vw_id = r.get("view_id")
            if vw_id is None:
                raise ValueError(f"row {i}: coords='view' needs a view_id on rows with a box or mask")
            if vw_id not in views_map:
                raise ValueError(f"row {i}: unknown view '{vw_id}'")
            v = views_map[vw_id]
            if v.video_id != vid:
                raise ValueError(f"row {i}: view '{vw_id}' belongs to video '{v.video_id}', not '{vid}'")
            if mask is not None:
                if isinstance(mask, (bytes, bytearray, memoryview)):
                    raise ValueError(f"row {i}: coords='view' cannot convert an already-encoded mask; pass a boolean array")
                m = np.asarray(mask, dtype=bool)
                if v.region is not None:
                    rh, rw = v.region.height, v.region.width
                    if m.shape != (rh, rw):
                        if v.output_size and m.shape == (v.output_size[1], v.output_size[0]):
                            m = _masks.resize_nearest(m, rh, rw)
                        else:
                            raise ValueError(f"row {i}: mask shape {m.shape} matches neither the view region ({rh}x{rw}) "
                                             f"nor its output_size")
                    m = _masks.place_in_frame(m, v.region.x, v.region.y, W, H)
                r["mask"] = m  # box is derived from the mask when the row is written
            else:
                r["box"] = v.to_source_box(box)
            out.append(r)
        return out

    def _check_ids_exist(self, t: pa.Table, video_id: str, ids: list[int], missing: str) -> None:
        if missing not in ("raise", "ignore"):
            raise ValueError("missing must be 'raise' or 'ignore'")
        if missing == "raise" and ids:
            gone = sorted(set(ids) - set(t.column("annotation_id").to_pylist()))
            if gone:
                raise KeyError(f"no annotation with id {gone[:5]}{' ...' if len(gone) > 5 else ''} in video '{video_id}' "
                               f"({len(gone)} of {len(ids)} ids; pass missing='ignore' to skip ids that are not there)")

    def delete_annotations(self, video_id: str, annotation_ids: Iterable[int], *, missing: str = "raise") -> int:
        """Remove annotations of one video by id. Returns how many rows were removed. Ids that are not in this video raise
        KeyError (and nothing is removed) unless `missing="ignore"`."""
        self.video(video_id)
        ids = [int(i) for i in annotation_ids]
        t = self._existing(video_id)
        self._check_ids_exist(t, video_id, ids, missing)
        keep = pc.invert(pc.is_in(t.column("annotation_id"), value_set=pa.array(ids, type=pa.int64())))
        out = t.filter(keep)
        n = t.num_rows - out.num_rows
        if n:
            write_parquet(out, self._ann_path(video_id))
        return n

    def update_annotations(self, video_id: str, annotation_ids: Iterable[int], *, missing: str = "raise", **values: Any) -> int:
        """Set columns (e.g. status='approved', label='x') on the given annotations. Returns rows changed. Ids that are not in
        this video raise KeyError (and nothing changes) unless `missing="ignore"`; giving no column to change is an error."""
        self.video(video_id)
        if not values:
            raise ValueError("give at least one column to change, e.g. status='approved'")
        bad = set(values) - (set(COLUMN_NAMES) - _IMMUTABLE)
        if bad:
            hint = "; set x0, y0, x1, y1 instead of box" if "box" in bad else ""
            raise ValueError(f"cannot update {sorted(bad)}{hint}")
        converted = {}
        for k, v in values.items():                       # check the values before looking for rows, so a typo is never a quiet 0
            if k == "extra" and not isinstance(v, str) and v is not None:
                v = _json_extra(v)
            elif k == "kind":
                v = _enum_value(v, Kind, k)
            elif k == "origin":
                v = _origin_value(v)
            elif k == "status":
                v = _enum_value(v, Status, k)
            converted[k] = v
        ids = [int(i) for i in annotation_ids]
        t = self._existing(video_id)
        self._check_ids_exist(t, video_id, ids, missing)
        sel = pc.is_in(t.column("annotation_id"), value_set=pa.array(ids, type=pa.int64()))
        n = int(pc.sum(sel.cast(pa.int8())).as_py() or 0)
        if not n:
            return 0
        for k, v in converted.items():
            idx = t.schema.get_field_index(k)
            new_col = pc.if_else(sel, pa.scalar(v, type=t.schema.field(k).type), t.column(k))
            t = t.set_column(idx, t.schema.field(k), new_col)
        meta = self.video(video_id)
        bad_issues = errors(check_annotation_table(t, frame_count=meta.frame_count or None, size=meta.display_size,
                                                   vocab_names={l.name for l in self.vocab.labels},
                                                   views=self._views_map(), video_id=video_id))
        if bad_issues:
            raise ValidationError(bad_issues)
        write_parquet(t.sort_by([("start_frame", "ascending"), ("annotation_id", "ascending")]), self._ann_path(video_id))
        return n

    def derive_view_rows(self, video_id: str, view_id: str, *, labels: Sequence[str] | None = None,
                         origins: Sequence[str] | None = None, statuses: Sequence[str] = ("unreviewed", "approved"),
                         dry_run: bool = False) -> dict[str, Any]:
        """Copy whole-frame rows into a view, for the rows that belong there: frames inside the view's time windows, box or mask
        entirely inside its region, label allowed in the view. Each copy keeps its columns (including `track_id`), gets the
        view's `view_id` and a new `annotation_id`, and records its source in `extra["derived_from"]`, so running this again
        copies only what is new. The originals stay. Rows that only partly overlap the region are not copied (cutting a box
        or mask would change it): they are counted in `skipped_partial`.

        Returns `copied`, `ids`, `already_derived`, and the skip counts `skipped_partial`, `skipped_outside`, `skipped_label`,
        `skipped_frames`. With `dry_run=True` nothing is written and `copied` is how many rows would be. Whole-frame samples (no `view_id`) then see both the original and the copy: build whole-frame releases
        with `build(view_ids=[None])` to leave the copies out."""
        view = self.view(view_id)
        if view.video_id != video_id:
            raise ValueError(f"view '{view_id}' belongs to video '{view.video_id}', not '{video_id}'")
        have = set()
        for r in self.annotations(video_id, view_id=[view_id], columns=["extra"]).column("extra").to_pylist():
            if r:
                try:
                    src = json.loads(r).get("derived_from")
                except (ValueError, AttributeError):
                    src = None
                if src is not None:
                    have.add(int(src))
        src_rows = self.annotations(video_id, view_id=[None], label=labels, origin=origins, status=list(statuses)).to_pylist()
        stats = {"already_derived": 0, "skipped_partial": 0, "skipped_outside": 0, "skipped_label": 0, "skipped_frames": 0}
        rows: list[dict] = []
        eps = 0.5
        for r in src_rows:
            if r["annotation_id"] in have:
                stats["already_derived"] += 1
                continue
            if not view.contains_frames(r["start_frame"], r["end_frame"]):
                stats["skipped_frames"] += 1
                continue
            if view.allowed_labels is not None and r["label"] not in view.allowed_labels:
                stats["skipped_label"] += 1
                continue
            if view.region is not None and r["kind"] != "marker" and r["x0"] is not None:
                g = view.region
                inside = (r["x0"] >= g.x - eps and r["y0"] >= g.y - eps and r["x1"] <= g.x + g.width + eps
                          and r["y1"] <= g.y + g.height + eps)
                if not inside:
                    overlaps = r["x0"] < g.x + g.width and r["x1"] > g.x and r["y0"] < g.y + g.height and r["y1"] > g.y
                    stats["skipped_partial" if overlaps else "skipped_outside"] += 1
                    continue
            row = {k: v for k, v in r.items() if v is not None and k not in ("annotation_id", "video_id", "extra")}
            extra = json.loads(r["extra"]) if r.get("extra") else {}
            extra["derived_from"] = r["annotation_id"]
            row.update(view_id=view_id, extra=extra)
            rows.append(row)
        ids = self.add_annotations(rows, video_id) if rows and not dry_run else []
        return {"copied": len(rows) if dry_run else len(ids), "ids": ids, "dry_run": dry_run, **stats}

    # ---- tracks: review one object at a time ----
    def _track_ids(self, video_id: str, track_id: int, view_id: str | None) -> list[int]:
        t = self.annotations(video_id, track_id=track_id, view_id=[view_id], columns=["annotation_id"])
        ids = t.column("annotation_id").to_pylist()
        if not ids:
            where = f"view '{view_id}'" if view_id else "the whole frame (no view)"
            raise KeyError(f"video '{video_id}' has no track {track_id} in {where}; track ids are per video and view, "
                           f"see tracks('{video_id}')")
        return ids

    def tracks(self, video_id: str, *, view_id: str | None = None) -> list[dict[str, Any]]:
        """One summary row per track of this video and view (`view_id=None`: whole-frame rows), sorted by track id: `track_id`,
        `labels`, `n_rows`, `first_frame`, `last_frame`, `mean_confidence` (None if unscored), `statuses` (counts) and `origins`."""
        t = self.annotations(video_id, view_id=[view_id], columns=["track_id", "label", "start_frame", "end_frame", "confidence",
                                                                   "status", "origin"]).to_pylist()
        groups: dict[int, list[dict]] = {}
        for r in t:
            if r["track_id"] is not None:
                groups.setdefault(r["track_id"], []).append(r)
        out = []
        for tid in sorted(groups):
            rows = groups[tid]
            conf = [r["confidence"] for r in rows if r["confidence"] is not None]
            out.append({"track_id": tid, "labels": sorted({r["label"] for r in rows}), "n_rows": len(rows),
                        "first_frame": min(r["start_frame"] for r in rows), "last_frame": max(r["end_frame"] for r in rows),
                        "mean_confidence": sum(conf) / len(conf) if conf else None,
                        "statuses": dict(Counter(r["status"] for r in rows)), "origins": sorted({r["origin"] for r in rows})})
        return out

    def update_track(self, video_id: str, track_id: int, *, view_id: str | None = None, **values: Any) -> int:
        """Change columns on every row of one track at once, e.g. `update_track("clip", 7, status="rejected")`. A track id means
        one object only within one video and view, so pass `view_id` for a track made in a view. KeyError if there is no such track."""
        return self.update_annotations(video_id, self._track_ids(video_id, track_id, view_id), **values)

    def delete_track(self, video_id: str, track_id: int, *, view_id: str | None = None) -> int:
        """Remove every row of one track. To keep it out of releases without losing it, use `update_track(..., status="rejected")`."""
        return self.delete_annotations(video_id, self._track_ids(video_id, track_id, view_id))

    # ---- annotations: reading ----
    def annotations(self, video_id: str | None = None, *, kind: str | Sequence[str] | None = None,
                    label: str | Sequence[str] | None = None, origin: str | Sequence[str] | None = None,
                    status: str | Sequence[str] | None = None, view_id: str | Sequence[str | None] | None = None,
                    track_id: int | Sequence[int] | None = None,
                    frames: tuple[int, int] | None = None, columns: Sequence[str] | None = None) -> pa.Table:
        """Query annotations as a pyarrow Table. `frames=(a, b)` keeps annotations overlapping [a, b] inclusive.

        `view_id` keeps rows labelled in that view (or any of several). Include None in a list to also (or only)
        get whole-frame rows that have no view, e.g. `view_id=[None]`. `track_id` keeps rows of those tracks; a track id means
        one object only within one video *and* view, so combine it with `view_id`.
        """
        if video_id is not None:
            self.video(video_id)                      # an unknown video is an error on reads too, as it is on writes
        expr = None

        def add(e):
            nonlocal expr
            expr = e if expr is None else expr & e

        for name, val in (("kind", kind), ("label", label), ("origin", origin), ("status", status)):
            if val is None:
                continue
            vals = [val] if isinstance(val, str) else list(val)
            allowed = {"kind": [e.value for e in Kind], "status": [e.value for e in Status]}.get(name)
            if allowed is not None and (bad := [v for v in vals if v not in allowed]):
                raise ValueError(f"unknown {name} {bad}; use one of {allowed}")
            add(pads.field(name).isin(vals))
        if view_id is not None:
            vals = [view_id] if isinstance(view_id, str) else list(view_id)
            named = [v for v in vals if v is not None]
            e = pads.field("view_id").isin(named) if named else None
            if None in vals:
                e = pads.field("view_id").is_null() if e is None else (e | pads.field("view_id").is_null())
            add(e)
        if track_id is not None:
            add(pads.field("track_id").isin([int(track_id)] if isinstance(track_id, (int, np.integer)) else [int(t) for t in track_id]))
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
        """Counts for `vidlabel info`: totals, and breakdowns by kind, label, origin, status and view, plus one entry per video."""
        t = self.annotations(columns=["kind", "label", "origin", "status", "video_id", "view_id"])
        c = lambda n: dict(Counter(t.column(n).to_pylist()))
        per_video = c("video_id")
        details = []
        for vid in self.manifest.videos:
            m = self.video(vid)
            details.append({"video_id": vid, "width": m.width, "height": m.height, "fps": round(m.fps.value, 3),
                            "codec": m.codec, "bit_rate": m.bit_rate, "frame_count": m.frame_count,
                            "annotations": per_video.get(vid, 0)})
        by_view = {("(whole frame)" if k is None else k): v for k, v in c("view_id").items()}
        return {"videos": len(self.manifest.videos), "views": len(self.manifest.views), "annotations": t.num_rows,
                "labels_in_vocab": len(self.vocab.labels), "by_kind": c("kind"), "by_label": c("label"),
                "by_origin": c("origin"), "by_status": c("status"), "by_view": by_view, "video_details": details}

    # ---- convenience ----
    def validate(self, deep: bool = False) -> list[Issue]:
        """Check the whole dataset against the spec and return a list of Issues (empty = valid). Never raises for bad data.
        `deep=True` also decodes every mask and checks it against its box and the frame size."""
        from .validate import validate_dataset
        return validate_dataset(self.path, deep=deep)

    def build(self, **kwargs):
        """Compile a training release: `build(**kwargs)` forwards to `vidlabel.build.build` and returns `(Release, path)`."""
        from .build import build as _build
        return _build(self, **kwargs)


def _json_extra(ex: Any) -> str:
    """`extra` as JSON text. NumPy numbers and arrays are converted; NaN and infinity (not valid JSON) are refused."""
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"{type(o).__name__} is not JSON-serializable")
    try:
        return json.dumps(ex, sort_keys=True, allow_nan=False, default=default)
    except ValueError:
        raise ValueError("extra holds NaN or infinity, which JSON cannot store; use None or leave the key out") from None
    except TypeError as e:
        raise ValueError(f"extra must be JSON: {e} (convert it, e.g. float(x) or str(x))") from None


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
        r["origin"] = _origin_value(r["origin"])
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
            r["extra"] = _json_extra(ex)
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
