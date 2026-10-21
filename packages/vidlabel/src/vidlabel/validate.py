"""Validation: the thing that keeps many people's contributions consistent."""
from __future__ import annotations

import json
from typing import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import ValidationError as PydanticValidationError
from vidlabel_spec import (COLUMN_NAMES, REQUIRED_COLUMNS, SPEC_VERSION, Kind, Manifest, MaskCodec, Origin, Status,
                           VideoMetadata, View, Vocab, is_valid_origin)

from . import masks


@dataclass
class Issue:
    level: str  # "error" | "warning"
    code: str
    message: str
    count: int = 1
    sample_ids: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        s = f"[{self.level}] {self.code}: {self.message}"
        if self.count > 1:
            s += f" ({self.count} rows)"
        if self.sample_ids:
            s += f" e.g. annotation_id {self.sample_ids[:3]}"
        return s


class ValidationError(ValueError):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("\n".join(str(i) for i in issues if i.level == "error"))


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == "error"]


def _np(table: pa.Table, name: str, fill=0, dtype=None) -> np.ndarray:
    a = table.column(name).combine_chunks()
    if a.null_count:
        a = pc.fill_null(a, fill)
    r = a.to_numpy(zero_copy_only=False)
    return r.astype(dtype) if dtype else r


def _nulls(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(pc.is_null(table.column(name).combine_chunks()).to_numpy(zero_copy_only=False))


def _not_in(table: pa.Table, name: str, allowed: list[str]) -> np.ndarray:
    col = table.column(name).combine_chunks()
    bad = pc.invert(pc.is_in(col, value_set=pa.array(allowed, type=pa.string())))
    return np.asarray(pc.fill_null(bad, False).to_numpy(zero_copy_only=False))


def _factorize(values) -> np.ndarray:
    """Small integer code per distinct value (None gets its own code)."""
    codes: dict = {}
    return np.array([codes.setdefault(v, len(codes)) for v in values], dtype=np.int64)


def _check_tracks(table: pa.Table, report, *, geom: np.ndarray, is_mask: np.ndarray, start: np.ndarray) -> None:
    """Track rules. A track is one object: identified by (view_id, track_id) within a video. Rejected rows are ignored,
    since a reviewer who rejects a box normally replaces it under the same id.

      track_label_conflict     (warning)  one track carries more than one label
      duplicate_track_in_frame (error)    one track has two boxes (or two masks) in the same frame
    """
    status = np.asarray(table.column("status").to_pylist(), dtype=object)
    sel = geom & ~_nulls(table, "track_id") & (status != "rejected")
    if not sel.any():
        return
    idx = np.flatnonzero(sel)
    track = _np(table, "track_id", dtype=np.int64)[idx]
    view = _factorize(table.column("view_id").to_pylist())[idx]
    label = _factorize(table.column("label").to_pylist())[idx]
    _, group = np.unique(np.stack([view, track], axis=1), axis=0, return_inverse=True)
    group = group.ravel()
    pairs = np.unique(np.stack([group, label], axis=1), axis=0)
    labels_per_track = np.bincount(pairs[:, 0], minlength=int(group.max()) + 1)
    bad = np.zeros(len(table), dtype=bool)
    bad[idx[labels_per_track[group] > 1]] = True
    report("warning", "track_label_conflict", "a track_id is used with more than one label (a track should be one object)", bad)
    _, inv, counts = np.unique(np.stack([view, track, start[idx], is_mask[idx].astype(np.int64)], axis=1), axis=0,
                               return_inverse=True, return_counts=True)
    bad = np.zeros(len(table), dtype=bool)
    bad[idx[counts[inv.ravel()] > 1]] = True
    report("error", "duplicate_track_in_frame", "the same track_id appears more than once in one frame (same kind, same view)", bad)


def check_tracks(table: pa.Table) -> list[Issue]:
    """Only the track rules, for a table that already has the standard columns (used when appending to existing rows)."""
    issues: list[Issue] = []
    if table.num_rows == 0:
        return issues
    ids = _np(table, "annotation_id", dtype=np.int64)

    def report(level, code, msg, bad):
        n = int(bad.sum())
        if n:
            issues.append(Issue(level, code, msg, n, [int(x) for x in ids[bad][:5]]))

    kind = np.asarray(table.column("kind").to_pylist(), dtype=object)
    _check_tracks(table, report, geom=(kind == "box") | (kind == "mask"), is_mask=kind == "mask",
                  start=_np(table, "start_frame", dtype=np.int64))
    return issues


def check_annotation_table(
    table: pa.Table, *, frame_count: int | None = None, size: tuple[int, int] | None = None,
    vocab_names: set[str] | None = None, views: Mapping[str, View] | None = None, video_id: str | None = None,
    deep_masks: bool = False,
) -> list[Issue]:
    """Check one annotation table. `size` is (width, height) in display orientation.

    `views` maps view_id -> View; rows tagged with a view must lie inside it (region, frame range, allowed labels).
    """
    issues: list[Issue] = []
    if table.num_rows == 0 and set(COLUMN_NAMES) <= set(table.column_names):
        return issues
    missing = [c for c in COLUMN_NAMES if c not in table.column_names]
    if missing:
        return [Issue("error", "missing_column", f"missing column(s): {missing}")]
    unknown = [c for c in table.column_names if c not in COLUMN_NAMES]
    if unknown:
        issues.append(Issue("error", "unknown_column", f"unknown column(s) {unknown}; custom data belongs in 'extra'"))

    ids = _np(table, "annotation_id", dtype=np.int64)

    def report(level, code, msg, bad):
        n = int(bad.sum())
        if n:
            issues.append(Issue(level, code, msg, n, [int(x) for x in ids[bad][:5]]))

    for name in REQUIRED_COLUMNS:
        report("error", "null_required", f"required column '{name}' has nulls", _nulls(table, name))

    report("error", "bad_kind", f"kind must be one of {[k.value for k in Kind]}", _not_in(table, "kind", [k.value for k in Kind]))
    bad_origin = np.array([o is not None and not is_valid_origin(o) for o in table.column("origin").to_pylist()], dtype=bool)
    report("error", "bad_origin", f"origin must be one of {[o.value for o in Origin]} or a custom value like 'x-crowd'", bad_origin)
    report("error", "bad_status", f"status must be one of {[s.value for s in Status]}", _not_in(table, "status", [s.value for s in Status]))

    uniq, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        issues.append(Issue("error", "duplicate_id", "annotation_id is not unique", int((counts > 1).sum()),
                            [int(x) for x in uniq[counts > 1][:5]]))

    start, end = _np(table, "start_frame", dtype=np.int64), _np(table, "end_frame", dtype=np.int64)
    report("error", "negative_frame", "start_frame < 0", start < 0)
    report("error", "frame_order", "end_frame < start_frame", end < start)
    if frame_count:
        report("error", "frame_out_of_range", f"frame beyond video length ({frame_count} frames)", end >= frame_count)

    if vocab_names is not None:
        col = table.column("label").combine_chunks()
        bad = pc.invert(pc.is_in(col, value_set=pa.array(sorted(vocab_names), type=pa.string())))
        report("error", "unknown_label", "label not in vocab.json", np.asarray(pc.fill_null(bad, False).to_numpy(zero_copy_only=False)))

    kind = np.asarray(table.column("kind").to_pylist(), dtype=object)
    is_marker, is_box, is_mask = kind == "marker", kind == "box", kind == "mask"
    geom = is_box | is_mask

    coords = {n: _np(table, n, fill=np.nan, dtype=np.float64) for n in ("x0", "y0", "x1", "y1")}
    x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]
    box_null = _nulls(table, "x0") | _nulls(table, "y0") | _nulls(table, "x1") | _nulls(table, "y1")
    mask_null = _nulls(table, "mask")
    codec_null = _nulls(table, "mask_codec")

    report("error", "missing_box", "box/mask annotation without a full box", geom & box_null)
    ok = geom & ~box_null
    report("error", "bad_box", "box must be finite with x1 > x0 and y1 > y0",
           ok & ~(np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1) & (x1 > x0) & (y1 > y0)))
    if size:
        w, h = size
        report("warning", "box_outside_frame", f"box extends outside the {w}x{h} frame",
               ok & ((x0 < 0) | (y0 < 0) | (x1 > w) | (y1 > h)))
    report("error", "marker_has_geometry", "marker must not have a box or mask", is_marker & (~box_null | ~mask_null))
    report("error", "box_has_mask", "box annotation must not carry a mask", is_box & ~mask_null)
    report("error", "geometry_span", "box/mask annotations are single-frame (start_frame == end_frame)", geom & (start != end))
    report("error", "missing_mask", "mask annotation without mask bytes", is_mask & mask_null)
    report("error", "missing_codec", "mask bytes without mask_codec", ~mask_null & codec_null)
    codec_bad = ~codec_null & _not_in(table, "mask_codec", [c.value for c in MaskCodec])
    report("error", "unknown_codec", f"mask_codec must be one of {[c.value for c in MaskCodec]}", codec_bad)

    conf = _np(table, "confidence", fill=0.5, dtype=np.float64)
    report("error", "bad_confidence", "confidence must be within 0..1", (conf < 0) | (conf > 1) | ~np.isfinite(conf))
    report("error", "bad_track_id", "track_id must be >= 0", _np(table, "track_id", fill=0) < 0)
    _check_tracks(table, report, geom=geom, is_mask=is_mask, start=start)

    if views is not None:
        vcol = np.asarray(table.column("view_id").to_pylist(), dtype=object)
        lcol = np.asarray(table.column("label").to_pylist(), dtype=object)
        has_view = np.array([v is not None for v in vcol], dtype=bool)
        report("error", "unknown_view", "view_id not found in views/", has_view & ~np.isin(vcol, list(views)))
        eps = 0.5  # px tolerance for float rounding after scaling
        for vid_, v in views.items():
            sel = has_view & (vcol == vid_)
            if not sel.any():
                continue
            if video_id is not None and v.video_id != video_id:
                report("error", "view_wrong_video", f"view '{vid_}' belongs to video '{v.video_id}', not '{video_id}'", sel)
            if v.region is not None:
                r = v.region
                outside = sel & geom & ~box_null & ((x0 < r.x - eps) | (y0 < r.y - eps) |
                                                    (x1 > r.x + r.width + eps) | (y1 > r.y + r.height + eps))
                report("error", "box_outside_view", f"box extends outside view '{vid_}' region", outside)
            if v.frame_range is not None:
                report("error", "frame_outside_view", f"frame outside view '{vid_}' frame_range {list(v.frame_range)}",
                       sel & ((start < v.frame_range[0]) | (end > v.frame_range[1])))
            if v.allowed_labels is not None:
                report("error", "label_not_allowed_in_view", f"label not allowed in view '{vid_}' (allowed: {v.allowed_labels})",
                       sel & ~np.isin(lcol, v.allowed_labels))

    extra = table.column("extra").to_pylist()
    bad_extra = np.zeros(len(extra), dtype=bool)
    for i, v in enumerate(extra):
        if v is None:
            continue
        try:
            bad_extra[i] = not isinstance(json.loads(v), dict)
        except (ValueError, TypeError):
            bad_extra[i] = True
    report("error", "bad_extra", "extra must be a JSON object", bad_extra)

    if deep_masks and size:
        blobs = table.column("mask").to_pylist()
        bad_dec = np.zeros(len(blobs), dtype=bool)
        bad_box = np.zeros(len(blobs), dtype=bool)
        for i, b in enumerate(blobs):
            if b is None or not is_mask[i] or codec_bad[i]:
                continue
            try:
                inf = masks.info(b)
                masks.decode_crop(b)
                if (inf["width"], inf["height"]) != size:
                    bad_dec[i] = True
                bb = masks.bbox(b)
                if bb is None or not np.allclose(bb, (x0[i], y0[i], x1[i], y1[i])):
                    bad_box[i] = True
            except Exception:
                bad_dec[i] = True
        report("error", "mask_undecodable", "mask cannot be decoded or has the wrong frame size", bad_dec)
        report("error", "mask_box_mismatch", "mask's box columns differ from its pixel bounding box", bad_box)
    return issues


def validate_dataset(path: str | Path, deep: bool = False) -> list[Issue]:
    """Validate a whole dataset directory. Never raises for bad data; returns issues."""
    root = Path(path)
    issues: list[Issue] = []

    def err(code, msg):
        issues.append(Issue("error", code, msg))

    def warn(code, msg):
        issues.append(Issue("warning", code, msg))

    if not (root / "manifest.json").exists():
        return [Issue("error", "no_manifest", f"{root} has no manifest.json (not a vidlabel dataset)")]
    try:
        manifest = Manifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))
    except (PydanticValidationError, ValueError) as e:
        return [Issue("error", "bad_manifest", f"manifest.json: {e}")]
    if manifest.spec_version.split(".")[0] != SPEC_VERSION.split(".")[0]:
        err("spec_major", f"dataset spec {manifest.spec_version}, library supports {SPEC_VERSION}")

    vocab_names: set[str] | None = None
    if (root / "vocab.json").exists():
        try:
            vocab_names = {l.name for l in Vocab.model_validate_json((root / "vocab.json").read_text(encoding="utf-8")).labels}
        except (PydanticValidationError, ValueError) as e:
            err("bad_vocab", f"vocab.json: {e}")
    else:
        err("no_vocab", "vocab.json is missing")

    views_map: dict[str, View] = {}
    views_dir = root / "views"
    view_files = sorted(views_dir.glob("*.json")) if views_dir.exists() else []
    metas: dict[str, VideoMetadata] = {}
    for vf in view_files:
        try:
            v = View.model_validate_json(vf.read_text(encoding="utf-8"))
            if v.view_id != vf.stem:
                err("view_id_mismatch", f"views/{vf.name} contains view_id '{v.view_id}'")
            views_map[v.view_id] = v
            if v.allowed_labels is not None and vocab_names is not None:
                unk = sorted(set(v.allowed_labels) - vocab_names)
                if unk:
                    err("view_labels", f"view '{v.view_id}' allows label(s) not in vocab.json: {unk}")
        except (PydanticValidationError, ValueError) as e:
            err("bad_view", f"views/{vf.name}: {e}")
    if set(manifest.views) != set(views_map):
        warn("manifest_views", "manifest.json 'views' differs from files in views/")

    vdir = root / "videos"
    on_disk = {p.name for p in vdir.iterdir() if p.is_dir()} if vdir.exists() else set()
    for vid in sorted(set(manifest.videos) - on_disk):
        err("missing_video_dir", f"video '{vid}' is in the manifest but videos/{vid}/ does not exist")
    for vid in sorted(on_disk - set(manifest.videos)):
        warn("orphan_video_dir", f"videos/{vid}/ exists but is not listed in manifest.json")

    all_ids: list[np.ndarray] = []
    for vid in sorted(set(manifest.videos) & on_disk):
        mp = vdir / vid / "metadata.json"
        if not mp.exists():
            err("no_metadata", f"videos/{vid}/metadata.json is missing")
            continue
        try:
            meta = VideoMetadata.model_validate_json(mp.read_text(encoding="utf-8"))
        except (PydanticValidationError, ValueError) as e:
            err("bad_metadata", f"videos/{vid}/metadata.json: {e}")
            continue
        if meta.video_id != vid:
            err("video_id_mismatch", f"videos/{vid}/metadata.json has video_id '{meta.video_id}'")
        metas[vid] = meta
        ap = vdir / vid / "annotations.parquet"
        if not ap.exists():
            continue
        try:
            table = pq.read_table(ap)
        except Exception as e:  # corrupt file
            err("bad_parquet", f"videos/{vid}/annotations.parquet unreadable: {e}")
            continue
        for i in check_annotation_table(table, frame_count=meta.frame_count or None, size=meta.display_size,
                                        vocab_names=vocab_names, views=views_map, video_id=vid, deep_masks=deep):
            i.message = f"{vid}: {i.message}"
            issues.append(i)
        if "video_id" in table.column_names and table.num_rows:
            other = pc.invert(pc.equal(table.column("video_id"), vid))
            if pc.any(pc.fill_null(other, False)).as_py():
                err("video_id_column", f"videos/{vid}/annotations.parquet has rows for other videos")
        if "annotation_id" in table.column_names:
            all_ids.append(_np(table, "annotation_id", dtype=np.int64))
    if all_ids:
        cat = np.concatenate(all_ids)
        u, c = np.unique(cat, return_counts=True)
        if (c > 1).any():
            issues.append(Issue("error", "duplicate_id_across_videos", "annotation_id repeated across videos",
                                int((c > 1).sum()), [int(x) for x in u[c > 1][:5]]))
    for v in view_files:
        try:
            vw = View.model_validate_json(v.read_text(encoding="utf-8"))
        except Exception:
            continue
        if vw.video_id not in metas:
            err("view_video", f"view '{vw.view_id}' refers to unknown video '{vw.video_id}'")
        elif vw.region:
            w, h = metas[vw.video_id].display_size
            if vw.region.x + vw.region.width > w or vw.region.y + vw.region.height > h:
                err("view_region", f"view '{vw.view_id}' region extends outside the {w}x{h} video")
    return issues
