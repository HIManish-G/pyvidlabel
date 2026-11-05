"""Slow or multi-step operations as functions that return plain data, for the web interface's jobs.

They mirror the matching commands in `vidlabel.cli` (add-video, index, check-video, cache, validate, build, preview) but return
dictionaries instead of printing, and report progress through a `JobContext`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..dataset import Dataset
from .jobs import JobContext


def _issue(i) -> dict:
    return {"level": i.level, "code": i.code, "message": i.message, "count": i.count}


def validate(ds: Dataset, *, deep: bool = False, ctx: JobContext | None = None) -> dict:
    from ..validate import validate_dataset
    ctx and ctx.progress(None, "checking every table" + (" and decoding every mask" if deep else ""))
    issues = validate_dataset(ds.path, deep=deep)
    return {"deep": deep, "ok": not any(i.level == "error" for i in issues), "issues": [_issue(i) for i in issues]}


def build(ds: Dataset, *, statuses: Sequence[str] | None = None, origins: Sequence[str] | None = None,
          kinds: Sequence[str] | None = None, video_ids: Sequence[str] | None = None, view_ids: Sequence[str | None] | None = None,
          ctx: JobContext | None = None) -> dict:
    kw: dict[str, Any] = {}
    for k, v in (("statuses", statuses), ("origins", origins), ("kinds", kinds), ("video_ids", video_ids), ("view_ids", view_ids)):
        if v:
            kw[k] = list(v)
    ctx and ctx.progress(None, "validating and compiling")
    rel, path = ds.build(**kw)
    notes = []
    if "rejected" in kw.get("statuses", ()):
        notes.append("this release includes rejected rows, so 'latest' was not moved to it")
    if rel.n_rows == 0:
        notes.append("the release is empty")
    return {"release_id": rel.release_id, "n_rows": rel.n_rows, "path": str(path), "notes": notes}


def add_video(ds: Dataset, file: str | Path, *, video_id: str | None = None, hash: bool = True, index: bool = True,
              ctx: JobContext | None = None) -> dict:
    from ..probe import probe_frame_index, probe_video, reconcile_frame_count
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"no file at {file}")
    ctx and ctx.progress(None, "reading the video's facts" + (" and hashing the file" if hash else ""))
    meta, idx, notes = probe_video(file, video_id, hash=hash), None, []
    if index:
        ctx and ctx.check()
        ctx and ctx.progress(None, "reading frame timestamps and keyframes")
        try:
            idx = probe_frame_index(file, content_hash=meta.content_hash)
            meta = reconcile_frame_count(meta, idx)
        except ValueError as e:
            notes.append(f"no frame index: {e}")
    same = [v for v in ds.video_ids() if meta.content_hash and ds.video(v).content_hash == meta.content_hash and v != meta.video_id]
    if same:
        notes.append(f"this file is already registered as '{same[0]}'; adding it again makes a second video")
    try:
        meta = ds.add_video(meta, frame_index=idx)
    except FileExistsError as e:
        raise FileExistsError(f"{e}; give this one another name") from None
    out = {"video_id": meta.video_id, "width": meta.width, "height": meta.height, "frame_count": meta.frame_count, "notes": notes}
    if idx is not None:
        out["keyframe_median"] = idx.interval_stats()["median"]
    return out


def index_videos(ds: Dataset, video_ids: Sequence[str] | None = None, *, force: bool = False, roots: Sequence[str] = (),
                 verify: bool = True, ctx: JobContext | None = None) -> dict:
    from ..cli import _resolve_video
    from ..probe import file_hash, probe_frame_index, reconcile_frame_count
    ids = list(video_ids or [v for v in ds.video_ids() if force or ds.frame_index(v) is None])
    results = []
    for n, vid in enumerate(ids):
        ctx and ctx.check()
        ctx and ctx.progress(n / max(len(ids), 1), f"indexing {vid}")
        try:
            meta = ds.video(vid)
            file = _resolve_video(ds, vid, None, [Path(r) for r in roots])
            if meta.content_hash and verify and file_hash(file) != meta.content_hash:
                raise ValueError(f"{file.name} is not the file that was registered (hash differs)")
            index = probe_frame_index(file, content_hash=meta.content_hash)
            fixed = reconcile_frame_count(meta, index)
            if fixed is not meta:
                ds.add_video(fixed, exist_ok=True)
            ds.set_frame_index(vid, index)
            st = index.interval_stats()
            results.append({"video_id": vid, "ok": True, "frames": index.frame_count, "keyframe_median": st["median"],
                            "keyframe_max": st["max"]})
        except (KeyError, FileNotFoundError, ValueError, RuntimeError) as e:
            results.append({"video_id": vid, "ok": False, "error": str(e)})
    return {"results": results, "nothing_to_do": not ids}


def check_video(ds: Dataset, video_id: str, *, video_path: str | None = None, roots: Sequence[str] = (), hash: bool = False,
                seek: bool = False, ctx: JobContext | None = None) -> dict:
    from ..cli import _LONG_GOP, _resolve_video, _same_times, reencode_command
    from ..frames import check_video as _check, seek_mismatches
    from ..probe import probe_frame_index
    file = _resolve_video(ds, video_id, Path(video_path) if video_path else None, [Path(r) for r in roots])
    meta = ds.video(video_id)
    ctx and ctx.progress(None, "comparing the file with the metadata" + (" and hashing it" if hash else ""))
    problems = list(_check(file, meta, verify_hash=hash))
    notes: list[str] = []
    stored = ds.frame_index(video_id)
    index = None
    try:
        index = probe_frame_index(file, content_hash=meta.content_hash)
    except (ValueError, RuntimeError) as e:
        notes.append(f"could not read the file's keyframes and timestamps: {e}")
    if index is not None and stored is not None and not _same_times(stored, index):
        problems.append("frame timestamps differ from the stored frame index: this file is a different encoding, or was re-timed")
    if not problems and not hash and meta.content_hash:
        notes.append("the content hash was not checked: a re-encode with the same frames, size and timing would pass; "
                     "run the check with the hash to confirm it is the identical file")
    if stored is None and index is not None:
        notes.append("this video has no stored frame index; index it to seek in variable-frame-rate video and to plan reads")
    elif stored is not None and index is not None and stored.keyframes != index.keyframes:
        notes.append("this file's keyframes differ from the stored index: it was re-encoded")
    keyframes = None
    if index is not None:
        st = index.interval_stats()
        keyframes = {"median": st["median"], "min": st["min"], "max": st["max"], "mean_cost": st["mean_cost"]}
        if st["mean_cost"] > _LONG_GOP:
            notes.append("keyframes are far apart, so random access is slow. To make it faster, re-encode with a keyframe every "
                         f"{_LONG_GOP} frames: {reencode_command(meta, file.name)}")
    seek_result = None
    if seek:
        ctx and ctx.check()
        ctx and ctx.progress(None, "testing random access (decodes the video once)")
        n = meta.frame_count
        positions = sorted({int(n * q / 40) for q in range(40)} | {n - 1})
        bad = seek_mismatches(file, meta, positions, index=stored)
        exact = not bad and not (meta.is_vfr and stored is None)
        seek_result = {"exact": exact, "positions": len(positions), "mismatches": bad[:20], "used_index": stored is not None}
    return {"ok": not problems, "file": str(file), "problems": problems, "notes": notes, "keyframes": keyframes, "seek": seek_result}


def build_cache(ds: Dataset, out: str | Path, *, release: str = "latest", view_id: str | None = None, frames: str = "annotated",
                format: str = "jpg", quality: int = 95, roots: Sequence[str] = (), overwrite: bool = False,
                ctx: JobContext | None = None) -> dict:
    from ..cache import FrameCache
    from ..samples import FrameSamples
    ctx and ctx.progress(None, "choosing frames")
    samples = FrameSamples(ds, release=None if release == "none" else release, view_id=view_id, frames=frames,
                           roots=[Path(r) for r in roots])
    ctx and ctx.progress(None, f"decoding {len(samples)} frames")
    _, st = FrameCache.build(samples, Path(out), format=format, quality=quality, overwrite=overwrite)
    return {"written": st.written, "skipped": st.skipped, "seconds": round(st.seconds, 1), "megabytes": round(st.bytes / 1e6, 1),
            "out": str(out), "notes": ["jpg is lossy: pixels differ slightly from the video; use png for exact pixels"] if format == "jpg" else []}


def export_preview(ds: Dataset, video_id: str, out_dir: str | Path, *, frames: Sequence[int] | None = None, view_id: str | None = None,
                   mp4: bool = False, roots: Sequence[str] = (), ctx: JobContext | None = None) -> dict:
    from ..preview import render_preview
    ctx and ctx.progress(None, "drawing annotations on frames")
    files = render_preview(ds, video_id, out_dir, frames=frames, view_id=view_id, mp4=mp4, roots=[Path(r) for r in roots])
    return {"files": [Path(f).name for f in files], "out": str(out_dir)}
