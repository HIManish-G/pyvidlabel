from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .dataset import Dataset

app = typer.Typer(no_args_is_help=True, add_completion=False, help="vidlabel: video annotation datasets.")


@app.command()
def init(path: Path):
    """Create an empty dataset."""
    Dataset.create(path)
    typer.echo(f"created {path}")


@app.command()
def info(path: Path):
    """Show what is in a dataset."""
    s = Dataset.open(path).stats()
    typer.echo(f"videos: {s['videos']}   views: {s['views']}   annotations: {s['annotations']}   vocab: {s['labels_in_vocab']}")
    for key in ("by_kind", "by_origin", "by_status"):
        typer.echo(f"{key[3:]:>8}: " + ", ".join(f"{k}={v}" for k, v in sorted(s[key].items())))
    if s["by_view"]:
        typer.echo("   views: " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_view"].items())))
    for v in s["video_details"]:
        fps = f"{v['fps']:.3f}".rstrip("0").rstrip(".")
        br = f"  {v['bit_rate'] / 1000:.0f} kb/s" if v["bit_rate"] else ""
        note = "  (no annotations yet)" if not v["annotations"] else ""
        typer.echo(f"  video {v['video_id']}: {v['width']}x{v['height']}  {fps} fps  {v['codec'] or '?'}{br}  "
                   f"{v['frame_count']} frames  {v['annotations']} annotations{note}")
    top = sorted(s["by_label"].items(), key=lambda kv: -kv[1])[:10]
    if top:
        typer.echo("  labels: " + ", ".join(f"{k}={v}" for k, v in top))


@app.command()
def validate(path: Path, deep: bool = typer.Option(False, help="Also decode every mask and cross-check its box.")):
    """Check a dataset against the spec. Exit code 1 if there are errors."""
    issues = Dataset.open(path).validate(deep=deep)
    for i in issues:
        typer.echo(str(i))
    n_err = sum(i.level == "error" for i in issues)
    typer.echo(f"{n_err} error(s), {len(issues) - n_err} warning(s)")
    raise typer.Exit(1 if n_err else 0)


@app.command()
def build(path: Path, include_rejected: bool = typer.Option(False, help="Also put rejected annotations in the release.")):
    """Compile the sorted training table; prints the release id to pin."""
    statuses = ("unreviewed", "approved", "rejected") if include_rejected else ("unreviewed", "approved")
    rel, out = Dataset.open(path).build(statuses=statuses)
    typer.echo(f"{rel.release_id}  rows={rel.n_rows} videos={rel.n_videos}  {out}")


@app.command("add-video")
def add_video(path: Path, video_file: Path,
              video_id: Optional[str] = typer.Option(None, help="Id to register it under. Default: the file name without extension."),
              no_hash: bool = typer.Option(False, "--no-hash", help="Skip hashing the file (faster, weaker identity).")):
    """Probe a video file with ffprobe and register it."""
    from .probe import probe_video
    meta = Dataset.open(path).add_video(probe_video(video_file, video_id, hash=not no_hash))
    typer.echo(f"added {meta.video_id}: {meta.width}x{meta.height}, {meta.frame_count} frames")


def _ints(text: str, n: int, what: str) -> list[int]:
    try:
        vals = [int(x) for x in text.split(",")]
    except ValueError:
        vals = []
    if len(vals) != n:
        raise typer.BadParameter(f"{what} must be {n} comma-separated integers")
    return vals


@app.command("add-labels")
def add_labels(path: Path, labels: list[str]):
    """Add labels to the shared vocabulary."""
    added = Dataset.open(path).add_labels(labels)
    typer.echo(f"added {len(added)} label(s): {', '.join(added) or '(all already existed)'}")


@app.command("add-view")
def add_view(path: Path, view_id: str, video_id: str,
             region: Optional[str] = typer.Option(None, help="x,y,width,height in displayed pixels"),
             frames: Optional[str] = typer.Option(None, help="first,last frame (inclusive)"),
             output_size: Optional[str] = typer.Option(None, help="width,height the crop is resized to, if resized"),
             labels: Optional[str] = typer.Option(None, help="comma-separated labels allowed in this view")):
    """Define a crop / time range of a video."""
    spec: dict = {"view_id": view_id, "video_id": video_id}
    if region:
        x, y, w, h = _ints(region, 4, "--region")
        spec["region"] = {"x": x, "y": y, "width": w, "height": h}
    if frames:
        spec["frame_range"] = tuple(_ints(frames, 2, "--frames"))
    if output_size:
        spec["output_size"] = tuple(_ints(output_size, 2, "--output-size"))
    if labels:
        spec["allowed_labels"] = [l.strip() for l in labels.split(",") if l.strip()]
    v = Dataset.open(path).add_view(spec)
    typer.echo(f"added view {v.view_id} on {v.video_id}")


def _resolve_video(ds: Dataset, video_id: str, video_path: Optional[Path], roots: list[Path]) -> Path:
    p = video_path or ds.resolve_video_path(video_id, roots)
    if p is None:
        raise typer.BadParameter(f"cannot find the video for '{video_id}'; pass --video-path or --root")
    return Path(p)


@app.command()
def preview(path: Path, video_id: str, out: Path = typer.Option(Path("preview_out"), help="Output folder."),
            frames: Optional[str] = typer.Option(None, help="e.g. '0-100,250'. Default: a spread of annotated frames."),
            every: int = typer.Option(1, help="With --frames, keep every Nth frame."),
            view: Optional[str] = typer.Option(None, help="Show this view's crop and only its annotations."),
            mp4: bool = typer.Option(False, help="Write one MP4 instead of PNG images."),
            video_path: Optional[Path] = typer.Option(None, help="Video file, if the recorded path is stale."),
            root: list[Path] = typer.Option([], help="Folder(s) to search for the video by file name."),
            color: Optional[str] = typer.Option(None, help="Box colour: 'red', '#ff8800' or '255,136,0' for all labels, "
                                                            "or per label: 'person=red;car=#00ff00'. Default: one colour per label."),
            thickness: Optional[int] = typer.Option(None, help="Box line width in pixels. Default: scales with the video."),
            font_scale: Optional[float] = typer.Option(None, help="Label text size, e.g. 0.8. Default: scales with the video."),
            labels: bool = typer.Option(True, "--labels/--no-labels", help="Write label text next to boxes."),
            outline: bool = typer.Option(True, "--outline/--no-outline", help="Dark border around boxes and text.")):
    """Draw the annotations on frames of the real video, to check they line up. Needs OpenCV (the video extra)."""
    from .preview import PreviewStyle, count_drawn, parse_colors, parse_frames, render_preview
    ds = Dataset.open(path)
    meta = ds.video(video_id)
    chosen = parse_frames(frames, meta.frame_count)[::max(every, 1)] if frames else None
    try:
        default_color, label_colors = parse_colors(color) if color else (None, {})
        style = PreviewStyle(color=default_color, label_colors=label_colors, thickness=thickness, font_scale=font_scale,
                             show_labels=labels, outline=outline)
        written = render_preview(ds, video_id, out, frames=chosen, video_path=_resolve_video(ds, video_id, video_path, root),
                                 view_id=view, mp4=mp4, style=style)
    except (ImportError, FileNotFoundError, ValueError, IndexError) as e:
        typer.echo(f"[error] {e}")
        raise typer.Exit(1)
    typer.echo(f"wrote {len(written)} file(s) to {out}")
    rows = ds.annotations(video_id, view_id=view).to_pylist()
    shown = chosen if chosen is not None else None
    if shown is not None:
        n, busy = count_drawn(rows, shown)
        typer.echo(f"{n} of {len(rows)} annotation(s) are active on the {len(shown)} frame(s) shown ({busy} frame(s) have any)")
        if rows and not n:
            first = sorted({r["start_frame"] for r in rows})[:8]
            typer.echo(f"[note] none of the shown frames has an annotation. Annotated frames start at: {first}. "
                       f"With --every N only every Nth frame is drawn, so a single-frame box can be skipped.")
    elif not rows:
        typer.echo("[note] this video has no annotations" + (f" in view '{view}'" if view else "") + ", so frames are plain.")


@app.command("check-video")
def check_video_cmd(path: Path, video_id: str, video_path: Optional[Path] = typer.Option(None, help="Video file, if the recorded path is stale."),
                    root: list[Path] = typer.Option([], help="Folder(s) to search for the video by file name."),
                    hash: bool = typer.Option(False, help="Also verify the content hash (reads the whole file)."),
                    seek: bool = typer.Option(False, help="Also test that random-access seeking returns exactly the "
                                                          "right frames (decodes the whole video once).")):
    """Check that the video file on disk matches the metadata registered for it."""
    from .frames import check_video, seek_mismatches
    ds = Dataset.open(path)
    try:
        problems = check_video(_resolve_video(ds, video_id, video_path, root), ds.video(video_id), verify_hash=hash)
    except (ImportError, FileNotFoundError, ValueError) as e:
        typer.echo(f"[error] {e}")
        raise typer.Exit(1)
    for p in problems:
        typer.echo(f"[error] {p}")
    typer.echo("ok: file matches the metadata" if not problems else f"{len(problems)} problem(s)")
    if seek and not problems:
        meta = ds.video(video_id)
        n = meta.frame_count
        positions = sorted({int(n * q / 40) for q in range(40)} | {n - 1})
        try:
            bad = seek_mismatches(_resolve_video(ds, video_id, video_path, root), meta, positions)
        except (ImportError, FileNotFoundError, ValueError) as e:
            typer.echo(f"[error] {e}")
            raise typer.Exit(1)
        if bad or meta.is_vfr:
            why = "the file is variable frame rate" if meta.is_vfr else f"frames {bad[:6]} differ"
            typer.echo(f"[note] seeking is not frame-exact for this file ({why}). Random access will decode from the "
                       f"start and be slow; VideoStream is unaffected. Re-encoding as constant frame rate fixes it.")
        else:
            typer.echo(f"ok: seeking is frame-exact at {len(positions)} positions, so random access is fast")
    raise typer.Exit(1 if problems else 0)


@app.command()
def schema(out: Optional[Path] = typer.Option(None, help="Write the JSON Schema to this file instead of printing it.")):
    """Print (or write) the JSON Schema of the spec."""
    from vidlabel_spec import json_schema, write_json_schema
    if out:
        write_json_schema(out)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(json.dumps(json_schema(), indent=2))


if __name__ == "__main__":
    app()
