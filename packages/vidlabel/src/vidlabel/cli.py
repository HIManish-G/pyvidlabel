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
def build(path: Path, include_rejected: bool = False):
    """Compile the sorted training table; prints the release id to pin."""
    statuses = ("unreviewed", "approved", "rejected") if include_rejected else ("unreviewed", "approved")
    rel, out = Dataset.open(path).build(statuses=statuses)
    typer.echo(f"{rel.release_id}  rows={rel.n_rows} videos={rel.n_videos}  {out}")


@app.command("add-video")
def add_video(path: Path, video_file: Path, video_id: Optional[str] = None,
              no_hash: bool = typer.Option(False, help="Skip hashing the file (faster, weaker identity).")):
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


@app.command()
def schema(out: Optional[Path] = None):
    """Print (or write) the JSON Schema of the spec."""
    from vidlabel_spec import json_schema, write_json_schema
    if out:
        write_json_schema(out)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(json.dumps(json_schema(), indent=2))


if __name__ == "__main__":
    app()
