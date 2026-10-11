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


@app.command("import-premiere")
def import_premiere(path: Path, video_id: str, markers_csv: Path,
                    start_tc: str = "00:00:00:00", label_from: str = "name", author: Optional[str] = None,
                    default_label: Optional[str] = None,
                    create_labels: bool = typer.Option(False, help="Add unseen labels to vocab.json.")):
    """Import a Premiere Pro marker export (format not yet verified against a real file)."""
    from .importers.premiere import import_markers
    ids = import_markers(Dataset.open(path), video_id, markers_csv, start_tc=start_tc, label_from=label_from,
                         author=author, default_label=default_label, create_labels=create_labels)
    typer.echo(f"imported {len(ids)} markers")


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
