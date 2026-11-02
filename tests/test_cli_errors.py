"""The mistakes a user is most likely to make (wrong path, not a video, duplicate name, bad numbers) give one `[error]` line
and exit code 1, never a Python traceback. Found by running the CLI as a new user would, on real files."""
import shutil
import warnings

import pytest
from typer.testing import CliRunner

from vidlabel import Dataset, FrameSamples, VideoMetadata
from vidlabel.cli import app

runner = CliRunner()
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")


def run(*args):
    r = runner.invoke(app, [str(a) for a in args])
    assert "Traceback (most recent call last)" not in r.output, r.output
    return r


@pytest.fixture
def ds(tmp_path):
    d = Dataset.create(tmp_path / "ds")
    d.add_video(VideoMetadata(video_id="clip", frame_count=100, fps={"num": 25, "den": 1}, width=64, height=48))
    return d


def test_not_a_dataset(tmp_path):
    for cmd in ("info", "build"):
        r = run(cmd, tmp_path / "nothing")
        assert r.exit_code == 1 and r.output.startswith("[error]") and "vidlabel init" in r.output
    r = run("validate", tmp_path / "nothing")          # reported as the documented issue, not raised
    assert r.exit_code == 1 and "no_manifest" in r.output and "1 error(s)" in r.output


def test_deleted_manifest_is_reported_by_validate(ds):
    (ds.path / "manifest.json").unlink()
    r = run("validate", ds.path)
    assert r.exit_code == 1 and "no_manifest" in r.output


def test_init_twice(ds):
    r = run("init", ds.path)
    assert r.exit_code == 1 and "already contains a dataset" in r.output


@needs_ffmpeg
def test_add_video_problems(ds, tmp_path):
    (tmp_path / "fake.mp4").write_text("not a video")
    r = run("add-video", ds.path, tmp_path / "fake.mp4")
    assert r.exit_code == 1 and "ffprobe could not read 'fake.mp4' as a video" in r.output
    r = run("add-video", ds.path, tmp_path / "missing.mp4")
    assert r.exit_code == 1 and "no such video file" in r.output
    assert ds.video_ids() == ["clip"]


@needs_ffmpeg
def test_add_video_duplicate_id_says_what_to_do(ds, tmp_path):
    from test_frames import make_video
    f = make_video(tmp_path / "a.mp4", n=30)
    assert run("add-video", ds.path, f, "--video-id", "a").exit_code == 0
    r = run("add-video", ds.path, f, "--video-id", "a")
    assert r.exit_code == 1 and "already exists" in r.output and "--video-id" in r.output


def test_add_labels_splits_commas_and_rejects_nothing(ds):
    r = run("add-labels", ds.path, "fly,Fly, ,fly", "goal")
    assert r.exit_code == 0 and "added 3 label(s): fly, Fly, goal" in r.output
    assert {l.name for l in Dataset.open(ds.path).vocab.labels} == {"fly", "Fly", "goal"}
    assert run("add-labels", ds.path, " , ").exit_code != 0


def test_add_view_mistakes(ds):
    cases = [(("v", "clip", "--region", "10,10,0,5"), "greater than 0"),
             (("v", "clip", "--output-size", "100,100"), "needs a region"),
             (("v", "ghost"), "unknown video 'ghost'"),
             (("v", "clip", "--region", "40,0,60,10"), "extends outside"),
             (("v", "clip", "--frames", "50,500"), "beyond the video")]
    for args, text in cases:
        r = run("add-view", ds.path, *args)
        assert r.exit_code == 1 and r.output.startswith("[error]") and text in r.output, (args, r.output)
    assert Dataset.open(ds.path).manifest.views == []


@needs_ffmpeg
def test_preview_frames_that_select_nothing(ds, tmp_path):
    from test_frames import make_video
    f = make_video(tmp_path / "p.mp4", n=30)
    ds.add_video({**ds.video("clip").model_dump(), "video_id": "p", "frame_count": 30})
    for spec, text in (("5-1", "select none"), ("500-600", "select none"), ("abc", "cannot read frames")):
        r = run("preview", ds.path, "p", "--frames", spec, "--out", tmp_path / "out", "--video-path", f)
        assert r.exit_code == 1 and text in r.output, (spec, r.output)


def test_empty_release_is_noted(ds):
    r = run("build", ds.path)
    assert r.exit_code == 0 and "rows=0" in r.output and "no annotations" in r.output


def test_debug_variable_restores_the_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDLABEL_DEBUG", "1")
    r = runner.invoke(app, ["info", str(tmp_path / "nothing")])
    assert isinstance(r.exception, FileNotFoundError)


def test_view_that_selects_nothing_warns(ds):
    ds.add_labels(["thing"])
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 32, "height": 24}})
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 3, "box": (1, 1, 9, 9), "origin": "human"}], "clip")
    with pytest.warns(RuntimeWarning, match="view 'crop' selects no samples.*whole frame"):
        assert len(FrameSamples(ds, release=None, view_id="crop")) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # no warning when the view does select rows, or when none is asked for
        FrameSamples(ds, release=None)
        ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 4, "box": (1, 1, 9, 9), "origin": "human",
                             "view_id": "crop"}], "clip")
        assert len(FrameSamples(ds, release=None, view_id="crop")) == 1


# ---- is this the video the labels were made on? ----
@needs_ffmpeg
def test_check_video_looks_at_the_file_it_is_given(tmp_path):
    import subprocess
    from test_frames import make_video
    src = make_video(tmp_path / "a.mp4", n=60)
    d = Dataset.create(tmp_path / "ds")
    assert run("add-video", d.path, src).exit_code == 0
    base = run("check-video", d.path, "a")
    assert base.exit_code == 0 and "content hash was not checked" in base.output
    regop = tmp_path / "b.mp4"                        # same frames and timing, re-encoded with a keyframe every 5 frames
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-g", "5", "-keyint_min", "5", str(regop)], check=True)
    r = run("check-video", d.path, "a", "--video-path", regop)
    assert r.exit_code == 0 and "keyframes differ from the stored index" in r.output and "every 5 frames" in r.output
    assert run("check-video", d.path, "a", "--video-path", regop, "--hash").exit_code == 1
    short = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(src), "-c:v", "libx264", str(short)], check=True)
    r = run("check-video", d.path, "a", "--video-path", short)
    assert r.exit_code == 1 and "metadata says 60" in r.output


@needs_ffmpeg
def test_reader_refuses_a_file_of_another_size(tmp_path):
    from test_frames import make_video
    from vidlabel.frames import FrameReader
    from vidlabel.probe import probe_video
    meta = probe_video(make_video(tmp_path / "a.mp4", n=60), "a")
    other = make_video(tmp_path / "o.mp4", n=60, w=128, h=80)
    with pytest.raises(ValueError, match="this is not the file that was registered"):
        with FrameReader(other, meta) as r:
            r.read(3)


@needs_ffmpeg
def test_reader_warns_about_a_trimmed_copy(tmp_path):
    import subprocess
    from test_frames import make_video
    from vidlabel.frames import FrameReader
    from vidlabel.probe import probe_video
    src = make_video(tmp_path / "a.mp4", n=60)
    meta = probe_video(src, "a")
    short = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(src), "-c:v", "libx264", str(short)], check=True)
    with pytest.warns(RuntimeWarning, match="has about \d+ frames but 60 were registered"):
        with FrameReader(short, meta) as r:
            r.read(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # the registered file itself is silent
        with FrameReader(src, meta) as r:
            r.read(3)


def test_wrong_size_mask_is_rejected_when_written(ds):
    import numpy as np
    with pytest.raises(ValueError, match="mask is 20x10 but video 'clip' is 64x48"):
        ds.add_annotations([{"kind": "mask", "label": "thing", "start_frame": 1, "mask": np.ones((10, 20), bool), "origin": "human"}],
                           "clip", create_labels=True)
    assert len(ds.annotations("clip")) == 0


def test_update_box_message_and_info_on_empty_dataset(ds):
    with pytest.raises(ValueError, match="set x0, y0, x1, y1"):
        ds.update_annotations("clip", [1], box=(1, 2, 3, 4))
    r = run("info", ds.path)
    assert r.exit_code == 0 and "kind:" not in r.output


def test_validation_errors_say_which_frame_and_track(ds):
    from vidlabel import ValidationError
    ds.add_labels(["thing"])
    rows = [{"kind": "box", "label": "thing", "start_frame": 500, "box": (1, 1, 9, 9), "track_id": 7, "origin": "human"}]
    with pytest.raises(ValidationError, match=r"frame_out_of_range.*at frame 500 \(track 7\)"):
        ds.add_annotations(rows, "clip")


def test_unknown_release_lists_the_ones_that_exist(ds):
    ds.add_labels(["thing"])
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 1, "box": (1, 1, 9, 9), "origin": "human"}], "clip")
    rel, _ = ds.build()
    with pytest.raises(FileNotFoundError, match=f"releases here: {rel.release_id}"):
        FrameSamples(ds, release="rdoesnotexist")


@needs_ffmpeg
def test_add_video_notes_a_file_that_is_already_registered(ds, tmp_path):
    from test_frames import make_video
    f = make_video(tmp_path / "a.mp4", n=30)
    assert "already registered" not in run("add-video", ds.path, f).output
    r = run("add-video", ds.path, f, "--video-id", "again")
    assert r.exit_code == 0 and "already registered as 'a'" in r.output


# ---- found by the earth-observation user test ----
def test_extra_takes_numpy_numbers_and_refuses_nan(ds):
    import numpy as np
    ds.add_labels(["thing"])
    row = {"kind": "box", "label": "thing", "start_frame": 1, "box": (1, 1, 9, 9), "origin": "model",
           "extra": {"mean_k": np.float32(2.5), "cells": np.int64(7), "lat": np.array([1.0, 2.0])}}
    ds.add_annotations([row], "clip")
    import json
    got = json.loads(ds.annotations("clip").to_pylist()[0]["extra"])
    assert got == {"mean_k": 2.5, "cells": 7, "lat": [1.0, 2.0]}
    with pytest.raises(ValueError, match="NaN or infinity"):
        ds.add_annotations([{**row, "start_frame": 2, "extra": {"x": float("nan")}}], "clip")
    with pytest.raises(ValueError, match="extra must be JSON"):
        ds.add_annotations([{**row, "start_frame": 3, "extra": {"x": object()}}], "clip")
    assert len(ds.annotations("clip")) == 1


def test_filter_typos_raise_instead_of_returning_nothing(ds):
    with pytest.raises(ValueError, match="unknown kind \\['boxx'\\]"):
        ds.annotations("clip", kind="boxx")
    with pytest.raises(ValueError, match="unknown status \\['maybe'\\]"):
        ds.annotations("clip", status=["approved", "maybe"])


def test_a_view_never_samples_other_videos(ds):
    ds.add_video(VideoMetadata(video_id="other", frame_count=20, fps={"num": 25, "den": 1}, width=64, height=48))
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 32, "height": 24}})
    s = FrameSamples(ds, release=None, view_id="crop", frames="all")
    assert s.video_ids() == ["clip"] and len(s) == 100
    assert FrameSamples(ds, release=None, frames="all").video_ids() == ["clip", "other"]


def test_a_release_with_rejected_rows_does_not_become_latest(ds):
    ds.add_labels(["thing"])
    ids = ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (1, 1, 9, 9), "origin": "model"} for f in (1, 2)], "clip")
    first, _ = ds.build()
    ds.update_annotations("clip", ids[:1], status="rejected")
    inspect, _ = ds.build(statuses=("unreviewed", "approved", "rejected"))
    assert inspect.release_id != first.release_id
    assert FrameSamples(ds, release="latest", kinds=("box",)).release_id == first.release_id
    r = run("build", ds.path, "--include-rejected")
    assert "'latest' was not moved" in r.output
