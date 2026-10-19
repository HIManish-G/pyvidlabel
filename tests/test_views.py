import json
import subprocess

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import make_meta
from vidlabel import Compiled, Dataset, ValidationError, VideoMetadata, View, masks
from vidlabel.cli import app


def _codes(exc):
    return {i.code for i in exc.value.issues}


@pytest.fixture
def two_views(tmp_path):
    """1280x720 clip with a 'upper' view (top) and a 'lower' view (bottom strip)."""
    d = Dataset.create(tmp_path / "c")
    d.add_video(make_meta("clip", frames=300, w=1280, h=720, fps=(25, 1)))
    d.add_video(make_meta("other", frames=100, w=1280, h=720, fps=(25, 1)))
    d.add_labels(["car", "person", "caption_text", "logo", "event"])
    d.add_view({"view_id": "upper", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 1280, "height": 620},
                "allowed_labels": ["car", "person"]})
    d.add_view({"view_id": "lower", "video_id": "clip", "region": {"x": 0, "y": 620, "width": 1280, "height": 100},
                "allowed_labels": ["caption_text", "logo"]})
    return d


# ---------- View math ----------
def test_view_box_conversion_offset_and_scale():
    v = View(view_id="t", video_id="v", region={"x": 100, "y": 620, "width": 400, "height": 100})
    assert v.to_source_box((10, 20, 50, 60)) == (110, 640, 150, 680)
    assert v.to_view_box((110, 640, 150, 680)) == (10, 20, 50, 60)
    s = View(view_id="t", video_id="v", region={"x": 100, "y": 620, "width": 400, "height": 100}, output_size=(200, 50))
    assert s.scale == (0.5, 0.5)
    assert s.to_source_box((10, 10, 20, 20)) == (120, 640, 140, 660)  # undo 2x downscale, then offset
    back = s.to_view_box(s.to_source_box((3, 4, 9, 11)))
    assert np.allclose(back, (3, 4, 9, 11))
    assert View(view_id="w", video_id="v").to_source_box((1, 2, 3, 4)) == (1, 2, 3, 4)  # no region: identity


def test_view_spec_rules():
    with pytest.raises(ValueError):
        View(view_id="a", video_id="v", output_size=(10, 10))  # needs a region
    with pytest.raises(ValueError):
        View(view_id="a", video_id="v", frame_range=(-1, 5))
    with pytest.raises(ValueError):
        View(view_id="a", video_id="v", frame_range=(9, 5))


# ---------- coords="view" ----------
def test_two_pipelines_boxes_land_in_source_coords(two_views):
    two_views.add_annotations([{"kind": "box", "label": "caption_text", "start_frame": 5, "box": (40, 30, 200, 70),
                              "origin": "model", "view_id": "lower"}], "clip", coords="view")
    two_views.add_annotations([{"kind": "box", "label": "car", "start_frame": 5, "box": (500, 300, 520, 320),
                              "origin": "model", "view_id": "upper"}], "clip", coords="view")
    t = two_views.annotations("clip", view_id="lower").to_pylist()[0]
    assert (t["x0"], t["y0"], t["x1"], t["y1"]) == (40, 650, 200, 690)
    b = two_views.annotations("clip", view_id="upper").to_pylist()[0]
    assert (b["x0"], b["y0"]) == (500, 300)
    assert two_views.validate() == []


def test_scaled_view_boxes(two_views):
    two_views.add_view({"view_id": "lower_small", "video_id": "clip", "output_size": (640, 50),
                      "region": {"x": 0, "y": 620, "width": 1280, "height": 100}})
    two_views.add_annotations([{"kind": "box", "label": "event", "start_frame": 1, "box": (10, 5, 100, 25),
                              "origin": "model", "view_id": "lower_small"}], "clip", coords="view")
    r = two_views.annotations("clip").to_pylist()[0]
    assert (r["x0"], r["y0"], r["x1"], r["y1"]) == (20, 630, 200, 670)


def test_view_mask_pasted_and_resized(two_views):
    m = np.zeros((100, 1280), bool)
    m[10:30, 100:140] = True
    two_views.add_annotations([{"kind": "mask", "label": "caption_text", "start_frame": 2, "mask": m, "origin": "model",
                              "view_id": "lower"}], "clip", coords="view")
    row = two_views.annotations("clip", kind="mask").to_pylist()[0]
    full = masks.decode(row["mask"])
    assert full.shape == (720, 1280) and full[630:650, 100:140].all() and full.sum() == 800
    assert (row["x0"], row["y0"], row["x1"], row["y1"]) == (100, 630, 140, 650)
    assert two_views.validate(deep=True) == []

    two_views.add_view({"view_id": "lower_half", "video_id": "clip", "output_size": (640, 50),
                      "region": {"x": 0, "y": 620, "width": 1280, "height": 100}, "allowed_labels": ["event"]})
    small = np.zeros((50, 640), bool)
    small[5:15, 50:70] = True
    two_views.add_annotations([{"kind": "mask", "label": "event", "start_frame": 3, "mask": small, "origin": "model",
                              "view_id": "lower_half"}], "clip", coords="view")
    r2 = two_views.annotations("clip", label="event").to_pylist()[0]
    assert (r2["x0"], r2["y0"], r2["x1"], r2["y1"]) == (100, 630, 140, 650)


def test_view_coords_errors(two_views):
    row = {"kind": "box", "label": "car", "start_frame": 1, "box": (1, 1, 5, 5), "origin": "human"}
    with pytest.raises(ValueError, match="view_id"):
        two_views.add_annotations([row], "clip", coords="view")
    with pytest.raises(ValueError, match="unknown view"):
        two_views.add_annotations([{**row, "view_id": "nope"}], "clip", coords="view")
    with pytest.raises(ValueError, match="encoded"):
        two_views.add_annotations([{**row, "box": None, "view_id": "upper", "mask": b"xx"}], "clip", coords="view")
    with pytest.raises(ValueError, match="shape"):
        two_views.add_annotations([{**row, "box": None, "view_id": "upper", "mask": np.ones((3, 3), bool)}], "clip", coords="view")
    with pytest.raises(ValueError):
        two_views.add_annotations([row], "clip", coords="bogus")
    assert two_views.annotations().num_rows == 0
    # markers need no view when converting
    two_views.add_annotations([{"kind": "marker", "label": "event", "start_frame": 4, "origin": "human"}], "clip", coords="view")
    assert two_views.annotations().num_rows == 1


# ---------- inside-view validation ----------
@pytest.mark.parametrize("row,code", [
    # a lower box that wanders into the upper area
    ({"kind": "box", "label": "caption_text", "start_frame": 1, "box": (10, 500, 100, 700), "view_id": "lower"}, "box_outside_view"),
    # label that belongs to the other pipeline
    ({"kind": "box", "label": "car", "start_frame": 1, "box": (10, 640, 30, 660), "view_id": "lower"}, "label_not_allowed_in_view"),
    ({"kind": "marker", "label": "event", "start_frame": 1, "view_id": "upper"}, "label_not_allowed_in_view"),
])
def test_inside_view_checks(two_views, row, code):
    with pytest.raises(ValidationError) as e:
        two_views.add_annotations([{"origin": "human", **row}], "clip")
    assert code in _codes(e)
    assert two_views.annotations().num_rows == 0


def test_frame_range_and_wrong_video(two_views):
    two_views.add_view({"view_id": "early", "video_id": "clip", "frame_range": [0, 50]})
    with pytest.raises(ValidationError) as e:
        two_views.add_annotations([{"kind": "marker", "label": "event", "start_frame": 40, "end_frame": 60,
                                  "origin": "human", "view_id": "early"}], "clip")
    assert "frame_outside_view" in _codes(e)
    with pytest.raises(ValidationError) as e:
        two_views.add_annotations([{"kind": "marker", "label": "event", "start_frame": 1, "origin": "human",
                                  "view_id": "upper"}], "other")
    assert "view_wrong_video" in _codes(e)


def test_view_definition_checks(two_views):
    with pytest.raises(ValueError, match="allowed_labels"):
        two_views.add_view({"view_id": "bad", "video_id": "clip", "allowed_labels": ["not_in_vocab"]})
    with pytest.raises(ValueError, match="frame_range"):
        two_views.add_view({"view_id": "bad", "video_id": "clip", "frame_range": [0, 999]})
    # a hand-edited view file that references a missing label is caught by the dataset validator
    p = two_views.path / "views" / "upper.json"
    d = json.loads(p.read_text())
    d["allowed_labels"] = ["car", "ghost"]
    p.write_text(json.dumps(d))
    assert "view_labels" in {i.code for i in two_views.validate()}


# ---------- filtering / training ----------
def _fill_two_pipelines(c):
    c.add_annotations([{"kind": "box", "label": "car", "start_frame": f, "box": (5, 5, 20, 20), "origin": "model",
                        "view_id": "upper"} for f in range(0, 100, 2)], "clip", coords="view")
    c.add_annotations([{"kind": "box", "label": "caption_text", "start_frame": f, "box": (5, 5, 50, 30), "origin": "model",
                        "view_id": "lower"} for f in range(0, 100, 10)], "clip", coords="view")
    c.add_annotations([{"kind": "marker", "label": "event", "start_frame": 40, "end_frame": 45, "origin": "human"}], "clip")


def test_view_filter(two_views):
    _fill_two_pipelines(two_views)
    assert two_views.annotations("clip", view_id="upper").num_rows == 50
    assert two_views.annotations("clip", view_id="lower").num_rows == 10
    assert two_views.annotations("clip", view_id=["upper", "lower"]).num_rows == 60
    assert two_views.annotations("clip", view_id=[None]).num_rows == 1
    assert two_views.annotations("clip", view_id=["lower", None]).num_rows == 11
    assert two_views.annotations("clip").num_rows == 61
    assert two_views.annotations("clip", view_id="upper", frames=(0, 9)).num_rows == 5


def test_build_per_view_and_clip_view(two_views):
    _fill_two_pipelines(two_views)
    rel_all, p_all = two_views.build(row_group_size=20)
    rel_t, p_t = two_views.build(view_ids=["lower"])
    assert rel_all.n_rows == 61 and rel_t.n_rows == 10 and rel_all.release_id != rel_t.release_id
    assert rel_t.extra["view_ids"] == ["lower"]
    assert set(Compiled(p_t).read().column("view_id").to_pylist()) == {"lower"}
    c = Compiled(p_all)
    assert c.clip("clip", 0, 99, view_id="lower").num_rows == 10
    assert c.clip("clip", 0, 99, columns=["label"], view_id="upper").column_names == ["label"]
    assert c.clip("clip", 40, 45).num_rows == c.clip("clip", 40, 45, view_id=None).num_rows


def test_update_annotation_still_checks_views(two_views):
    ids = two_views.add_annotations([{"kind": "box", "label": "caption_text", "start_frame": 1, "box": (1, 650, 9, 690),
                                    "origin": "model", "view_id": "lower"}], "clip")
    with pytest.raises(ValidationError):
        two_views.update_annotations("clip", ids, label="car")  # not allowed in the lower view


# ---------- bit_rate / probe / info / CLI ----------
def test_bit_rate_field_and_probe_parsing(monkeypatch, tmp_path):
    base = dict(video_id="a", frame_count=1, fps={"num": 1, "den": 1}, width=2, height=2)
    assert VideoMetadata(**base).bit_rate is None
    assert VideoMetadata(**base, bit_rate=2_500_000).bit_rate == 2_500_000
    with pytest.raises(Exception):
        VideoMetadata(**base, bit_rate=-1)

    from vidlabel import probe

    f = tmp_path / "x.mp4"
    f.write_bytes(b"data")
    fake = {"streams": [{"width": 1280, "height": 720, "avg_frame_rate": "25/1", "r_frame_rate": "25/1", "nb_frames": "550",
                         "duration": "22.0", "codec_name": "h264", "pix_fmt": "yuv420p", "time_base": "1/12800",
                         "bit_rate": "1834000"}], "format": {"bit_rate": "1900000"}}
    monkeypatch.setattr(probe.shutil, "which", lambda n: "/usr/bin/ffprobe")
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(fake), ""))
    assert probe.probe_video(f).bit_rate == 1_834_000
    fake["streams"][0].pop("bit_rate")  # fall back to the container's figure
    assert probe.probe_video(f).bit_rate == 1_900_000
    fake["format"].pop("bit_rate")
    assert probe.probe_video(f).bit_rate is None


def test_info_lists_videos_and_cli_views(tmp_path):
    r = CliRunner()
    p = str(tmp_path / "d")
    assert r.invoke(app, ["init", p]).exit_code == 0
    d = Dataset.open(p)
    d.add_video(VideoMetadata(video_id="test1", frame_count=550, fps={"num": 25, "den": 1}, width=1280, height=720,
                              codec="h264", bit_rate=1_834_000))
    out = r.invoke(app, ["info", p]).output
    assert "video test1: 1280x720  25 fps  h264  1834 kb/s  550 frames  0 annotations  (no annotations yet)" in out
    assert r.invoke(app, ["add-labels", p, "car", "caption"]).exit_code == 0
    res = r.invoke(app, ["add-view", p, "lower", "test1", "--region", "0,620,1280,100", "--frames", "0,100",
                         "--output-size", "640,50", "--labels", "caption"])
    assert res.exit_code == 0, res.output
    v = d.view("lower")
    assert v.region.y == 620 and v.output_size == (640, 50) and v.allowed_labels == ["caption"]
    assert r.invoke(app, ["add-view", p, "bad", "test1", "--region", "1,2,3"]).exit_code != 0
    d.add_annotations([{"kind": "box", "label": "caption", "start_frame": 1, "box": (1, 1, 5, 5), "origin": "human",
                        "view_id": "lower"}], "test1", coords="view")
    out = r.invoke(app, ["info", p]).output
    assert "lower=1" in out and "1 annotations" in out and "(no annotations yet)" not in out
    assert r.invoke(app, ["validate", p, "--deep"]).exit_code == 0


def test_datasets_written_by_0_1_still_validate(tmp_path):
    """A 0.1.0 dataset has no bit_rate, output_size or allowed_labels and an older spec_version."""
    d = Dataset.create(tmp_path / "old")
    (d.path / "manifest.json").write_text(json.dumps({"spec_version": "0.1.0", "videos": ["v"], "views": ["half"], "extra": {}}))
    (d.path / "videos" / "v").mkdir(parents=True)
    (d.path / "videos" / "v" / "metadata.json").write_text(json.dumps(
        {"video_id": "v", "frame_count": 10, "fps": {"num": 25, "den": 1}, "width": 64, "height": 48, "rotation": 0,
         "is_vfr": False, "sources": [], "extra": {}}))
    (d.path / "views" / "half.json").write_text(json.dumps(
        {"view_id": "half", "video_id": "v", "region": {"x": 0, "y": 0, "width": 32, "height": 48}, "extra": {}}))
    assert Dataset.open(d.path).validate() == []
    d.add_labels(["a"])
    d.add_annotations([{"kind": "box", "label": "a", "start_frame": 0, "box": (1, 1, 9, 9), "origin": "human",
                        "view_id": "half"}], "v")
    assert Dataset.open(d.path).validate(deep=True) == []
