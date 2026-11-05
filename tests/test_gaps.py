"""Gaps found by the user tests: mask size in plain validate, track-level review, deriving view rows, several frame ranges per
view, no silent no-ops, and a readable preview."""
import numpy as np
import pytest

from vidlabel import Dataset, FrameSamples, ValidationError, VideoMetadata
from vidlabel import masks as masklib
from vidlabel.validate import validate_dataset


@pytest.fixture
def ds(tmp_path):
    d = Dataset.create(tmp_path / "ds")
    d.add_video(VideoMetadata(video_id="clip", frame_count=100, fps={"num": 25, "den": 1}, width=64, height=48))
    d.add_labels(["thing", "other"])
    return d


def codes(path, deep=False):
    return {i.code for i in validate_dataset(path, deep=deep)}


# ---- 1. mask size in plain validate ----
def test_plain_validate_sees_a_mask_made_for_another_frame_size(ds):
    import pyarrow as pa
    import pyarrow.parquet as pq
    m = np.zeros((48, 64), bool)
    m[5:20, 5:20] = True
    ds.add_annotations([{"kind": "mask", "label": "thing", "start_frame": 3, "mask": m, "origin": "human"}], "clip")
    assert codes(ds.path) == set()
    # a blob encoded for a 128x96 frame, written behind the library's back (as another tool might)
    big = np.zeros((96, 128), bool)
    big[5:20, 5:20] = True
    p = ds.path / "videos" / "clip" / "annotations.parquet"
    t = pq.read_table(p)
    rows = t.to_pylist()
    rows[0]["mask"] = masklib.encode(big)
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), p)
    assert "mask_frame_size" in codes(ds.path)                 # no --deep needed
    assert "mask_frame_size" in {i.code for i in Dataset.open(ds.path).validate()}


# ---- silent no-ops ----
def _boxes(ds, frames, **kw):
    return ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (1, 1, 9, 9), "origin": "model", **kw}
                               for f in frames], "clip")


def test_unknown_ids_and_empty_updates_are_errors(ds):
    ids = _boxes(ds, [1, 2, 3])
    with pytest.raises(KeyError, match="no annotation with id.*999"):
        ds.update_annotations("clip", [ids[0], 999], status="approved")
    assert {r["status"] for r in ds.annotations("clip").to_pylist()} == {"unreviewed"}       # nothing changed
    with pytest.raises(KeyError):
        ds.delete_annotations("clip", [999])
    assert len(ds.annotations("clip")) == 3
    assert ds.delete_annotations("clip", [999, ids[0]], missing="ignore") == 1
    assert ds.update_annotations("clip", [999], status="approved", missing="ignore") == 0
    with pytest.raises(ValueError, match="at least one column"):
        ds.update_annotations("clip", ids[1:])
    with pytest.raises(ValueError, match="status"):
        ds.update_annotations("clip", [999], status="maybe", missing="ignore")                # the typo is caught, not a quiet 0
    with pytest.raises(ValueError, match="missing must be"):
        ds.delete_annotations("clip", ids, missing="skip")


def test_an_unknown_video_is_an_error_on_reads_and_writes(ds):
    with pytest.raises(KeyError, match="unknown video 'clpi'"):
        ds.annotations("clpi")
    with pytest.raises(KeyError):
        ds.delete_annotations("clpi", [1])
    assert len(ds.annotations()) == 0


# ---- tracks ----
def test_track_level_filter_summary_and_review(ds):
    ds.add_view({"view_id": "v", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 32, "height": 24}})
    for f in range(10):
        _boxes(ds, [f], track_id=1, confidence=0.9)
        _boxes(ds, [f], track_id=2, confidence=0.4, label="thing")
    _boxes(ds, [5], track_id=1, view_id="v")                                  # track 1 of the view is another object
    t = ds.tracks("clip")
    assert [r["track_id"] for r in t] == [1, 2] and t[0]["n_rows"] == 10 and t[0]["first_frame"] == 0 and t[0]["last_frame"] == 9
    assert abs(t[1]["mean_confidence"] - 0.4) < 1e-6 and t[0]["statuses"] == {"unreviewed": 10}
    assert len(ds.annotations("clip", track_id=2, view_id=[None])) == 10
    assert len(ds.annotations("clip", track_id=[1, 2])) == 21
    assert ds.update_track("clip", 2, status="rejected") == 10
    assert ds.tracks("clip")[1]["statuses"] == {"rejected": 10}
    assert ds.tracks("clip")[0]["statuses"] == {"unreviewed": 10}
    assert ds.tracks("clip", view_id="v")[0]["statuses"] == {"unreviewed": 1}                   # the view's track 1 untouched
    assert ds.delete_track("clip", 1) == 10 and len(ds.annotations("clip", view_id=["v"])) == 1
    with pytest.raises(KeyError, match="no track 9 in the whole frame"):
        ds.update_track("clip", 9, status="approved")
    with pytest.raises(KeyError, match="in view 'v'"):
        ds.update_track("clip", 2, view_id="v", status="approved")


def test_track_commands(ds):
    from typer.testing import CliRunner
    from vidlabel.cli import app
    for f in range(5):
        _boxes(ds, [f], track_id=3, confidence=0.5)
    r = CliRunner().invoke(app, ["tracks", str(ds.path), "clip"])
    assert r.exit_code == 0 and "track     3" in r.output and "unreviewed=5" in r.output
    r = CliRunner().invoke(app, ["review", str(ds.path), "clip", "--track", "3", "--status", "approved"])
    assert r.exit_code == 0 and "set 5 row(s) of 1 track(s) to approved" in r.output
    r = CliRunner().invoke(app, ["review", str(ds.path), "clip", "--track", "8", "--status", "approved"])
    assert r.exit_code == 1 and r.output.startswith("[error]") and "no track 8" in r.output


# ---- several frame ranges per view ----
def test_view_model_accepts_ranges_and_refuses_bad_ones():
    from vidlabel_spec import View
    v = View(view_id="w", video_id="clip", frame_ranges=[(0, 9), (20, 29)])
    assert v.ranges == [(0, 9), (20, 29)] and v.contains_frames(22, 25) and not v.contains_frames(8, 21)
    assert View(view_id="w", video_id="clip", frame_range=(3, 5)).ranges == [(3, 5)]
    assert View(view_id="w", video_id="clip").ranges is None and View(view_id="w", video_id="clip").contains_frames(0, 99)
    for bad, text in (({"frame_ranges": [(0, 9), (5, 12)]}, "overlap"), ({"frame_ranges": [(20, 29), (0, 9)]}, "increasing"),
                      ({"frame_ranges": []}, "must not be empty"), ({"frame_ranges": [(9, 3)]}, "first <= last"),
                      ({"frame_range": (0, 5), "frame_ranges": [(7, 9)]}, "not both")):
        with pytest.raises(ValueError, match=text):
            View(view_id="w", video_id="clip", **bad)


def test_view_with_two_windows_end_to_end(ds, tmp_path):
    ds.add_view({"view_id": "w", "video_id": "clip", "frame_ranges": [[0, 9], [50, 59]]})
    with pytest.raises(ValueError, match="beyond the video"):
        ds.add_view({"view_id": "bad", "video_id": "clip", "frame_ranges": [[0, 9], [90, 120]]})
    _boxes(ds, [0, 5, 9, 55], view_id="w")
    assert codes(ds.path) == set()
    ds.add_annotations([{"kind": "marker", "label": "thing", "start_frame": 7, "end_frame": 52, "origin": "human"}], "clip")
    marker_row = {"kind": "marker", "label": "thing", "start_frame": 7, "end_frame": 52, "origin": "human", "view_id": "w"}
    with pytest.raises(ValidationError, match="frame_outside_view.*ranges"):       # spans two windows: not inside one
        ds.add_annotations([marker_row], "clip")
    with pytest.raises(ValidationError, match="frame_outside_view"):
        _boxes(ds, [30], view_id="w")
    s = FrameSamples(ds, release=None, view_id="w", kinds=("box",))
    assert s.frames_of("clip") == [0, 5, 9, 55]
    assert FrameSamples(ds, release=None, view_id="w", frames="all").frames_of("clip") == list(range(10)) + list(range(50, 60))


def test_add_view_command_repeats_frames(ds):
    from typer.testing import CliRunner
    from vidlabel.cli import app
    r = CliRunner().invoke(app, ["add-view", str(ds.path), "winters", "clip", "--frames", "0,9", "--frames", "50,59"])
    assert r.exit_code == 0, r.output
    assert Dataset.open(ds.path).view("winters").ranges == [(0, 9), (50, 59)]
    r = CliRunner().invoke(app, ["add-view", str(ds.path), "one", "clip", "--frames", "5,8"])
    assert r.exit_code == 0 and Dataset.open(ds.path).view("one").frame_range == (5, 8)
    r = CliRunner().invoke(app, ["add-view", str(ds.path), "bad", "clip", "--frames", "5,8", "--frames", "6,9"])
    assert r.exit_code == 1 and "overlap" in r.output


# ---- derive view rows ----
def test_derive_view_rows(ds):
    from vidlabel_spec import Region
    ds.add_labels(["event"])
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 32, "height": 24},
                 "frame_ranges": [[0, 9], [50, 59]], "allowed_labels": ["thing", "event"]})
    m = np.zeros((48, 64), bool)
    m[2:10, 2:10] = True
    ds.add_annotations([
        {"kind": "box", "label": "thing", "start_frame": 1, "box": (1, 1, 9, 9), "track_id": 4, "origin": "model", "confidence": 0.7},   # in
        {"kind": "box", "label": "thing", "start_frame": 2, "box": (20, 10, 40, 20), "origin": "model"},                                    # partly out
        {"kind": "box", "label": "thing", "start_frame": 3, "box": (40, 30, 60, 40), "origin": "model"},                                    # outside
        {"kind": "box", "label": "other", "start_frame": 4, "box": (1, 1, 9, 9), "origin": "model"},                                        # label
        {"kind": "box", "label": "thing", "start_frame": 30, "box": (1, 1, 9, 9), "origin": "model"},                                       # frames
        {"kind": "box", "label": "thing", "start_frame": 6, "box": (1, 1, 9, 9), "origin": "model", "status": "rejected"},                 # rejected
        {"kind": "mask", "label": "thing", "start_frame": 5, "mask": m, "origin": "human"},                                                 # in (mask)
        {"kind": "marker", "label": "event", "start_frame": 52, "end_frame": 55, "origin": "human"},                                         # in (marker)
    ], "clip")
    r = ds.derive_view_rows("clip", "crop")
    assert r["copied"] == 3 and (r["skipped_partial"], r["skipped_outside"], r["skipped_label"], r["skipped_frames"]) == (1, 1, 1, 1)
    copies = ds.annotations("clip", view_id=["crop"]).to_pylist()
    box = next(c for c in copies if c["kind"] == "box")
    assert box["track_id"] == 4 and abs(box["confidence"] - 0.7) < 1e-6 and box["origin"] == "model"
    import json
    assert json.loads(box["extra"])["derived_from"] in {x["annotation_id"] for x in ds.annotations("clip", view_id=[None]).to_pylist()}
    assert {c["kind"] for c in copies} == {"box", "mask", "marker"} and len(ds.annotations("clip", view_id=[None])) == 8
    assert codes(ds.path, deep=True) == set()
    again = ds.derive_view_rows("clip", "crop")                       # nothing new: safe to repeat
    assert again["copied"] == 0 and again["already_derived"] == 3
    s = FrameSamples(ds, release=None, view_id="crop", kinds=("box", "mask", "marker"))
    assert s.frames_of("clip") == [1, 5, 52, 53, 54, 55]
    with pytest.raises(ValueError, match="belongs to video"):
        ds.add_video(VideoMetadata(video_id="b", frame_count=5, fps={"num": 25, "den": 1}, width=64, height=48))
        ds.derive_view_rows("b", "crop")


def test_derive_view_command(ds):
    from typer.testing import CliRunner
    from vidlabel.cli import app
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 32, "height": 24}})
    _boxes(ds, [1, 2])
    r = CliRunner().invoke(app, ["derive-view", str(ds.path), "clip", "crop"])
    assert r.exit_code == 0 and "copied 2 row(s) into view crop" in r.output
    r = CliRunner().invoke(app, ["derive-view", str(ds.path), "clip", "crop"])
    assert "copied 0 row(s)" in r.output and "already there 2" in r.output


# ---- preview polish ----
def _row(**kw):
    base = dict(kind="box", label="thing", start_frame=0, end_frame=0, track_id=None, confidence=None, status="approved",
                x0=20, y0=20, x1=80, y1=70, mask=None)
    base.update(kw)
    return base


def test_preview_rejected_rows_look_different():
    pytest.importorskip("cv2")
    from vidlabel.preview import draw_annotations, PreviewStyle
    frame = np.full((120, 200, 3), 90, np.uint8)
    st = PreviewStyle(color=(255, 0, 0), show_labels=False)
    ok = draw_annotations(frame, [_row()], 0, style=st)
    rej = draw_annotations(frame, [_row(status="rejected")], 0, style=st)
    assert (ok != rej).any()
    top_ok = ok[20, 20:80]; top_rej = rej[20, 20:80]
    solid = lambda row: (row != 90).any(axis=1).mean()
    assert solid(top_ok) > 0.9 and 0.2 < solid(top_rej) < 0.8      # dashed has gaps


def test_preview_labels_stay_inside_and_do_not_overprint():
    pytest.importorskip("cv2")
    from vidlabel.preview import draw_annotations, PreviewStyle, _place_label
    cv2 = pytest.importorskip("cv2")
    # a box in the top-left corner: the label cannot go above it, and must still be inside the frame
    x, y = _place_label((60, 12), 3, (0, 0, 40, 40), [], (120, 200))
    assert 0 <= x and x + 60 <= 200 and y - 12 - 2 >= 0 and y + 3 <= 120
    # a box at the right edge: the label is moved left to fit
    x, y = _place_label((60, 12), 3, (190, 50, 200, 90), [], (120, 200))
    assert x + 60 <= 200
    # two boxes with the same top-left: second label avoids the first
    first = _place_label((60, 12), 3, (30, 50, 90, 90), [], (120, 200))
    taken = [(first[0], first[1] - 14, first[0] + 60, first[1] + 5)]
    second = _place_label((60, 12), 3, (30, 50, 90, 90), taken, (120, 200))
    assert second != first
    r2 = (second[0], second[1] - 14, second[0] + 60, second[1] + 5)
    assert not (r2[0] < taken[0][2] and r2[2] > taken[0][0] and r2[1] < taken[0][3] and r2[3] > taken[0][1])
    out = draw_annotations(np.full((120, 200, 3), 90, np.uint8), [_row(x0=0, y0=0, x1=30, y1=30), _row(x0=0, y0=0, x1=30, y1=30, track_id=1)], 0)
    assert out.shape == (120, 200, 3)


# ---- ids cannot point outside their folder ----
@pytest.mark.parametrize("bad", [".", "..", "a/b", "a\\b", "a\n", ""])
def test_ids_cannot_escape_the_dataset_folder(tmp_path, bad):
    from vidlabel import Dataset, VideoMetadata
    ds = Dataset.create(tmp_path / "d")
    with pytest.raises(Exception):
        ds.add_video(VideoMetadata(video_id=bad, frame_count=2, fps={"num": 1, "den": 1}, width=4, height=4))
    assert sorted(p.name for p in (tmp_path / "d").iterdir()) == ["manifest.json", "videos", "views", "vocab.json"]
    assert list((tmp_path / "d" / "videos").iterdir()) == []
    for call in (lambda: ds.video(bad), lambda: ds.annotations(bad), lambda: ds.view(bad)):
        with pytest.raises((KeyError, ValueError)):
            call()


def test_a_manifest_naming_a_path_is_reported_not_followed(tmp_path):
    import json
    from vidlabel import Dataset
    ds = Dataset.create(tmp_path / "d")
    m = json.loads((tmp_path / "d" / "manifest.json").read_text())
    m["videos"] = ["../../etc"]
    (tmp_path / "d" / "manifest.json").write_text(json.dumps(m))
    issues = Dataset.open(tmp_path / "d").validate()
    assert [i.code for i in issues] == ["bad_manifest"]
