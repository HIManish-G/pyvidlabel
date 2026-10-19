import json

import pytest

from conftest import blob_mask, make_meta
from vidlabel import Dataset, ValidationError, masks


def test_create_open(tmp_path):
    d = Dataset.create(tmp_path / "x")
    assert Dataset.open(tmp_path / "x").video_ids() == []
    with pytest.raises(FileExistsError):
        Dataset.create(tmp_path / "x")
    with pytest.raises(FileNotFoundError):
        Dataset.open(tmp_path / "nope")


def test_unified_rows_roundtrip(ds):
    ds.add_annotations([
        {"kind": "marker", "label": "cut", "start_frame": 10, "end_frame": 40, "origin": "imported", "tool": "some-converter", "extra": {"mytool": {"name": "x"}}},
        {"kind": "box", "label": "person", "start_frame": 5, "box": (1, 2, 20, 30), "track_id": 0, "origin": "model",
         "tool": "yolo", "tool_version": "8", "confidence": 0.9},
        {"kind": "mask", "label": "car", "start_frame": 7, "mask": blob_mask(), "origin": "human", "author": "a", "status": "approved"},
    ], "vid1")
    t = ds.annotations("vid1")
    assert t.num_rows == 3
    assert t.column("start_frame").to_pylist() == [5, 7, 10]  # sorted
    row = {k: v[0] for k, v in ds.annotations("vid1", kind="box").to_pydict().items()}
    assert row["end_frame"] == 5 and row["origin"] == "model" and row["x1"] == 20.0
    m = ds.annotations("vid1", kind="mask").to_pylist()[0]
    assert m["mask_codec"] == "bitpack_zstd"
    assert (m["x0"], m["y0"]) == masks.bbox(m["mask"])[:2]  # box derived from mask
    assert json.loads(ds.annotations("vid1", kind="marker").to_pylist()[0]["extra"])["mytool"]["name"] == "x"
    assert ds.validate(deep=True) == []


def test_queries(ds):
    ds.add_annotations([{"kind": "marker", "label": "cut", "start_frame": s, "end_frame": e, "origin": "human"}
                        for s, e in [(0, 5), (10, 20), (30, 30)]], "vid1")
    assert ds.annotations("vid1", frames=(15, 25)).num_rows == 1
    assert ds.annotations("vid1", frames=(5, 10)).num_rows == 2  # touches both ends
    assert ds.annotations(label="car").num_rows == 0
    assert ds.annotations("vid1", columns=["start_frame"]).column_names == ["start_frame"]
    assert ds.annotations().num_rows == 3


@pytest.mark.parametrize("row,code", [
    ({"kind": "marker", "label": "nope", "start_frame": 1, "origin": "human"}, "unknown_label"),
    ({"kind": "marker", "label": "cut", "start_frame": 500, "origin": "human"}, "frame_out_of_range"),
    ({"kind": "marker", "label": "cut", "start_frame": 9, "end_frame": 3, "origin": "human"}, "frame_order"),
    ({"kind": "box", "label": "car", "start_frame": 1, "origin": "human"}, "missing_box"),
    ({"kind": "box", "label": "car", "start_frame": 1, "box": (5, 5, 2, 9), "origin": "human"}, "bad_box"),
    ({"kind": "box", "label": "car", "start_frame": 1, "end_frame": 4, "box": (1, 1, 2, 2), "origin": "human"}, "geometry_span"),
    ({"kind": "marker", "label": "cut", "start_frame": 1, "origin": "human", "confidence": 3}, "bad_confidence"),
    ({"kind": "marker", "label": "cut", "start_frame": 1, "origin": "human", "view_id": "ghost"}, "unknown_view"),
    ({"kind": "marker", "label": "cut", "start_frame": 1, "origin": "human", "extra": "[1]"}, "bad_extra"),
])
def test_bad_rows_rejected_and_nothing_written(ds, row, code):
    with pytest.raises(ValidationError) as e:
        ds.add_annotations([row], "vid1")
    assert code in {i.code for i in e.value.issues}
    assert ds.annotations("vid1").num_rows == 0


def test_bad_enum_and_unknown_field(ds):
    with pytest.raises(ValueError):
        ds.add_annotations([{"kind": "blob", "label": "cut", "start_frame": 1, "origin": "human"}], "vid1")
    with pytest.raises(ValueError, match="extra"):
        ds.add_annotations([{"kind": "marker", "label": "cut", "start_frame": 1, "origin": "human", "colour": "red"}], "vid1")


def test_box_outside_frame_is_warning_only(ds):
    ds.add_annotations([{"kind": "box", "label": "car", "start_frame": 1, "box": (-3, 0, 100, 10), "origin": "human"}], "vid1")
    assert {i.code for i in ds.validate()} == {"box_outside_frame"}


def test_create_labels(ds):
    ds.add_annotations([{"kind": "marker", "label": "brand_new", "start_frame": 1, "origin": "human"}], "vid1", create_labels=True)
    assert "brand_new" in {l.name for l in ds.vocab.labels}


def test_update_and_delete(ds):
    ids = ds.add_annotations([{"kind": "marker", "label": "cut", "start_frame": i, "origin": "model"} for i in range(4)], "vid1")
    assert ds.update_annotations("vid1", ids[:2], status="approved", label="highlight") == 2
    t = ds.annotations("vid1")
    assert sorted(t.column("status").to_pylist()) == ["approved", "approved", "unreviewed", "unreviewed"]
    with pytest.raises(ValueError):
        ds.update_annotations("vid1", ids, annotation_id=5)
    with pytest.raises(ValidationError):
        ds.update_annotations("vid1", ids, label="not_in_vocab")
    assert ds.delete_annotations("vid1", ids[:3]) == 3
    assert ds.annotations("vid1").num_rows == 1


def test_views_and_rotation(tmp_path):
    d = Dataset.create(tmp_path / "r")
    d.add_video(make_meta("rot", w=64, h=48, rotation=90))  # displayed as 48 wide, 64 tall
    d.add_labels(["a"])
    d.add_view({"view_id": "v1", "video_id": "rot", "region": {"x": 0, "y": 0, "width": 40, "height": 60}})
    with pytest.raises(ValueError):
        d.add_view({"view_id": "v2", "video_id": "rot", "region": {"x": 0, "y": 0, "width": 60, "height": 40}})
    d.add_annotations([{"kind": "box", "label": "a", "start_frame": 0, "box": (0, 0, 40, 60), "origin": "human", "view_id": "v1"}], "rot")
    assert d.validate() == []


def test_resolve_video_path(ds, tmp_path):
    (tmp_path / "moved").mkdir()
    (tmp_path / "moved" / "vid1.mp4").write_bytes(b"x")
    assert ds.resolve_video_path("vid1") is None
    assert ds.resolve_video_path("vid1", roots=[tmp_path / "moved"]).name == "vid1.mp4"


def test_stats(ds):
    ds.add_annotations([{"kind": "marker", "label": "cut", "start_frame": 1, "origin": "human"}], "vid1")
    s = ds.stats()
    assert s["annotations"] == 1 and s["by_origin"] == {"human": 1}
