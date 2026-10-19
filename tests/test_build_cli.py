import json

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import blob_mask, make_meta
from vidlabel import Compiled, Dataset, ValidationError
from vidlabel.cli import app
from vidlabel_spec import json_schema, VideoMetadata


def _fill(ds, n=400):
    rng = np.random.default_rng(0)
    for vid in ds.video_ids():
        ds.add_annotations([{"kind": "box", "label": "car", "start_frame": int(f), "box": (1, 1, 9, 9), "origin": "model",
                             "track_id": 0} for f in rng.choice(190, size=min(n, 150), replace=False)], vid)
        ds.add_annotations([{"kind": "marker", "label": "cut", "start_frame": 20, "end_frame": 60, "origin": "human"},
                            {"kind": "mask", "label": "person", "start_frame": 3, "mask": blob_mask(), "origin": "human",
                             "status": "rejected"}], vid)


def test_build_excludes_rejected_and_pins_release(ds):
    _fill(ds)
    rel, path = ds.build()
    c = Compiled(path)
    assert rel.release_id == c.release_id and rel.n_videos == 2
    t = c.read()
    assert "rejected" not in set(t.column("status").to_pylist())
    assert t.num_rows == 2 * 151
    assert t.sort_by([("video_id", "ascending"), ("start_frame", "ascending"), ("annotation_id", "ascending")]).equals(t)
    rel2, _ = ds.build()  # same data -> same id
    assert rel2.release_id == rel.release_id
    ds.add_annotations([{"kind": "marker", "label": "cut", "start_frame": 99, "origin": "human"}], "vid1")
    assert ds.build()[0].release_id != rel.release_id
    assert ds.build(statuses=("rejected",))[0].n_rows == 2


def test_clip_matches_bruteforce_and_prunes(ds):
    _fill(ds)
    _, path = ds.build(row_group_size=40)
    c = Compiled(path / "..")  # compiled/ dir resolves LATEST
    full = c.read().to_pylist()
    for vid, a, b in [("vid1", 0, 5), ("vid2", 100, 120), ("vid1", 55, 70), ("vid2", 0, 199), ("zzz", 0, 9)]:
        got = sorted(r["annotation_id"] for r in c.clip(vid, a, b).to_pylist())
        want = sorted(r["annotation_id"] for r in full if r["video_id"] == vid and r["start_frame"] <= b and r["end_frame"] >= a)
        assert got == want
    c.clip("vid2", 100, 101)
    assert len(c.last_row_groups_read) < c._pf.num_row_groups
    assert c.clip("vid1", 0, 5, columns=["label"]).column_names == ["label"]


def test_build_refuses_invalid_dataset(ds):
    _fill(ds, 5)
    (ds.path / "vocab.json").write_text(json.dumps({"version": "0", "labels": []}))
    with pytest.raises(ValidationError):
        ds.build()


def test_validate_detects_tampering(ds):
    _fill(ds, 5)
    assert ds.validate(deep=True) == []
    (ds.path / "videos" / "vid2" / "metadata.json").write_text("{not json")
    assert "bad_metadata" in {i.code for i in ds.validate()}


def test_cli_smoke(tmp_path):
    r = CliRunner()
    p = str(tmp_path / "d")
    assert r.invoke(app, ["init", p]).exit_code == 0
    ds = Dataset.open(p)
    ds.add_video(make_meta("a"))
    ds.add_labels(["x"])
    ds.add_annotations([{"kind": "marker", "label": "x", "start_frame": 1, "origin": "human"}], "a")
    out = r.invoke(app, ["info", p])
    assert out.exit_code == 0 and "annotations: 1" in out.output
    assert r.invoke(app, ["validate", p, "--deep"]).exit_code == 0
    assert r.invoke(app, ["build", p]).exit_code == 0
    (tmp_path / "d" / "vocab.json").unlink()
    assert r.invoke(app, ["validate", p]).exit_code == 1


def test_spec_schema_and_strictness():
    assert {"manifest", "video_metadata", "annotation_columns"} <= set(json_schema())
    base = dict(video_id="a", frame_count=1, fps={"num": 1, "den": 1}, width=2, height=2)
    VideoMetadata(**base)
    for bad in ({"colour": 1}, {"rotation": 45}, {"video_id": "a b"}, {"content_hash": "XYZ"}, {"width": 0}):
        with pytest.raises(Exception):
            VideoMetadata(**{**base, **bad})
    assert VideoMetadata(**{**base, "rotation": 90, "width": 10}).display_size == (2, 10)
