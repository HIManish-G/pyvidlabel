"""The frame index as data: the model, storing it in a dataset, and validating it. (Using it to read video is in test_frame_index_video.py.)"""
import json

import pytest
from pydantic import ValidationError

from conftest import make_meta
from vidlabel import Dataset
from vidlabel.validate import validate_dataset
from vidlabel_spec import FrameIndex

TB = {"num": 1, "den": 1000}


def idx(n=100, step=40, keys=(0, 25, 50, 75), **kw):
    return FrameIndex(timebase=TB, pts=[i * step for i in range(n)], keyframes=list(keys), **kw)


def test_model_rejects_inconsistent_data():
    with pytest.raises(ValidationError, match="strictly increasing"):
        FrameIndex(timebase=TB, pts=[0, 40, 40], keyframes=[0])
    with pytest.raises(ValidationError, match="strictly increasing"):
        FrameIndex(timebase=TB, pts=[0, 40, 80], keyframes=[0, 0])
    with pytest.raises(ValidationError, match="inside the video"):
        FrameIndex(timebase=TB, pts=[0, 40, 80], keyframes=[0, 3])
    with pytest.raises(ValidationError):
        FrameIndex(timebase=TB, pts=[], keyframes=[])
    with pytest.raises(ValidationError):
        FrameIndex(timebase=TB, pts=[0], keyframes=[0], surprise=1)   # unknown fields are rejected like everywhere else


def test_lookups_and_costs():
    i = idx()
    assert i.frame_count == 100
    assert [i.keyframe_before(f) for f in (0, 24, 25, 74, 99)] == [0, 0, 25, 50, 75]
    assert [i.decode_cost(f) for f in (0, 24, 25, 99)] == [1, 25, 1, 25]
    assert i.frame_at(40.0) == 1 and i.frame_at(40.2) == 1 and i.frame_at(60.0) is None and i.frame_at(-5) is None
    assert i.seconds(10) == pytest.approx(0.4)
    st = i.interval_stats()
    assert (st["min"], st["median"], st["max"]) == (25, 25, 25)
    assert st["mean_cost"] == pytest.approx(13.0)         # intervals of 25 frames cost 1..25, mean 13


def test_stored_with_the_video_and_checked_against_it(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    meta = make_meta("v", frames=100).model_copy(update={"content_hash": "sha256:" + "ab" * 32})
    assert ds.add_video(meta, frame_index=idx(content_hash=meta.content_hash)).video_id == "v"
    assert ds.frame_index("v") == idx(content_hash=meta.content_hash)
    ds.add_video(make_meta("w", frames=100))
    assert ds.frame_index("w") is None
    with pytest.raises(KeyError):
        ds.frame_index("nope")
    with pytest.raises(ValueError, match="has 50 frames|100"):
        ds.set_frame_index("w", idx(n=50, keys=(0,)))
    with pytest.raises(ValueError, match="different file"):
        ds.set_frame_index("v", idx(content_hash="sha256:" + "cd" * 32))
    assert not [x for x in validate_dataset(ds.path) if x.level == "error"]


def test_validation_catches_bad_stale_and_wrong_length_indexes(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    ds.add_video(make_meta("v", frames=100).model_copy(update={"content_hash": "sha256:" + "ab" * 32}), frame_index=idx())
    path = ds.path / "videos" / "v" / "frame_index.json"
    good = json.loads(path.read_text(encoding="utf-8"))

    def codes():
        return {x.code: x.level for x in validate_dataset(ds.path)}

    assert codes() == {}
    path.write_text(json.dumps({**good, "pts": good["pts"][:60], "keyframes": [0, 25, 50]}), encoding="utf-8")
    assert codes() == {"frame_index_length": "error"}
    path.write_text(json.dumps({**good, "pts": list(reversed(good["pts"]))}), encoding="utf-8")
    assert codes() == {"bad_frame_index": "error"}
    path.write_text(json.dumps({**good, "content_hash": "sha256:" + "cd" * 32}), encoding="utf-8")
    assert codes() == {"frame_index_stale": "warning"}
    path.write_text("not json", encoding="utf-8")
    assert codes() == {"bad_frame_index": "error"}


def test_schema_includes_the_frame_index():
    from vidlabel_spec import SPEC_VERSION, json_schema
    assert "frame_index" in json_schema() and SPEC_VERSION == "0.5.0"
