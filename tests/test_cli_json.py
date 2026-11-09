"""`--json` output and the `query` command: what a script or an agent reads from the command line."""
import json

import pytest
from typer.testing import CliRunner
from vidlabel import Dataset, VideoMetadata
from vidlabel.cli import app

run = CliRunner().invoke


@pytest.fixture
def path(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    ds.add_video(VideoMetadata(video_id="v", frame_count=50, fps={"num": 25, "den": 1}, width=64, height=48))
    ds.add_labels(["a", "b"])
    ds.add_annotations([{"kind": "box", "label": "a", "start_frame": f, "box": (1, 1, 9, 9), "track_id": 1, "origin": "model", "confidence": 0.5} for f in range(10)]
                       + [{"kind": "marker", "label": "b", "start_frame": 20, "end_frame": 30, "origin": "human"}], "v")
    return str(ds.path)


def _json(r):
    assert r.output.strip().count("\n") == 0, r.output          # exactly one line: one JSON document
    return json.loads(r.output)


def test_info_validate_tracks_build_as_json(path):
    r = run(app, ["info", path, "--json"])
    assert r.exit_code == 0 and _json(r)["annotations"] == 11
    r = run(app, ["validate", path, "--json"])
    d = _json(r)
    assert r.exit_code == 0 and d["ok"] is True and d["errors"] == 0 and isinstance(d["issues"], list)
    r = run(app, ["tracks", path, "v", "--json"])
    assert _json(r)["tracks"][0]["track_id"] == 1
    r = run(app, ["build", path, "--json"])
    d = _json(r)
    assert d["release_id"].startswith("r") and d["rows"] == 11 and d["empty"] is False


def test_validate_json_keeps_the_exit_code_and_lists_problems(tmp_path):
    (tmp_path / "x").mkdir()
    r = run(app, ["validate", str(tmp_path / "x"), "--json"])
    d = _json(r)
    assert r.exit_code == 1 and d["ok"] is False and d["errors"] >= 1 and d["issues"][0]["level"] == "error"


def test_errors_are_json_when_json_is_asked_for(path):
    r = run(app, ["tracks", path, "nope", "--json"])
    assert r.exit_code == 1 and _json(r)["ok"] is False and "nope" in _json(r)["error"]
    r = run(app, ["tracks", path, "nope"])
    assert r.exit_code == 1 and r.output.startswith("[error]")        # without --json, as before


def test_query_filters_pages_and_never_sends_mask_pixels(path):
    d = _json(run(app, ["query", path, "v"]))
    assert d["total"] == 11 and d["returned"] == 11 and isinstance(d["rows"][0]["annotation_id"], str)
    d = _json(run(app, ["query", path, "v", "--kind", "marker"]))
    assert d["total"] == 1 and d["rows"][0]["label"] == "b" and d["rows"][0]["end_frame"] == 30
    d = _json(run(app, ["query", path, "v", "--track", "1", "--frames", "3-5"]))
    assert d["total"] == 3
    d = _json(run(app, ["query", path, "v", "--limit", "2", "--offset", "9"]))
    assert d["total"] == 11 and d["returned"] == 2
    r = run(app, ["query", path, "v", "--frames", "x-y"])
    assert r.exit_code == 1 and "--frames must look like" in r.output
    assert run(app, ["query", path, "nope"]).exit_code == 1


def test_mcp_command_refuses_a_folder_that_is_not_a_dataset(tmp_path):
    r = run(app, ["mcp", str(tmp_path)])
    assert r.exit_code == 1 and "[error]" in r.output


def test_query_errors_are_json_too_and_odd_filters_are_refused(path):
    for args, text in ((["nope"], "nope"), (["v", "--kind", "circle"], "unknown kind"), (["v", "--status", "zzz"], "unknown status"),
                       (["v", "--frames", "x-y"], "--frames must look like")):
        r = run(app, ["query", path, *args]) if args[0] != "nope" else run(app, ["query", path, "nope"])
        d = _json(r)
        assert r.exit_code == 1 and d["ok"] is False and text in d["error"]
    d = _json(run(app, ["query", path, "v", "--frames", "4"]))
    assert d["total"] == 1                                              # a single frame
    r = run(app, ["query", path, "v", "--label", "nope"])
    assert r.exit_code == 1 and "no label 'nope'" in _json(r)["error"]
    assert _json(run(app, ["query", path, "v", "--origin", "human"]))["total"] == 1
    assert _json(run(app, ["query", path, "v", "--limit", "0"]))["returned"] == 1       # a limit below 1 still returns one row
    assert _json(run(app, ["query", path, "v", "--offset", "-4", "--limit", "3"]))["offset"] == 0
    r = run(app, ["query", path, "v", "--view", "nope"])
    assert r.exit_code == 1 and "no view 'nope'" in _json(r)["error"]


def test_query_shows_masks_without_their_pixels(tmp_path):
    import numpy as np
    ds = Dataset.create(tmp_path / "m")
    ds.add_video(VideoMetadata(video_id="v", frame_count=5, fps={"num": 25, "den": 1}, width=32, height=24))
    ds.add_labels(["a"])
    m = np.zeros((24, 32), bool)
    m[5:15, 5:20] = True
    ds.add_annotations([{"kind": "mask", "label": "a", "start_frame": 1, "mask": m, "origin": "human"}], "v")
    d = _json(run(app, ["query", str(ds.path), "v"]))
    row = d["rows"][0]
    assert row["has_mask"] is True and "mask" not in row and "mask_codec" not in row and len(json.dumps(d)) < 2000


def test_query_says_which_video_ids_exist_when_given_a_file_path(path):
    for given in ("C:\\Users\\me\\Downloads\\clip.mp4", "videos/clip.mp4", "clip.mp4"):
        r = run(app, ["query", path, given])
        err = _json(r)["error"]
        assert r.exit_code == 1 and "not a file path" in err and "['v']" in err
    err = _json(run(app, ["query", path, "nope"]))["error"]
    assert "Video ids: ['v']" in err and "file path" not in err
