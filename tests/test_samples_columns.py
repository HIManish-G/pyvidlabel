"""FrameSamples keeps its rows as columns and a sorted (video, frame) index. These tests compare what it returns for every
frame against a plain scan of the table, on a random mix of single-frame rows, spans, markers and several videos, and
check that opening it on many rows stays cheap. No video files are needed: only targets are read."""
import random
import time
import tracemalloc

import numpy as np

from vidlabel import Dataset, FrameSamples, VideoMetadata

LABELS = ["thing", "other", "event"]


def make(tmp_path, n_videos=4, n_frames=120, per_video=150, seed=3):
    rnd = random.Random(seed)
    ds = Dataset.create(tmp_path / "ds")
    ds.add_labels(LABELS)
    for v in range(n_videos):
        ds.add_video(VideoMetadata(video_id=f"v{v}", frame_count=n_frames, fps={"num": 25, "den": 1}, width=200, height=100))
        rows = []
        for i in range(per_video):
            kind = rnd.choice(["box", "box", "box", "marker"])
            a = rnd.randrange(n_frames)
            row = {"kind": kind, "label": "event" if kind == "marker" else rnd.choice(["thing", "other"]), "start_frame": a,
                   "origin": "human"}
            if kind == "marker":
                row["end_frame"] = min(n_frames - 1, a + rnd.choice([0, 0, 1, 5, 20]))
            else:
                x, y = rnd.uniform(0, 150), rnd.uniform(0, 60)
                row["box"] = (x, y, x + rnd.uniform(5, 40), y + rnd.uniform(5, 30))
                if rnd.random() < 0.6:
                    row["track_id"] = rnd.randrange(1000)
            rows.append(row)
        ds.add_annotations(rows, f"v{v}", validate=False)      # duplicate tracks are fine here: we only compare selections
    return ds


def expected(ds, samples, video_id, frame):
    """The rows active on a frame, found by scanning the table: points in table order first, then spans in table order."""
    t = ds.annotations(video_id).to_pylist()
    pts = [r for r in t if r["start_frame"] == r["end_frame"] == frame]
    spans = [r for r in t if r["start_frame"] != r["end_frame"] and r["start_frame"] <= frame <= r["end_frame"]]
    return pts + spans


def test_targets_match_a_plain_scan(tmp_path):
    ds = make(tmp_path)
    s = FrameSamples(ds, release=None, kinds=("box", "marker"))
    assert len(s) > 100
    seen_marker = seen_span = 0
    for video_id, frame in s.index:
        rows = expected(ds, s, video_id, frame)
        t = s.targets(video_id, frame, (200, 100))
        boxes = [r for r in rows if r["kind"] == "box"]
        assert t["annotation_ids"].tolist() == [r["annotation_id"] for r in boxes]
        assert t["track_ids"].tolist() == [-1 if r["track_id"] is None else r["track_id"] for r in boxes]
        assert t["labels"].tolist() == [LABELS.index(r["label"]) for r in boxes]
        np.testing.assert_allclose(t["boxes"], np.array([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in boxes]).reshape(-1, 4), rtol=1e-6)
        marked = {r["label"] for r in rows if r["kind"] == "marker"}
        assert t["frame_labels"].tolist() == [1.0 if n in marked else 0.0 for n in LABELS]
        seen_marker += bool(marked)
        seen_span += any(r["start_frame"] != r["end_frame"] for r in rows)
    assert seen_marker and seen_span              # the random data really exercised markers and spans


def test_annotated_frames_are_exactly_those_with_a_selected_kind(tmp_path):
    ds = make(tmp_path, seed=5)
    for kinds in (("box",), ("marker",), ("box", "marker")):
        s = FrameSamples(ds, release=None, kinds=kinds)
        for v in ds.video_ids():
            want = set()
            for r in ds.annotations(v).to_pylist():
                if r["kind"] in kinds:
                    want.update(range(r["start_frame"], r["end_frame"] + 1))
            assert s.frames_of(v) == sorted(want)


def test_unknown_video_and_frame_have_no_targets(tmp_path):
    ds = make(tmp_path, n_videos=2)
    s = FrameSamples(ds, release=None)
    assert s.targets("nope", 3, (200, 100))["boxes"].shape == (0, 4)
    assert s.targets("v0", 10_000, (200, 100))["labels"].tolist() == []


def test_empty_selection_is_fine(tmp_path):
    ds = make(tmp_path, n_videos=1, per_video=10)
    s = FrameSamples(ds, release=None, labels=["thing"], origins=["model"])     # nothing is "model"
    assert len(s) == 0 and s.video_ids() == []


def test_many_rows_open_quickly_and_stay_small(tmp_path):
    """Opening used to build one Python dict per row (about 1.5 KB each) and scan every row once per video."""
    ds = make(tmp_path, n_videos=60, n_frames=5000, per_video=1500)       # 90,000 rows
    tracemalloc.start()
    t0 = time.perf_counter()
    s = FrameSamples(ds, release=None, kinds=("box",))
    took = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(s) > 50_000
    assert peak < 40 * 1024 * 1024, f"{peak / 1e6:.0f} MB for 90,000 rows"
    assert took < 10, f"{took:.1f}s"
