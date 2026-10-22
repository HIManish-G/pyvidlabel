"""Running a ReadPlan: right frames, every request exactly once, each keyframe interval decoded once."""
import multiprocessing as mp
import shutil
from collections import Counter

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

from test_frame_index_video import N, _count, _work, cfr  # noqa: E402,F401  (cfr is a fixture)
from test_frames import index_of  # noqa: E402

from vidlabel import Dataset  # noqa: E402
from vidlabel.cache import FrameCache  # noqa: E402
from vidlabel.frames import FrameReader  # noqa: E402
from vidlabel.probe import probe_frame_index, probe_video  # noqa: E402
from vidlabel.samples import FrameSamples  # noqa: E402


@pytest.fixture
def ds(tmp_path, cfr):
    path, _, _ = cfr
    d = Dataset.create(tmp_path / "ds")
    for v in ("a", "b"):
        m = probe_video(path, v)
        d.add_video(m, frame_index=probe_frame_index(path, content_hash=m.content_hash))
    d.add_labels(["thing"])
    for v in ("a", "b"):
        d.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (4, 4, 30, 30), "origin": "human"}
                           for f in (10, 260, 600)], v)
    return d


def requests(n=150, seed=0):
    rng = np.random.default_rng(seed)
    return [(("a", "b")[int(v)], int(f)) for v, f in zip(rng.integers(0, 2, n), rng.integers(0, N, n))]


def test_every_request_comes_back_once_with_the_right_pixels_and_targets(ds):
    fs = FrameSamples(ds, release=None)
    reqs = requests() + [("a", 10), ("a", 10), ("a", 10), ("b", 7)]       # repeats, and a frame nobody annotated
    plan = fs.plan_reads(reqs)
    got = list(fs.iter_plan(plan))
    assert len(got) == len(reqs) == len(plan)
    assert Counter((s["video_id"], s["frame_index"]) for s in got) == Counter(reqs)
    for s in got:
        assert index_of(s["image"]) == s["frame_index"]
        if s["frame_index"] == 10:
            assert len(s["boxes"]) == 1
        if (s["video_id"], s["frame_index"]) == ("b", 7):
            assert s["boxes"].shape == (0, 4)
    tens = [s for s in got if (s["video_id"], s["frame_index"]) == ("a", 10)]
    tens[0]["image"][:] = 0                                                # repeats are separate arrays
    assert all(index_of(t["image"]) == 10 for t in tens[1:])


def test_each_keyframe_interval_is_decoded_once(ds, monkeypatch):
    fs = FrameSamples(ds, release=None)
    reqs = [r for r in requests(300, seed=2) if r[0] == "a"]
    idx = ds.frame_index("a")
    plan = fs.plan_reads(reqs)
    est = plan.summary()
    with monkeypatch.context() as m:
        made = _count(m)
        assert len(list(fs.iter_plan(plan))) == len(reqs)
        planned = _work(made, idx)
    with monkeypatch.context() as m:
        made = _count(m)
        with FrameReader(fs.video_path("a"), ds.video("a"), index=idx) as r:
            for _, f in reqs:                                               # the same requests, one at a time as asked
                r.read(f)
        one_by_one = _work(made, idx)
    assert planned <= est["est_decoded_frames"] * 1.1 + 5 and planned <= N + 5   # about each interval once
    assert one_by_one > 2 * planned, (planned, one_by_one)
    assert est["est_decoded_frames_one_by_one"] > 2 * est["est_decoded_frames"]


@pytest.mark.parametrize("num_shards", [1, 3])
def test_shards_together_give_every_sample_once_and_epochs_reorder(ds, num_shards):
    fs = FrameSamples(ds, release=None)
    reqs = requests(120, seed=3)
    plan = fs.plan_reads(reqs, seed=9)
    orders = []
    for epoch in (0, 1):
        seq = [[(s["video_id"], s["frame_index"]) for s in fs.iter_plan(plan, epoch=epoch, shard=k, num_shards=num_shards)]
               for k in range(num_shards)]
        assert Counter(sum(seq, [])) == Counter(reqs)
        orders.append(sum(seq, []))
    assert orders[0] != orders[1]
    again = [(s["video_id"], s["frame_index"]) for k in range(num_shards)
             for s in fs.iter_plan(plan, epoch=0, shard=k, num_shards=num_shards)]
    assert again == orders[0]                                               # same seed and epoch, same order


def test_shuffle_buffer_mixes_but_keeps_the_samples_and_is_deterministic(ds):
    fs = FrameSamples(ds, release=None)
    reqs = requests(100, seed=4)
    plan = fs.plan_reads(reqs)
    ids = lambda **kw: [(s["video_id"], s["frame_index"]) for s in fs.iter_plan(plan, **kw)]
    plain, mixed = ids(), ids(shuffle_buffer=16)
    assert Counter(mixed) == Counter(plain) == Counter(reqs) and mixed != plain and mixed == ids(shuffle_buffer=16)


def test_only_a_few_videos_are_open_at_a_time(ds, monkeypatch):
    fs = FrameSamples(ds, release=None)
    plan = fs.plan_reads(requests(80, seed=5))
    with monkeypatch.context() as m:
        made = _count(m)
        got = list(fs.iter_plan(plan, max_open=1))
    assert Counter((s["video_id"], s["frame_index"]) for s in got) == Counter(plan_requests(plan))
    assert all(index_of(s["image"]) == s["frame_index"] for s in got)
    assert all(c.cap.isOpened() is False for c in made)                     # everything was closed again


def plan_requests(plan):
    return [(c.video_id, f) for c in plan.chunks for f, n in zip(c.frames, c.repeats) for _ in range(n)]


def test_a_frame_cache_serves_a_plan(ds, tmp_path):
    fs = FrameSamples(ds, release=None)
    reqs = [("a", 10), ("a", 260), ("a", 10), ("b", 600)]
    cache, _ = FrameCache.build(fs, tmp_path / "c", format="png")
    cached = FrameSamples(ds, release=None, cache=cache)
    got = list(cached.iter_plan(cached.plan_reads(reqs)))
    assert Counter((s["video_id"], s["frame_index"]) for s in got) == Counter(reqs)
    assert all(index_of(s["image"]) == s["frame_index"] for s in got)


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_planned_stream_with_dataloader_workers(ds, ctx):
    if ctx not in mp.get_all_start_methods():
        pytest.skip(f"{ctx} not available")
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from vidlabel.torch_data import PlannedStream, collate_frames
    fs = FrameSamples(ds, release=None)
    reqs = requests(90, seed=6)
    stream = PlannedStream(fs, fs.plan_reads(reqs, seed=1), shuffle_buffer=8)
    assert len(stream) == 90
    runs = []
    for epoch in (0, 1):
        stream.set_epoch(epoch)
        seen = []
        for b in DataLoader(stream, batch_size=5, num_workers=2, collate_fn=collate_frames, multiprocessing_context=ctx):
            assert b["image"].dtype == torch.uint8
            for img, v, f in zip(b["image"], b["video_id"], b["frame_index"]):
                assert index_of(img.permute(1, 2, 0).numpy()) == f
                seen.append((v, int(f)))
        assert Counter(seen) == Counter(reqs)
        runs.append(seen)
    assert runs[0] != runs[1]


def test_planned_stream_applies_the_transform_and_ranks_split_the_work(ds):
    pytest.importorskip("torch")
    from vidlabel.torch_data import PlannedStream
    fs = FrameSamples(ds, release=None)
    reqs = requests(60, seed=7)
    plan = fs.plan_reads(reqs)
    parts = [[(s["video_id"], s["frame_index"], s["marked"]) for s in
              PlannedStream(fs, plan, lambda s: {**s, "marked": True}, rank=r, world_size=2)] for r in range(2)]
    assert all(m for part in parts for *_, m in part)
    assert Counter((v, f) for part in parts for v, f, _ in part) == Counter(reqs)
