"""ReadPlan as pure planning: no video is opened here."""
import numpy as np
import pytest

from conftest import make_meta
from vidlabel import Dataset
from vidlabel.readplan import ReadPlan
from vidlabel.samples import FrameSamples
from vidlabel_spec import FrameIndex

TB = {"num": 1, "den": 1000}


def index(n=600, every=100):
    return FrameIndex(timebase=TB, pts=[i * 40 for i in range(n)], keyframes=list(range(0, n, every)))


def test_requests_are_grouped_by_keyframe_interval_and_deduplicated():
    reqs = [("a", 350), ("a", 30), ("a", 305), ("a", 30), ("b", 210), ("a", 399), ("a", 30)]
    plan = ReadPlan(reqs, indexes={"a": index(), "b": index()})
    got = [(c.video_id, c.start, c.frames, c.repeats) for c in plan.chunks]
    assert got == [("a", 0, (30,), (3,)), ("a", 300, (305, 350, 399), (1, 1, 1)), ("b", 200, (210,), (1,))]
    assert len(plan) == 7 and sum(c.n_samples for c in plan.chunks) == 7
    # cost: reach the first frame from its keyframe, then read on to the last
    assert [c.cost for c in plan.chunks] == [31, 6 + 94, 11]


def test_summary_compares_with_reading_one_at_a_time():
    rng = np.random.default_rng(0)
    reqs = [("a", int(f)) for f in rng.integers(0, 600, 200)]
    s = ReadPlan(reqs, indexes={"a": index()}).summary()
    assert s["samples"] == 200 and s["videos"] == 1 and s["chunks"] == 6
    assert s["est_decoded_frames"] <= 600                      # at most the whole video once
    assert s["est_decoded_frames_one_by_one"] > 3 * s["est_decoded_frames"]


def test_videos_without_an_index_are_cut_into_blocks():
    plan = ReadPlan([("x", 10), ("x", 260), ("x", 255), ("x", 499)], block=250)
    assert [(c.start, c.frames) for c in plan.chunks] == [(0, (10,)), (250, (255, 260, 499))]


def test_requests_outside_the_video_are_rejected():
    with pytest.raises(ValueError, match="outside video 'a'.*600 frames"):
        ReadPlan([("a", 600)], indexes={"a": index()})
    with pytest.raises(ValueError, match="outside video"):
        ReadPlan([("a", -1)])
    with pytest.raises(ValueError, match="60 frames"):
        ReadPlan([("a", 60)], frame_counts={"a": 60})


def _plan(n_chunks_videos=4, seed=3):
    rng = np.random.default_rng(1)
    reqs = [(f"v{v}", int(f)) for v in range(n_chunks_videos) for f in rng.integers(0, 600, 120)]
    return ReadPlan(reqs, indexes={f"v{v}": index() for v in range(n_chunks_videos)}, seed=seed)


def test_chunk_order_is_reproducible_changes_each_epoch_and_keeps_every_chunk():
    plan = _plan()
    assert plan.order(0) == plan.order(0) and plan.order(0) != plan.order(1) and plan.order(0) != _plan(seed=4).order(0)
    assert sorted(map(id, plan.order(2))) == sorted(map(id, plan.chunks))


@pytest.mark.parametrize("num_shards", [1, 2, 3, 5, 8])
def test_every_chunk_goes_to_exactly_one_shard_whatever_the_number_of_shards(num_shards):
    plan = _plan()
    for epoch in (0, 1):
        parts = [plan.shard(epoch, s, num_shards) for s in range(num_shards)]
        assert sorted(map(id, sum(parts, []))) == sorted(map(id, plan.chunks))
        sizes = plan.shard_sizes(epoch, num_shards)
        assert sum(sizes) == len(plan) and max(sizes) - min(sizes) <= max(c.n_samples for c in plan.chunks)


def test_shard_arguments_are_checked():
    plan = _plan()
    with pytest.raises(ValueError):
        plan.shard(0, 2, 2)
    with pytest.raises(ValueError):
        plan.shard(0, 0, 0)


def test_empty_plan():
    plan = ReadPlan([])
    assert len(plan) == 0 and plan.chunks == [] and plan.shard(0, 0, 3) == [] and plan.summary()["chunks"] == 0


def test_samples_make_plans_from_positions_or_pairs(tmp_path):
    ds = Dataset.create(tmp_path / "d")
    ds.add_video(make_meta("a", frames=300))
    ds.add_video(make_meta("b", frames=300), frame_index=index(300))
    ds.add_labels(["thing"])
    for v in ("a", "b"):
        ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": f, "box": (1, 1, 9, 9), "origin": "human"}
                            for f in (5, 150, 250)], v)
    fs = FrameSamples(ds, release=None)
    assert len(fs) == 6
    positions = np.random.default_rng(0).choice(len(fs), 20)                     # what a weighted draw returns
    p1 = fs.plan_reads(positions)
    p2 = fs.plan_reads([fs.index[i] for i in positions])
    assert p1.chunks == p2.chunks and len(p1) == 20
    assert fs.plan_reads([("b", 7)]).chunks[0].start == 0                        # frames nobody annotated can be requested too
    with pytest.raises(ValueError, match="outside video"):
        fs.plan_reads([("a", 300)])
    with pytest.raises(KeyError):
        fs.plan_reads([("zzz", 1)])
    with pytest.raises(IndexError):
        fs.plan_reads([99])
