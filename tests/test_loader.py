import shutil

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

from test_frames import H, N, W, index_of, make_video  # noqa: E402

from vidlabel import Dataset  # noqa: E402
from vidlabel.boxes import convert_boxes  # noqa: E402
from vidlabel.probe import probe_video  # noqa: E402
from vidlabel.samples import FrameSamples  # noqa: E402
from conftest import blob_mask  # noqa: E402


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("lv")
    return [make_video(d / f"v{i}.mp4", n=N - 10 * i) for i in range(3)]


@pytest.fixture
def ds(tmp_path, videos):
    d = Dataset.create(tmp_path / "ds")
    for i, v in enumerate(videos):
        d.add_video(probe_video(v, f"v{i}"))
    d.add_labels(["thing", "other", "event"])
    d.add_view({"view_id": "crop", "video_id": "v0", "region": {"x": 16, "y": 8, "width": 48, "height": 32},
                "output_size": (96, 64)})
    for i in range(3):
        d.add_annotations([
            {"kind": "box", "label": "thing", "start_frame": 10, "box": (8, 16, 40, 48), "track_id": 7, "origin": "human"},
            {"kind": "box", "label": "other", "start_frame": 10, "box": (50, 10, 90, 30), "origin": "model", "confidence": 0.4},
            {"kind": "box", "label": "thing", "start_frame": 20, "box": (1, 2, 3, 4), "origin": "human"},
            {"kind": "mask", "label": "thing", "start_frame": 30, "mask": blob_mask(H, W), "origin": "human"},
            {"kind": "marker", "label": "event", "start_frame": 8, "end_frame": 12, "origin": "human"},
        ], f"v{i}")
    d.add_annotations([{"kind": "box", "label": "thing", "start_frame": 10, "box": (2, 2, 10, 10), "view_id": "crop",
                        "origin": "human"}], "v0", coords="view")
    return d


def test_convert_boxes():
    xyxy = np.array([[10, 20, 50, 80]], np.float32)
    assert convert_boxes(xyxy, "xyxy", "xywh").tolist() == [[10, 20, 40, 60]]
    assert convert_boxes(xyxy, "xyxy", "cxcywh").tolist() == [[30, 50, 40, 60]]
    for fmt in ("xywh", "cxcywh"):
        back = convert_boxes(convert_boxes(xyxy, "xyxy", fmt), fmt, "xyxy")
        assert np.allclose(back, xyxy)
    yolo = convert_boxes(xyxy, "xyxy", "cxcywh", size=(100, 200), dst_normalized=True)
    assert np.allclose(yolo, [[0.3, 0.25, 0.4, 0.3]])
    assert np.allclose(convert_boxes(yolo, "cxcywh", "xyxy", size=(100, 200), src_normalized=True), xyxy)
    assert convert_boxes([], "xyxy", "xywh").shape == (0, 4)
    with pytest.raises(ValueError):
        convert_boxes(xyxy, "xyxy", "ltrb")
    with pytest.raises(ValueError):
        convert_boxes(xyxy, "xyxy", "xyxy", dst_normalized=True)


def test_index_targets_and_images(ds):
    ds.build()
    fs = FrameSamples(ds, release="latest")
    assert fs.class_names == ["thing", "other", "event"]
    assert fs.release_id and len(fs) == len({(v, f) for v, f in fs.index})
    # frames with a box or mask: 10, 20, 30 in each of 3 videos (the view row is on frame 10 as well)
    assert [f for _, f in fs.index if _ == "v1"] == [10, 20, 30]
    assert fs.video_ids() == ["v0", "v1", "v2"]
    with fs.reader("v1") as r:
        img = r.read(10)
    assert index_of(img) == 10
    s = fs.sample("v1", 10, img)
    names = fs.class_names
    assert s["image"].shape == (H, W, 3) and s["frame_index"] == 10 and s["video_id"] == "v1"
    assert s["boxes"].dtype == np.float32 and s["boxes"].shape == (2, 4)
    assert sorted(s["labels"].tolist()) == sorted([names.index("thing"), names.index("other")])
    assert 7 in s["track_ids"].tolist() and -1 in s["track_ids"].tolist()
    assert s["frame_labels"][names.index("event")] == 1.0 and s["frame_labels"].sum() == 1.0   # marker 8-12 covers 10
    s20 = fs.sample("v1", 20, img)
    assert s20["frame_labels"].sum() == 0 and s20["boxes"].tolist() == [[1, 2, 3, 4]]


def test_options_stride_labels_offset_box_format_masks(ds):
    fs = FrameSamples(ds, release=None, labels=["thing", "other"], label_offset=1, box_format="cxcywh", normalized=True,
                      kinds=("box",), stride=2, videos=["v2"])
    assert fs.class_names == ["thing", "other"] and fs.release_id is None
    assert fs.frames_of("v2") == [10]                      # box frames 10, 20 -> every 2nd
    img = np.zeros((H, W, 3), np.uint8)
    s = fs.sample("v2", 10, img)
    assert s["labels"].min() >= 1 and s["labels"].max() <= 2
    thing = s["boxes"][s["labels"] == 1][0]
    assert np.allclose(thing, [24 / W, 32 / H, 32 / W, 32 / H])  # (8,16,40,48) as normalized cx,cy,w,h
    m = FrameSamples(ds, release=None, kinds=("mask",), with_masks=True).sample("v1", 30, img)
    assert m["masks"].shape == (1, H, W) and m["masks"].dtype == np.uint8 and m["masks"].sum() > 0
    assert FrameSamples(ds, release=None, frames="all", videos=["v2"]).frames_of("v2") == list(range(N - 20))


def test_view_samples_are_in_the_crop(ds):
    fs = FrameSamples(ds, release=None, view_id="crop")
    ds.add_annotations([{"kind": "box", "label": "other", "start_frame": 11, "box": (20, 12, 28, 20), "view_id": "crop",
                         "origin": "human"}], "v0")      # given in source pixels
    fs = FrameSamples(ds, release=None, view_id="crop")
    assert fs.index == [("v0", 10), ("v0", 11)]           # only rows labelled in that view
    with fs.reader("v0") as r:
        img = r.read(10)
        img11 = r.read(11)
    assert img.shape == (64, 96, 3)                       # the crop, resized to output_size
    assert np.allclose(fs.sample("v0", 10, img)["boxes"], [[2, 2, 10, 10]])      # was added in view pixels
    assert np.allclose(fs.sample("v0", 11, img11)["boxes"], [[8, 8, 24, 24]])    # source (20,12)-(28,20): shift, then 2x


def test_pinned_release_does_not_move(ds):
    rel, _ = ds.build()
    before = FrameSamples(ds, release=rel.release_id)
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 40, "box": (1, 1, 5, 5), "origin": "human"}], "v0")
    ds.build()
    assert len(FrameSamples(ds, release=rel.release_id)) == len(before)
    assert len(FrameSamples(ds, release="latest")) == len(before) + 1
    assert len(FrameSamples(ds, release=None)) == len(before) + 1
    assert before.describe()["release_id"] == rel.release_id


def test_errors_are_clear(ds):
    with pytest.raises(FileNotFoundError, match="vidlabel build"):
        FrameSamples(ds, release="latest")
    ds.build()
    with pytest.raises(FileNotFoundError, match="not found"):
        FrameSamples(ds, release="r000")
    for bad in ({"labels": ["nope"]}, {"frames": "some"}, {"stride": 0}, {"kinds": ("pixel",)}, {"box_format": "ltrb"}):
        with pytest.raises(ValueError):
            FrameSamples(ds, release=None, **bad)


# ---- PyTorch ----
torch = pytest.importorskip("torch")


def _stream_ids(it):
    return [(s["video_id"], s["frame_index"]) for s in it]


def test_torch_map_style_and_collate(ds):
    from vidlabel.torch_data import VideoFrames, collate_frames
    fs = FrameSamples(ds, release=None, videos=["v1", "v2"])
    vf = VideoFrames(fs)
    assert len(vf) == 6
    for i in (5, 0, 3, 1):                                  # random order works (reopens as needed)
        item = vf[i]
        v, f = fs.index[i]
        assert (item["video_id"], item["frame_index"]) == (v, f)
        assert item["image"].dtype == torch.uint8 and tuple(item["image"].shape) == (3, H, W)
        assert index_of(item["image"].permute(1, 2, 0).numpy()) == f
    batch = collate_frames([vf[0], vf[1]])
    assert tuple(batch["image"].shape) == (2, 3, H, W) and len(batch["boxes"]) == 2 and batch["frame_labels"].shape[0] == 2


def test_torch_stream_is_ordered_complete_and_seeded(ds):
    from vidlabel.torch_data import VideoStream
    fs = FrameSamples(ds, release=None)
    plain = VideoStream(fs, shuffle_videos=False)
    assert _stream_ids(plain) == fs.index                 # front to back, nothing missing or duplicated
    a = VideoStream(fs, shuffle_videos=True, seed=3)
    assert _stream_ids(a) == _stream_ids(a)               # deterministic
    assert sorted(_stream_ids(a)) == sorted(fs.index)
    a.set_epoch(1)
    assert sorted(_stream_ids(a)) == sorted(fs.index)
    buffered = VideoStream(fs, shuffle_videos=False, shuffle_buffer=4, seed=1)
    ids = _stream_ids(buffered)
    assert sorted(ids) == sorted(fs.index) and ids != fs.index


def test_torch_stream_sharding_covers_everything_once(ds):
    from vidlabel.torch_data import VideoStream
    fs = FrameSamples(ds, release=None)
    parts = [_stream_ids(VideoStream(fs, shuffle_videos=True, seed=5, rank=r, world_size=2)) for r in range(2)]
    assert not set(parts[0]) & set(parts[1])
    assert sorted(parts[0] + parts[1]) == sorted(fs.index)


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_dataloader_with_workers(ds, ctx):
    import multiprocessing as mp
    if ctx not in mp.get_all_start_methods():
        pytest.skip(f"{ctx} not available")
    from torch.utils.data import DataLoader
    from vidlabel.torch_data import VideoFrames, VideoStream, collate_frames
    fs = FrameSamples(ds, release=None)
    for data in (VideoStream(fs, shuffle_videos=True, seed=1), VideoFrames(fs)):
        dl = DataLoader(data, batch_size=2, num_workers=2, collate_fn=collate_frames, multiprocessing_context=ctx)
        seen = []
        for b in dl:
            assert b["image"].dtype == torch.uint8
            for img, v, f in zip(b["image"], b["video_id"], b["frame_index"]):
                assert index_of(img.permute(1, 2, 0).numpy()) == f
                seen.append((v, int(f)))
        assert sorted(seen) == sorted(fs.index)


def test_pinned_release_pins_the_class_indices(ds):
    rel, path = ds.build()
    assert (path / "vocab.json").exists() and "vocab.json" in rel.files
    pinned = FrameSamples(ds, release=rel.release_id).class_names
    ds.add_labels(["brand_new"])
    ds.build()                                           # same rows -> same release id, vocabulary snapshot kept
    assert FrameSamples(ds, release=rel.release_id).class_names == pinned == ["thing", "other", "event"]
    assert FrameSamples(ds, release=None).class_names == ["thing", "other", "event", "brand_new"]   # live: follows the dataset
    (path / "vocab.json").unlink()                       # releases made before this existed fall back to the live vocabulary
    assert FrameSamples(ds, release=rel.release_id).class_names[-1] == "brand_new"
