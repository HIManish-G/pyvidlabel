"""Reading video with a frame index: exactness (also on variable frame rate), less decoding, batch order, the frame cache."""
import shutil
import subprocess
import warnings

import numpy as np
import pytest
from typer.testing import CliRunner

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

from test_frames import H, W, index_of, make_video  # noqa: E402
from test_seek import Counting  # noqa: E402

from vidlabel import Dataset  # noqa: E402
from vidlabel.cache import CachedReader, FrameCache  # noqa: E402
from vidlabel.cli import app  # noqa: E402
from vidlabel.frames import FrameReader, read_frames, seek_mismatches  # noqa: E402
from vidlabel.probe import probe_frame_index, probe_video, reconcile_frame_count  # noqa: E402
from vidlabel.samples import FrameSamples  # noqa: E402
from vidlabel_spec import FrameIndex  # noqa: E402

N = 700   # frames in the constant-frame-rate test video: three keyframe intervals with x264's default


@pytest.fixture(scope="module")
def cfr(tmp_path_factory):
    p = make_video(tmp_path_factory.mktemp("fi") / "cfr.mp4", n=N)
    meta = probe_video(p, "cfr")
    return p, meta, probe_frame_index(p, content_hash=meta.content_hash)


@pytest.fixture(scope="module")
def vfr(tmp_path_factory):
    """Real variable frame rate: random frames dropped, the survivors keep their original timestamps."""
    d = tmp_path_factory.mktemp("fv")
    raw = d / "cfr.mp4"
    make_video(raw, n=1500)
    out = d / "vfr.mp4"
    for flag in ("-fps_mode", "-vsync"):                 # the flag was renamed in ffmpeg 5.1
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf", "select='gt(random(1),0.25)'",
                            flag, "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", str(out)], check=True)
            break
        except subprocess.CalledProcessError:
            continue
    meta = probe_video(out, "vfr")
    return out, meta, probe_frame_index(out, content_hash=meta.content_hash)


class Jumps(Counting):
    """Counting wrapper that also remembers where it was told to jump."""

    def __init__(self, cap):
        super().__init__(cap)
        self.jumps: list[int] = []

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.jumps.append(int(value))
        return super().set(prop, value)


def _count(monkeypatch):
    from vidlabel import frames
    made: list[Jumps] = []
    real = frames._open
    monkeypatch.setattr(frames, "_open", lambda path: made.append(Jumps(real(path))) or made[-1])
    return made


def _work(made, idx):
    """Frames decoded: the ones the library asked for, plus what OpenCV decodes inside each jump (from the keyframe before it)."""
    jumps = [g for c in made for g in c.jumps]
    return sum(c.decoded for c in made) + sum(min(g, idx.frame_count - 1) - idx.keyframe_before(min(g, idx.frame_count - 1)) + 1
                                              for g in jumps)


def test_probe_reads_timestamps_and_keyframes(cfr, vfr):
    _, meta, idx = cfr
    assert idx.frame_count == meta.frame_count == N and idx.keyframes[0] == 0 and len(idx.keyframes) >= 2
    assert idx.content_hash == meta.content_hash
    gaps = {b - a for a, b in zip(idx.pts, idx.pts[1:])}
    assert len(gaps) == 1                                   # constant frame rate: equal spacing
    _, vmeta, vidx = vfr
    assert vidx.frame_count == vmeta.frame_count and vmeta.is_vfr
    assert len({b - a for a, b in zip(vidx.pts, vidx.pts[1:])}) > 2   # frames were dropped, so spacing varies


def test_estimated_frame_count_is_replaced_by_the_exact_one(cfr):
    _, meta, idx = cfr
    guess = meta.model_copy(update={"frame_count": N + 3, "extra": {"frame_count_estimated": True}})
    fixed = reconcile_frame_count(guess, idx)
    assert fixed.frame_count == N and "frame_count_estimated" not in fixed.extra
    assert reconcile_frame_count(meta, idx) is meta
    with pytest.raises(ValueError, match="frames"):
        reconcile_frame_count(meta.model_copy(update={"frame_count": N + 3}), idx)


@pytest.mark.parametrize("which", ["cfr", "vfr"])
def test_indexed_reads_are_exact_in_any_order(which, cfr, vfr):
    path, meta, idx = cfr if which == "cfr" else vfr
    n = meta.frame_count
    rng = np.random.default_rng(1)
    targets = [int(x) for x in rng.integers(0, n, 60)] + [0, n - 1, n // 2, 1, n - 2]
    truth = {i: f for i, f in read_frames(path, sorted(set(targets)), meta)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # a fallback or a "not exact" warning would fail the test
        with FrameReader(path, meta, index=idx) as r:
            for t in targets:
                frame = r.read(t)
                assert np.array_equal(frame, truth[t]), t
    assert seek_mismatches(path, meta, range(0, n, 17), index=idx) == []


def test_variable_frame_rate_can_seek_with_an_index_and_decodes_less(vfr, monkeypatch):
    path, meta, idx = vfr
    n = meta.frame_count
    targets = [int(n * q) for q in (0.97, 0.03, 0.8, 0.81, 0.002, 0.9, 0.45, 0.01)]
    truth = dict(read_frames(path, sorted(targets), meta))     # pixels show the original frame number, so compare with decoding
    work = {}
    for label, index in (("indexed", idx), ("sequential", None)):
        with monkeypatch.context() as m:
            made = _count(m)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with FrameReader(path, meta, index=index) as r:
                    for t in targets:
                        assert np.array_equal(r.read(t), truth[t]), t
            work[label] = _work(made, idx)
    assert work["indexed"] * 2 < work["sequential"], work


def test_reads_inside_one_keyframe_interval_never_seek(cfr, monkeypatch):
    path, meta, idx = cfr
    end = idx.keyframes[1]                                   # the first interval
    with monkeypatch.context() as m:
        made = _count(m)
        with FrameReader(path, meta, index=idx) as r:
            for t in (3, 50, 120, end - 1):
                assert index_of(r.read(t)) == t
        assert sum(c.seeks for c in made) == 0
        assert _work(made, idx) <= end + 2


def test_read_many_keeps_the_requested_order_and_decodes_each_interval_once(cfr, monkeypatch):
    path, meta, idx = cfr
    req = [600, 30, 610, 31, 30, 305, 595]                  # scrambled, with a repeat, three intervals
    with monkeypatch.context() as m:
        made = _count(m)
        with FrameReader(path, meta, index=idx) as r:
            frames = r.read_many(req)
        grouped = _work(made, idx)
    assert [index_of(f) for f in frames] == req
    frames[0][:] = 0                                         # repeats are copies, not shared buffers
    assert index_of(frames[4]) == 30 and frames[1].any()
    with monkeypatch.context() as m:
        made = _count(m)
        with FrameReader(path, meta, index=idx) as r:
            for t in req:
                r.read(t)
        one_by_one = _work(made, idx)
    assert grouped < one_by_one, (grouped, one_by_one)


def test_a_wrong_index_is_never_trusted(cfr):
    path, meta, idx = cfr
    other_count = FrameIndex(timebase=idx.timebase, pts=idx.pts[:-5], keyframes=[0])
    with pytest.warns(RuntimeWarning, match="frames"):
        r = FrameReader(path, meta, index=other_count)
    assert r.index is None and index_of(r.read(200)) == 200
    stale = idx.model_copy(update={"content_hash": "sha256:" + "ee" * 32})
    with pytest.warns(RuntimeWarning, match="different file"):
        assert FrameReader(path, meta, index=stale).index is None
    # right size and hash, but timestamps that do not belong to this file: caught on the first read, still correct
    lying = FrameIndex(timebase=idx.timebase, pts=[p + (i * 37) % 200 for i, p in enumerate(idx.pts)],
                       keyframes=idx.keyframes, content_hash=idx.content_hash)
    with pytest.warns(RuntimeWarning, match="do not match the frame index"):
        with FrameReader(path, meta, index=lying) as r:
            assert index_of(r.read(321)) == 321 and index_of(r.read(5)) == 5
            assert r.index is None


def test_seek_off_ignores_the_index(cfr, monkeypatch):
    path, meta, idx = cfr
    with monkeypatch.context() as m:
        made = _count(m)
        with FrameReader(path, meta, index=idx, seek="off") as r:
            assert index_of(r.read(300)) == 300 and index_of(r.read(10)) == 10
        assert sum(c.seeks for c in made) == 0


# ---- the dataset and command line --------------------------------------------------------------------------------

def test_add_video_and_index_commands(tmp_path, cfr):
    path, meta, _ = cfr
    d = Dataset.create(tmp_path / "d")
    r = CliRunner()
    res = r.invoke(app, ["add-video", str(d.path), str(path), "--video-id", "a"])
    assert res.exit_code == 0 and "keyframe every" in res.output, res.output
    assert d.frame_index("a").frame_count == N
    res = r.invoke(app, ["add-video", str(d.path), str(path), "--video-id", "b", "--no-index"])
    assert res.exit_code == 0 and d.frame_index("b") is None
    res = r.invoke(app, ["index", str(d.path)])                      # only the video without one
    assert res.exit_code == 0 and "indexed b" in res.output and "indexed a" not in res.output
    assert d.frame_index("b") is not None
    assert "nothing to do" in r.invoke(app, ["index", str(d.path)]).output
    assert r.invoke(app, ["index", str(d.path), "--force"]).exit_code == 0
    wrong = make_video(tmp_path / "other.mp4", n=N, w=64, h=48)
    res = r.invoke(app, ["index", str(d.path), "a", "--video-path", str(wrong), "--force"])
    assert res.exit_code == 1 and "not the file that was registered" in res.output


def test_check_video_reports_keyframes_and_suggests_a_fix(tmp_path, cfr):
    path, _, _ = cfr
    d = Dataset.create(tmp_path / "d")
    d.add_video(probe_video(path, "a"))                               # registered without an index
    r = CliRunner()
    res = r.invoke(app, ["check-video", str(d.path), "a"])
    assert res.exit_code == 0, res.output
    assert "no stored frame index" in res.output and "keyframes: every" in res.output
    assert "ffmpeg -i cfr.mp4" in res.output and "-g 30" in res.output and "different file" in res.output
    assert r.invoke(app, ["index", str(d.path)]).exit_code == 0
    res = r.invoke(app, ["check-video", str(d.path), "a", "--seek"])
    assert res.exit_code == 0 and "no stored frame index" not in res.output
    assert "seeking is frame-exact" in res.output and "using the frame index" in res.output


def test_check_video_stays_quiet_when_keyframes_are_close(tmp_path):
    p = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=25", "-frames:v", "90",
                    "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(p)], check=True)
    d = Dataset.create(tmp_path / "d")
    r = CliRunner()
    assert r.invoke(app, ["add-video", str(d.path), str(p), "--video-id", "s"]).exit_code == 0
    res = r.invoke(app, ["check-video", str(d.path), "s"])
    assert res.exit_code == 0 and "keyframes: every 10 frames" in res.output and "ffmpeg" not in res.output


# ---- training reads -----------------------------------------------------------------------------------------------

@pytest.fixture
def ds(tmp_path, cfr):
    path, _, _ = cfr
    d = Dataset.create(tmp_path / "ds")
    for v in ("a", "b"):
        d.add_video(probe_video(path, v), frame_index=probe_frame_index(path, content_hash=probe_video(path, v).content_hash))
    d.add_labels(["thing"])
    rng = np.random.default_rng(0)
    for v in ("a", "b"):
        d.add_annotations([{"kind": "box", "label": "thing", "start_frame": int(f), "box": (4, 4, 30, 30), "origin": "human"}
                           for f in sorted(rng.choice(N, 40, replace=False))], v)
    return d


def test_sampler_readers_use_the_stored_index(ds):
    fs = FrameSamples(ds, release=None)
    with fs.reader("a") as r:
        assert r.index is not None and index_of(r.read(500)) == 500


def test_map_style_batches_are_grouped_but_come_back_in_order(ds, monkeypatch):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from vidlabel.torch_data import VideoFrames, collate_frames
    fs = FrameSamples(ds, release=None)
    vf = VideoFrames(fs)
    order = [70, 3, 55, 12, 79, 40, 41, 0]
    batch = vf.__getitems__(order)
    for i, item in zip(order, batch):
        v, f = fs.index[i]
        assert (item["video_id"], item["frame_index"]) == (v, f)
        assert index_of(item["image"].permute(1, 2, 0).numpy()) == f
    singles = [vf[i] for i in order]
    assert all(torch.equal(a["image"], b["image"]) for a, b in zip(batch, singles))
    loader = DataLoader(vf, batch_size=8, shuffle=True, collate_fn=collate_frames, generator=torch.Generator().manual_seed(0))
    seen = 0
    for b in loader:
        for img, f in zip(b["image"], b["frame_index"]):
            assert index_of(img.permute(1, 2, 0).numpy()) == f
            seen += 1
    assert seen == len(vf)


# ---- the frame cache ----------------------------------------------------------------------------------------------

def test_cache_png_is_exact_and_jpg_is_close(ds, tmp_path):
    fs = FrameSamples(ds, release=None)
    for fmt in ("png", "jpg"):
        cache, st = FrameCache.build(fs, tmp_path / fmt, format=fmt)
        assert st.written == len(fs) and st.skipped == 0 and st.bytes > 0
        cached = FrameSamples(ds, release=None, cache=cache)
        with cached.reader("a") as r, FrameReader(fs.video_path("a"), ds.video("a")) as ref:
            assert isinstance(r, CachedReader)
            for f in fs.frames_of("a")[:10]:
                got, want = r.read(f), ref.read(f)
                assert index_of(got) == f
                if fmt == "png":
                    assert np.array_equal(got, want)
                else:
                    assert np.abs(got.astype(int) - want.astype(int)).mean() < 3
    # a second build does nothing; --overwrite redoes it
    _, again = FrameCache.build(fs, tmp_path / "png", format="png")
    assert again.written == 0 and again.skipped == len(fs)
    assert FrameCache.build(fs, tmp_path / "png", format="png", overwrite=True)[1].written == len(fs)
    with pytest.raises(ValueError, match="already holds"):
        FrameCache.build(fs, tmp_path / "png", format="jpg")


def test_cache_serves_samples_crops_to_views_and_falls_back_to_the_video(ds, tmp_path):
    fs = FrameSamples(ds, release=None, videos=["a"])
    cache, _ = FrameCache.build(fs, tmp_path / "c", format="png")
    cached = FrameSamples(ds, release=None, videos=["a"], cache=tmp_path / "c")        # a folder is accepted
    f = cached.frames_of("a")[5]
    with cached.reader("a") as r:
        assert r.read(f).shape == (H, W, 3)
        missing = next(i for i in range(N) if not cache.has("a", i))
        assert index_of(r.read(missing)) == missing                                      # not cached: decoded from the video
        got = r.read_many([missing + 1, f, missing])
        assert [index_of(x) for x in got] == [missing + 1, f, missing]
    ds.add_view({"view_id": "crop", "video_id": "a", "region": {"x": 8, "y": 8, "width": 40, "height": 24}})
    vs = FrameSamples(ds, release=None, videos=["a"], view_id="crop", cache=cache)
    with vs.reader("a") as r:
        assert r.read(f).shape == (24, 40, 3)


def test_cache_for_another_file_is_ignored(ds, tmp_path):
    fs = FrameSamples(ds, release=None, videos=["a"])
    cache, _ = FrameCache.build(fs, tmp_path / "c", format="png")
    other = ds.video("a").model_copy(update={"content_hash": "sha256:" + "dd" * 32})
    with pytest.warns(RuntimeWarning, match="different file"):
        assert not cache.usable("a", other)
    assert cache.usable("a", ds.video("a")) and not cache.usable("zzz", ds.video("a"))
    with pytest.raises(FileNotFoundError, match="not a frame cache"):
        FrameCache(tmp_path / "empty")


def test_cache_command(ds, tmp_path):
    r = CliRunner()
    res = r.invoke(app, ["cache", str(ds.path), str(tmp_path / "c"), "--release", "none", "--format", "png"])
    assert res.exit_code == 0 and "cached 80 frame(s)" in res.output and "lossy" not in res.output, res.output
    res = r.invoke(app, ["cache", str(ds.path), str(tmp_path / "c2"), "--release", "none"])
    assert res.exit_code == 0 and "lossy" in res.output
    res = r.invoke(app, ["cache", str(ds.path), str(tmp_path / "c"), "--release", "none", "--format", "jpg"])
    assert res.exit_code == 1 and "already holds png" in res.output
