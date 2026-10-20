import shutil
import subprocess
import warnings

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

from test_frames import index_of, make_video  # noqa: E402

from vidlabel.frames import FrameReader, read_frames, seek_mismatches  # noqa: E402
from vidlabel.probe import probe_video  # noqa: E402

N = 700


class Counting:
    """Wraps an OpenCV capture and counts how many frames it was asked to decode."""

    def __init__(self, cap):
        self.cap, self.decoded, self.seeks = cap, 0, 0

    def grab(self):
        self.decoded += 1
        return self.cap.grab()

    def read(self):
        self.decoded += 1
        return self.cap.read()

    def set(self, prop, value):
        self.seeks += prop == cv2.CAP_PROP_POS_FRAMES
        return self.cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self.cap, name)


@pytest.fixture(scope="module")
def cfr(tmp_path_factory):
    p = make_video(tmp_path_factory.mktemp("seek") / "cfr.mp4", n=N)
    return p, probe_video(p, "cfr")


@pytest.fixture(scope="module")
def vfr(tmp_path_factory):
    """Real variable frame rate: random frames dropped, the survivors keep their original timestamps."""
    d = tmp_path_factory.mktemp("vfr")
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
    return out, probe_video(out, "vfr")


def _count_decoding(monkeypatch):
    """Every capture the library opens is wrapped; returns the list of wrappers so work can be summed afterwards."""
    from vidlabel import frames
    made: list[Counting] = []
    real = frames._open
    monkeypatch.setattr(frames, "_open", lambda path: made.append(Counting(real(path))) or made[-1])
    return made


def test_seeking_is_exact_and_decodes_far_less(cfr, monkeypatch):
    path, meta = cfr
    truth = dict(read_frames(path, range(N), meta))
    targets = [650, 12, 400, 399, 3, 699, 200, 201, 520, 100]
    work = {}
    for mode in ("auto", "off"):
        with monkeypatch.context() as m:
            made = _count_decoding(m)
            with FrameReader(path, meta, seek=mode) as r:
                if mode == "auto":
                    assert r.seek_enabled
                    made.clear()                      # ignore the one-off verification decode
                for t in targets:
                    f = r.read(t)
                    assert np.array_equal(f, truth[t]) and (mode == "off" or index_of(f) == t)
            work[mode] = (sum(c.decoded for c in made), sum(c.seeks for c in made))
    assert work["auto"][1] > 0 and work["off"][1] == 0             # seeks were used, and only when allowed
    assert work["auto"][0] < work["off"][0] / 3, work              # ...and they skipped most of the decoding


def test_seek_modes_validate_and_short_clips_do_not_seek(cfr):
    path, meta = cfr
    with pytest.raises(ValueError):
        FrameReader(path, meta, seek="sometimes")
    assert FrameReader(path, meta.model_copy(update={"frame_count": 50}), seek="auto").seek_enabled is False
    assert FrameReader(path, meta, seek="off").seek_enabled is False
    assert FrameReader(path, meta, seek="always").seek_enabled is True


def test_variable_frame_rate_never_seeks_and_stays_correct(vfr):
    path, meta = vfr
    assert meta.is_vfr
    n = meta.frame_count
    targets = sorted({int(n * q) for q in (0.8, 0.05, 0.95, 0.01, 0.5)})
    truth = dict(read_frames(path, targets, meta))
    with FrameReader(path, meta) as r:
        assert r.seek_enabled is False
        for t in [targets[3], targets[0], targets[4], targets[1], targets[2]]:       # out of order: forces going backwards
            assert np.array_equal(r.read(t), truth[t])


def test_the_check_catches_bad_seeking_even_if_metadata_says_constant_rate(vfr):
    """The metadata flag is only a first guard: the pixel comparison is what actually protects against misalignment."""
    path, meta = vfr
    liar = meta.model_copy(update={"is_vfr": False})
    assert seek_mismatches(path, liar, range(100, 1000, 100))            # OpenCV lands on the wrong frames here
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = FrameReader(path, liar)
        assert r.seek_enabled is False
        assert any("not frame-exact" in str(x.message) for x in w)
    r.close()


def test_exact_seeking_has_no_mismatches(cfr):
    path, meta = cfr
    assert seek_mismatches(path, meta, [0, 1, 99, 350, 699, 5000, -3]) == []   # out-of-range positions are ignored


def test_check_video_seek_option(cfr, vfr, tmp_path):
    from typer.testing import CliRunner

    from vidlabel import Dataset
    from vidlabel.cli import app

    d = Dataset.create(tmp_path / "ds")
    d.add_video(cfr[1])
    d.add_video(vfr[1])
    r = CliRunner()
    good = r.invoke(app, ["check-video", str(d.path), "cfr", "--seek"])
    assert good.exit_code == 0 and "seeking is frame-exact" in good.output
    odd = r.invoke(app, ["check-video", str(d.path), "vfr", "--seek"])
    assert odd.exit_code == 0 and "not frame-exact" in odd.output and "variable frame rate" in odd.output
