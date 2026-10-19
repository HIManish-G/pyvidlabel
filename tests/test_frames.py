import shutil
import subprocess

import numpy as np
import pytest
from typer.testing import CliRunner

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")

from vidlabel import Dataset, VideoSource, masks  # noqa: E402
from vidlabel.cli import app  # noqa: E402
from vidlabel.frames import check_video, crop_to_view, read_frame, read_frames  # noqa: E402
from vidlabel.preview import choose_frames, draw_annotations, label_color, parse_frames, render_preview  # noqa: E402
from vidlabel.probe import probe_video  # noqa: E402

W, H, N = 96, 64, 60


def make_video(path, n=N, w=W, h=H, fps=25):
    """H.264 video where frame i shows its own index as 12 black/white 8px cells along the top row, plus a red block
    at the left edge (rows 16-32) that makes orientation visible."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        f = np.full((h, w, 3), 60, np.uint8)
        f[0:8, :] = 0
        for b in range(12):
            if i >> b & 1:
                f[0:8, b * 8:(b + 1) * 8] = 255
        f[16:32, 0:16] = (255, 0, 0)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


def index_of(frame):
    return sum(int(frame[4, b * 8 + 4].mean() > 128) << b for b in range(12))


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("vid") / "clip.mp4")


@pytest.fixture
def ds_with_video(tmp_path, video):
    d = Dataset.create(tmp_path / "ds")
    d.add_video(probe_video(video, "clip"))
    d.add_labels(["thing", "other", "event"])
    return d


def test_probe_real_encoded_video(video):
    m = probe_video(video, "clip")
    assert (m.width, m.height, m.frame_count, m.rotation) == (W, H, N, 0)
    assert (m.fps.num, m.fps.den, m.codec, m.is_vfr) == (25, 1, "h264", False)
    assert m.bit_rate and m.bit_rate > 0


def test_frames_are_exactly_the_requested_ones(video):
    meta = probe_video(video, "clip")
    asked = [59, 0, 1, 2, 17, 30, 30, 58]
    got = list(read_frames(video, asked, meta))
    assert [i for i, _ in got] == sorted(set(asked))
    assert all(index_of(f) == i for i, f in got)
    assert got[0][1].shape == (H, W, 3) and got[0][1].dtype == np.uint8
    assert index_of(read_frame(video, 41, meta)) == 41
    assert list(read_frames(video, [], meta)) == []


def test_frame_past_end_and_negative(video):
    meta = probe_video(video, "clip")
    with pytest.raises(IndexError, match="past the end"):
        list(read_frames(video, [5, N + 10], meta))
    with pytest.raises(IndexError):
        list(read_frames(video, [-1], meta))


def test_wrong_file_is_detected(video, tmp_path):
    meta = probe_video(video, "clip")
    other = make_video(tmp_path / "other.mp4", n=5, w=64, h=48)
    with pytest.raises(ValueError, match="not the file"):
        list(read_frames(other, [0], meta))
    with pytest.raises(FileNotFoundError):
        list(read_frames(tmp_path / "missing.mp4", [0], meta))


def test_check_video(video, tmp_path):
    meta = probe_video(video, "clip")
    assert check_video(video, meta) == []
    assert check_video(video, meta, verify_hash=True) == []
    assert any("frame_count" in p or "decoded" in p for p in check_video(video, meta.model_copy(update={"frame_count": N + 1})))
    other = make_video(tmp_path / "o.mp4", n=N, w=W, h=H)
    shifted = tmp_path / "shifted.mp4"
    shifted.write_bytes(other.read_bytes() + b"\0")  # same video, different bytes
    assert any("hash" in p for p in check_video(shifted, meta, verify_hash=True))
    assert check_video(tmp_path / "nope.mp4", meta)[0].endswith("does not exist")
    assert any("frame size" in p for p in check_video(make_video(tmp_path / "s.mp4", n=N, w=64, h=48), meta))


@pytest.mark.parametrize("flag,expected_rotation", [(90, 270), (-90, 90), (180, 180)])
def test_rotation_convention_matches_ffmpeg_and_opencv(video, tmp_path, flag, expected_rotation):
    """End to end on a real file: ffmpeg -display_rotation (counter-clockwise) -> probe -> our rotation -> pixels
    identical to what OpenCV's own auto-rotation produces."""
    rot = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", str(flag), "-i", str(video), "-c", "copy",
                    str(rot)], check=True)
    meta = probe_video(rot, "rot")
    assert meta.rotation == expected_rotation and (meta.width, meta.height) == (W, H)
    ours = read_frame(rot, 3, meta)
    assert (ours.shape[1], ours.shape[0]) == meta.display_size

    cap = cv2.VideoCapture(str(rot))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    for _ in range(4):
        ok, bgr = cap.read()
    cap.release()
    assert ok
    theirs = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if theirs.shape == (H, W, 3) and expected_rotation in (90, 270):
        pytest.skip("this OpenCV build does not auto-rotate")
    assert ours.shape == theirs.shape and np.array_equal(ours, theirs)
    red = np.argwhere((ours[..., 0] > 200) & (ours[..., 1] < 60))
    assert len(red) > 100  # the red block survived, now in its rotated position


def test_crop_to_view(video):
    from vidlabel import View
    meta = probe_video(video, "clip")
    frame = read_frame(video, 0, meta)
    v = View(view_id="v", video_id="clip", region={"x": 8, "y": 16, "width": 48, "height": 32})
    assert crop_to_view(frame, v).shape == (32, 48, 3)
    assert np.array_equal(crop_to_view(frame, v), frame[16:48, 8:56])
    vs = View(view_id="s", video_id="clip", region={"x": 8, "y": 16, "width": 48, "height": 32}, output_size=(24, 16))
    assert read_frame(video, 5, meta, view=vs).shape == (16, 24, 3)
    assert crop_to_view(frame, None) is frame


def test_parse_and_choose_frames():
    assert parse_frames("3,10-12,50-", 53) == [3, 10, 11, 12, 50, 51, 52]
    assert parse_frames("0-5,200", 10) == [0, 1, 2, 3, 4, 5]
    rows = [{"kind": "box", "start_frame": f, "end_frame": f} for f in range(0, 300, 3)]
    picked = choose_frames(rows, 300, limit=12)
    assert len(picked) == 12 and picked[0] == 0 and picked[-1] == 297
    assert choose_frames([], 50, limit=5) == [0, 12, 25, 37, 49] or len(choose_frames([], 50, limit=5)) == 5
    assert choose_frames([{"kind": "marker", "start_frame": 4, "end_frame": 9}], 20) == [4, 9]


def test_label_color_is_stable():
    assert label_color("thing") == label_color("thing") and label_color("thing") != label_color("other")


def test_draw_annotations_box_mask_marker():
    frame = np.full((64, 96, 3), 100, np.uint8)
    m = np.zeros((64, 96), bool)
    m[40:50, 60:80] = True
    blob = masks.encode(m)
    rows = [
        {"kind": "box", "label": "thing", "start_frame": 5, "end_frame": 5, "x0": 10, "y0": 20, "x1": 40, "y1": 50,
         "track_id": 3, "confidence": 0.9},
        {"kind": "mask", "label": "other", "start_frame": 5, "end_frame": 5, "x0": 60, "y0": 40, "x1": 80, "y1": 50, "mask": blob},
        {"kind": "marker", "label": "event", "start_frame": 2, "end_frame": 9},
        {"kind": "box", "label": "thing", "start_frame": 6, "end_frame": 6, "x0": 0, "y0": 0, "x1": 5, "y1": 5},
    ]
    out = draw_annotations(frame, rows, 5)
    assert out is not frame and np.array_equal(frame, np.full((64, 96, 3), 100, np.uint8))  # input untouched
    assert tuple(out[30, 10]) == label_color("thing")            # left edge of the box
    assert tuple(out[49, 25]) == label_color("thing")            # bottom edge (y1 - 1)
    assert not np.array_equal(out[45, 70], frame[45, 70])        # mask blended
    assert tuple(out[45, 62]) != (100, 100, 100)
    assert not np.array_equal(out[0:20, 0:60], frame[0:20, 0:60])  # marker banner
    assert np.array_equal(draw_annotations(frame, rows[3:], 5), frame)  # nothing active at frame 5


def test_preview_shows_the_right_frames_with_boxes_in_place(ds_with_video, tmp_path):
    ds = ds_with_video
    m = np.zeros((H, W), bool)
    m[40:55, 30:60] = True
    ds.add_annotations([
        {"kind": "box", "label": "thing", "start_frame": 10, "box": (10, 20, 40, 50), "origin": "human"},
        {"kind": "mask", "label": "other", "start_frame": 40, "mask": m, "origin": "human"},
    ], "clip")
    files = render_preview(ds, "clip", tmp_path / "out", video_path=ds.video("clip").sources[0].uri)
    assert [f.name for f in files] == ["clip_f000010.png", "clip_f000040.png"]
    img = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2RGB)
    assert index_of(img) == 10                                   # the frame on screen is frame 10
    assert tuple(img[30, 10]) == label_color("thing")            # box drawn where the annotation says
    img2 = cv2.cvtColor(cv2.imread(str(files[1])), cv2.COLOR_BGR2RGB)
    assert index_of(img2) == 40

    mp4 = render_preview(ds, "clip", tmp_path / "out2", frames=range(8, 14), mp4=True)
    assert len(mp4) == 1 and mp4[0].stat().st_size > 0


def test_preview_of_a_view(ds_with_video, tmp_path):
    ds = ds_with_video
    ds.add_view({"view_id": "crop", "video_id": "clip", "region": {"x": 8, "y": 16, "width": 48, "height": 32},
                 "output_size": (96, 64)})
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 7, "box": (10, 10, 30, 30), "origin": "model",
                         "view_id": "crop"}], "clip", coords="view")
    ds.add_annotations([{"kind": "box", "label": "other", "start_frame": 7, "box": (20, 20, 50, 40), "origin": "model"}], "clip")
    files = render_preview(ds, "clip", tmp_path / "v", view_id="crop")
    img = cv2.imread(str(files[0]))
    assert img.shape[:2] == (64, 96)                              # the view's output size
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert tuple(rgb[20, 10]) == label_color("thing")             # drawn in the crop's own pixels
    assert not (rgb == np.array(label_color("other"))).all(axis=-1).any()  # whole-frame rows are not shown in a view


def test_preview_errors(ds_with_video, tmp_path):
    ds = ds_with_video
    meta_src = ds.video("clip").sources[0].uri
    ds.add_video(ds.video("clip").model_copy(update={"video_id": "lost", "sources": [VideoSource(uri=str(tmp_path / "gone.mp4"))]}))
    with pytest.raises(FileNotFoundError, match="cannot find"):
        render_preview(ds, "lost", tmp_path / "o")
    assert render_preview(ds, "lost", tmp_path / "o", video_path=meta_src, frames=[0])  # explicit path works


def test_cli_preview_and_check_video(ds_with_video, tmp_path):
    ds = ds_with_video
    ds.add_annotations([{"kind": "marker", "label": "event", "start_frame": 3, "end_frame": 6, "origin": "human"}], "clip")
    r = CliRunner()
    out = tmp_path / "pv"
    res = r.invoke(app, ["preview", str(ds.path), "clip", "--out", str(out), "--frames", "0-9", "--every", "3"])
    assert res.exit_code == 0, res.output
    assert sorted(p.name for p in out.iterdir()) == ["clip_f000000.png", "clip_f000003.png", "clip_f000006.png", "clip_f000009.png"]
    assert r.invoke(app, ["preview", str(ds.path), "clip", "--out", str(tmp_path / "pv2")]).exit_code == 0  # auto frames
    assert r.invoke(app, ["check-video", str(ds.path), "clip"]).exit_code == 0
    assert r.invoke(app, ["check-video", str(ds.path), "clip", "--hash"]).exit_code == 0
    wrong = make_video(tmp_path / "w.mp4", n=5)
    bad = r.invoke(app, ["check-video", str(ds.path), "clip", "--video-path", str(wrong)])
    assert bad.exit_code == 1 and "decoded 5 frames" in bad.output
