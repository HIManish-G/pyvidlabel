import json
import subprocess

import pytest

from vidlabel import probe


@pytest.mark.parametrize("side_data,tags,expected,odd", [
    ([{"rotation": -90}], None, 90, None),      # typical portrait phone video: apply 90 clockwise
    ([{"rotation": 90}], None, 270, None),      # ffmpeg -display_rotation 90 (counter-clockwise) = 270 clockwise
    ([{"rotation": 180}], None, 180, None),
    ([{"rotation": -180}], None, 180, None),
    ([{"rotation": 0}], None, 0, None),
    ([{"rotation": -90.0001}], None, 90, None),  # float noise
    ([], {"rotate": "90"}, 90, None),            # older files: clockwise tag
    ([], {"rotate": "270"}, 270, None),
    ([], None, 0, None),
    ([{"rotation": 45}], None, 0, -45.0),        # not a multiple of 90: reported, not silently dropped
])
def test_rotation_conversion(side_data, tags, expected, odd):
    stream = {"side_data_list": side_data}
    if tags:
        stream["tags"] = tags
    assert probe._rotation(stream) == (expected, odd)


def _fake_ffprobe(monkeypatch, stream):
    base = {"width": 1080, "height": 1920, "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001",
            "nb_frames": "309", "duration": "10.31", "codec_name": "h264", "pix_fmt": "yuv420p", "time_base": "1/30000"}
    out = json.dumps({"streams": [{**base, **stream}], "format": {}})
    monkeypatch.setattr(probe.shutil, "which", lambda n: "/usr/bin/ffprobe")
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))


def test_probe_rotated_file_and_display_size(monkeypatch, tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    _fake_ffprobe(monkeypatch, {"side_data_list": [{"rotation": 90}]})
    m = probe.probe_video(f)
    assert m.rotation == 270 and (m.width, m.height) == (1080, 1920) and m.display_size == (1920, 1080)
    assert m.fps.num == 30000 and m.fps.den == 1001 and "unsupported_rotation_degrees" not in m.extra


def test_probe_records_unsupported_rotation(monkeypatch, tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    _fake_ffprobe(monkeypatch, {"side_data_list": [{"rotation": 45}]})
    m = probe.probe_video(f)
    assert m.rotation == 0 and m.extra["unsupported_rotation_degrees"] == -45.0
