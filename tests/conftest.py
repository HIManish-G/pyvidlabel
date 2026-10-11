import numpy as np
import pytest

from vidlabel import Dataset, VideoMetadata


def make_meta(video_id="vid1", frames=300, w=64, h=48, fps=(30000, 1001), rotation=0):
    return VideoMetadata(video_id=video_id, frame_count=frames, fps={"num": fps[0], "den": fps[1]},
                         width=w, height=h, rotation=rotation, sources=[{"uri": f"/data/{video_id}.mp4"}])


@pytest.fixture
def ds(tmp_path):
    d = Dataset.create(tmp_path / "ds")
    d.add_video(make_meta("vid1"))
    d.add_video(make_meta("vid2", frames=200))
    d.add_labels(["cut", "person", "car", "highlight"])
    return d


def blob_mask(h=48, w=64, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), bool)
    y, x = rng.integers(0, h // 2), rng.integers(0, w // 2)
    m[y:y + 10, x:x + 13] = rng.random((10, 13)) > 0.3
    return m
