"""Build a small sample dataset: python examples/make_sample.py /tmp/sample"""
import sys

import numpy as np

from vidlabel import Compiled, Dataset, VideoMetadata

ds = Dataset.create(sys.argv[1] if len(sys.argv) > 1 else "sample_ds", exist_ok=True)
ds.add_video(VideoMetadata(video_id="clip1", frame_count=300, fps={"num": 30, "den": 1}, width=320, height=240,
                           sources=[{"uri": "/path/to/clip1.mp4"}]), exist_ok=True)
ds.add_labels(["goal", "player", "ball"])

mask = np.zeros((240, 320), bool)
mask[100:140, 150:190] = True
ds.add_annotations([
    {"kind": "marker", "label": "goal", "start_frame": 120, "end_frame": 150, "origin": "premiere"},
    {"kind": "box", "label": "player", "start_frame": 10, "box": (20, 30, 80, 150), "track_id": 1, "origin": "model",
     "tool": "detector", "tool_version": "1.0", "confidence": 0.93},
    {"kind": "mask", "label": "ball", "start_frame": 10, "mask": mask, "track_id": 2, "origin": "human", "author": "me",
     "status": "approved"},
], "clip1")

print(ds.stats())
release, path = ds.build()
print("release", release.release_id)
print(Compiled(path).clip("clip1", 100, 130).to_pydict()["label"])
