# Tutorial: tracking animals in lab video

A worked example of a typical biology workflow: a model proposes boxes and tracks for two animals in an arena camera video, a person
adds behaviour spans, you review the model's work, and you build a release to train on. The data here is simulated so that every
step runs without a video file; with a real file you would register it with `vidlabel add-video` instead (see
[getting started](getting-started.md#1-make-a-dataset-and-register-a-video)).

You will use: videos, labels, boxes with tracks and confidence, behaviour markers, review by track, validation, a release, and
training targets.

## 1. A dataset with one trial

<!-- doctest -->
```python
import numpy as np
import pyarrow.compute as pc
from vidlabel import Dataset, VideoMetadata, FrameSamples

ds = Dataset.create("arena")
ds.add_video(VideoMetadata(video_id="trial01", frame_count=600, fps={"num": 30, "den": 1}, width=640, height=480))
ds.add_labels(["animal", "grooming", "rearing"])
```

One video is one recording. A session of many trials is many videos in the same dataset, sharing one label list.

## 2. Load the model's detections

Imagine a detector that gives a box and a confidence every 10th frame, and a tracker that keeps an id per animal. Whatever tool
you use, the shape of the rows is the same. Mark them `origin: model` and say which tool made them, so they can always be told
apart from human work.

<!-- doctest -->
```python
rng = np.random.default_rng(0)
rows = []
for track_id, (x, y) in enumerate([(100.0, 200.0), (400.0, 300.0)], start=1):
    for frame in range(0, 600, 10):
        x += rng.normal(0, 4)
        y += rng.normal(0, 4)
        rows.append({"kind": "box", "label": "animal", "start_frame": frame, "box": (x - 20, y - 12, x + 20, y + 12),
                     "track_id": track_id, "origin": "model", "tool": "my-detector", "tool_version": "1.2",
                     "confidence": float(rng.uniform(0.5, 1.0))})
ids = ds.add_annotations(rows, "trial01")
print(len(ids), "boxes added")
```

All rows are checked together; if one is bad nothing is written, and the error names the row. Track ids belong to one video, so
animal `1` here is a different object from animal `1` in another trial.

## 3. A person adds behaviour spans

A span of time has no geometry. `end_frame` is inclusive.

<!-- doctest -->
```python
ds.add_annotations([
    {"kind": "marker", "label": "grooming", "start_frame": 120, "end_frame": 210, "origin": "human", "author": "ana"},
    {"kind": "marker", "label": "rearing", "start_frame": 330, "end_frame": 360, "origin": "human", "author": "ana"},
], "trial01")
```

## 4. Look at what the model did

See the tracks (one summary per object), and find the shaky detections.

<!-- doctest -->
```python
for t in ds.tracks("trial01"):
    print(t["track_id"], t["labels"], t["n_rows"], "rows, frames", t["first_frame"], "to", t["last_frame"],
          f"mean confidence {t['mean_confidence']:.2f}")

boxes = ds.annotations("trial01", kind="box")
shaky = boxes.filter(pc.less(boxes.column("confidence"), 0.6))
print(shaky.num_rows, "boxes below 0.6 confidence")
```

On the command line the same overview is `vidlabel tracks arena trial01`.

## 5. Review

Suppose you watched the clip with the boxes drawn on it (`vidlabel preview arena trial01 --out out`) and animal 2 was followed
correctly, while the low-confidence boxes of animal 1 need a second look. Review a whole track in one call, and reject single rows
you do not trust:

<!-- doctest -->
```python
ds.update_track("trial01", 2, status="approved", author="ana")
bad = shaky.column("annotation_id").to_pylist()[:3]
ds.update_annotations("trial01", bad, status="rejected", author="ana")
print(ds.stats()["by_status"])
```

On the command line: `vidlabel review arena trial01 --track 2 --status approved`. Rejected rows stay in the file, so the
decision is recorded, and they are left out of releases.

Updating or deleting an id that does not exist raises an error instead of doing nothing, so a stale id list cannot silently skip work.

## 6. Validate and build

<!-- doctest -->
```python
assert ds.validate() == []
release, path = ds.build()
print(release.release_id, release.n_rows, "rows")
```

Pin `release.release_id` in your training config. Only human-confirmed work? Build with
`ds.build(origins=["human"])`, or require approval: `ds.build(statuses=["approved"])`. See [releases](releases.md).

## 7. Targets for training

`FrameSamples` turns a release into per-frame targets. Here we ask only for the box targets of approved tracks and look at one
frame without any video file; in a real run `FrameSamples` also reads the pixels (see [training](training.md)).

<!-- doctest -->
```python
release, _ = ds.build(statuses=["approved"])
samples = FrameSamples(ds, release=release.release_id, kinds=("box",))
print(len(samples), "sampled frames from", samples.video_ids())
t = samples.targets("trial01", 10, (640, 480))
print(t["boxes"].shape, t["labels"], t["track_ids"])
print("frame-level labels active at frame 10:", t["frame_labels"])
```

`frame_labels` is a multi-hot vector over the vocabulary built from the markers active at that frame; ask for frame 130 and the
"grooming" entry is on.

## What to try next

- Crop an arena out of a larger frame with a [view](views.md), so a model sees only the arena.
- Add per-animal masks (see [masks](masks.md)) and check them with `vidlabel validate --deep`.
- Fix the model's boxes by hand, and keep both versions: set `origin: human` on your corrected row and reject the model's, so you can
  later measure how often the model needed correcting.
- Load several trials into one dataset and split train and validation by video (`FrameSamples(videos=[...])`).
