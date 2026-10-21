# Getting started

This page takes you from nothing to a validated dataset and a training release. Every Python block marked as
runnable is executed by the test suite, so what you read here works.

## Install

vidlabel needs Python 3.10 or newer. From a checkout of the repository:

```bash
pip install -e packages/vidlabel-spec -e packages/vidlabel         # the format and the library
pip install -e "packages/vidlabel[video]"                          # + OpenCV: read frames, previews, check-video
pip install -e "packages/vidlabel[torch]"                          # + OpenCV and PyTorch: the training loader
```

`vidlabel[torch]` installs whatever PyTorch pip picks by default; for a GPU build follow
[pytorch.org](https://pytorch.org/get-started/locally/) first.

Registering a video file with `vidlabel add-video` (or `probe_video` in Python) also needs **ffprobe**, which comes with
[ffmpeg](https://ffmpeg.org/download.html): `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian or
Ubuntu, or a download on Windows. Everything else works without it. You can describe a video by hand if you cannot
install it (shown below).

## 1. Make a dataset and register a video

From the command line:

```bash
vidlabel init my_dataset
vidlabel add-video my_dataset path/to/clip.mp4 --video-id clip
vidlabel info my_dataset
```

`add-video` runs ffprobe and records frame rate (as an exact fraction), frame count, stored size, rotation, codec, bit
rate and a SHA-256 hash of the file. The video is **not copied**: the dataset keeps its path as a hint, and the hash lets
anyone detect a re-encoded or wrong file later (`vidlabel check-video`).

From Python, without a video file at all:

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata

ds = Dataset.create("my_dataset")
ds.add_video(VideoMetadata(
    video_id="clip", frame_count=300, fps={"num": 30, "den": 1}, width=320, height=240,
    sources=[{"uri": "/path/to/clip.mp4"}],
))
```

With a real file, `from vidlabel.probe import probe_video; ds.add_video(probe_video("clip.mp4", "clip"))` does the same
thing as the command.

## 2. Labels

Labels live in one shared list, `vocab.json`. An annotation may only use a label that is in it, which catches typos.

<!-- doctest -->
```python
ds.add_labels(["person", "car", "scene_change"])
```

(`add_annotations(..., create_labels=True)` adds unseen labels for you; leave it off when you want a typo to be an error.)

## 3. Annotations

Every annotation is a row. The three kinds:

<!-- doctest -->
```python
import numpy as np

mask = np.zeros((240, 320), bool)
mask[100:140, 150:190] = True

ids = ds.add_annotations([
    # a span of time: no geometry. end_frame is inclusive.
    {"kind": "marker", "label": "scene_change", "start_frame": 120, "end_frame": 150, "origin": "human"},
    # a box on one frame: corners x0, y0, x1, y1 in pixels, origin top-left, x1 and y1 exclusive.
    {"kind": "box", "label": "person", "start_frame": 10, "box": (20, 30, 80, 150), "track_id": 1,
     "origin": "model", "tool": "my-detector", "tool_version": "1.0", "confidence": 0.93},
    # a mask on one frame: pass a boolean array; its box is filled in for you.
    {"kind": "mask", "label": "car", "start_frame": 10, "mask": mask, "track_id": 2,
     "origin": "human", "author": "me", "status": "approved"},
], "clip")
print(len(ids), "annotations added")
```

Things worth knowing right away:

- **Frames are 0-based** and `end_frame` is inclusive. A point in time has `start_frame == end_frame`; boxes and masks
  are always a single frame.
- **Coordinates are pixels of the displayed frame** (after any rotation), top-left origin. [Concepts](concepts.md#coordinates-and-rotation) explains rotation.
- **`origin` is required**: `human`, `model`, `imported`, `other`, or your own value starting with `x-` such as
  `x-crowd`. Say *which* tool or model in `tool`, and keep anything else under your own key in `extra`.
- Nothing is written unless **every** row is valid. A bad row raises `ValidationError` naming the check that failed.

## 4. Read them back

`annotations()` returns a [pyarrow](https://arrow.apache.org/docs/python/) Table; filters are applied while reading the
Parquet file, so they are cheap.

<!-- doctest -->
```python
boxes = ds.annotations("clip", kind="box", frames=(0, 100))     # frames=(a, b): overlapping a..b, inclusive
print(boxes.num_rows, boxes.to_pylist()[0]["label"])
print(ds.annotations(label="person", origin="model").num_rows)  # across all videos
```

## 5. Review and fix

<!-- doctest -->
```python
ds.update_annotations("clip", ids[:1], status="approved")       # status: unreviewed, approved, rejected
ds.delete_annotations("clip", ids[2:])
print(ds.stats()["by_status"])
```

`rejected` rows stay in the file (so a reviewer's decision is not lost) but are left out of releases unless you ask.

## 6. Validate

<!-- doctest -->
```python
issues = ds.validate()
print(issues or "valid")
```

```bash
vidlabel validate my_dataset          # exit code 1 if there are errors
vidlabel validate my_dataset --deep   # also decodes every mask and checks it against its box
```

Validation checks required fields, enums, frame ranges, labels, boxes, masks, views, tracks and more; the full list
is in the [spec](SPEC.md#validation-codes). Writes run the same checks on the rows you add, and `build` refuses to
compile an invalid dataset.

## 7. Check against the real video

This needs OpenCV (`vidlabel[video]`) and the video file:

```bash
vidlabel check-video my_dataset clip                       # does the file match the metadata?
vidlabel preview my_dataset clip --frames 5-15 --out out   # draws boxes, masks and markers onto real frames
```

Open the PNGs in `out`. If a box is not where the object is, the coordinates are in the wrong frame of reference:
see [rotation and coordinates](concepts.md#coordinates-and-rotation). `preview` only draws frames you ask for (or a
spread of annotated ones), and it tells you how many annotations landed on the frames it drew.

## 8. Build a release for training

<!-- doctest -->
```python
release, path = ds.build()
print(release.release_id, "->", path)
```

```bash
vidlabel build my_dataset        # prints the release id, e.g. r3fa91c0b2d44
```

A release is the whole dataset merged into one table, sorted, with rejected rows left out, plus a snapshot of the label
list. Its id is a hash of the contents, so building twice with no changes gives the same id. Put that id in your
training config: it names exactly the data a run used. Continue with [training](training.md).

## The same thing from the command line

<!-- clitest -->
```bash
vidlabel init cli_dataset
vidlabel add-labels cli_dataset person car
vidlabel info cli_dataset
vidlabel validate cli_dataset
vidlabel build cli_dataset
```

## Next

- [Concepts](concepts.md): the model in detail
- [Views](views.md): when one video is processed as several crops
- [Training](training.md): from a release to batches
- [Writing a converter](converters.md): bring in annotations you already have
