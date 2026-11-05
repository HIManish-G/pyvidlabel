# How-to recipes

Short answers to "how do I...". Each recipe is self-contained. The Python blocks share one dataset and run in order, and the test
suite executes them.

<!-- doctest -->
```python
import numpy as np
import pyarrow.compute as pc
from vidlabel import Dataset, VideoMetadata, FrameSamples

ds = Dataset.create("recipes")
for i in range(3):
    ds.add_video(VideoMetadata(video_id=f"clip{i}", frame_count=200, fps={"num": 25, "den": 1}, width=320, height=240))
ds.add_labels(["person", "dog", "goal"])
```

## Register many videos

From the command line, one call per file (a shell loop does many); from Python, with ffprobe available:

```python
from pathlib import Path
from vidlabel.probe import probe_video
for f in sorted(Path("videos").glob("*.mp4")):
    ds.add_video(probe_video(f, f.stem))
```

Ids must contain only letters, digits, `.`, `_` and `-`. Registering the same file twice is noted rather than silent.

## Add boxes from a model, in bulk

Rows are plain dictionaries. Build the whole list, then add it in one call: the rows are checked together and written once.

<!-- doctest -->
```python
rows = [{"kind": "box", "label": "person", "start_frame": f, "box": (10 + f, 20, 60 + f, 120),
         "track_id": 1, "origin": "model", "tool": "my-detector", "tool_version": "2.0", "confidence": 0.9}
        for f in range(0, 100, 5)]
ids = ds.add_annotations(rows, "clip0")
```

If your tool gives `x, y, width, height`, convert first; vidlabel stores corners. `vidlabel.boxes.convert_boxes` converts between
`xyxy`, `xywh` and `cxcywh`, and `normalized` fractions need multiplying by the image size.

## Add a span of time

<!-- doctest -->
```python
ds.add_annotations([{"kind": "marker", "label": "goal", "start_frame": 40, "end_frame": 60, "origin": "human", "author": "sam"}], "clip1")
```

## Add a mask from a segmentation array

Pass a boolean array of the full frame size; the box is filled in from it.

<!-- doctest -->
```python
m = np.zeros((240, 320), bool)
m[50:100, 80:160] = True
ds.add_annotations([{"kind": "mask", "label": "dog", "start_frame": 7, "mask": m, "origin": "model"}], "clip0")
```

A mask of the wrong size is refused at write time with the expected size. Masks from a model that saw a cropped, resized frame: see
[views](views.md) and use `coords="view"`.

## Find what needs review

<!-- doctest -->
```python
boxes = ds.annotations("clip0", kind="box")
uncertain = boxes.filter(pc.less(boxes.column("confidence"), 0.5))
unreviewed = ds.annotations(status="unreviewed", origin="model")      # across all videos
print(uncertain.num_rows, unreviewed.num_rows)
```

`annotations()` takes `kind`, `label`, `origin`, `status`, `frames=(a, b)`, `view_id` and `track_id`; list values match any. Misspelled
filter values raise instead of returning nothing.

## Review one object at a time

<!-- doctest -->
```python
for t in ds.tracks("clip0"):
    print(t["track_id"], t["n_rows"], t["statuses"])
ds.update_track("clip0", 1, status="approved")
```

`ds.delete_track("clip0", 1)` removes a whole track. Both take `view_id=` for a track inside a view. Command line: `vidlabel tracks` and
`vidlabel review --track N --status approved`.

## Correct a model and keep both versions

Reject the model's row and add yours, rather than overwriting. Then you can measure later how often the model needed correcting.

<!-- doctest -->
```python
bad = ds.annotations("clip0", kind="box").to_pylist()[0]
ds.update_annotations("clip0", [bad["annotation_id"]], status="rejected", author="ana")
ds.add_annotations([{"kind": "box", "label": "person", "start_frame": bad["start_frame"], "box": (12, 22, 62, 122),
                     "track_id": bad["track_id"], "origin": "human", "author": "ana", "status": "approved"}], "clip0")
```

A rejected row does not count in the track checks, so the corrected row can reuse the track id.

## Keep your own data on a row

Put it under your own key in `extra`; core columns reject unknown fields.

<!-- doctest -->
```python
ds.add_annotations([{"kind": "marker", "label": "goal", "start_frame": 90, "end_frame": 95, "origin": "human",
                     "extra": {"mylab": {"camera": 2, "note": "replay"}}}], "clip2")
```

For a source other than the four built-in origins use `x-` values (`origin: "x-crowd"`), and name the specific tool in `tool`.

## Count what you have

<!-- doctest -->
```python
s = ds.stats()
print(s["by_kind"], s["by_label"], s["by_status"])
```

`vidlabel info PATH` prints the same.

## Split into train and validation

Split by video, never by frame, or neighbouring frames leak between sets.

<!-- doctest -->
```python
train = FrameSamples(ds, release=None, videos=["clip0", "clip1"], kinds=("box", "mask", "marker"))
val = FrameSamples(ds, release=None, videos=["clip2"], kinds=("box", "mask", "marker"))
print(len(train), len(val))
```

For a reproducible split, build a release per split (`ds.build(video_ids=[...])`) and pin both ids.

## Train only on human work, or only on confident model output

<!-- doctest -->
```python
human_only = FrameSamples(ds, release=None, origins=["human"], kinds=("box",))
confident = FrameSamples(ds, release=None, min_confidence=0.8, kinds=("box",))
print(len(human_only), len(confident))
```

Rows with no confidence (human work) are kept by `min_confidence`.

## Use class names and indices

<!-- doctest -->
```python
samples = FrameSamples(ds, release=None, kinds=("box",), label_offset=1)    # 0 stays free for background
print(samples.class_names)
```

## Look at the data on real frames

```bash
vidlabel preview recipes clip0 --frames 0-50 --every 10 --out out
vidlabel preview recipes clip0 --view crop --mp4 --out out
```

Rejected rows are drawn dashed and prefixed `rej`, and drawn under the others. More in the [CLI reference](cli.md#preview).

## Check that a video file is still the one you labelled

```bash
vidlabel check-video recipes clip0 --video-path /new/location/clip0.mp4 --hash
```

Without `--hash` the check compares frame count, size and per-frame timestamps, and says it did not check the content.

## Make random access faster

If training is slow because it seeks in video: (1) re-encode with more frequent keyframes (`vidlabel check-video` prints the ffmpeg
command), (2) read with a [`ReadPlan`](training.md#reading-a-list-of-frames-you-chose-readplan) and a shuffle buffer of about 1,000
samples, or (3) build a [frame cache](training.md#making-random-access-fast). Measurements are in [performance](performance.md).

## Move a dataset

Copy the folder. Videos are not inside it; their recorded paths are hints. On the new machine pass `roots=[...]` to `FrameSamples`
or `--root` to commands so videos are found by file name, and run `vidlabel check-video` to confirm they are the same files.

## Delete or relabel

There is no label rename or delete command: labels are referenced by name from rows. To rename, add the new name, update rows
(`ds.update_annotations(video, ids, label="new")` for each video), and leave the old name in the vocabulary (it does no harm).
Deleting an annotation: `ds.delete_annotations(video_id, ids)`. Asking to change or delete an id that does not exist is an error.
