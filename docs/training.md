# Training

The loader turns a dataset into **samples**: one frame of a video plus the boxes, masks, labels and time markers that
apply to it. It has two layers:

- `vidlabel.FrameSamples` works with plain NumPy and needs no deep-learning framework. It decides *which* frames
  to use, from the annotation table alone, and builds the targets for each.
- `vidlabel.torch_data` wraps it as PyTorch datasets (`VideoStream`, `VideoFrames`). Install with `pip install -e "packages/vidlabel[torch]"`.

Reading pixels needs OpenCV (`vidlabel[video]`, included in `[torch]`) and the video files at the paths the dataset
records (or under `roots=[...]`).

## Pin a release

```bash
vidlabel build my_dataset          # prints e.g. r3fa91c0b2d44
```

Pass that id as `release=` and the run uses exactly those rows and exactly that label list, however the dataset changes
afterwards. `release="latest"` follows the newest build; `release=None` reads the live annotations (unreviewed and
approved), which can change between runs. `samples.describe()` returns a dict to log next to the run.

## A minimal PyTorch loop

```python
from torch.utils.data import DataLoader
from vidlabel import Dataset, FrameSamples
from vidlabel.torch_data import VideoStream, collate_frames

if __name__ == "__main__":                      # required on Windows and macOS when num_workers > 0
    ds = Dataset.open("my_dataset")
    samples = FrameSamples(ds, release="r3fa91c0b2d44", kinds=("box",), label_offset=1)   # 0 = background
    loader = DataLoader(VideoStream(samples, shuffle_buffer=256), batch_size=8, num_workers=4,
                        collate_fn=collate_frames)
    for epoch in range(10):
        loader.dataset.set_epoch(epoch)         # a different video order each epoch
        for batch in loader:
            images = batch["image"]             # uint8 (B, 3, H, W) if the frames share a size
            boxes, labels = batch["boxes"], batch["labels"]    # one array per sample: boxes differ in count
            ...
```

## What a sample contains

| key | shape / type | meaning |
|---|---|---|
| `image` | `(H, W, 3)` uint8 RGB (NumPy); `(3, H, W)` uint8 (PyTorch) | the frame in displayed orientation; cropped and resized to the view if one is set |
| `boxes` | `(N, 4)` float32 | in `box_format`, in pixels of `image` (or fractions with `normalized=True`) |
| `labels` | `(N,)` int64 | class index: position in `class_names`, plus `label_offset` |
| `track_ids` | `(N,)` int64 | `-1` when the annotation has no track |
| `annotation_ids` | `(N,)` int64 | to trace a target back to its row |
| `masks` | `(N, H, W)` uint8 | only with `with_masks=True`; aligned with `boxes` |
| `frame_labels` | `(C,)` float32 | multi-hot of the time **markers** active on this frame, indexed by `class_names` |
| `video_id`, `view_id`, `frame_index` | | where it came from |

Markers never become boxes. They fill `frame_labels`, which is how "this part of the video is a goal" reaches a
classifier or a temporal model alongside the box targets.

## Choosing what to train on

`FrameSamples(ds, ...)` takes:

| argument | default | meaning |
|---|---|---|
| `release` | `"latest"` | a release id, `"latest"`, or `None` for live annotations |
| `view_id` | `None` | train on that view's crop with only that view's annotations; `None` = full frames, all annotations |
| `videos` | all | restrict to these video ids (a train/validation split is just two lists) |
| `kinds` | `("box", "mask")` | which kinds define the samples and the targets: `"box"`, `"mask"`, `"marker"`. Markers always fill `frame_labels`; add `"marker"` to also sample frames that *only* have a marker |
| `labels` | the vocabulary | keep only these labels; class indices then follow this order |
| `origins` | all | e.g. `["human"]` to train on human labels only |
| `min_confidence` | `None` | drop model rows below this confidence (rows without a confidence are kept) |
| `frames` | `"annotated"` | `"annotated"`: frames that have a selected annotation; `"all"`: every frame |
| `stride` | `1` | keep every Nth frame of that list |
| `with_masks` | `False` | decode masks into the sample |
| `box_format` | `"xyxy"` | `"xyxy"` (torchvision, Pascal VOC), `"xywh"` (COCO), `"cxcywh"` (YOLO) |
| `normalized` | `False` | boxes as fractions of the image size (YOLO files use `cxcywh` normalized) |
| `label_offset` | `0` | added to class indices, e.g. `1` for detectors that reserve 0 for background |
| `roots` | `()` | folders to search for videos whose recorded path is stale |

Stored boxes are always pixel corners; `box_format` only changes what training code receives
([why](design-decisions.md#boxes-are-stored-as-pixel-corners)). Which frames to use is decided **from the annotation
table alone**, without opening a video, so class balancing, rare-label selection or "only low-confidence model boxes"
are table questions. The two layers are visible here without any video:

<!-- doctest -->
```python
import numpy as np
from vidlabel import Dataset, FrameSamples, VideoMetadata

ds = Dataset.create("train_demo")
ds.add_video(VideoMetadata(video_id="clip", frame_count=300, fps={"num": 30, "den": 1}, width=320, height=240,
                           sources=[{"uri": "/path/to/clip.mp4"}]))
ds.add_labels(["person", "car", "scene_change"])
ds.add_annotations([
    {"kind": "box", "label": "person", "start_frame": 10, "box": (20, 30, 80, 150), "track_id": 1, "origin": "model"},
    {"kind": "box", "label": "car", "start_frame": 10, "box": (100, 60, 220, 140), "origin": "human"},
    {"kind": "marker", "label": "scene_change", "start_frame": 8, "end_frame": 12, "origin": "human"},
], "clip")
release, _ = ds.build()

samples = FrameSamples(ds, release=release.release_id, kinds=("box",))
print(len(samples), samples.index)                    # 1 [('clip', 10)]   (only frames with a box)
print(samples.class_names)                            # ['person', 'car', 'scene_change']

image = np.zeros((240, 320, 3), np.uint8)             # normally samples.reader("clip").read(10)
s = samples.sample("clip", 10, image)
print(s["boxes"].tolist(), s["labels"].tolist(), s["frame_labels"].tolist())
assert sorted(s["labels"].tolist()) == [0, 1] and s["frame_labels"].tolist() == [0.0, 0.0, 1.0]   # (box order can vary)

yolo = FrameSamples(ds, release=release.release_id, box_format="cxcywh", normalized=True).sample("clip", 10, image)
print(np.round(yolo["boxes"], 3).tolist())            # centre x, centre y, width, height as fractions
```

Without PyTorch, loop over `samples.index` yourself: `samples.reader(video_id)` gives a `FrameReader` for the pixels, and
`samples.sample(video_id, frame, image)` the full sample. That is all the PyTorch classes do.

## Reading video quickly: stream, do not shuffle

Video is stored as chains of frames, so reading frame N normally means decoding from the start of the file. The two
PyTorch datasets reflect that:

- **`VideoStream`** (recommended) is an iterable dataset that visits each video **once, front to back**, so every frame
  is decoded at most once per epoch. Whole videos are shared between DataLoader workers and distributed ranks (use at least
  as many videos as workers). Randomness comes from `shuffle_videos=True` (video order, reseeded by `set_epoch(n)`) and a
  `shuffle_buffer` of samples. Seeded and deterministic. Measured: about 90 to 110 samples per second in one process on a
  720p clip.
- **`VideoFrames`** is map-style (`dataset[i]`), so any sampler works. Going backwards in a video costs a re-decode, so it
  is for validation, debugging and small datasets. Measured on a 720p clip, shuffled access ran at 2 to 16 samples per
  second depending on how often the video has keyframes.

Because streaming gives neighbouring frames of one video in a row, batches are less varied than a full shuffle;
`shuffle_buffer` and many videos reduce that.

### Seeking, and when it is not used

`FrameReader` (what `VideoFrames` uses) can jump to the nearest keyframe instead of decoding from the start: reading
frame 17,000 of a 10-minute clip dropped from 3.8 s to 0.01 s. OpenCV's seek is exact for ordinary constant-frame-rate
files but was measurably **wrong on variable-frame-rate video** (about three frames in four were off by several frames in
our test), which would silently misalign every box. So seeking is used only after a one-off check, per video and
process (about a second), that it returns exactly the frames sequential decoding does; videos flagged variable-rate, or
that fail the check, are read sequentially with a warning. `vidlabel check-video my_dataset clip --seek` runs a wider check.

## Several pipelines: views

`FrameSamples(ds, view_id="lower_strip")` yields the cropped (and `output_size`-resized) frame, with boxes and masks already
in its pixels and only that view's annotations. Use one `FrameSamples` per view ([views](views.md)).

## Transforms and augmentation

Pass `transform=fn` to `VideoStream` or `VideoFrames`: it receives the sample dict (as tensors) and returns one. Anything that
changes the image geometry (flips, crops, resizes) must change `boxes` and `masks` to match; vidlabel does not do this for you.

## Not built yet

`FrameSamples` builds one fixed list of `(video, frame)` pairs. There is no built-in per-epoch plan for class-balanced or
weighted sampling. The intended shape is to pick the frames from the table first, then read them in sorted order per
video, which gets arbitrary sampling at streaming speed. Until then, narrow what is sampled with `videos=`, `labels=`, `origins=`,
`min_confidence=` and `stride=`, or write your own loop over the table and use `samples.reader()` and `samples.sample()` for the pixels and targets.

## Running on Windows, macOS and Linux

The loader is plain Python, NumPy and OpenCV and is tested in CI on all three, including DataLoader workers started with
`fork` (the Linux default) and `spawn` (the default on Windows and macOS). With `num_workers > 0`, keep your training code
under `if __name__ == "__main__":` on Windows and macOS: that is a PyTorch requirement, not a vidlabel one.
