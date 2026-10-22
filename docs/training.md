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

Video is stored as chains of frames: most frames are only the *difference* from earlier ones, and decoding can only start at
a **keyframe**. Reading frame N therefore costs a decode from the keyframe before it. With a keyframe every 250 frames (common
for H.264), an average random frame means decoding about 125 frames. The two PyTorch datasets reflect that:

- **`VideoStream`** (recommended) is an iterable dataset that visits each video **once, front to back**, so every frame
  is decoded at most once per epoch. Whole videos are shared between DataLoader workers and distributed ranks (use at least
  as many videos as workers). Randomness comes from `shuffle_videos=True` (video order, reseeded by `set_epoch(n)`) and a
  `shuffle_buffer` of samples. Seeded and deterministic. Measured: about 90 to 110 samples per second in one process on a
  720p clip.
- **`VideoFrames`** is map-style (`dataset[i]`), so any sampler works, but every random frame pays the keyframe cost above.
  It is for validation, debugging, small datasets, and (with the cache below) random sampling.

Because streaming gives neighbouring frames of one video in a row, batches are less varied than a full shuffle;
`shuffle_buffer` and many videos reduce that.

### What random access costs

Measured on one machine with a synthetic 1280x720 H.264 clip (3000 frames, a keyframe every 250), one CPU process, 120 random
frames. Expect different absolute numbers on real footage and hardware; the ratios are what to take away.

| how the frames are read | frames per second |
|---|---|
| every frame in order (what `VideoStream` does) | about 400 |
| random frames, one at a time | about 6 (170 ms each) |
| random frames read as a batch of 8, 32, 120 (`read_many`) | about 7, 10, 28 |
| random frames from a JPEG frame cache | about 350 (3 ms each) |
| random frames from a PNG frame cache | about 135 (7 ms each) |

### The frame index

`vidlabel add-video` also writes a **frame index**, and `vidlabel index my_dataset` adds one to videos registered without it:
the timestamp of every frame and which frames are keyframes, read from the container without decoding (seconds, even for long
videos). `FrameReader` (what the loaders use) picks it up automatically and then:

- **jumps only when it saves work.** Inside the keyframe interval it is already in, it reads on; it seeks when the target is
  behind it or a keyframe is much nearer than reading on.
- **checks every frame it returns** against the index: the decoder's timestamp must be the one stored for that frame. If
  they ever disagree it warns, ignores the index and decodes sequentially, so a wrong frame is never returned silently.
- **works on variable-frame-rate video.** OpenCV's seek-by-frame-number was measured to be wrong there (29 of 40 jumps landed on
  the wrong frame); the reader aims by time instead and corrects itself from the index.

The index does **not** make decoding cheaper: a single random read is as slow as before (166 ms against 173 ms in the table
above). What it buys is correctness on variable-frame-rate video, no more one-off pixel check per file, and the ability to
plan reads (next section). An index for another file (hash or frame count differs) is ignored with a warning;
`vidlabel validate` reports `frame_index_stale` and `frame_index_length`.

Without an index the reader behaves as before: it seeks only after a one-off per-file pixel comparison with sequential
decoding, never in variable-frame-rate video, and warns if the check fails.

### Reading a batch: `read_many`

`FrameReader.read_many(frames)` returns the frames in the order asked but reads them in increasing order, so each keyframe
interval is decoded once however the request is ordered. A `DataLoader` over `VideoFrames` with `batch_size > 1` uses this
automatically (`VideoFrames.__getitems__`), grouping the batch by video. The gain depends on how many frames of a batch share a
keyframe interval, which is why the table shows 1.2x at 8 frames and 4.8x at 120 for a video with only 12 intervals. Large
batches from few videos benefit most; a batch of 8 frames from 8 different videos benefits not at all.

### Making random access fast

If you need uniform random sampling, make the frames cheap to reach. `vidlabel check-video my_dataset clip` shows the keyframe
spacing and the average decode cost per random frame, and prints the fix when it is high. Two options:

1. **A frame cache** (nothing about the dataset changes):

   ```bash
   vidlabel cache my_dataset /fast/disk/frames --release latest
   ```

   decodes exactly the frames a run will use, in order and at full speed, and saves them as JPEG (about 88 KB each at 720p)
   or PNG (`--format png`, exact pixels, about 175 KB). Then `FrameSamples(ds, cache="/fast/disk/frames")` reads them as image
   files, and anything not cached falls back to the video. Building runs at about 50 (JPEG) or 25 (PNG) frames per second
   in our test. The cache is *derived* data: it lives outside the dataset, can be deleted and rebuilt any time, records the
   content hash of each video and is ignored for a video whose file changed, and holds whole frames so one cache serves every view.
   JPEG is lossy: pixels differ slightly from decoding the video.
2. **Re-encode with more keyframes** (`-g 30`, shown by `check-video`): about 8 times less decoding per random frame, at the
   cost of a larger file. It makes a *different* file with slightly different pixels, so register it as a new video.

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
