# vidlabel

An open, directory-based format and Python library for **video annotation data**: time markers and spans, bounding
boxes, masks, tracks and labels, all in one annotation table. It is meant to be the thing that sits between your
labelling tools and your training code.

> **Status:** version 0.5.0, pre-release. "vidlabel" is a working name. It is installed from a source checkout for now
> (nothing is published to PyPI yet). The format and the library are tested on Linux, Windows and macOS; see
> [what is and is not verified](#what-is-and-is-not-verified).

## The idea in one minute

- **One annotation table.** A marker ("the ball is hit here"), a box, a mask and a track are all rows with the same
  columns. *Who made a row* (a person, a model, an imported file) is a column (`origin`, `tool`, `author`), not a
  separate folder or format.
- **Video pixels are never stored.** A dataset records where each video is, plus facts about it (frame rate, size,
  rotation, a content hash), and the annotations that refer to its frames.
- **Plain files.** A dataset is a folder of JSON and [Parquet](https://parquet.apache.org/) files, in the spirit of
  Zarr and GEFF: a small spec, a validator, an `extra` field for your own data, and a spec package separate from the
  library. No server, no database.
- **Built for training.** `vidlabel build` compiles a sorted, content-hashed release you can pin in a config, and a
  PyTorch loader turns a release into frames with boxes, masks and labels.
- **Crops are first-class.** If one pipeline looked at the top of the frame and another at a ticker along the bottom,
  each is a *view*. Annotations always live in full-frame coordinates; the library converts for you.

```
my_dataset/
  manifest.json                 videos, views, spec version
  vocab.json                    the one shared label list
  videos/<video_id>/
    metadata.json               fps, frame count, size, rotation, content hash, where the file has been seen
    annotations.parquet         every annotation for this video
    frame_index.json            optional: per-frame timestamps and keyframes, for fast and exact seeking
  views/<view_id>.json          a crop / time range of a video
  compiled/<release_id>/        sorted training table + vocabulary snapshot + release.json (made by `vidlabel build`)
```

## A first look

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata

ds = Dataset.create("tour")
ds.add_video(VideoMetadata(video_id="clip", frame_count=300, fps={"num": 30, "den": 1}, width=320, height=240,
                           sources=[{"uri": "/path/to/clip.mp4"}]))
ds.add_labels(["person", "scene_change"])
ds.add_annotations([
    {"kind": "box", "label": "person", "start_frame": 10, "box": (20, 30, 80, 150), "track_id": 1,
     "origin": "model", "tool": "my-detector", "confidence": 0.93},
    {"kind": "marker", "label": "scene_change", "start_frame": 120, "end_frame": 150, "origin": "human"},
], "clip")
assert ds.validate() == []
release, path = ds.build()            # a sorted, content-hashed table; release.release_id is what you pin
print(release.release_id, release.n_rows)
```

## Where to go next

| I want to... | Read |
|---|---|
| install it and make my first dataset | [Getting started](getting-started.md) |
| understand the model: columns, coordinates, rotation, tracks, status, releases | [Concepts](concepts.md) |
| handle several crops of one video (main area plus a ticker, say) | [Views](views.md) |
| store and use masks | [Masks](masks.md) |
| feed a PyTorch (or any) training loop | [Training](training.md) |
| import annotations from another tool, or export to another format | [Writing a converter](converters.md) |
| look up a command or a function | [Command line](cli.md), [Python API](python-api.md) |
| see the exact file format | [Specification](SPEC.md) |
| know why it was built this way | [Design decisions](design-decisions.md) |
| fix a problem | [Troubleshooting and FAQ](faq.md) |

## What is and is not verified

Tested in CI on Linux, Windows and macOS: the spec and validator, reading and writing, views, masks, releases, the
command line, frame reading and previews on real encoded H.264 video, and the PyTorch loader including DataLoader
workers started with both `fork` and `spawn`.

Not yet verified: reading and writing speed at large scale (Parquet was chosen without a head-to-head benchmark against
alternatives such as Lance), several people editing the same video at once (the rule today is one writer per video),
the loader on very large datasets or many GPUs, and variable-frame-rate video in real datasets. These are listed again,
with the reasoning, in [design decisions](design-decisions.md#open-questions).

## Not included, on purpose

There are **no importers or exporters** for other formats (COCO, YOLO, MOT, CVAT, and so on). Formats vary too much for
one library to cover well, and every converter is a small amount of code on top of `add_annotations`. The guide to
[writing your own](converters.md) shows how, with a tested example.
