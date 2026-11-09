# vidlabel wiki

An open, directory-based format and Python library for **video annotation data**: time markers and spans, bounding boxes, masks,
tracks and labels in one annotation table, with views for crops, releases you can pin, and loaders for training. It sits between your
labelling tools and your training code.

> **Status:** version 0.6.0, alpha, published on PyPI (`pip install vidlabel`). The test suite passes in CI on
> Linux, Windows and macOS ([limitations](limitations.md)).

## The idea in one minute

- **One annotation table.** A marker ("the ball is hit here"), a box, a mask and a track are rows with the same columns. *Who made a row*
  (a person, a model, an imported file) is a column, not a separate format.
- **Video pixels are never stored.** A dataset records where each video is and a hash of the file; annotations refer to its frames.
- **Plain files.** A folder of JSON and Parquet. No server, no database; share it by copying the folder.
- **Built for training.** `vidlabel build` compiles a sorted, hashed release you can pin; loaders turn it into frames with targets.
- **Crops are first-class.** Each region or time window a pipeline looks at is a *view*; annotations always live in full-frame coordinates.

## Find your way

| I want to... | go to |
|---|---|
| install it | [Installation](installation.md) |
| make my first dataset in ten minutes | [Getting started](getting-started.md) |
| understand the model (columns, coordinates, tracks, status) | [Concepts](concepts.md), [Glossary](glossary.md) |
| follow a complete example | [Animal tracking](tutorial-animal-tracking.md), [Satellite tiles](tutorial-satellite-tiles.md) |
| do a specific task | [How-to recipes](howto.md) |
| handle several crops or time windows of one video | [Views](views.md) |
| store masks | [Masks](masks.md) |
| fix a validation error | [Validation](validation.md) |
| pin the data a run used | [Releases](releases.md) |
| train with PyTorch, and read video fast | [Training](training.md), [Performance](performance.md) |
| bring in annotations from another tool | [Writing a converter](converters.md) |
| look up a command or function | [Command line](cli.md), [Python API](python-api.md) |
| read the exact format | [Specification](SPEC.md) |
| know why it is built this way | [Design decisions](design-decisions.md), [Architecture](architecture.md) |
| check what works, what is unverified | [Limitations and roadmap](limitations.md), [Compatibility](compatibility.md) |
| share a dataset safely | [Security](security.md) |
| fix a problem | [Troubleshooting and FAQ](faq.md) |
| change vidlabel | [Development](development.md), [Changelog](changelog.md) |

## What a dataset looks like

```
my_dataset/
  manifest.json                 videos, views, spec version
  vocab.json                    the one shared label list
  videos/<video_id>/
    metadata.json               fps, frame count, size, rotation, content hash, where the file has been seen
    annotations.parquet         every annotation for this video
    frame_index.json            optional: per-frame timestamps and keyframes, for fast and exact seeking
  views/<view_id>.json          a crop / time ranges of a video
  compiled/<release_id>/        sorted training table + vocabulary snapshot + release.json (made by `vidlabel build`)
```

## A first look

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata

ds = Dataset.create("tour")
ds.add_video(VideoMetadata(video_id="clip", frame_count=300, fps={"num": 30, "den": 1}, width=320, height=240))
ds.add_labels(["person", "scene_change"])
ds.add_annotations([
    {"kind": "box", "label": "person", "start_frame": 10, "box": (20, 30, 80, 150), "track_id": 1,
     "origin": "model", "tool": "my-detector", "confidence": 0.93},
    {"kind": "marker", "label": "scene_change", "start_frame": 120, "end_frame": 150, "origin": "human"},
], "clip")
assert ds.validate() == []
release, path = ds.build()
print(release.release_id, release.n_rows)
```

## Where this wiki comes from

These pages are generated from the `docs/` folder of the repository, where every runnable example is executed by the test suite and
every link is checked. To fix a page, change `docs/`, not the wiki ([Development](development.md#the-github-wiki)).
