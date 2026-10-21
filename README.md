# vidlabel (working name)

An open, directory-based format and Python library for **video annotation data**: time markers and spans, bounding
boxes, masks, tracks and labels, all in one annotation table. *Who made a row* (a person, a model, an imported file) is
a column, not a separate structure. Video pixels are **not** stored; datasets point at your videos.

Inspired by Zarr and GEFF: a tiny spec, a validator, an `extra` field for custom data, and a spec package separate from
the library. A dataset is a folder of JSON and Parquet files; there is no server.

```
my_dataset/
  manifest.json                 videos, views, spec version
  vocab.json                    the one shared label list
  videos/<video_id>/
    metadata.json               fps, frame count, size, rotation, content hash, where the file has been seen
    annotations.parquet         every annotation for this video
  views/<view_id>.json          a crop / time range of a video; annotations stay in full-frame coordinates
  compiled/<release_id>/        sorted training table + vocabulary snapshot (made by `vidlabel build`)
```

## Quick start

```bash
pip install -e packages/vidlabel-spec -e packages/vidlabel
python examples/make_sample.py /tmp/sample
vidlabel info /tmp/sample ; vidlabel validate /tmp/sample --deep ; vidlabel build /tmp/sample
```

```python
from vidlabel import Dataset, Compiled
ds = Dataset.open("my_dataset")
ds.add_annotations([{"kind": "box", "label": "person", "start_frame": 10, "box": (20, 30, 80, 150),
                     "origin": "model", "tool": "detector"}], "clip1")
ds.annotations("clip1", kind="box", frames=(0, 100))          # a pyarrow Table
release, path = ds.build()                                    # pin release.release_id in your training config
```

Optional extras: `pip install -e "packages/vidlabel[video]"` (OpenCV: previews, frame reading) and
`pip install -e "packages/vidlabel[torch]"` (adds PyTorch for the training loader). Registering a video file also needs
`ffprobe` (part of ffmpeg).

## What it does

- **One table for everything.** Markers, boxes, masks and tracks share columns; `origin`, `tool` and `author` say who made them.
- **Views.** Several crops of one video, each processed by its own pipeline, stored in one set of full-frame coordinates.
- **Validation.** Strict models and over fifty documented checks; writes are all-or-nothing and `build` refuses invalid data.
- **Masks** stored losslessly (crop, bit-pack, zstd): smaller and about 6 times faster to decode than COCO RLE in our measurements.
- **Releases.** `vidlabel build` makes a sorted table whose id is a hash of its contents, with a label snapshot, so a run can be reproduced exactly.
- **Check against the real video.** `vidlabel check-video` and `vidlabel preview` (OpenCV) draw annotations on actual frames.
- **Training.** A framework-free sample builder and PyTorch datasets (streaming and map-style), including crops and box formats.
- **Runs on Linux, Windows and macOS** (tested in CI, with and without DataLoader workers).

## Documentation

Start at **[docs/index.md](docs/index.md)**.

| | |
|---|---|
| [Getting started](docs/getting-started.md) | install, first dataset, validate, build |
| [Concepts](docs/concepts.md) | columns, coordinates and rotation, origin, tracks, releases |
| [Views](docs/views.md) | several crops of one video |
| [Masks](docs/masks.md) | the codec and helpers |
| [Training](docs/training.md) | from a release to batches |
| [Writing a converter](docs/converters.md) | import from and export to other formats |
| [Command line](docs/cli.md) · [Python API](docs/python-api.md) | reference, generated from the code |
| [Specification](docs/SPEC.md) | the file format, with every validation code |
| [Design decisions](docs/design-decisions.md) | why it is this way, what was measured, what is open |
| [Troubleshooting and FAQ](docs/faq.md) | common problems |

## Status

Version 0.4.0, pre-release; the name is provisional and nothing is on PyPI yet. Tested: spec and validator, reading and
writing, views, masks, releases, the command line, frame reading and previews on real H.264 video, and the PyTorch loader.
**Not yet verified:** Parquet against other storage at scale, several people editing one video at once, the loader on very
large datasets or many GPUs, variable-frame-rate video in real datasets. There are **no importers or exporters** for other
formats on purpose; [the converter guide](docs/converters.md) shows how to write one.

MIT licensed. Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
