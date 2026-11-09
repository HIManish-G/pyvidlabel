# vidlabel

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
    frame_index.json            optional: per-frame timestamps and keyframes, for fast and exact seeking
  views/<view_id>.json          a crop / time range of a video; annotations stay in full-frame coordinates
  compiled/<release_id>/        sorted training table + vocabulary snapshot (made by `vidlabel build`)
```

## Quick start

```bash
pip install vidlabel                      # the format and the library (installs vidlabel-spec too)
python examples/make_sample.py /tmp/sample      # from a checkout of this repository
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

Optional extras: `pip install "vidlabel[video]"` (OpenCV: previews, frame reading), `"vidlabel[ui]"` (web interface),
`"vidlabel[mcp]"` (agent server) and `"vidlabel[torch]"` (adds PyTorch for the training loader). Registering a video file also needs
`ffprobe` (part of ffmpeg).

## What it does

- **One table for everything.** Markers, boxes, masks and tracks share columns; `origin`, `tool` and `author` say who made them.
- **Views.** Several crops of one video, each processed by its own pipeline, stored in one set of full-frame coordinates.
- **Validation.** Strict models and over fifty documented checks; writes are all-or-nothing and `build` refuses invalid data.
- **Masks** stored losslessly (crop, bit-pack, zstd): smaller and about 6 times faster to decode than COCO RLE in the project's benchmarks (see [masks](docs/masks.md)).
- **Releases.** `vidlabel build` makes a sorted table whose id is a hash of its contents, with a label snapshot, so a run can be reproduced exactly.
- **Check against the real video.** `vidlabel check-video` and `vidlabel preview` (OpenCV) draw annotations on actual frames.
- **For AI agents.** `vidlabel mcp` lets an agent read a dataset and propose boxes, spans and masks over MCP; they wait as unreviewed proposals until a person takes them in and commits. Every read command also prints JSON (`--json`). See [AI agents](docs/agents.md).
- **Web interface.** `pip install "vidlabel[ui]"` then `vidlabel ui my_dataset`: browse, review tracks, validate, build releases, draw boxes, spans and masks, add videos, manage labels and views, and revert changes without the command line.
- **Training.** A framework-free sample builder and PyTorch datasets (streaming and map-style), including crops and box formats.
- **Fast, exact frame access.** A per-video frame index makes seeking safe (even in variable-frame-rate video) and plans reads by keyframe; an optional frame cache turns random access into a file read.
- **Built for Linux, Windows and macOS.** The test suite runs in CI on Linux, Windows and macOS (with and without DataLoader workers). Windows and macOS are covered by CI only.

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

Version 0.6.0, alpha release on PyPI; the format and API can still change between minor versions. Tested: spec and validator, reading and
writing, views, masks, releases, the command line, frame reading and previews on real H.264 video, and the PyTorch loader.
**Not yet verified:** Parquet against other storage at scale, several people editing one video at once, the loader on very
large datasets or many GPUs, variable-frame-rate video in real datasets. There are **no importers or exporters** for other
formats on purpose; [the converter guide](docs/converters.md) shows how to write one.

MIT licensed. Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
