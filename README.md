# vidlabel (working name)

An open, directory-based format and Python library for **video annotation data**: time markers and spans, bounding
boxes, masks, tracks, labels. Everything lives in one annotation table; *who made it* (a person, a model, an imported file)
is a column, not a separate structure. Video pixels are **not** stored; datasets point at your videos.

Inspired by Zarr / GEFF: a tiny spec, a validator, an `extra` field for custom data, and a spec package separate
from the library.

```
my_dataset/
  manifest.json                 videos, views, spec version
  vocab.json                    the one shared label list
  videos/<video_id>/
    metadata.json               fps, frame count, size, content hash, where the file has been seen
    annotations.parquet         every annotation for this video (zstd, row groups, page index)
  views/<view_id>.json          a crop / time range of a video; annotations stay in source coordinates
  compiled/<release_id>/        sorted training table + release.json (made by `vidlabel build`)
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
ds.annotations("clip1", kind="box", frames=(0, 100))          # pyarrow Table
release, path = ds.build()                                    # pin release.release_id in your training config
Compiled(path).clip("clip1", 100, 130)                        # fast: only matching row groups are read
```

## Views: several crops of one video

```python
ds.add_view({"view_id": "lower_strip", "video_id": "clip",
             "region": {"x": 0, "y": 620, "width": 1280, "height": 100}, "allowed_labels": ["caption_text"]})
# boxes straight from a model that only saw the strip, in the crop's own pixels; stored in full-frame coordinates
ds.add_annotations([{"kind": "box", "label": "caption_text", "start_frame": 5, "box": (40, 30, 200, 70),
                     "origin": "model", "view_id": "lower_strip"}], "clip", coords="view")
ds.annotations("clip", view_id="lower_strip")
ds.build(view_ids=["lower_strip"])        # a release for just that view
```

## Check your annotations against the real video

```bash
pip install "vidlabel[video]"                       # adds OpenCV (source checkout: pip install -e "packages/vidlabel[video]")
vidlabel check-video my_dataset clip                # file on disk matches the metadata? (size, frame count, --hash)
vidlabel preview my_dataset clip --out preview_out  # draws boxes, masks and markers on real frames
vidlabel preview my_dataset clip --view lower_strip # a view's crop, with only that view's annotations
```

From Python: `vidlabel.frames.read_frames(path, [0, 10, 250], meta)` returns frame-accurate RGB frames in displayed
orientation, optionally cropped to a view.

## Masks

Stored losslessly as: crop to the mask's bounding box, bit-pack rows, zstd (`bitpack_zstd`). In benchmarks this was
0.64-0.84x the size of COCO RLE with about 6x faster decoding. COCO RLE can be converted with `masks.from_coco_rle` /
`masks.to_coco_rle` but is never stored. A smaller, slower arithmetic-coded codec is a possible future addition
(`mask_codec` is a per-row column, so codecs can mix).

## Status

v0.1, early. Tested: spec, mask codec, read/write/validate, views (coordinate conversion, inside-view checks), build + clip reads, CLI.
**Not yet verified:** Parquet-vs-Lance speed at scale,
multi-writer conflicts (one writer per video at a time for now), PyTorch loader.

See `docs/SPEC.md` for the format. MIT licensed.
