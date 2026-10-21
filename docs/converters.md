# Writing a converter

vidlabel ships **no importers or exporters** for other formats. COCO, YOLO, MOT, CVAT, a spreadsheet someone made: they
differ in coordinates, frame numbering, label names, how tracks and time spans are written, and what is even recorded.
One library cannot cover that well, and a converter is usually a page of code on top of
[`Dataset.add_annotations`](python-api.md#datasetadd_annotations). This guide shows how to write one that is correct, and
the example below is a real script that the test suite runs.

## The shape of every converter

1. **Register the video** (once): `ds.add_video(probe_video("clip.mp4", "clip"))`.
2. **Read your source** and turn each thing in it into a *row*: a dict using the [column names](concepts.md#the-annotation-table).
3. **Call `ds.add_annotations(rows, video_id)`** once per video. It validates every row first and writes **nothing** if
   any is invalid, raising `ValidationError` that says which check failed and for how many rows.
4. **Check the result**: `vidlabel validate`, then `vidlabel preview` to see the boxes on real frames.

## A complete example

The made-up input is a CSV of tracked boxes (`frame,track,class,x,y,w,h,score`). It lives in
[`examples/convert_csv.py`](../examples/convert_csv.py):

```python
"""Example converter: read a simple CSV of tracked boxes and add it to a vidlabel dataset.

vidlabel ships no importers because every source format differs; this is the pattern to copy. The made-up input is:

    frame,track,class,x,y,w,h,score
    0,1,Car,100,50,40,30,0.91

(frame is 0-based, x/y/w/h are pixels of the displayed frame, track and score may be empty.)

Run it:  python examples/convert_csv.py my_dataset clip detections.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from vidlabel import Dataset


def convert(ds: Dataset, csv_path: str | Path, video_id: str, *, tool: str = "csv-converter", tool_version: str = "1",
            replace: bool = True) -> list[int]:
    """Add the boxes in `csv_path` to `video_id`. Returns the new annotation ids.

    replace=True first removes what an earlier run of this converter added to the video, so re-running does not
    duplicate rows.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for line_no, rec in enumerate(csv.DictReader(f), start=2):        # start=2: line 1 is the header
            x, y, w, h = (float(rec[k]) for k in ("x", "y", "w", "h"))
            rows.append({
                "kind": "box",
                "label": rec["class"].strip().lower(),                    # map the source's names to your vocabulary here
                "start_frame": int(rec["frame"]),                         # add or subtract 1 here if the source counts from 1
                "box": (x, y, x + w, y + h),                              # vidlabel stores corners: x0, y0, x1, y1
                "track_id": int(rec["track"]) if rec.get("track") else None,
                "confidence": float(rec["score"]) if rec.get("score") else None,
                "origin": "imported",                                     # where the row came from; details go in tool/extra
                "tool": tool,
                "tool_version": tool_version,
                "extra": {"csv_line": line_no},                           # anything else worth keeping, under your own key
            })
    if replace:
        old = ds.annotations(video_id, origin="imported", columns=["annotation_id", "tool"])
        mine = [i for i, t in zip(old.column("annotation_id").to_pylist(), old.column("tool").to_pylist()) if t == tool]
        ds.delete_annotations(video_id, mine)
    # create_labels=True adds unseen labels to the vocabulary; leave it off to make unknown labels an error instead.
    # Nothing is written if any row is invalid: the ValidationError says which check failed and for how many rows.
    return ds.add_annotations(rows, video_id, create_labels=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ids = convert(Dataset.open(sys.argv[1]), sys.argv[3], sys.argv[2])
    print(f"added {len(ids)} annotations")
```

Run it with `python examples/convert_csv.py my_dataset clip detections.csv`.

## The checklist: decisions every converter must make

**Frame numbering.** vidlabel frames are 0-based. Some formats count from 1 (MOT does). Check against
`ds.video(video_id).frame_count`: the validator rejects frames past the end (`frame_out_of_range`).

**Coordinates.** Stored boxes are pixel corners `(x0, y0, x1, y1)`, origin top-left, `x1`/`y1` exclusive, **in the
displayed orientation**. Convert whatever your source uses with `convert_boxes`:

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata
from vidlabel.boxes import convert_boxes

ds = Dataset.create("convert_demo")
ds.add_video(VideoMetadata(video_id="clip", frame_count=300, fps={"num": 30, "den": 1}, width=640, height=360))
ds.add_labels(["person", "car", "goal"])
W, H = ds.video("clip").display_size              # use the displayed size, not width/height, for rotated video

# A YOLO-style line: "class cx cy w h", all as fractions of the image size.
names = ["person", "car"]
rows = []
for line in ["0 0.5 0.5 0.25 0.5", "1 0.25 0.25 0.1 0.1"]:
    cls, *vals = line.split()
    box = convert_boxes([[float(v) for v in vals]], "cxcywh", "xyxy", size=(W, H), src_normalized=True)[0]
    rows.append({"kind": "box", "label": names[int(cls)], "start_frame": 42, "box": tuple(float(v) for v in box),
                 "origin": "imported", "tool": "yolo-txt"})
ids = ds.add_annotations(rows, "clip")
first = ds.annotations("clip").sort_by("x0").to_pylist()[0]
print(first["label"], first["x0"])                  # car 128.0 = (0.25 - 0.1 / 2) * 640
assert (first["label"], first["x0"]) == ("car", 128.0)
```

Formats offered by `convert_boxes`: `xyxy`, `xywh` (COCO), `cxcywh` (YOLO), each in pixels or normalized.
If your source drew on the **unrotated stored frame of a rotated phone video**, run the boxes through
`rotate_boxes(boxes, meta.rotation, (meta.width, meta.height))` first ([why](concepts.md#coordinates-and-rotation)).

**Labels.** Map the source's names to your vocabulary. `create_labels=True` adds unseen names; leave it off so a typo is an error.

**Time spans become markers.** A marker has `start_frame` and an **inclusive** `end_frame`. From seconds:

<!-- doctest -->
```python
fps = ds.video("clip").fps.value                       # the exact fraction, as a float
start, end = 2.4, 5.0                                  # seconds; the span is [start, end): frames 72 to 149 at 30 fps
ds.add_annotations([{"kind": "marker", "label": "goal", "start_frame": round(start * fps),
                     "end_frame": round(end * fps) - 1, "origin": "imported", "tool": "event-log"}], "clip")
```

**Tracks.** Put the source's object id in `track_id` (a non-negative integer). One id must mean one object, with one
label, and at most one box per frame ([rules](concepts.md#tracks)). If the source reuses ids across videos or restarts
numbering, that is fine: ids are per video and per view.

**Masks.** Pass a boolean `(height, width)` array the size of the **displayed** frame. From COCO run-length encoding,
`vidlabel.masks.from_coco_rle(rle)` returns that array. Polygons are not rasterised for you; draw them with OpenCV
(`cv2.fillPoly`) or Pillow first. The box is filled in from the mask.

**Provenance: `origin`, `tool`, `author`, `confidence`, `status`, `extra`.**

| set | to |
|---|---|
| `origin` | `"imported"` for converted data (or `"human"` / `"model"` if you know which; or an `x-` value of your own) |
| `tool`, `tool_version` | the source format or program, so you can find and replace what a converter added |
| `author` | the person, if the source records one |
| `confidence` | 0 to 1, if the source has a score |
| `status` | `"unreviewed"` (default) unless the source records a review decision |
| `extra` | anything else worth keeping, **under your own top-level key**, e.g. `{"csv_line": 42}` |

**If the source was labelled on a crop**, add the crop as a [view](views.md) and pass `view_id=...` with
`coords="view"`; vidlabel converts to full-frame coordinates.

**Re-running.** Running a converter twice should not double the data. Either remove what the previous run added (the
example matches on `origin="imported"` and `tool`) or pass your own stable `annotation_id` values (a repeat then fails
loudly instead of duplicating).

**Large imports.** Group rows by video and call `add_annotations` once per video: every call rewrites that video's
file, so many small calls are slow. Rows are validated as a batch.

## Writing an exporter

Reading is a query returning a pyarrow table; filter on `status` to leave out rejected rows, then convert. For example,
YOLO label lines for one frame:

<!-- doctest -->
```python
def yolo_lines(ds, video_id, frame):
    names = [l.name for l in ds.vocab.labels]
    W, H = ds.video(video_id).display_size
    rows = ds.annotations(video_id, kind="box", frames=(frame, frame), status=["unreviewed", "approved"]).to_pylist()
    boxes = convert_boxes([[r["x0"], r["y0"], r["x1"], r["y1"]] for r in rows], "xyxy", "cxcywh", size=(W, H),
                          dst_normalized=True)
    return [f"{names.index(r['label'])} " + " ".join(f"{v:.6f}" for v in b) for r, b in zip(rows, boxes)]

lines = yolo_lines(ds, "clip", 42)
print(lines)
assert len(lines) == 2 and all(len(l.split()) == 5 for l in lines)
```

## Sharing a converter

Publish it as its own package (for example `vidlabel-coco`) that depends only on vidlabel's public API: that keeps the
core small and lets each format move at its own pace. A converter should have a test with a tiny input file and should
call `ds.validate()` on the result. It never needs a spec change: custom data goes in `extra`.

## Common errors

| message | meaning and fix |
|---|---|
| `frame beyond video length` (`frame_out_of_range`) | a frame number past `frame_count`: wrong video, or 1-based numbering |
| `box extends outside the WxH frame` (warning) | coordinates in the wrong orientation or units, or not clipped |
| `label not in vocab.json` (`unknown_label`) | add it with `add_labels` or pass `create_labels=True` |
| `box must be finite with x1 > x0 and y1 > y0` (`bad_box`) | a zero-size box, or `w`/`h` treated as corners |
| `origin must be one of ...` (`bad_origin`) | use `imported`, or an `x-` value |
| `unknown field(s) [...]; put custom data inside 'extra'` | a key that is not a column name: nest it under `extra` |
| `annotation_id N already exists` | you passed an id that is already stored |
| `the same track_id appears more than once in one frame` | duplicate objects share an id, or the file was converted twice |
