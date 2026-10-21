# Concepts

The model in detail. Everything here is normative in the [specification](SPEC.md); this page explains the reasons and
the practical consequences.

## A dataset on disk

```
my_dataset/
  manifest.json                 spec version, list of video ids, list of view ids
  vocab.json                    the one shared label list (names, optional parent/description)
  videos/<video_id>/
    metadata.json               facts about the video file (below)
    annotations.parquet         every annotation of this video, all kinds together
  views/<view_id>.json          a named crop and/or frame range of one video
  compiled/<release_id>/        made by `vidlabel build`: annotations.parquet, vocab.json, release.json
  compiled/LATEST               the id of the newest release
```

IDs (video, view) use letters, digits, `.`, `_` and `-` only. Every file is ordinary JSON or Parquet, so any tool that
reads those can read a dataset; the library adds validation, queries and the training loader.

Each video has its **own** annotation file. That is deliberate: people or jobs working on different videos never
touch the same file. Writes are atomic (a temporary file is renamed into place) but not locked, so the rule is
**one writer per video at a time**.

## Videos

A video is registered once and described by `metadata.json`. The video file is never copied or modified.

| field | meaning |
|---|---|
| `video_id` | the name annotations refer to |
| `frame_count` | number of frames |
| `fps` | frame rate as an exact fraction `{num, den}`: 30000/1001 is 29.97, which a decimal cannot hold exactly |
| `timebase` | the container's clock tick as a fraction (optional) |
| `is_vfr` | true if the frame rate varies; see [troubleshooting](faq.md#variable-frame-rate-video) |
| `width`, `height` | the **stored** frame size |
| `rotation` | clockwise degrees (0, 90, 180, 270) to apply to the stored frame for display; see below |
| `codec`, `pix_fmt`, `bit_rate`, `duration_s` | descriptive, from ffprobe |
| `content_hash` | `sha256:<hex>` of the file: detects a re-encoded or wrong file |
| `sources` | where the file has been seen (`uri`, optional `note`): paths are *hints* and can go stale |
| `extra` | your own data |

`Dataset.resolve_video_path(video_id, roots)` tries each recorded source and then searches `roots` for the same file
name, which is how a dataset moved to another machine finds its videos again (`--root` on the commands that read video).

## Frames and time

Frames are **0-based**, in display order. `end_frame` is **inclusive**. A single moment has `start_frame == end_frame`.
A frame index is a position in the decoded stream, not a timestamp; for constant-frame-rate video time is
`frame * den / num` seconds (`fps` is `num/den` frames per second).

## Coordinates and rotation

Phones record video "sideways" and set a rotation flag so players turn it upright. vidlabel keeps both facts: `width` and
`height` are the **stored** size, and `rotation` is how many degrees **clockwise** to turn the stored frame to display it.
For 90 and 270 the displayed width and height are swapped (`VideoMetadata.display_size`).

**All annotation coordinates are in the displayed orientation**, the picture a person sees in a player: pixels, origin at
the top-left, `x1` and `y1` exclusive, and always in *full-frame* coordinates even for [views](views.md). So a box
`(x0, y0, x1, y1)` covers pixels `x0 <= x < x1`, `y0 <= y < y1`.

```
stored 1920x1080, rotation 90          displayed 1080x1920
+----------------------+                +-----------+
| top is the left side |   turn 90      |           |
|  of what you see     |  clockwise ->  |  upright  |
+----------------------+                |           |
                                        +-----------+
```

ffprobe reports the display matrix counter-clockwise; vidlabel negates it, so a file that ffmpeg marked
`-display_rotation 90` is stored as `rotation: 270`. If your source drew boxes on the **unrotated stored** frame,
convert them with `vidlabel.boxes.rotate_boxes(boxes, rotation, (stored_width, stored_height))`.
The frame readers apply the rotation themselves, so what they return matches the coordinates. If boxes look misplaced in
a preview of a phone video, rotation is the first thing to check.

## The annotation table

One row per annotation, the same columns for every kind. Stored as Parquet (zstd, row groups of about 20,000 rows,
statistics and page index on), sorted by `start_frame` then `annotation_id` within each video's file.

| column | type | required | meaning |
|---|---|---|---|
| `annotation_id` | int64 | yes | Stable unique id. Random 63-bit integer assigned on creation; never reused. |
| `video_id` | string | yes | Which video (matches videos/<video_id>/). |
| `start_frame` | int32 | yes | First frame, 0-indexed in display order. A point in time has start_frame == end_frame. |
| `end_frame` | int32 | yes | Last frame, inclusive. |
| `kind` | string | yes | marker / box / mask |
| `label` | string | yes | A label name from vocab.json. |
| `track_id` | int32 |  | Ties boxes/masks of one object across frames. One id means one object within a video and view (rows labelled in different views have separate id spaces). |
| `x0` | float32 |  | Box left edge, pixels, source-video coordinates. |
| `y0` | float32 |  | Box top edge, pixels (origin top-left). |
| `x1` | float32 |  | Box right edge (exclusive). |
| `y1` | float32 |  | Box bottom edge (exclusive). |
| `mask` | binary |  | Encoded mask bytes, interpreted by mask_codec. |
| `mask_codec` | string |  | How `mask` is encoded. |
| `view_id` | string |  | Optional: the view (crop / focused version) this was labelled in. Geometry is always in source-video coordinates. |
| `origin` | string | yes | human / model / imported / other, or a custom value starting with 'x-' (e.g. x-crowd) |
| `tool` | string |  | Tool or model name that produced it. |
| `tool_version` | string |  | Tool or model version. |
| `author` | string |  | Person responsible, if any. |
| `confidence` | float32 |  | 0..1, usually only for model output. |
| `status` | string | yes | unreviewed / approved / rejected |
| `extra` | string |  | Optional JSON object (as text) for anything custom. Use your own top-level key, e.g. extra.mytool. |

### Kinds

| kind | frames | geometry | rules |
|---|---|---|---|
| `marker` | a point or a span (`start_frame <= end_frame`) | none | must not carry a box or mask |
| `box` | exactly one (`start_frame == end_frame`) | `x0, y0, x1, y1` | `x1 > x0`, `y1 > y0`, finite |
| `mask` | exactly one | a mask, and the box columns hold its bounding box | the box equals the mask's pixel bounding box (checked by `validate --deep`) |

To say "this object is in frames 10 to 20", add a box (or mask) per frame sharing a `track_id`; to say "this
*segment of the video* is a goal", add a marker over the span. Markers are how full-frame or time-only labelling is
stored; they are not special-cased anywhere else.

### Origin, tool, author and `extra`

*Who or what made a row* is a column, never a separate structure, so human and model output sit in one table and one
query can compare them.

- `origin` (required): `human`, `model`, `imported` (converted from somewhere else), `other`, or **your own value that
  starts with `x-`** (lowercase letters, digits, `.`, `_`, `-`; for example `x-crowd`). The fixed set keeps filters
  reliable and the `x-` prefix catches typos like `Human`.
- `tool` and `tool_version`: the specific model, labelling tool, or (for `imported`) the source the data came from.
- `author`: the person responsible, if any.
- `confidence`: 0 to 1, normally only for model output.
- `extra`: a JSON object (stored as text) for anything else. Use **your own top-level key** (`extra.mytool`) so
  different tools never collide. Pass a dict; the library serialises it.

Core objects reject unknown fields on purpose (a typo should be an error, not silently stored); `extra` is the one place
for custom data, in the annotation table and in every metadata object.

### Status

`unreviewed` (the default), `approved`, `rejected`. A rejected annotation is kept in the file, so a reviewer's decision is
not lost, but `build` leaves it out unless you pass `--include-rejected`.

### Tracks

`track_id` ties the boxes and masks of **one object** across frames. A track id identifies one object within one video
**and view** (rows labelled in different views have separate id spaces, so two pipelines can each number from 1). The
validator checks that

- one track does not carry more than one label (`track_label_conflict`, a warning, since some pipelines reclassify);
- one track does not have two boxes, or two masks, in the same frame (`duplicate_track_in_frame`, an error).

Rejected rows are ignored by both checks, because a reviewer who rejects a box normally redraws it under the same id.
A box and a mask of the same track in the same frame are allowed. Appending rows also checks them against the rows
already stored for the same tracks, so a later call cannot silently duplicate an earlier one.

### Ids

`annotation_id` is a random 63-bit integer assigned when a row is created, unique across the whole dataset, and never
reused. You may supply your own (for stable ids from another system); it must not already exist.

## Vocabulary

`vocab.json` is the single shared list of label names, each with an optional `parent` (a label that must itself be in
the list), `description` and `extra`. Every annotation's `label` must be in it. The library only ever **appends** labels, so a
label's position, and the class index the training loader derives from it, does not change; releases also snapshot the list.

## Releases

`vidlabel build` merges every video's annotations into one table sorted by (`video_id`, `start_frame`,
`annotation_id`), leaves out rejected rows by default, writes it to `compiled/<release_id>/annotations.parquet` and
snapshots `vocab.json` next to it. The **release id is derived from the table's contents** (`r` plus the first 12 hex
characters of its SHA-256), so the same data and filters always give the same id, and a changed annotation gives a new
one. A training run that records its release id can be reproduced exactly.

You can build a subset (`origins`, `kinds`, `video_ids`, `view_ids`, `statuses`), and `Compiled.clip(video, start, end)`
reads a frame range quickly by using the Parquet statistics to skip row groups that cannot match. `build` refuses to run
if validation finds errors.
