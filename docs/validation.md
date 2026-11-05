# Validation: what is checked and how to fix it

`vidlabel validate PATH` (or `ds.validate()`) reads the dataset and reports **issues**. Each issue has a `level` (`error` or
`warning`), a `code`, and a message that names the video, the number of rows affected and, for row problems, where the first one is
("at frame 120 (track 3)"). Errors make the command exit with status 1 and stop `build`; warnings do not.

Validation also runs on the rows you pass to `add_annotations`, so most problems are refused before they reach the file, with a
`ValidationError` that names the failing check. The table of all codes with their meaning is in the [spec](SPEC.md#validation-codes);
this page says what usually causes each and what to do.

```bash
vidlabel validate my_dataset            # fast: reads tables, checks every rule except decoding masks
vidlabel validate my_dataset --deep     # also decodes every mask and checks its box (slower)
```

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata

ds = Dataset.create("v")
ds.add_video(VideoMetadata(video_id="a", frame_count=10, fps={"num": 25, "den": 1}, width=64, height=48))
ds.add_labels(["thing"])
try:
    ds.add_annotations([{"kind": "box", "label": "thing", "start_frame": 99, "box": (1, 1, 5, 5), "origin": "human"}], "a")
except Exception as e:
    print(type(e).__name__, e)
for issue in ds.validate():
    print(issue.level, issue.code, issue.message)
```

## Dataset-level problems

| code | usual cause | fix |
|---|---|---|
| `no_manifest`, `bad_manifest` | the folder is not a dataset, or `manifest.json` was edited by hand and broke | check the path; restore the file from backup; for a new dataset use `vidlabel init` |
| `spec_major` | the dataset was written by an incompatible major version of the spec | use the library version that wrote it; there is no automatic migration across major versions |
| `no_vocab`, `bad_vocab` | `vocab.json` missing or invalid | restore it; labels used by annotations must be listed (`vidlabel add-labels`) |
| `missing_video_dir`, `no_metadata`, `bad_metadata`, `video_id_mismatch` | a video listed in the manifest has no folder, no `metadata.json`, invalid metadata, or metadata for another id | restore the folder from backup, or re-register the video |
| `orphan_video_dir` (warning) | a folder under `videos/` the manifest does not list | register it or delete the folder |
| `manifest_views` (warning) | `manifest.json` lists different views than `views/` holds | re-add the views |

## Frame index

| code | usual cause | fix |
|---|---|---|
| `bad_frame_index`, `frame_index_length` | an index file is malformed or has a different number of frames than the metadata | rebuild it: `vidlabel index PATH VIDEO --force` |
| `frame_index_stale` (warning) | the index was made from a different file (hash differs) than the metadata describes | rebuild it from the right file, or re-register the video |

## Annotation tables

| code | usual cause | fix |
|---|---|---|
| `bad_parquet`, `video_id_column` | a damaged file, or rows copied between videos | restore the file; each video's table must hold only its own rows |
| `duplicate_id`, `duplicate_id_across_videos` | two files merged by hand, or ids chosen by the user | give the rows new ids (`add_annotations` assigns random ones) |
| `missing_column`, `unknown_column` | a table written by another tool | custom data belongs in the `extra` JSON column, not in new columns |
| `null_required` | an empty `kind`, `label`, `start_frame`, `end_frame`, `origin` or `status` | fill the required fields |
| `bad_kind`, `bad_origin`, `bad_status` | a typo (`"Human"`, `"boxx"`) | use `marker`/`box`/`mask`; `human`/`model`/`imported`/`other` or an `x-` value; `unreviewed`/`approved`/`rejected` |
| `negative_frame`, `frame_order`, `frame_out_of_range` | frames count from 0; `end_frame` is inclusive and must be at least `start_frame`; a frame beyond the video | a frame number past the end is often a **1-based** source (a tool that counts from 1): subtract 1 |
| `unknown_label` | a label not in `vocab.json` | `vidlabel add-labels`, or fix the typo |
| `bad_confidence` | confidence outside 0 to 1 | rescale (a model that outputs 0 to 100 needs dividing) |
| `bad_extra` | `extra` is not a JSON object | store a dictionary, e.g. `{"mytool": {...}}` |

## Geometry

| code | usual cause | fix |
|---|---|---|
| `missing_box`, `bad_box` | a box or mask without four finite corners, or an empty box | check for NaN and zero-area boxes in the source |
| `geometry_span` | a box or mask whose `end_frame` is not its `start_frame` | boxes and masks are single-frame; repeat the row per frame, or use a marker for a span |
| `box_outside_frame` (warning) | a box reaches outside the frame, usually a detector's rounding | clip it, or ignore the warning |
| `marker_has_geometry`, `box_has_mask` | geometry on a marker, or a mask attached to a `box` row | use `kind: "mask"` for masks |
| `missing_mask`, `missing_codec`, `unknown_codec` | a mask row without bytes or codec | pass a boolean array to `add_annotations`; it encodes it |
| `mask_frame_size` | a stored mask was encoded for another frame size than the video | re-encode at the video's size; common when a mask from a resized crop was stored without `coords="view"` |
| `mask_undecodable`, `mask_box_mismatch` (`--deep`) | damaged mask bytes, or the stored box differs from the mask's pixel bounding box | re-write the mask through `add_annotations`, which fills the box from the mask |

## Tracks

| code | usual cause | fix |
|---|---|---|
| `bad_track_id` | negative track id | use non-negative integers |
| `track_label_conflict` (warning) | one track has more than one label (an ID switch, or a classifier that changed its mind) | review the track (`vidlabel tracks`), then relabel or split it |
| `duplicate_track_in_frame` | two boxes (or two masks) with the same track id in one frame | give one a different id, or reject one; a box and a mask of one track may share a frame |

Track ids are scoped to one video **and view**, so the same number in different views is not a conflict. Rejected rows are ignored by
both track checks, so a reviewer who rejects a box can redraw it under the same id.

## Views

| code | usual cause | fix |
|---|---|---|
| `bad_view`, `view_id_mismatch`, `view_video`, `view_region`, `view_labels` | a view file is invalid, named differently from its id, points at an unknown video, extends outside the video, or allows labels not in the vocabulary | fix the view definition |
| `unknown_view`, `view_wrong_video` | a row's `view_id` does not exist or belongs to another video | add the view first |
| `box_outside_view`, `frame_outside_view`, `label_not_allowed_in_view` | a row tagged with a view lies outside its region, its frame ranges, or its allowed labels | move the row to the whole frame (no `view_id`), widen the view, or fix the row |

## Working through a failure

1. Run `vidlabel validate PATH` and read the first message; fix that class of problem and run again, since one root cause often
   produces many rows of the same code.
2. For problems about frames, check with the real video: `vidlabel preview PATH VIDEO --frames A-B --out out`.
3. For anything about the file itself (wrong frame count, re-encoded video), use `vidlabel check-video PATH VIDEO` (see
   [troubleshooting](faq.md#checking-against-the-video)).
4. If you inherited a dataset that fails and you do not know why, validate with `VIDLABEL_DEBUG=1` to see the underlying traceback.
