# vidlabel format, spec 0.4.0

Authoritative definitions: `packages/vidlabel-spec` (pydantic models; `vidlabel schema` prints JSON Schema). This page is the
normative summary; [concepts](concepts.md) explains the reasons and [design decisions](design-decisions.md) records the choices.

## Layout
`manifest.json`, `vocab.json`, `videos/<video_id>/metadata.json`, `videos/<video_id>/annotations.parquet`,
`views/<view_id>.json`, optional `compiled/<release_id>/{annotations.parquet,vocab.json,release.json}`, `compiled/LATEST`.
IDs match `[A-Za-z0-9._-]+`.

## Conventions
- **Frames** are 0-indexed in display order. `end_frame` is inclusive. A point has `start_frame == end_frame`.
- **Rotation**: `rotation` is the *clockwise* angle (0, 90, 180, 270) applied to the stored frame for display; `width`/`height`
  are the stored size, and 90/270 swap them for display. (ffprobe reports the display matrix counter-clockwise, so it is negated.)
- **Coordinates** are pixels, origin top-left, in the video's *displayed* orientation (after `rotation`), always in
  full source-video coordinates, even when labelled in a view. `x1`/`y1` are exclusive.
- **Video identity**: `content_hash` (`sha256:<hex>`) identifies the file; `sources` lists places it has been seen.
  Paths are hints and can go stale; `Dataset.resolve_video_path` searches given roots by file name.
- **Custom data** goes in `extra` (metadata objects, and a JSON-object string in the annotation table). Use your
  own top-level key (e.g. `extra.mytool`). Core objects reject unknown fields.

## Views (crops and time ranges)
A view is a named region and/or frame range of one video, stored in `views/<view_id>.json`:
`region` (x, y, width, height in displayed pixels), `output_size` (width, height the crop is resized to before a
pipeline sees it; omit if not resized), `frame_range` ([first, last], inclusive), `allowed_labels` (optional list
from `vocab.json`). Several views of one video may coexist and overlap (e.g. a main area and a caption strip, each
processed by its own model). No video file is created.

Annotations are **always stored in full source-video coordinates**. A row labelled in a view carries that `view_id`.
To convert from the crop a pipeline saw: `source = region.xy + view_pixels / scale`, where
`scale = output_size / region size` (1 if there is no `output_size`). The library does this for you:
`View.to_source_box()`, `View.to_view_box()`, and `add_annotations(..., coords="view")` (also pastes crop masks into a
full-frame canvas). The validator rejects rows whose box lies outside the view's region, whose frames lie outside its
`frame_range`, whose label is not in its `allowed_labels`, or whose view belongs to a different video.
Rows with no `view_id` refer to the whole frame. Filter with `ds.annotations(view_id=...)` (`[None]` = whole-frame rows),
`Compiled.clip(..., view_id=...)`, and `ds.build(view_ids=[...])` for a per-view release.

## Annotation table (one row per annotation, all kinds together)
See `ANNOTATION_COLUMNS`. Kinds: `marker` (time only), `box` (single frame), `mask` (single frame, also has a box).
`origin`: human | model | imported | other, or your own value starting with `x-` (lowercase letters, digits, `.`, `_`, `-`; e.g. `x-crowd`).
Built-in values keep filters reliable; the `x-` prefix marks a custom one and still catches typos like `Human`. Name the specific
source in `tool` (for `imported`, the tool the data came from). `status`: unreviewed | approved | rejected.
Rules checked by the validator (all codes are listed [below](#validation-codes)): required fields present, enums valid,
`end >= start`, frames inside the video, label in `vocab.json`, boxes finite and non-empty, markers carry no geometry, masks
carry bytes + codec, `view_id` exists, `confidence` in 0..1, `extra` is a JSON object, `annotation_id` unique (also across
videos), and the track rules below. `--deep` additionally decodes every mask, checks its frame size and that the box equals
the mask's bounding box.

**Tracks.** `track_id` (a non-negative integer, optional) identifies one object within one video **and view**: rows with
different `view_id` (including none) are separate id spaces. Among rows of kind `box` or `mask` that are not `rejected`, one
track must not carry more than one label (warning `track_label_conflict`) and must not have two rows of the same kind in
the same frame (error `duplicate_track_in_frame`). A box and a mask of one track may share a frame. Appending annotations
applies the duplicate check against the rows already stored for the same tracks.

Storage: Parquet, zstd, row groups of about 20,000 rows, statistics and page index on. Rows sorted by
(`start_frame`, `annotation_id`) within a video file.

## Mask codec `bitpack_zstd`
Header (little endian): `u8 version=1; u32 frame_h, frame_w, crop_y0, crop_x0, crop_h, crop_w` then
`zstd(np.packbits(crop, axis=-1))`. `crop_h = crop_w = 0` means an empty mask with no payload. Lossless.

## Compiled releases
`vidlabel build` merges per-video files into one table sorted by (`video_id`, `start_frame`, `annotation_id`),
excluding `rejected` by default. `release_id = "r" + first 12 hex of the file's sha256`, so identical data and
filters give the same id. Pin it in training configs. `Compiled.clip()` prunes row groups by statistics.
Each release folder also holds `vocab.json`, a snapshot of the label vocabulary at build time, so class indices derived from
a pinned release stay fixed when labels are added later. Releases made before this was added fall back to the live vocabulary.

## Validation codes
Returned by `Dataset.validate()` as `Issue(level, code, message)`; errors fail `vidlabel validate` and block `build`.

| code | level | meaning |
|---|---|---|
| `no_manifest`, `bad_manifest` | error | `manifest.json` missing or invalid |
| `spec_major` | error | dataset written by an incompatible spec major version |
| `no_vocab`, `bad_vocab` | error | `vocab.json` missing or invalid |
| `bad_view`, `view_id_mismatch`, `view_video`, `view_region`, `view_labels` | error | a view file is invalid, is named differently from its `view_id`, refers to an unknown video, extends outside the video, or allows labels not in the vocabulary |
| `manifest_views` | warning | `manifest.json` lists different views than `views/` contains |
| `missing_video_dir`, `no_metadata`, `bad_metadata`, `video_id_mismatch` | error | a listed video has no folder or metadata, invalid metadata, or metadata naming another video |
| `orphan_video_dir` | warning | a `videos/` folder not listed in the manifest |
| `bad_parquet`, `video_id_column` | error | an annotation file cannot be read, or has rows for other videos |
| `duplicate_id_across_videos` | error | an `annotation_id` appears in two videos |
| `missing_column`, `unknown_column` | error | the annotation table lacks a standard column or has an extra one (custom data belongs in `extra`) |
| `null_required` | error | a required column has nulls |
| `bad_kind`, `bad_origin`, `bad_status` | error | value not in the allowed set (origin may also be an `x-` value) |
| `duplicate_id` | error | `annotation_id` repeated within a video |
| `negative_frame`, `frame_order`, `frame_out_of_range` | error | `start_frame < 0`, `end_frame < start_frame`, or a frame beyond the video |
| `unknown_label` | error | label not in `vocab.json` |
| `missing_box`, `bad_box`, `geometry_span` | error | a box or mask without a full box, a non-finite or empty box, or a box or mask spanning more than one frame |
| `box_outside_frame` | warning | a box extends outside the frame |
| `marker_has_geometry`, `box_has_mask` | error | a marker with a box or mask; a box row carrying a mask |
| `missing_mask`, `missing_codec`, `unknown_codec` | error | a mask row without bytes, bytes without a codec, or an unknown codec |
| `bad_confidence` | error | `confidence` outside 0..1 |
| `bad_track_id` | error | `track_id < 0` |
| `track_label_conflict` | warning | one track carries more than one label |
| `duplicate_track_in_frame` | error | one track has two boxes (or two masks) in one frame |
| `unknown_view`, `view_wrong_video` | error | `view_id` not found, or the view belongs to another video |
| `box_outside_view`, `frame_outside_view`, `label_not_allowed_in_view` | error | a row labelled in a view lies outside its region, frame range or allowed labels |
| `bad_extra` | error | `extra` is not a JSON object |
| `mask_undecodable`, `mask_box_mismatch` | error | with `--deep`: a mask cannot be decoded or has the wrong frame size; its box differs from its pixel bounding box |

## Editing rules (0.1)
Each video has its own annotation file, so people working on different videos never conflict. Writes are atomic
(temp file + rename) but not locked: one writer per video at a time. Ids are random 63-bit integers.
