# vidlabel format, spec 0.4.0

Authoritative definitions: `packages/vidlabel-spec` (pydantic models; `vidlabel schema` prints JSON Schema).

## Layout
`manifest.json`, `vocab.json`, `videos/<video_id>/metadata.json`, `videos/<video_id>/annotations.parquet`,
`views/<view_id>.json`, optional `compiled/<release_id>/{annotations.parquet,release.json}`, `compiled/LATEST`.
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
Rules checked by the validator: required fields present, enums valid, `end >= start`, frames inside the video,
label in `vocab.json`, boxes finite and non-empty, markers carry no geometry, masks carry bytes + codec,
`view_id` exists, `confidence` in 0..1, `extra` is a JSON object, `annotation_id` unique (also across videos).
`--deep` additionally decodes every mask, checks its frame size and that the box equals the mask's bounding box.

Storage: Parquet, zstd, row groups of about 20,000 rows, statistics and page index on. Rows sorted by
(`start_frame`, `annotation_id`) within a video file.

## Mask codec `bitpack_zstd`
Header (little endian): `u8 version=1; u32 frame_h, frame_w, crop_y0, crop_x0, crop_h, crop_w` then
`zstd(np.packbits(crop, axis=-1))`. `crop_h = crop_w = 0` means an empty mask with no payload. Lossless.

## Compiled releases
`vidlabel build` merges per-video files into one table sorted by (`video_id`, `start_frame`, `annotation_id`),
excluding `rejected` by default. `release_id = "r" + first 12 hex of the file's sha256`, so identical data and
filters give the same id. Pin it in training configs. `Compiled.clip()` prunes row groups by statistics.

## Editing rules (0.1)
Each video has its own annotation file, so people working on different videos never conflict. Writes are atomic
(temp file + rename) but not locked: one writer per video at a time. Ids are random 63-bit integers.
