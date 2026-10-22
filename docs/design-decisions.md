# Design decisions

Why vidlabel is the way it is, what was measured, and what is still open. Each entry says what was decided and the
reason, so a decision can be challenged on its merits.

## One annotation table, and origin is a column

Time markers, boxes, masks and tracks are rows of **one table** with the same columns. *Who or what made a row* is a column
(`origin`, `tool`, `tool_version`, `author`), not a separate file, folder or format.

*Why:* in practice human labels, model predictions and imported data end up being compared, mixed and re-reviewed. When
they live in different structures every one of those operations needs glue code and the structures drift apart. In one
table, "show me where the model disagrees with the humans" is one filter.

## Video pixels are never stored

A dataset records where a video is and facts about it (frame rate as an exact fraction, size, rotation, a content hash),
and the annotations that refer to its frames.

*Why:* video is large, usually already lives somewhere managed, and is often shared by many datasets. Copying it would make
the dataset heavy and create a second copy that can disagree with the first. The content hash makes it detectable when a
path now points at a different file.

*Cost:* paths go stale. `resolve_video_path` and `--root` search by file name, and `check-video` verifies a file against its
metadata.

## A folder of plain files, with Parquet for the table

A dataset is JSON for the small things and Parquet (zstd, row groups, statistics and a page index) for annotations, one file
per video.

*Why:* any tool that reads Parquet can read the data; columnar storage with statistics lets a frame range be read without
scanning everything; one file per video means people working on different videos never touch the same file.

*What was not verified:* Parquet was chosen **without a completed head-to-head benchmark** against Lance. That comparison was
started and cut short, and the decision was to proceed with Parquet and revisit speed differences later. See
[open questions](#open-questions).

## The spec is a separate, strict package

`vidlabel-spec` holds pydantic models for the metadata objects and the annotation column definitions (and can print a JSON
Schema); `vidlabel` is the library. Core objects **reject unknown fields**; custom data goes in `extra` under your own key.

*Why:* following the Zarr and GEFF approach, a tiny contract that other implementations can target without importing the
whole library. Strictness turns a typo (`frame_cuont`) into an error instead of silently lost data, and `extra` gives
everyone a safe place for what the spec does not cover.

## Boxes are stored as pixel corners

Boxes are `x0, y0, x1, y1` in pixels, origin top-left, `x1` and `y1` exclusive, as float32: not YOLO's normalized
centre-and-size.

*Why:* the stored form should be **unambiguous and lossless**, and every other format derivable from it.

- A normalized box needs a reference size. Annotations here can be labelled on the full frame, a crop, or a resized crop,
  on a possibly rotated video. Absolute pixels in the displayed full frame mean one thing.
- Pixels are exact: a human drew on pixels. Normalizing turns 200 into 0.15625, which float32 rounds, and a crop-and-resize
  round trip makes it worse.
- Corners make the common operations direct: clipping to a view region, checking a box is inside the frame, overlap.
- There is no universal format (COCO is top-left plus size, VOC and torchvision are corners, YOLO is normalized centre plus
  size), so any choice needs conversions. `convert_boxes` and the loader's `box_format` / `normalized` options do them at the edges.

## Coordinates are in the displayed orientation; rotation is clockwise

*Why:* the picture a person sees in a player is the one they label, and it is what every reader of the data wants. Storing
the stored-frame size and a clockwise `rotation` keeps both facts, and the readers apply rotation exactly once. ffprobe
reports the display matrix counter-clockwise, so it is negated on the way in; a test checks that our rotation equals
OpenCV's own auto-rotation for 90, 180 and 270. A rotation that is not a multiple of 90 cannot be represented: it is
recorded in `extra["unsupported_rotation_degrees"]` and treated as 0, not silently dropped.

## Views store full-frame coordinates

A view is a named crop and/or frame range; annotations labelled in it are stored in the full frame's pixels with a
`view_id`. *Why:* the same row can then be drawn on the original video, compared with another view's output, or used by a
tool that knows nothing about views. The conversion (offset, plus scaling for a resized crop) happens at the boundary
and is tested, including for masks. Track ids are scoped per view because pipelines number objects independently.

## Origin is a fixed set plus `x-` values

`origin` is `human`, `model`, `imported`, `other`, or a value starting with `x-`. *Why:* a closed set keeps filters
reliable (a typo like `Human` is an error), but nobody should be stuck with four words: the `x-` prefix (as in HTTP and
MIME) marks a deliberate custom value. Details of the specific source go in `tool`, `author` and `extra`.

## Masks: crop, bit-pack, zstd

*Why:* measured against COCO run-length encoding on real masks (SA-V, plus DAVIS predictions), it was 16 to 36 percent smaller and about 6 times faster
to decode, and decoding CPU is the likelier training bottleneck. A context-modelled arithmetic coder was about 4 times
smaller than COCO RLE but slower to decode, so it stays a possible archival codec (the codec is a per-row column).
Temporal delta coding did **not** help on this data in any form.
*Caveats:* one SA-V video; DAVIS data are predictions, whose boundary noise may understate temporal redundancy in
human-edited masks; single thread; no I/O measurement. Re-test temporal coding on real tracked ground truth before
dropping it for good.

## Releases are content-addressed and carry their vocabulary

`build` writes a sorted table whose id is a hash of its contents, plus a snapshot of `vocab.json`. *Why:* a training run that
records the id can be reproduced exactly, and class indices cannot drift when labels are added later.

## Frames are read with OpenCV, and every frame is checked

Decoding is sequential by default because that is frame-accurate. Seeking to a keyframe is far faster, but OpenCV's
seek-by-frame-number was measured to be exact on constant-frame-rate files and **wrong on variable-frame-rate video** (29 of 40
jumps landed on the wrong frame) and on a broken-start MPEG-TS file, which would silently misalign boxes. Without a frame
index, seeking is therefore used only after a one-off pixel comparison with sequential decoding, and never for videos flagged
variable-rate. *Why not a different decoder:* a second decoder can produce slightly different pixels (colour conversion) than the
one used elsewhere, which is a reproducibility problem of its own.

## A frame index, not stored patches

Random access is slow because of how video is compressed, not because we cannot find the keyframe: finding it takes almost
nothing, decoding up to the frame takes ~125 frames on average at a keyframe interval of 250. Two ideas that look like they
should help do not:

- **Storing a byte offset or "better seek position"** only replaces the cheap part. The decode cost remains.
- **Storing a patch from the nearest keyframe to a frame** (so the frames in between could be skipped): the file does not
  contain such a patch (every frame depends on a few earlier ones, which depend on others), so making one means decoding up to
  the frame, and storing it means storing image data about as large as the frame itself. That is storing pixels in a harder-to-read
  form. Saving the decoder's internal state instead means saving several raw reference frames (about 1.4 MB each at 720p) in a
  decoder-specific format.

What a per-frame index *does* give: the keyframe positions (so reads can be planned: decode each keyframe interval once, jump
only when it saves work), and the timestamp of every frame, so a seek can be **checked** (the decoder's timestamp must equal
the stored one) and so variable-frame-rate video, where frame number and time disagree, can be seeked by time. Other video
tooling made the same choice: TorchCodec's "exact" seek mode scans the file once to build the same frame and keyframe index.
Measured effect (one machine, synthetic 720p clip): a single random read is unchanged (166 ms against 173 ms); batches of 8, 32
and 120 frames read through `read_many` run 1.2x, 1.8x and 4.8x faster than one at a time. The index is stored as its own optional
file (`frame_index.json`), not in `metadata.json`, so the metadata stays small and readable for long videos.

## Faster random access is derived data, not part of the dataset

When uniform random sampling matters, the options are re-encoding the video with a keyframe every 30 frames (about 8 times less
decoding, bigger files, a different file) or caching the decoded frames a run will use (a file read per frame: about 3 ms for
JPEG against about 170 ms to seek). The cache is built from the annotation table, because the table already says exactly which
frames training needs. It is kept **outside** the dataset, can be deleted and rebuilt any time, and records the content hash of
the video it came from. *Why not store frames in the dataset:* videos are large, already stored somewhere, and the dataset's
promise is that it never holds pixels; a derived cache gives the speed without breaking that. A *proxy video* (a re-encoded copy
registered next to the original) was considered and not built: it would be a second file with its own hash and slightly different
pixels, and `check-video` prints the one-line `ffmpeg` command for people who want one.

## Training reads by streaming

`VideoStream` visits each video once, front to back. *Why:* decoding is the dominant cost and random access defeats it. Choosing
*which* frames to train on is a question for the annotation table and needs no video access, so sampling strategies belong
on the table side (see [open questions](#open-questions)).

## Choosing frames is the user's job; reading them cheaply is ours

Class balance, weighting, how many unannotated frames to include and how long an epoch is depend on the data and the model, so the
library does not decide them. `ReadPlan` takes any list of `(video, frame)` requests and only arranges the reading: chunks by
keyframe interval, increasing order inside a chunk, repeated frames decoded once, chunks (not frames) shuffled per epoch and
shared between workers with nearly equal sample counts. *Why chunks:* a keyframe interval is the smallest unit that is cheap to
read on its own, so it is also the natural unit to shuffle and to hand to a worker. *Trade-offs:* samples inside a chunk are
neighbours in time until a shuffle buffer mixes them; a video with few keyframe intervals is few chunks, which limits how evenly
work can be split; videos without a frame index are cut into 250-frame blocks, which only approximates their real keyframes. How
much the reduced mixing matters for training quality is not measured.

## No importers or exporters in the library

*Why:* every source format differs in coordinates, frame numbering, naming, tracks and time spans; supporting each properly
would make the library a maze of special cases, and it could never be complete. What the library does provide is a small,
strict, well-validated target ([writing a converter](converters.md)) and the box and mask helpers converters need. Adapters
can live in their own packages.

## One writer per video, no locks

*Why:* one file per video makes conflicts rare, and atomic writes (temporary file, then rename) mean a crash cannot leave a
half-written file. A real multi-writer story (locks, or merging) is not designed yet.

## Open questions

- **Parquet versus other storage at scale.** No completed benchmark against Lance or others; speed at millions of rows and
  many files is unmeasured.
- **Several editors at once.** Nothing prevents two writers on one video from overwriting each other; a locking or merge story is needed.
- **Sampling recipes.** Which frames to draw is left to users on purpose, and reading a drawn list cheaply is built (`ReadPlan`). Worked
  examples of common choices (class balance, weights) could be added to the docs; nothing about them is built into the library.
- **The loader at scale.** Many videos, hour-long videos, many GPUs and remote storage are untested.
- **Variable-frame-rate video in real datasets.** With a frame index it can be seeked and every frame is checked; tested on a synthetic variable-rate
  H.264 file only. The index relies on OpenCV reporting each frame's timestamp, which held for H.264 in MP4 in our tests; other codecs and
  containers are unmeasured (a mismatch falls back to sequential decoding with a warning, never to a wrong frame).
- **Frame cache at scale.** Measured on a few hundred frames of one clip. Build time, disk use and read speed on millions of frames or network storage are unknown.
- **Temporal mask coding and the arithmetic codec.** Deferred; see the caveats above.
- **Name and publishing.** "vidlabel" is a working name and nothing is on PyPI yet.
- **Spec strictness over time.** The track rules added in 0.5.0 make `validate` stricter than before (a duplicate track in
  a frame is now an error). Datasets written by earlier versions that contain such duplicates need fixing.
