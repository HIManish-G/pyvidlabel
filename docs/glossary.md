# Glossary

Terms as vidlabel uses them. Each links to the page that defines it fully.

| term | meaning |
|---|---|
| **annotation** | One row in a video's annotation table: a marker, a box or a mask, with a label and provenance. [Concepts](concepts.md#the-annotation-table) |
| **annotation table** | The Parquet file `videos/<video_id>/annotations.parquet`. One row per annotation, all kinds together. [Spec](SPEC.md#annotation-table-one-row-per-annotation-all-kinds-together) |
| **author** | Column naming the person who made or reviewed a row. |
| **box** | A rectangle on one frame, stored as pixel corners `x0, y0, x1, y1` with `x1` and `y1` exclusive. |
| **build** | Compile the per-video tables into one sorted, content-hashed release. [Releases](releases.md) |
| **chunk** | In a `ReadPlan`: the requested frames of one video that fall in one keyframe interval, read in one pass. [Training](training.md#reading-a-list-of-frames-you-chose-readplan) |
| **codec (mask)** | How a mask is stored in bytes. The only one is `bitpack_zstd`. [Masks](masks.md) |
| **compiled release** | See *release*. |
| **confidence** | Optional number from 0 to 1 a model gave a row. |
| **content hash** | `sha256:<hex>` of a video file; how a dataset recognises "the same file". |
| **dataset** | A folder in the vidlabel layout. [Concepts](concepts.md#a-dataset-on-disk) |
| **deep validation** | `validate --deep`: also decodes every mask and checks it against its box. |
| **extra** | A JSON-object column (and metadata field) for your own data under your own key. |
| **frame** | A 0-based index in display order. `end_frame` is inclusive. |
| **frame cache** | Optional, derived JPEG or PNG copies of frames so random access is a file read. Not part of the dataset. [Training](training.md) |
| **frame index** | Optional `frame_index.json`: a timestamp for every frame and the keyframe numbers. [Spec](SPEC.md#frame-index-optional) |
| **keyframe** | A frame that can be decoded without earlier frames. Random access costs a decode from the keyframe before the frame wanted. |
| **kind** | `marker`, `box` or `mask`. |
| **label** | A name from `vocab.json`. |
| **marker** | A span of time with no geometry (`start_frame` to `end_frame`). |
| **mask** | A boolean pixel region on one frame, stored cropped to its bounding box, bit-packed and zstd-compressed. |
| **origin** | Who produced a row: `human`, `model`, `imported`, `other`, or your own `x-...` value. |
| **ReadPlan** | An arrangement of reads for a list of frames you chose, so each keyframe interval is decoded once. [Training](training.md) |
| **release** | `compiled/<release_id>/`: the whole dataset merged into one sorted table plus a vocabulary snapshot. The id is a hash of the file. [Releases](releases.md) |
| **rotation** | Clockwise angle (0, 90, 180, 270) applied to the stored frame for display. Coordinates are in the displayed orientation. |
| **spec** | The format definition: `docs/SPEC.md` and the `vidlabel-spec` package. |
| **status** | `unreviewed`, `approved` or `rejected`. Rejected rows stay in the file but are left out of releases. |
| **tool, tool_version** | Which program (and version) produced a row. |
| **track** | A `track_id` shared by the rows of one object across frames, scoped to one video and view. [Concepts](concepts.md#tracks) |
| **view** | A named crop and/or set of time ranges of one video, optionally resized. Rows carry a `view_id` tag; their coordinates stay full-frame. [Views](views.md) |
| **vocab** | `vocab.json`: the one shared list of labels. |
| **writer** | Whoever modifies a video's annotation file. Rule today: one writer per video at a time. |
