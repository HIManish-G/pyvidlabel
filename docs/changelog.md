# Changelog

The spec version and the library version move together. "Spec" entries change the files; "Library" entries change behaviour only.
0.6.0 is the first release published to PyPI (alpha). Earlier versions were internal milestones.

## 0.6.1 (2026-11)

The format is unchanged (spec 0.6.0); the packages are versioned 0.6.1.

**Library**
- `vidlabel --help` names the `ui` extra correctly (the help renderer used to drop the bracketed extra name).
- `vidlabel query` names the available video ids when it is given something that is not one, and says when the argument looks like a file path.
- Web interface dialogs take focus as soon as they open, and Escape closes the topmost dialog even before focus has reached it.
- PyPI project pages: package descriptions, classifiers (alpha) and project links.
- Documentation reworded for readers: install instructions use `pip install vidlabel`.

## 0.6.0 (2026-11)

**Spec**
- Views accept `frame_ranges`: several time windows in one view (`frame_range` still works; not both). A 0.5 library cannot read a view
  that uses `frame_ranges`.
- New validation error `mask_frame_size`: a stored mask encoded for another frame size than the video, read from the mask header
  without decoding. `--deep` still decodes everything.
- Video and view ids cannot be `.` or `..`, and the manifest's video and view lists are checked the same way (`bad_manifest`).

**Library**
- Web interface: `vidlabel ui PATH` (extra `vidlabel[ui]`), a local server protected by a token.
  - Per-author draft with undo, redo and an all-or-nothing commit; history with revert (last 50); background jobs with progress.
  - Pages for overview, videos, viewer, track review, validation, releases and history; adding and checking videos, folders for moved
    files, frame cache, labels, views and copying labels into a view.
  - Viewer built around one current frame: frame cache and prefetch, playback clock, zoomable timeline with lanes and span editing,
    picture zoom and pan, keys `,` `.` to jump between annotation edges and `C` to copy a box forward.
  - Drawing: boxes, spans and masks (brush, eraser, polygon), moving and resizing boxes, reshaping masks with redo, and drawing inside a
    view (drawn in the view's pixels, stored in the video's own). The draft refuses, at the time of the edit, a second box or mask for
    one track on a frame and track changes that would collide.
  - **Sample inspector** page: the picture with the loader's targets drawn on it, a table, raw tensors, previous and next sample, and
    choice of release, view, box format and label offset.
  - Accessibility: contrast, accessible names, headings, dialogs, announcements, a skip link, a timeline slider, and adding or editing
    boxes and spans by typing numbers. Checked with axe-core.
  - Server endpoints `/api/annotations-range`, `/api/mask-png`, a windowed `/api/timeline`.
- AI agents: `vidlabel mcp` serves a dataset to an agent over MCP (extra `vidlabel[mcp]`): read tools, a frame picture, and
  `propose_boxes`, `propose_spans`, `propose_masks`, `retract`. Proposals go into the agent's own draft as `model`/`unreviewed` rows
  under `agent:<name>`; a person takes them in under *Proposals* in the web interface and commits. Agents cannot approve, reject,
  change, delete or commit. See [AI agents](agents.md).
- `--json` on `info`, `validate`, `build`, `tracks`, and a new `vidlabel query`.
- `Dataset.derive_view_rows(dry_run=True)` reports what would be copied.
- `Dataset.update_annotations(..., mask=...)` replaces the mask of mask rows and recomputes their box.
- Review by track: `Dataset.tracks()`, `update_track()`, `delete_track()`, a `track_id` filter on `annotations()`; commands
  `vidlabel tracks` and `vidlabel review --track`.
- `Dataset.derive_view_rows()` and `vidlabel derive-view` copy whole-frame labels into a view.
- Updating or deleting an unknown annotation id raises `KeyError` (opt out with `missing="ignore"`); reading an unknown video raises.
- `FrameSamples` stores annotations as NumPy columns: opening 1 M annotations takes 0.6 s and 0.5 GB instead of 6.9 s and 1.6 GB.
- A view samples only its own video; a view that selects no samples warns and explains view tagging.
- Command-line errors print one `[error]` line and exit with status 1 (`VIDLABEL_DEBUG=1` restores the traceback); `validate` reports a missing
  manifest as `no_manifest`.
- `check-video` probes the file it is given, compares timestamps with the stored index and states what it did not check.
- The loader warns when a video file's frame count clearly differs from the metadata.
- `build --include-rejected` leaves `LATEST` unchanged.
- Wrong-size mask arrays are refused at write time; `extra` accepts NumPy numbers and refuses NaN; misspelled filter values raise.
- `add-labels` splits commas; `preview --frames` that selects nothing is an error; an empty release is announced.
- Validation messages name the frame and track of the first offending rows.
- Preview: rejected rows are dashed, washed out and prefixed `rej`; labels stay inside the frame and avoid overprinting.

## 0.5.0

**Spec**
- Optional frame index `videos/<id>/frame_index.json` (timestamps and keyframes) and the codes `bad_frame_index`, `frame_index_length`,
  `frame_index_stale`.
- `track_id` is scoped to one video and view; new rules: warning `track_label_conflict`, error `duplicate_track_in_frame`. A 0.4 dataset
  opens unchanged but is validated against the new rules.

**Library**
- Frame index aware reading (indexed seeking, per-frame timestamp check, variable-frame-rate seeking), `read_many`.
- `ReadPlan` and `PlannedStream`; the optional frame cache (`vidlabel cache`).

## 0.4.0

First documented version.
