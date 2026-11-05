# Changelog

The spec version and the library version move together. "Spec" entries change the files; "Library" entries change behaviour only.
Nothing has been published to PyPI yet, so versions below are internal milestones.

## 0.6.0 (2026-11)

**Spec**
- Views accept `frame_ranges`: several time windows in one view (`frame_range` still works; not both). A 0.5 library cannot read a view
  that uses `frame_ranges`.
- New validation error `mask_frame_size`: a stored mask encoded for another frame size than the video, read from the mask header
  without decoding. `--deep` still decodes everything.
- Video and view ids cannot be `.` or `..`, and the manifest's video and view lists are checked the same way (`bad_manifest`).

**Library**
- Web interface, stage 1: `vidlabel ui PATH` (extra `vidlabel[ui]`). Local server with a token, a per-author draft with undo, redo and an
  all-or-nothing commit, history with revert (last 50), background jobs with progress, and pages for overview, videos, viewer, track review,
  validation, releases and history, plus (stage 2) adding and checking videos, folders for moved files, frame cache, labels, views and copying labels into a view. See [Web interface](ui.md). `Dataset.derive_view_rows(dry_run=True)` reports what would be copied.
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
- `build --include-rejected` no longer moves `LATEST`.
- Wrong-size mask arrays are refused at write time; `extra` accepts NumPy numbers and refuses NaN; misspelled filter values raise.
- `add-labels` splits commas; `preview --frames` that selects nothing is an error; an empty release is announced.
- Validation messages name the frame and track of the first offending rows.
- Preview: rejected rows are dashed, washed out and prefixed `rej`; labels stay inside the frame and avoid overprinting.
- Docs: Windows and macOS are described as not yet verified; the release id is described as a hash of file bytes.

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
