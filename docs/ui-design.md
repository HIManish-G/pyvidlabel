# Design: the local web interface

Status: **stages 1 to 3 are built** (server, draft and history, jobs, every route, and pages for browsing, review, validation, releases, history, videos, labels and views); stage 4 (polish) is
planned. This page is the plan the interface is built and tested against; the user guide is [Web interface](ui.md). If the behaviour here
and the code ever disagree, one of them is a bug.

## Goal

A web interface, started with `vidlabel ui PATH`, that does everything the command line and the library do, for people who do not want
to use either. It runs on your own computer, needs nothing installed beyond `pip install "vidlabel[ui]"` (no Node, no database, no
internet), and opens in the browser.

The model is MLflow's: it does not replace the library, it shows what you have and lets you manage it. It is not a replacement for a
full labelling product; drawing exists so that a dataset can be fixed and made inside it, with brush and polygon tools, not
AI-assisted segmentation.

## Principles

1. **The library is the only writer.** The interface calls `Dataset` methods. It never writes annotation files itself, so it cannot
   store anything the command line could not, and every rule in [validation](validation.md) applies.
2. **Frames come from the library's reader**, not the browser's video player, so what you see is exactly the frame the annotations refer to
   (frame-accurate, variable frame rate handled, any codec OpenCV can decode).
3. **Nothing changes until you commit** for annotation edits (the draft model below).
4. **Every operation has a screen**, and a test fails when a command or public `Dataset` method has none (the coverage table).
5. **Offline and self-contained.** The frontend is plain JavaScript shipped inside the Python package; no CDN.

## How it runs

```
vidlabel ui my_dataset                  # http://127.0.0.1:8765/?token=...  (opens the browser)
vidlabel ui my_dataset --port 9000
vidlabel ui my_dataset --host 0.0.0.0   # share on your network; requires the token
```

- A small Starlette server (served by uvicorn); the page and its script are bundled files.
- By default it listens on `127.0.0.1` only. Every request must carry a random token (given in the printed URL), and the `Host` and
  `Origin` headers are checked. A local server that edits files can otherwise be driven by any web page open in your browser.
- Sharing is opt-in. There are no accounts. A person enters a name once; it is stored as `author` on their changes.
- One server per dataset is the single writer. Edits are serialised with a lock, so several people using one server cannot overwrite
  each other. Running the server and the command line on the same dataset at once is detected at commit (see conflicts) but is not prevented.
- Videos and datasets on mounted drives work, since the server reads them by path. Network filesystems have weaker rename guarantees than
  local disks, which is another reason for one server as the only writer.

## The draft model (git-like)

Annotation edits are not applied to the dataset immediately. They go into a **draft**, which you review and **commit**.

| git | here |
|---|---|
| working tree | the dataset on disk |
| staging area | the **draft**: a list of pending edits |
| commit | **Commit**: apply the whole draft to the dataset, with a message |
| `git diff` | the **Changes** panel: "+12 boxes, 3 rows rejected, track 4 deleted", with each edit listed |
| `git restore` | **Discard** one edit, or all |
| `git revert` | **Revert** a past commit |
| history | the **History** page |

What a draft holds: adding, updating and deleting annotations, including whole-track operations. The pages show the dataset **with the
draft applied**, drawn so that pending changes are visibly different (dotted outline, "pending" tag), so you can see the result before committing.

**Undo and redo** work inside the draft: each edit is recorded as an operation, undoing pops the last one, redo puts it back. The
draft is saved on disk (`.vidlabel-ui/drafts/<author>.json` inside the dataset folder; the spec ignores that folder), so it survives
closing the browser or restarting the server. Each author has their own draft.

**Commit** first tries the whole draft on a scratch copy of the affected videos and validates the result, then applies it with the library.
It is **all or nothing across videos**: if anything fails, every touched video is put back exactly as it was and the draft is kept. The commit is
recorded in `.vidlabel-ui/history/` (one JSON file per commit with time, author, message and the ids it touched, plus a Parquet file with the
rows it changed or removed, so that it can be reverted). A message is required. The last 50 commits are kept.

**Conflicts.** A draft remembers a fingerprint of each video's annotation file when its first edit was made. If the file changed by the
time you commit (someone used the command line, or another person committed), the commit stops and says which videos changed.
You can reapply: edits that still make sense go through, edits that refer to rows that no longer exist are listed and skipped.

**Applied immediately, with confirmation and a log entry** (no draft): adding labels, views and videos (additive and harmless), building
a release, indexing, caching and other jobs. They have no inverse in the library (there is no remove-video or remove-label), and the
screens say so. Anything destructive is annotation-level, so it is in the draft.

## Slow operations

Hashing a large video, `validate --deep`, `check-video --seek`, indexing, building the frame cache and building releases run as
**jobs**: the request returns a job id at once, the page shows a progress bar and live messages, and the job can be cancelled. Jobs on
the same dataset run one at a time. A job that fails shows the same one-line error the command line prints.

## Pages and the coverage table

Every command and public `Dataset` method has a place:

| CLI / library | page | notes |
|---|---|---|
| `init` | New dataset | choose a folder |
| `info`, `stats`, `schema` | Overview | counts by kind, label, origin, status, view; schema link |
| `add-labels`, `vocab` | Labels | add, list, usage counts |
| `add-video`, `video`, `resolve_video_path` | Videos | pick a file or folder on the server's disk; set roots for moved videos |
| `index`, `frame_index`, `set_frame_index` | Video health | button, progress, result |
| `check-video` (`--hash`, `--seek`) | Video health | each check a button, each result explained |
| `cache` | Video health or Cache | sizes, format, quality, progress |
| `add-view`, `view`, `view_ids` | Views | draw the crop on a frame, set output size, add time windows |
| `derive-view` | Views | preview how many rows would be copied before doing it |
| `annotations` (all filters) | Browse | table and timeline with the same filters |
| `add_annotations` | Viewer | box, span and mask tools (below) |
| `update_annotations`, `delete_annotations` | Viewer, Browse | selection, bulk actions (in the draft) |
| `tracks`, `update_track`, `delete_track`, `review` | Review | track list, approve and reject, filters, jump to frames |
| `validate` (`--deep`) | Validation | issues with the fix text from [validation](validation.md); click to jump to the row |
| `build`, releases | Releases | filters as form fields, the id to copy, rows and size |
| `preview` | Viewer | export frames or an MP4 |
| `FrameSamples.targets`, `describe` | Sample inspector | what one training sample would contain |
| the loaders, `ReadPlan` | not in the interface | code-level; a test lists the intended exclusions |

## The viewer and its tools

- Frame-accurate scrubbing with the library reader, a timeline of markers and tracks, keyboard shortcuts (step, jump to next annotation).
- Overlays drawn in the browser from JSON; masks decoded on the server and sent as small images.
- **Box tool:** drag to draw, handles to resize, the label and track in a side panel.
- **Span tool:** drag on the timeline to make or resize a marker.
- **Mask tools:** brush, eraser and polygon, working on a canvas the size of the frame. Saved through `add_annotations` with a boolean array,
  so the codec, bounding box and size checks are the library's. A mask drawn in a view uses `coords="view"`.
- Not planned for version 1: AI-assisted segmentation (it would mean shipping a model), automatic tracking.

## Server interface

JSON over HTTP under `/api`, one route per operation, thin wrappers over the library; the same functions are covered by tests without a
browser. The page is a client of this interface, so a script could use it too. The frame and mask routes return images.

## Testing

- The server is tested through its HTTP interface against real datasets (including encoded test video), covering every route.
- A test fails if a CLI command or public `Dataset` method is neither mapped to a route nor listed as an intended exclusion.
- Draft tests: every edit type and its undo, commit, conflict detection, revert, and "the dataset is unchanged until commit".
- Frontend: scripted browser tests (headless Chromium) for the main flows: open, scrub, draw a box, review a track, commit, undo.
- Security tests: wrong token, wrong `Host`/`Origin`, path escapes through the folder picker.

## Stages

1. Server, draft and history core, jobs, and the whole interface of routes, with tests. Overview, Browse, Viewer (read only), Review, Validation, Releases pages.
2. Builder pages: New dataset, Labels, Videos and Video health, Views, derive-view.
3. Drawing: boxes and spans, then masks (brush, eraser, polygon).
4. Polish: undo and history screens, shortcuts, sample inspector, documentation and a screen-by-screen guide.

## Decisions

- **Drafts are per author**, with conflicts caught at commit. (Decided.)
- **History keeps the last 50 commits**, because reverting needs the removed rows and masks can be large. (Decided.)
- **Reverting** commits the inverse of a past commit, with the same row ids; rows changed again since are left alone and listed.
- **Open:** accessibility beyond the keyboard shortcuts and a light and dark theme (both exist; a review of contrast and screen-reader labels is part of stage 4).

## What is built

| stage | state |
|---|---|
| 1. server, draft, history, jobs, routes, tests; Overview, Videos, Viewer (read only plus approve, reject and delete of rows), Review, Validation, Releases, History, Changes panel | built |
| 2. builder pages: Labels, Videos (add, check, index, folders for moved files, frame cache), Views (draw a region, create, copy labels in with a preview) | built |
| 3. drawing: select, move and resize boxes, new boxes, spans on the timeline, masks with brush, eraser and polygon, editing an existing mask | built (whole frame only; views get rows through Copy labels in) |
| 4. polish and the sample inspector page | the sample inspector is on the Releases page; the rest is not built |

## Visual identity

Time is what this data is made of, so time-based things are drawn as horizontal bands on a shared frame ruler: the Overview "score" (one
band per video, coloured by review status), the viewer timeline, and the draft tape along the bottom of every page. The palette is cool
grey-green paper and ink-blue, with ultramarine for selection and pink reserved for changes that are not committed yet. One bundled
typeface (Bricolage Grotesque) carries the headings. Both a light and a dark theme follow the system setting.
