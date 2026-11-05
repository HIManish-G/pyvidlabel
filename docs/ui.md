# Web interface

`vidlabel ui` opens a web page on your own computer for looking at a dataset, reviewing what a model predicted, checking that the
dataset is valid, and building releases. It needs no database, no Node and no internet connection.

> **Status.** The interface can browse, review, draw and edit boxes, time spans and masks, validate, build releases, add and check
> videos, manage labels and views, and commit and revert changes. Still to come are polish and an accessibility review. The plan is in
> [Web interface design](ui-design.md).

## Start it

```
pip install "vidlabel[ui]"
vidlabel ui my_dataset
```

The command prints an address ending in `?token=...` and opens your browser. Open that exact address; the token is what lets the page talk
to the server. Press Ctrl+C in the terminal to stop it. If the folder is not a dataset yet, the page offers to create one.

| option | what it does |
|---|---|
| `--port 9000` | use another port (the next free one is used if the port is busy) |
| `--no-browser` | do not open a browser window |
| `--host 0.0.0.0` | share on your network (see below) |
| `--fs-root FOLDER` | the folder picker and file paths may only use this folder (repeat for more) |
| `--token TEXT` | use a fixed access token instead of a random one |

## Your name and your draft

The first time, the page asks for your name. It is saved with the changes you commit, and there are no accounts or passwords. Each person
has their own draft, so two people using one server do not see each other's unfinished edits.

## How changes work

Edits are **not** written to the dataset when you make them. They go into your **draft**, shown by the **Changes** button at the top right.
Pages show the dataset with your draft applied: changed rows carry a purple "pending" tag or a dashed outline.

- **Undo and redo** (the buttons, or Ctrl+Z and Ctrl+Y in the viewer) step through your edits. Removing one edit from the list also removes
  anything built on it.
- **Commit** needs a short message. The whole draft is tried on a copy and checked by the validator first; then it is applied to every
  video together or not at all. If anything goes wrong the dataset is left exactly as it was and your draft is kept.
- **Conflicts.** If the dataset changed after you began (you ran a command, or someone else committed), the commit stops and says which
  videos changed. You can apply your edits on top of the new version; edits to rows that no longer exist are dropped.
- **History** lists the last 50 commits. **Revert** adds a new commit that puts things back: added rows are removed, removed rows return with
  the same ids, changed values go back. Rows changed again since then are left alone and counted.

Drafts and history are stored inside the dataset folder in `.vidlabel-ui/`, which the validator ignores. Delete that folder and the dataset
is unaffected, but drafts and history are lost.

## Pages

| page | what it does |
|---|---|
| Overview | headline counts and **the score**: every video as a band on one ruler, coloured by review status (click a band to open the video at that frame); counts by status, kind, origin and label |
| Videos | add a video (a path or the file picker; optionally fingerprint it and read its keyframes), see each video's size, frame rate, whether its file is found and whether it has a frame index and fingerprint, **Check** it against the file, **Index** it, tell the server which folders to search for moved files, and build the optional frame cache |
| Labels | the label list with usage counts; add labels (several at once, separated by commas) |
| Views | the views of the dataset; create one by dragging a region on the first frame or typing numbers, with an optional resize, time windows and allowed labels; **Copy labels in** previews, then copies whole-frame rows into a view |
| Viewer (click a video) | frame-accurate frames read by the library, boxes and masks drawn over them, a timeline of annotation density and markers, a list of what is on the frame with Approve, Reject and Delete, and the drawing tools (below) |
| Review | the tracks of one video with a thumbnail, confidence, status and origin; filter, sort by lowest confidence, approve, reject or delete one track or many at once |
| Validation | the quick check at once; the full check (decodes every mask) as a background job with a progress bar |
| Releases | the existing releases, building a new one (choose which statuses to include), and an inspector showing the training targets for any frame |
| History | past commits and Revert |

Adding labels, views and videos, copying labels into a view, indexing and caching are applied at once (they only add, and the library has no
remove for them); only annotation edits go through the draft. The bar along the bottom always shows your draft: how much is added, changed
and deleted, with Undo, Redo and Commit.

Frames are read from the video file on the computer running the server. If the file has moved, the viewer still shows the annotations
over a grey frame and says the file was not found.

## Drawing and editing in the viewer

Open a video and use the tools above the frame. Every drawing goes into your draft (pink "pending" outline) and is written only when you commit.
Drawing works on the whole frame, so coordinates are always the video's own pixels. Inside a view the tools are off; draw on the whole frame
and use **Copy labels in** on the Views page.

| tool (key) | what it does |
|---|---|
| Select (V) | click a box or mask to select it; drag a box to move it, drag its handles to resize it; the panel on the right changes its label or track, or edits a span's frames |
| Box (B) | drag on the frame to draw a box at the current frame |
| Span (S) | drag on the timeline to label a stretch of frames; a click labels one frame |
| Brush (M), Eraser (E) | paint or erase a mask on a layer the size of the frame; `[` and `]` change the brush size |
| Polygon (P) | click corners; click the first point, double-click or press Enter to close the shape |

Choose the **label** and the **track** for new drawings next to the tools. "New track" gives the next free track number and then keeps using
it, so you can draw a box, press → and draw the next one. A mask is saved with **Save mask** (or Enter); a mask in progress can be undone
stroke by stroke with Ctrl+Z. To change an existing mask, select it and choose **Edit mask shape**, paint or erase, then **Apply to the mask**;
its box follows the new shape. New rows are marked as made by a person (`origin` human, status unreviewed) with your name as the author.

Masks: **Alt**-drag erases with the brush, **Shift**-click draws a straight line from the last point, `[` and `]` change the size, a slider sets
the opacity, and Undo and Redo step through strokes. A mask in progress belongs to the frame it was started on; if you move away the panel
offers to go back, and it can only be saved onto that frame. Polygon corners can be dragged before the shape is closed.

Keys: `,` and `.` jump to the previous or next frame where a span or track starts or ends; **C** copies the selected box to the next frame (same label and track) and moves there, so pressing it again follows an object along; it refuses if that track already has a box on the next frame (pressing it several times quickly makes one copy per frame). ← → step a frame, Shift 10 frames, Space plays, Home and End, Delete removes the selected row, Esc cancels a polygon or drag, Ctrl+Z and
Ctrl+Y undo and redo your draft (or your mask strokes while a mask is open).

A track can have only one box (and one mask) on a frame. The draft refuses a new box, or a change of track number, that would break this, with a
message naming the track and frame, so a draft you can see is always one that can be committed. If an older draft already holds such a pair, the
commit stops with `duplicate_track_in_frame`; remove one of the two entries from the Changes list (the **×** beside it) and commit again.

## Playing and moving around

The frame box, the time, the timeline and the picture always show the same frame. If a frame takes a moment to load, the previous picture
stays with its own annotations and the corner says "loading frame N". Drawing waits until the picture is the right one.

- **Timeline:** drag anywhere on it to move through the video. Scroll to zoom around the pointer, Shift-scroll to pan, drag the box in the
  small overview at the bottom, or use the −, + and Fit buttons. Counts by status, spans and tracks are shown in lanes; with the Select tool, drag
  a span's end to resize it or its middle to move it.
- **Picture:** scroll over it to zoom, middle-drag to pan, `0` (or **Reset zoom**) to fit. Drawing stays exact at any zoom.
- **Play** uses the video's frame rate (0.25× to 2×) and waits for a frame rather than skipping one; **Loop** starts again at the end.

## Slow work runs as jobs

Full validation, building a release and similar operations run in the background. The top bar shows a **Jobs** button with the current
progress; open it to see messages, cancel a job, or read an error. One job runs at a time per server. A job can only be cancelled between steps.

## Sharing

By default the server listens on this computer only. `--host 0.0.0.0` makes it reachable on your network; anyone with the address and token
can then change the dataset, so share it only with people you trust, and prefer a trusted network. Without `--fs-root` a shared server
refuses all file picking, so people cannot browse your disk. One server is the only writer: do not run two servers (or a server and a long
command-line edit) on the same dataset at once. A dataset on a mounted network drive works, since the server reads it by path.

See [Security](security.md) for what the token, host and origin checks do and do not protect against.

## Troubleshooting

- *"the web interface needs extra packages"*: run `pip install "vidlabel[ui]"`.
- *The page says the token is missing*: open the exact address that `vidlabel ui` printed.
- *Frames are grey*: the video file is not where the dataset recorded it (the viewer says so); run `vidlabel check-video`, or start the server on the computer that has the file.
- *A commit says the dataset changed*: use "Apply my edits on top", or discard your draft and start again.

## Bundled font

The interface uses the Bricolage Grotesque font, shipped inside the package (`static/fonts/`, SIL Open Font License; the licence text is next to the file), so nothing is loaded from the internet.
