# Web interface

`vidlabel ui` opens a web page on your own computer for looking at a dataset, reviewing what a model predicted, checking that the
dataset is valid, and building releases. It needs no database, no Node and no internet connection.

> **Status.** This is the first stage of the interface. It can browse, review, validate, build releases, and commit and revert changes.
> It cannot yet draw new boxes or masks, add videos, or create views from the page; use the command line or the library for those (the
> server already has routes for them). The plan is in [Web interface design](ui-design.md).

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
| Overview | counts by status, kind, origin and label; the videos; add labels |
| Videos | each video's size, frame rate, whether it has a frame index and content hash, and where it came from |
| Viewer (click a video) | frame-accurate frames read by the library, boxes and masks drawn over them, a timeline of annotation density and markers, a list of what is on the frame with Approve, Reject and Delete; keys: ← → step, Shift 10 frames, Space play, Home and End, Ctrl+Z undo |
| Review | the tracks of one video with a thumbnail, confidence, status and origin; filter, sort by lowest confidence, approve, reject or delete one track or many at once |
| Validation | the quick check at once; the full check (decodes every mask) as a background job with a progress bar |
| Releases | the existing releases, building a new one (choose which statuses to include), and an inspector showing the training targets for any frame |
| History | past commits and Revert |

Frames are read from the video file on the computer running the server. If the file has moved, the viewer still shows the annotations
over a grey frame and says the file was not found.

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
