# Security and safe handling of datasets

vidlabel reads files that other people may send you. This page says what it does with them, what it protects against, and what it
leaves to you. It is not a security audit; no one outside the project has reviewed it.

## What opening a dataset runs

- Dataset files are **data, not code**. JSON is parsed with pydantic models; annotation tables are read with PyArrow. The library does
  not use `pickle`, `eval` or `exec`, and nothing in a dataset is executed.
- The `extra` field is JSON text and is never interpreted.
- Opening an untrusted Parquet file means running PyArrow's Parquet reader on it. Keep PyArrow up to date as you would for any
  untrusted file.

## Paths

- Video and view ids are restricted to letters, digits, `.`, `_` and `-`, and cannot be `.` or `..`, so an id cannot make the library
  read or write outside its folder. `manifest.json` entries are checked the same way (a manifest naming a path is reported as
  `bad_manifest`, not followed).
- The **recorded source path of a video is only a hint**. It is read when you ask for frames (`FrameReader`, `preview`, `check-video`).
  A dataset from someone else can name any path on your machine as a video source; the worst effect is that `ffprobe` or OpenCV is
  asked to open that file. Do not run preview or training on a dataset whose video paths you have not looked at (they are in
  each `videos/<id>/metadata.json` under `sources`, and `ds.video(id).sources` in Python), and use `--root` or `roots=[...]` to resolve videos by file name inside a folder you choose.
- Text files are written and read as UTF-8 on every platform.

## Masks

A mask blob carries the frame size in its header, and decoding allocates that much. The default `vidlabel validate` compares the header
with the video size **without decoding**, so run it first on a dataset you did not make (`mask_frame_size` reports a mismatch). Then
decode. A corrupt payload is an error (`mask_undecodable` with `--deep`), not a crash.

## Video files

Decoding is done by OpenCV and ffmpeg, which parse complex untrusted formats. Treat video files from unknown sources like any other
untrusted media: keep the libraries updated, and consider sandboxing.

## Integrity, not authenticity

- `content_hash` (SHA-256) tells you whether a video file is the one a dataset was made for. It does not tell you who made the dataset.
- A release id is a hash of a table, useful for reproducing a run. It is **not a signature**: anyone can build a release with any id of
  their making, since the id is derived from the content they wrote.
- If provenance matters, sign or checksum the dataset folder (a tarball hash, or your version-control commit) outside vidlabel.

## Privacy

Datasets can contain personal data indirectly: annotators' names (`author`), identities in labels, and paths that reveal user names or
project names. Before sharing a folder, check `author`, `tool`, `extra` and the recorded video paths in `metadata.json`, and remove what
you do not want to disclose. The video content itself is not in the dataset.

## Reporting a vulnerability

Do not post a vulnerability in a public issue. Use GitHub's private vulnerability reporting for the repository (Security tab), or the
contact in the repository's security policy once one is added.

## The web interface

`vidlabel ui` starts a local server that can change your dataset, so it is protected against other web pages and other computers:

- it listens on `127.0.0.1` unless you pass `--host`;
- every request needs a random access token (a cookie set when you open the printed address, or the `X-Vidlabel-Token` header);
- on a local server the `Host` header must be this server and the `Origin` header, when present, must be this page, which stops DNS rebinding and other sites' requests;
- every request that changes something must carry `X-Requested-With: vidlabel`, which a cross-site form cannot send;
- file paths the page names must be inside `--fs-root` folders when given; a server shared on the network refuses file access unless a folder is given.

What it does not do: encrypt traffic (it is plain HTTP, so do not expose it to the internet; use an SSH tunnel or a VPN), tell people apart
(the name is not a login), or stop someone who has the token from editing. Treat the printed address like a password.

## AI agents

`vidlabel mcp` gives an agent no way to change the dataset. Its tools read, and propose rows into the agent's own draft: always `model`,
`unreviewed`, author `agent:<name>`, size-capped, checked like any other edit, and all or nothing. The agent cannot approve, reject, delete,
set `origin` or `author`, or commit; a person takes proposals in under *Proposals* and commits. The server runs over standard input and
output, opens no port, and reads only the dataset it was started on and the video files that dataset names. Treat what an agent proposes as
unreviewed model output, and its notes and labels as untrusted text.
