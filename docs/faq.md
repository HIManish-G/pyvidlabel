# Troubleshooting and FAQ

## Installing

**`pip install "vidlabel[video]"` says the package "does not provide the extra 'video'".**
That happens with an older or stale editable install. From the repository root run
`pip install -e "packages/vidlabel[video]"`, or just `pip install opencv-python-headless`.

**`ffprobe not found` when running `add-video`.**
`add-video` needs ffprobe, which comes with ffmpeg: `brew install ffmpeg` (macOS), `sudo apt install ffmpeg` (Debian/Ubuntu),
or a download from [ffmpeg.org](https://ffmpeg.org/download.html) added to your PATH (Windows). Only registering a video
needs it. You can also build the metadata by hand ([getting started](getting-started.md#1-make-a-dataset-and-register-a-video)).

**"reading frames needs OpenCV".**
`preview`, `check-video` and the loader read frames through OpenCV. Install the `video` or `torch` extra as above.

## Checking against the video

**`vidlabel preview` drew no annotations.**
Preview only draws the frames it is asked for. With `--frames 85-135 --every 10` it draws 85, 95, 105 and so on, so a box on
frame 100 is skipped. When you pass `--frames` it prints how many annotations were active on the frames it drew and lists
where your annotations are if none were. Without `--frames` it picks a spread of annotated frames.

**Boxes are in the wrong place on a phone video.**
Check rotation. Annotation coordinates are in the *displayed* orientation (what a player shows); `metadata.json` has the
stored size and a clockwise `rotation`. If your boxes were drawn on the unrotated stored frame, convert them with
`vidlabel.boxes.rotate_boxes`. See [coordinates and rotation](concepts.md#coordinates-and-rotation).

**`check-video` says "decoded N frames but the metadata says M".**
The file is not the one that was registered (re-encoded or trimmed), or the frame count was *estimated* at registration
(the message says so) because the container did not state it. Re-register with `add_video(..., exist_ok=True)` after
checking, or find the original file.

**"video frame is WxH but the metadata says ..."**
The file's frame size does not match what was registered: wrong file.

**The recorded video path is wrong on this machine.**
Paths are hints. Pass `--video-path file.mp4`, or `--root some/folder` to search by file name (`roots=[...]` in Python).

## Variable-frame-rate video

Some recordings (many phone and screen captures) do not have a constant frame rate. vidlabel flags them (`is_vfr: true`
in the metadata) and handles them safely, but a frame *index* in such a video is a position in the decoded stream, not
`time * fps`, so convert times to frame indices with care. Seeking is never used on them, because OpenCV's seek was
measured to land on the wrong frame in about three tries out of four. Random access to such a file is therefore slow,
and `VideoStream` (which reads front to back) is unaffected. Re-encoding to a constant frame rate
(`ffmpeg -i in.mp4 -r 30 out.mp4`, adjusting the rate) makes seeking available again, but then **re-register the video and
check that your annotations still line up**, because frame indices change.

**Warning: "seeking is not frame-exact here ...".**
The one-off check found that seeking returns different frames than sequential decoding for this file, so reads fall back to
sequential decoding and are slow. Use `VideoStream`, or re-encode. `vidlabel check-video my_dataset clip --seek` runs a wider check.

## Writing annotations

**`ValidationError: ... label not in vocab.json`.** Add the label with `ds.add_labels([...])`, or pass `create_labels=True`.

**`frame beyond video length`.** A frame number is past `frame_count`: wrong video, or a source that counts frames from 1.

**`unknown field(s) [...]; put custom data inside 'extra'`.** Row keys must be column names. Put your own data in `extra`.

**`the same track_id appears more than once in one frame`.** Two boxes (or two masks) of one track on a frame. Give the
objects different ids, or reject one. Rejected rows are ignored.

**`annotation_id N already exists`.** You passed an explicit id that is already stored. Omit it to get a fresh random one.

**Nothing was written.** Writes are all-or-nothing per call: if any row is invalid, none is saved. Read the error: it names the check.

**How do I rename a label?** There is no rename. Add the new label and change the rows:
`ds.update_annotations(video_id, ids, label="new_name")`. The vocabulary only grows, so old names stay listed.

**Can two people annotate at once?** Yes, on *different videos*. Two writers on the *same* video can overwrite each other: there
are no locks ([design decisions](design-decisions.md#one-writer-per-video-no-locks)).

**Are non-English labels fine?** Yes. All JSON is written as UTF-8 on every operating system.

## Training

**Do I need PyTorch?** No. `FrameSamples` returns NumPy arrays; the PyTorch classes are a thin wrapper.

**DataLoader hangs or errors on Windows or macOS with `num_workers > 0`.** Put the training code under
`if __name__ == "__main__":`. That is a PyTorch requirement on platforms that start workers with `spawn`.

**Why is `VideoFrames` slow when shuffled?** Video decoding is sequential; see [training](training.md#reading-video-quickly-stream-do-not-shuffle). Use `VideoStream`.

**Do I get the same data every run?** With `release="<id>"`, yes: the same rows and the same class indices. `release=None`
reads the live annotations and can change.

**Can I do class-balanced or weighted sampling?** Not built in yet ([training](training.md#not-built-yet)).

## Scope

**Can I import COCO / YOLO / MOT / CVAT?** Not with a built-in command, on purpose. Write a short converter:
[writing a converter](converters.md) has a tested example, and `convert_boxes` and `masks.from_coco_rle` do the format-specific maths.

**Does it store my videos?** No. It records where they are and a hash to recognise them.

**Is it on PyPI?** Not yet. Install from a source checkout ([getting started](getting-started.md#install)).
