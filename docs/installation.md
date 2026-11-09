# Installation

vidlabel is two Python packages. Both are published on PyPI; installing `vidlabel` also installs `vidlabel-spec`. The current release is an alpha.

| package | what it is | needs |
|---|---|---|
| `vidlabel-spec` | the format: strict models for metadata, views, vocabulary, releases, the frame index | `pydantic` |
| `vidlabel` | the library and the `vidlabel` command | the spec package, NumPy, PyArrow, zstandard, Typer |

Python 3.10 or newer. The test suite is run in CI on Python 3.10 to 3.13 on Linux, and on Python 3.12 on Windows and macOS
(see [compatibility](compatibility.md)).

## Choose what you need

```bash
pip install vidlabel                  # format + library + command line
pip install "vidlabel[video]"         # + OpenCV: read frames, previews, check-video, frame cache
pip install "vidlabel[ui]"            # + the web interface
pip install "vidlabel[mcp]"           # + the MCP server for AI agents
pip install "vidlabel[torch]"         # + OpenCV and PyTorch: the training loaders

To work on vidlabel itself, install from a checkout instead: `pip install -e packages/vidlabel-spec -e packages/vidlabel` ([development](development.md)).
```

| you want to | install |
|---|---|
| read, write, validate and build datasets; use the command line | the first line only |
| read the format from another program (a different language, a service) | `vidlabel-spec` alone gives you the models and JSON Schema (`vidlabel schema`) |
| draw annotations on real frames, check a file against its metadata, use the frame cache | `[video]` |
| feed PyTorch (`VideoStream`, `VideoFrames`, `PlannedStream`) | `[torch]` |

`[torch]` installs whatever PyTorch pip picks by default. For a GPU build, install PyTorch first by following
[pytorch.org](https://pytorch.org/get-started/locally/), then install vidlabel.

## ffmpeg (ffprobe)

`vidlabel add-video`, `vidlabel index` and `probe_video` call **ffprobe** to read the facts about a video file (frame rate as an
exact fraction, frame count, size, rotation, codec, and per-frame timestamps). It ships with [ffmpeg](https://ffmpeg.org/download.html).

| system | command |
|---|---|
| macOS | `brew install ffmpeg` |
| Debian, Ubuntu | `sudo apt install ffmpeg` |
| Windows | download a build from ffmpeg.org and add its `bin` folder to `PATH`, or `choco install ffmpeg` |

Without ffprobe everything still works except reading facts from a file: you can describe a video by hand with `VideoMetadata`
(see [getting started](getting-started.md#1-make-a-dataset-and-register-a-video)). A missing ffprobe is reported as one clear
error, not a traceback.

## Check the install

```bash
vidlabel --help
vidlabel init demo && vidlabel info demo && vidlabel validate demo
```

For a development install with the test suite, see [development](development.md).

## Uninstall and upgrade

Datasets are plain folders and are not touched by installing or removing the packages. To upgrade, run `pip install --upgrade vidlabel`. A newer library opens older datasets of the same spec major version; see
[compatibility](compatibility.md#upgrading) for what changes between versions.
