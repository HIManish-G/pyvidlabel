# vidlabel

An open, directory-based format and Python library for **video annotation data**: time markers and spans, bounding boxes,
masks, tracks and labels in one annotation table (JSON + Parquet). Video pixels are not stored; datasets point at your videos.

**Status: alpha (0.6.0).** The format and API can still change between minor versions; see the
[changelog](https://github.com/HIManish-G/pyvidlabel/blob/main/docs/changelog.md).

```
pip install vidlabel                # library and command line
pip install "vidlabel[video]"       # + OpenCV: read frames, preview, check-video
pip install "vidlabel[ui]"          # + local web interface:  vidlabel ui my_dataset
pip install "vidlabel[mcp]"         # + MCP server for AI agents:  vidlabel mcp my_dataset
pip install "vidlabel[torch]"       # + PyTorch datasets
```

Python 3.10 or newer. `vidlabel-spec` (the format's models and JSON Schema) is installed with it.

```python
from vidlabel import Dataset, VideoMetadata

ds = Dataset.create("my_dataset")
ds.add_video(VideoMetadata(video_id="clip", frame_count=100, fps={"num": 25, "den": 1}, width=640, height=360))
ds.add_labels(["cat"])
ds.add_annotations([{"kind": "box", "label": "cat", "start_frame": 0, "box": (10, 10, 200, 150), "origin": "human"}], "clip")
print(ds.validate())
```

Documentation, format specification and guides: <https://github.com/HIManish-G/pyvidlabel> ([wiki](https://github.com/HIManish-G/pyvidlabel/wiki)). MIT licensed.
