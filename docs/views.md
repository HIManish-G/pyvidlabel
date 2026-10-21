# Views: several crops of one video

## The problem

Sometimes one video is processed as more than one picture. A model looks at the main area; a different pipeline reads a
caption strip along the bottom, cropped and resized to the size *its* model wants. Their boxes are in different pixel
spaces, they use different labels, and they cover different frames. Storing each pipeline's output as a separate
dataset loses the fact that it is all one video; storing the raw crop coordinates makes them useless for anything else.

## What a view is

A **view** is a named region and/or frame range of one video, stored in `views/<view_id>.json`. No new video file is made.

| field | meaning |
|---|---|
| `view_id`, `video_id` | the view's name and the video it belongs to |
| `region` | `x, y, width, height` in displayed pixels: the crop; omit for the whole frame |
| `output_size` | `(width, height)` the crop is resized to before the pipeline sees it; omit if it is not resized. Needs a `region` |
| `frame_range` | `[first, last]` inclusive: only these frames are processed in this view |
| `allowed_labels` | if set, annotations in this view may only use these labels (all must be in the vocabulary) |

Views of one video may overlap and coexist.

## The one rule

**Annotations are always stored in full-frame coordinates.** A row labelled in a view carries that `view_id`, but its box
and mask are in the source video's pixels. That is what lets the same row be drawn on the original video, compared with
another view's output, or used by any tool that knows nothing about views.

The conversion from the pixels a pipeline saw is
`source = region.xy + view_pixels / scale`, where `scale = output_size / region size` (1 if there is no `output_size`).
You rarely do this yourself:

<!-- doctest -->
```python
from vidlabel import Dataset, VideoMetadata

ds = Dataset.create("views_demo")
ds.add_video(VideoMetadata(video_id="clip", frame_count=300, fps={"num": 25, "den": 1}, width=1280, height=720))
ds.add_labels(["person", "car", "caption_text", "logo"])

ds.add_view({"view_id": "main", "video_id": "clip", "region": {"x": 0, "y": 0, "width": 1280, "height": 620},
             "allowed_labels": ["person", "car"]})
ds.add_view({"view_id": "lower_strip", "video_id": "clip",
             "region": {"x": 0, "y": 620, "width": 1280, "height": 100}, "output_size": (640, 50),
             "allowed_labels": ["caption_text", "logo"]})

# A model that only saw the strip (shrunk to 640x50) found text at (40, 10)-(200, 30) in *its* pixels.
ds.add_annotations([{"kind": "box", "label": "caption_text", "start_frame": 5, "box": (40, 10, 200, 30),
                     "origin": "model", "tool": "caption-detector", "view_id": "lower_strip"}],
                   "clip", coords="view")

row = ds.annotations("clip").to_pylist()[0]
print([row[k] for k in ("x0", "y0", "x1", "y1")])   # stored in full-frame pixels
assert [row[k] for k in ("x0", "y0", "x1", "y1")] == [80.0, 640.0, 400.0, 680.0]
```

The strip starts at y = 620 and was shrunk by half, so the model's `(40, 10, 200, 30)` is `(80, 640, 400, 680)` in the
full frame. With `coords="view"` masks are converted too: pass a boolean array the size of the view's region or of its
`output_size`, and it is resized and pasted into a full-frame canvas.

The helpers behind this are on the view itself:

<!-- doctest -->
```python
view = ds.view("lower_strip")
print(view.scale)                                   # (0.5, 0.5)
print(view.to_view_box((80, 640, 400, 680)))        # (40.0, 10.0, 200.0, 30.0)
print(view.to_source_box((40, 10, 200, 30)))        # (80.0, 640.0, 400.0, 680.0)
```

## Rows without a view

A row with no `view_id` refers to the whole frame. Both kinds live in the same table:

<!-- doctest -->
```python
ds.add_annotations([{"kind": "box", "label": "person", "start_frame": 5, "box": (100, 100, 200, 300),
                     "origin": "model"}], "clip")

ds.annotations("clip", view_id="lower_strip").num_rows     # 1: only rows labelled in that view
ds.annotations("clip", view_id=[None]).num_rows            # 1: only whole-frame rows
ds.annotations("clip", view_id=["lower_strip", None]).num_rows   # 2: both
```

## What the validator checks

Rows tagged with a view must lie **inside** it: the box within the view's region, the frames within its `frame_range`,
the label in its `allowed_labels`, and the view must belong to the same video. A view whose region falls outside the
video, or whose `allowed_labels` are not in the vocabulary, is rejected when you add it.

## Using views downstream

- **Releases:** `ds.build(view_ids=["lower_strip"])` (Python) compiles a release of just that view's rows.
- **Fast reads:** `Compiled.clip(video, start, end, view_id="lower_strip")`.
- **Training:** `FrameSamples(ds, view_id="lower_strip")` yields the *cropped, resized* frame with boxes already converted
  to its pixels, and only that view's annotations: exactly what that pipeline sees. See [training](training.md).
- **Checking:** `vidlabel preview my_dataset clip --view lower_strip` draws that view's crop with its annotations.
- **Tracks** are numbered per view: two pipelines can each use `track_id` 1 without clashing
  ([concepts](concepts.md#tracks)).
