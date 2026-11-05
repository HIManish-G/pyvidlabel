# Tutorial: labelling a large scene in tiles

A worked example of a typical earth-observation workflow: a scene is too large to look at whole, so a segmentation model is run on
resized tiles, and you want labels stored once, on the full scene, while each tile sees its own crop. The scene is registered as a
one-frame "video"; a time series of scenes would be a video with many frames. The data is simulated so every step runs without a file.

You will use: views with `output_size`, writing masks in tile pixels, reading them back in scene pixels, and training on one tile.

## 1. A scene and its tiles

<!-- doctest -->
```python
import numpy as np
from vidlabel import Dataset, VideoMetadata, FrameSamples, masks

ds = Dataset.create("scene")
ds.add_video(VideoMetadata(video_id="scene_a", frame_count=1, fps={"num": 1, "den": 1}, width=2048, height=2048))
ds.add_labels(["water", "building"])

for row in range(2):
    for col in range(2):
        ds.add_view({"view_id": f"tile_{row}{col}", "video_id": "scene_a",
                     "region": {"x": col * 1024, "y": row * 1024, "width": 1024, "height": 1024},
                     "output_size": [512, 512]})
print(ds.view_ids())
```

A view stores no pixels. It is a named crop (`region`, in scene pixels) and the size it is resized to before the model sees it
(`output_size`). The four tiles here cover the scene without overlap; overlapping views are allowed.

## 2. Write a mask the way the model saw it

The model worked on a 512x512 resized tile and produced a 512x512 mask. Pass it with `coords="view"` and the library pastes it into
a full-scene canvas, scaling by the view's resize factor:

<!-- doctest -->
```python
tile_mask = np.zeros((512, 512), bool)
tile_mask[100:300, 50:200] = True                      # in tile pixels

ds.add_annotations([{"kind": "mask", "label": "water", "start_frame": 0, "mask": tile_mask,
                     "origin": "model", "tool": "my-segmenter", "view_id": "tile_01"}],
                   "scene_a", coords="view")
```

## 3. Read it back in scene pixels

Annotations are always stored in full-scene coordinates, whatever view they were drawn in:

<!-- doctest -->
```python
row = ds.annotations("scene_a", view_id=["tile_01"]).to_pylist()[0]
print("box in scene pixels:", row["x0"], row["y0"], row["x1"], row["y1"])
full = masks.decode(row["mask"])
print("mask canvas:", full.shape, "pixels set:", int(full.sum()))
```

The tile `tile_01` starts at x=1024, y=0 and halves the resolution, so tile pixel (50, 100) lands at scene pixel
(1024 + 50·2, 0 + 100·2) = (1124, 200). Each tile-pixel becomes a 2x2 block, so the mask has 4x the pixels.

To go the other way, a scene box into tile pixels:

<!-- doctest -->
```python
print(ds.view("tile_01").to_view_box((1124.0, 100.0, 1224.0, 200.0)))
```

## 4. Validate

The validator checks that a row labelled in a view lies inside the view's region and that its mask was made for the scene's size.

<!-- doctest -->
```python
assert ds.validate() == []
```

## 5. Train on one tile

Build a release and ask `FrameSamples` for a single view. The samples are the tile's crop, with only that tile's annotations, in
the tile's own (resized) pixels:

<!-- doctest -->
```python
ds.build()
tile = FrameSamples(ds, release="latest", view_id="tile_01", kinds=("mask",), with_masks=True)
t = tile.targets("scene_a", 0, (512, 512))
print(t["boxes"], "mask shape", t["masks"].shape, "pixels", int(t["masks"].sum()))
```

The mask comes back at 512x512 with 30,000 pixels: the 120,000 scene pixels shrunk by the same factor of four.

## Things to know

- A view is a **tag** on rows. A tile model's labels belong to its tile; labels drawn on the whole scene (`view_id` empty) are not
  seen by `FrameSamples(view_id="tile_01")` unless you copy them in with `ds.derive_view_rows("scene_a", "tile_01")`
  ([views](views.md#copying-whole-frame-labels-into-a-view)).
- Resizing is nearest-neighbour for masks, so a mask survives a round trip through a tile at the same size exactly; a mask drawn at a
  different resolution than the tile (a coarser model output pasted up) cannot become more detailed than it was.
- Rows that only partly overlap a tile are not copied by `derive_view_rows`, because cutting a mask would change it.
- For a time series (many dates of one area), register each date as a frame of one video, or each date as its own video if
  they have different sizes. Spans of time ("flood event, dates 12 to 15") are markers.
