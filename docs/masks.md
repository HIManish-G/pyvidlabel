# Masks

A mask annotation is a row with `kind="mask"`, the encoded mask in `mask`, the codec name in `mask_codec`, and the mask's
bounding box in `x0, y0, x1, y1`. You pass a boolean array; the library encodes it and fills in the box.

<!-- doctest -->
```python
import numpy as np
from vidlabel import Dataset, VideoMetadata, masks

ds = Dataset.create("masks_demo")
ds.add_video(VideoMetadata(video_id="clip", frame_count=100, fps={"num": 30, "den": 1}, width=320, height=240))
ds.add_labels(["car"])

m = np.zeros((240, 320), bool)          # (height, width), the displayed frame
m[100:140, 150:190] = True
ds.add_annotations([{"kind": "mask", "label": "car", "start_frame": 10, "mask": m, "origin": "human"}], "clip")

row = ds.annotations("clip").to_pylist()[0]
print(row["mask_codec"], (row["x0"], row["y0"], row["x1"], row["y1"]))      # bitpack_zstd (150, 100, 190, 140)
assert np.array_equal(masks.decode(row["mask"]), m)                         # lossless
```

## The codec: `bitpack_zstd`

Crop the mask to its bounding box, pack each row of the crop into bits (`np.packbits`), and compress with zstd (level 3).
A small header records the full frame size and where the crop sits, so `decode` rebuilds the full-frame array. It is
**lossless**, and an empty mask is just the header. The byte layout is in the [spec](SPEC.md#mask-codec-bitpack_zstd).

Why this and not COCO run-length encoding, which most tools use? On the data we measured (real masks from SA-V, plus
DAVIS-sequence predictions), crop + bit-pack + zstd was **16 to 36 percent smaller than COCO RLE and decoded about 6 times
faster**. Decoding, not size, is what costs CPU time during training. A context-modelled arithmetic coder was about 4
times smaller than COCO RLE but slower to decode, so it is left as a possible future *archival* codec;
`mask_codec` is a per-row column, so codecs can coexist in one table.

Be aware of the limits of that measurement: it used one SA-V video and DAVIS predictions (not human ground truth), a
single thread, and no storage or I/O timing. Treat the percentages as a good indication, not a guarantee for your data.

## Helpers (`vidlabel.masks`)

| function | does |
|---|---|
| `encode(mask)` / `decode(blob)` | boolean array to bytes and back (full frame) |
| `decode_crop(blob)` | only the cropped region, `(crop, y0, x0)`, without allocating the full frame |
| `bbox(blob)` | `(x0, y0, x1, y1)`, or `None` for an empty mask |
| `info(blob)` | header facts: frame `height`/`width`, crop `x0`, `y0`, `w`, `h`, `empty` |
| `place_in_frame(mask, x, y, frame_w, frame_h)` | paste a crop-sized mask into an empty full-frame canvas |
| `resize_nearest(mask, h, w)` | nearest-neighbour resize of a boolean mask |
| `from_coco_rle(rle)` / `to_coco_rle(mask)` | convert COCO run-length dicts; used only by converters, **never stored** |

<!-- doctest -->
```python
blob = masks.encode(m)
print(masks.info(blob))                 # {'height': 240, 'width': 320, 'y0': 100, 'x0': 150, 'h': 40, 'w': 40, 'empty': False}
print(masks.bbox(blob))                 # (150, 100, 190, 140)
crop, y0, x0 = masks.decode_crop(blob)
assert crop.shape == (40, 40) and (y0, x0) == (100, 150)
assert np.array_equal(masks.from_coco_rle(masks.to_coco_rle(m)), m)
```

## Things to know

- The mask must be the size of the **displayed** frame. `vidlabel validate --deep` decodes every mask and checks its
  frame size and that its box equals the mask's pixel bounding box.
- Masks from a view are pasted into the full frame, see [views](views.md).
- To read masks for training, use `FrameSamples(..., kinds=("mask",), with_masks=True)`: you get `(N, H, W)` uint8
  arrays aligned with the boxes ([training](training.md)).
- To use another codec later, add a value to `MaskCodec` in the spec package and a module next to `masks.py` with a
  lossless round-trip test ([CONTRIBUTING](../CONTRIBUTING.md)).
