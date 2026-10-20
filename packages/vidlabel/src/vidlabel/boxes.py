"""Convert boxes between the common layouts. Stored boxes are `xyxy` in pixels (x1/y1 exclusive); training code often
wants something else, so conversion happens when reading, never in storage.

  xyxy    x0, y0, x1, y1          corners (Pascal VOC, torchvision)
  xywh    x0, y0, width, height   top-left + size (COCO)
  cxcywh  cx, cy, width, height   centre + size (YOLO)

`normalized=True` means values are fractions of the image size (YOLO files use cxcywh normalized).
"""
from __future__ import annotations

import numpy as np

FORMATS = ("xyxy", "xywh", "cxcywh")


def convert_boxes(boxes, src: str = "xyxy", dst: str = "xyxy", *, size: tuple[int, int] | None = None,
                  src_normalized: bool = False, dst_normalized: bool = False) -> np.ndarray:
    """Convert an (N, 4) array of boxes. `size` is (width, height) of the image they refer to; it is required when
    `src_normalized` or `dst_normalized` is set. Returns float32."""
    for f in (src, dst):
        if f not in FORMATS:
            raise ValueError(f"unknown box format '{f}'; use one of {FORMATS}")
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if (src_normalized or dst_normalized) and size is None:
        raise ValueError("size=(width, height) is required to convert normalized boxes")
    scale = np.array([size[0], size[1], size[0], size[1]], dtype=np.float64) if size else None
    if src_normalized:
        b = b * scale
    a, c, d, e = b.T
    if src == "xywh":
        x0, y0, x1, y1 = a, c, a + d, c + e
    elif src == "cxcywh":
        x0, y0, x1, y1 = a - d / 2, c - e / 2, a + d / 2, c + e / 2
    else:
        x0, y0, x1, y1 = a, c, d, e
    if dst == "xywh":
        out = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)
    elif dst == "cxcywh":
        out = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=1)
    else:
        out = np.stack([x0, y0, x1, y1], axis=1)
    if dst_normalized:
        out = out / scale
    return out.astype(np.float32)
