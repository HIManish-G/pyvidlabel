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


def rotate_boxes(boxes, rotation: int, stored_size: tuple[int, int]) -> np.ndarray:
    """Move (N, 4) `xyxy` boxes from the stored frame to the displayed orientation of a rotated video.

    `rotation` is the video's clockwise rotation (0, 90, 180 or 270, as in VideoMetadata.rotation) and `stored_size` is
    (width, height) of the stored frame. Annotations in a vidlabel dataset use the displayed orientation, so a source that
    drew boxes on the unrotated stored frame needs this before `add_annotations`. x1/y1 stay exclusive.
    """
    if rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270")
    W, H = stored_size
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x0, y0, x1, y1 = b.T
    if rotation == 90:
        out = np.stack([H - y1, x0, H - y0, x1], axis=1)
    elif rotation == 180:
        out = np.stack([W - x1, H - y1, W - x0, H - y0], axis=1)
    elif rotation == 270:
        out = np.stack([y0, W - x1, y1, W - x0], axis=1)
    else:
        out = b
    return out.astype(np.float32)
