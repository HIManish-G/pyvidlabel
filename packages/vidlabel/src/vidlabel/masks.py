"""Mask codec: crop to bounding box, bit-pack rows, zstd.

Blob layout (little endian):
    u8  version (=1)
    u32 frame_height, frame_width      full frame size the mask belongs to
    u32 y0, x0, h, w                   crop rectangle (h = w = 0 means empty mask, no payload)
    ... zstd( np.packbits(crop, axis=-1) )   rows padded to whole bytes
Lossless. Chosen by benchmark: ~0.1 ms decode, smaller than COCO RLE on typical object masks.
"""
from __future__ import annotations

import struct

import numpy as np
import zstandard

CODEC = "bitpack_zstd"
_VERSION = 1
_HDR = struct.Struct("<BIIIIII")
_LEVEL = 3


def _as_bool2d(mask) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {m.shape}")
    return m if m.dtype == np.bool_ else m.astype(bool)


def encode(mask) -> bytes:
    """Encode a boolean (H, W) array: crop to its bounding box, bit-pack the rows, zstd-compress. Lossless."""
    m = _as_bool2d(mask)
    H, W = m.shape
    rows, cols = m.any(axis=1), m.any(axis=0)
    if not rows.any():
        return _HDR.pack(_VERSION, H, W, 0, 0, 0, 0)
    y0 = int(rows.argmax()); y1 = H - int(rows[::-1].argmax())
    x0 = int(cols.argmax()); x1 = W - int(cols[::-1].argmax())
    packed = np.packbits(m[y0:y1, x0:x1], axis=-1)
    payload = zstandard.ZstdCompressor(level=_LEVEL).compress(packed.tobytes())
    return _HDR.pack(_VERSION, H, W, y0, x0, y1 - y0, x1 - x0) + payload


def info(blob: bytes) -> dict:
    """Read only the header: frame size, crop rectangle. Cheap."""
    if len(blob) < _HDR.size:
        raise ValueError("mask blob too short")
    ver, H, W, y0, x0, h, w = _HDR.unpack_from(blob)
    if ver != _VERSION:
        raise ValueError(f"unsupported mask blob version {ver}")
    return {"height": H, "width": W, "y0": y0, "x0": x0, "h": h, "w": w, "empty": h == 0 or w == 0}


def bbox(blob: bytes) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the mask, or None if empty."""
    i = info(blob)
    return None if i["empty"] else (i["x0"], i["y0"], i["x0"] + i["w"], i["y0"] + i["h"])


def decode_crop(blob: bytes) -> tuple[np.ndarray, int, int]:
    """Decode only the cropped region: (crop[h, w] bool, y0, x0). Skips allocating the full frame."""
    i = info(blob)
    if i["empty"]:
        return np.zeros((0, 0), dtype=bool), 0, 0
    h, w = i["h"], i["w"]
    stride = (w + 7) // 8
    raw = zstandard.ZstdDecompressor().decompress(blob[_HDR.size:], max_output_size=h * stride)
    if len(raw) != h * stride:
        raise ValueError("corrupt mask payload (wrong decoded size)")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(h, stride), axis=1, count=w)
    return bits.astype(bool), i["y0"], i["x0"]


def decode(blob: bytes) -> np.ndarray:
    """Full-frame boolean mask (H, W)."""
    i = info(blob)
    out = np.zeros((i["height"], i["width"]), dtype=bool)
    if not i["empty"]:
        crop, y0, x0 = decode_crop(blob)
        out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out


def place_in_frame(mask, x: int, y: int, frame_w: int, frame_h: int) -> np.ndarray:
    """Paste a crop-sized mask into an empty full-frame canvas with its top-left at (x, y); parts outside are clipped."""
    m = _as_bool2d(mask)
    h, w = m.shape
    out = np.zeros((frame_h, frame_w), dtype=bool)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame_w), min(y + h, frame_h)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = m[y0 - y:y1 - y, x0 - x:x1 - x]
    return out


def resize_nearest(mask, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask to (h, w)."""
    m = _as_bool2d(mask)
    H, W = m.shape
    ys = np.minimum(((np.arange(h) + 0.5) * H / h).astype(int), H - 1)
    xs = np.minimum(((np.arange(w) + 0.5) * W / w).astype(int), W - 1)
    return m[np.ix_(ys, xs)]


# ---- COCO RLE interop (for importing from / exporting to the COCO ecosystem) ----

def from_coco_rle(rle: dict) -> np.ndarray:
    """Decode a COCO RLE dict {'size': [h, w], 'counts': list[int] | str} to a boolean (h, w) mask."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if isinstance(counts, str):
        counts = _counts_from_string(counts)
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() != h * w:
        raise ValueError("RLE counts do not sum to h*w")
    flat = np.repeat(np.arange(len(counts)) % 2 == 1, counts)
    return flat.reshape(w, h).T.copy()  # COCO runs are column-major


def to_coco_rle(mask, compressed: bool = False) -> dict:
    """A boolean mask as a COCO run-length dict (`counts`, `size`); `compressed=True` gives the compact string form."""
    m = _as_bool2d(mask)
    h, w = m.shape
    flat = m.ravel(order="F").astype(np.int8)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:
        runs = [0] + runs
    return {"size": [h, w], "counts": _counts_to_string(runs) if compressed else runs}


def _counts_from_string(s: str) -> list[int]:
    cnts: list[int] = []
    p = 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def _counts_to_string(cnts: list[int]) -> str:
    out = []
    for i, x in enumerate(cnts):
        if i > 2:
            x -= cnts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)
