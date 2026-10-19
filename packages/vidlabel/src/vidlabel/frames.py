"""Read frames from the videos a dataset points at. Needs OpenCV: pip install "vidlabel[video]".

Frames are returned as RGB uint8 arrays in the video's *displayed* orientation (rotation applied), which is the
coordinate system every annotation uses. If a View is given, the frame is cropped to its region and resized to its
`output_size`, i.e. exactly what a pipeline working on that view would see.

Decoding is sequential, so reading frame N costs a decode of frames 0..N. That is what makes it frame-accurate
(seeking in compressed video often is not); a faster indexed reader can come later behind the same functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from vidlabel_spec import VideoMetadata, View

from .probe import file_hash


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - exercised only without OpenCV
        raise ImportError('reading frames needs OpenCV. Install it with: pip install opencv-python-headless '
                          '(from a source checkout: pip install -e "packages/vidlabel[video]")') from e
    return cv2


def _open(path: str | Path):
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {path}")
    try:  # we apply rotation ourselves so behaviour does not depend on the OpenCV build
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except Exception:
        pass
    return cap


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    cv2 = _cv2()
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def crop_to_view(frame: np.ndarray, view: View | None) -> np.ndarray:
    """Crop a displayed-orientation frame to a view's region and resize to its output_size (if any)."""
    if view is None or view.region is None:
        return frame
    r = view.region
    out = frame[r.y:r.y + r.height, r.x:r.x + r.width]
    if view.output_size is not None and (out.shape[1], out.shape[0]) != tuple(view.output_size):
        out = _cv2().resize(out, tuple(view.output_size), interpolation=_cv2().INTER_AREA)
    return out


def read_frames(path: str | Path, frame_indices: Iterable[int], meta: VideoMetadata, *,
                view: View | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, RGB frame) for each requested index, in increasing order.

    Raises ValueError if the file's frame size does not match `meta` (wrong or re-encoded file) and IndexError if a
    requested frame is past the end of the video.
    """
    cv2 = _cv2()
    wanted = sorted(set(int(i) for i in frame_indices))
    if not wanted:
        return
    if wanted[0] < 0:
        raise IndexError("frame indices must be >= 0")
    cap = _open(path)
    try:
        cap_rotates = bool(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))  # True only if OpenCV would not let us disable it
        stored = (meta.width, meta.height)
        display = meta.display_size
        k, pos = 0, 0
        while k < len(wanted):
            if pos < wanted[k]:
                if not cap.grab():
                    break
                pos += 1
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if (w, h) == stored:
                already_rotated = cap_rotates
            elif (w, h) == display and stored != display:
                already_rotated = True  # this OpenCV build rotated for us
            else:
                raise ValueError(f"video frame is {w}x{h} but the metadata says {stored[0]}x{stored[1]} "
                                 f"(rotation {meta.rotation}): this is not the file that was registered")
            frame = bgr if already_rotated else _apply_rotation(bgr, meta.rotation)
            frame = crop_to_view(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), view)
            yield pos, frame
            pos += 1
            k += 1
        if k < len(wanted):
            raise IndexError(f"frame {wanted[k]} is past the end of the video (only {pos} frames could be decoded)")
    finally:
        cap.release()


def read_frame(path: str | Path, index: int, meta: VideoMetadata, *, view: View | None = None) -> np.ndarray:
    for _, f in read_frames(path, [index], meta, view=view):
        return f
    raise IndexError(index)  # pragma: no cover


def check_video(path: str | Path, meta: VideoMetadata, *, verify_hash: bool = False) -> list[str]:
    """Compare a video file with the metadata registered for it. Returns a list of problems (empty = consistent)."""
    problems: list[str] = []
    p = Path(path)
    if not p.exists():
        return [f"{p} does not exist"]
    if verify_hash and meta.content_hash:
        algo = meta.content_hash.split(":")[0]
        if algo != "sha256":
            problems.append(f"cannot verify hash algorithm '{algo}'")
        elif file_hash(p) != meta.content_hash:
            problems.append("content hash differs: this is not the same file that was registered")
    cap = _open(p)
    try:
        ok, bgr = cap.read()
        if not ok:
            return problems + ["could not decode the first frame"]
        h, w = bgr.shape[:2]
        n = 1
        while cap.grab():
            n += 1
    finally:
        cap.release()
    if (w, h) not in {(meta.width, meta.height), meta.display_size}:
        problems.append(f"frame size {w}x{h} differs from the metadata ({meta.width}x{meta.height})")
    if n != meta.frame_count:
        problems.append(f"decoded {n} frames but the metadata says {meta.frame_count}"
                        + (" (frame_count was estimated when registering)" if meta.extra.get("frame_count_estimated") else ""))
    return problems
