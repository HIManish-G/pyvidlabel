"""Read frames from the videos a dataset points at. Needs OpenCV: pip install "vidlabel[video]".

Frames are returned as RGB uint8 arrays in the video's *displayed* orientation (rotation applied), which is the
coordinate system every annotation uses. If a View is given, the frame is cropped to its region and resized to its
`output_size`, i.e. exactly what a pipeline working on that view would see.

Decoding is sequential, so reading frame N costs a decode of frames 0..N. That is what makes it frame-accurate
(seeking in compressed video often is not); a faster indexed reader can come later behind the same functions.
"""
from __future__ import annotations

import hashlib
import os
import warnings
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


def crop_mask_to_view(mask: np.ndarray, view: View | None) -> np.ndarray:
    """Crop a full-frame mask to a view's region and resize (nearest) to its output_size, like `crop_to_view` does for pixels."""
    from . import masks
    if view is None or view.region is None:
        return mask
    r = view.region
    m = mask[r.y:r.y + r.height, r.x:r.x + r.width]
    if view.output_size is not None and (m.shape[1], m.shape[0]) != tuple(view.output_size):
        m = masks.resize_nearest(m, view.output_size[1], view.output_size[0])
    return m


def _display_frame(bgr: np.ndarray, meta: VideoMetadata, cap_rotates: bool, view: View | None) -> np.ndarray:
    """A decoded BGR frame -> RGB in displayed orientation, cropped to `view`. Checks it is the registered file."""
    cv2 = _cv2()
    stored, display = (meta.width, meta.height), meta.display_size
    h, w = bgr.shape[:2]
    if (w, h) == stored:
        already_rotated = cap_rotates
    elif (w, h) == display and stored != display:
        already_rotated = True  # this OpenCV build rotated for us
    else:
        raise ValueError(f"video frame is {w}x{h} but the metadata says {stored[0]}x{stored[1]} "
                         f"(rotation {meta.rotation}): this is not the file that was registered")
    frame = bgr if already_rotated else _apply_rotation(bgr, meta.rotation)
    return crop_to_view(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), view)


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
            frame = _display_frame(bgr, meta, cap_rotates, view)
            yield pos, frame
            pos += 1
            k += 1
        if k < len(wanted):
            raise IndexError(f"frame {wanted[k]} is past the end of the video (only {pos} frames could be decoded)")
    finally:
        cap.release()


_SEEK_VERDICTS: dict[tuple, bool] = {}   # (path, size, mtime) -> "seeking returns exactly the frames sequential decoding does"
_SEEK_PROBES = (40, 130, 300)            # frames compared once per video before seeking is trusted


def _digest(frame: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=16).digest()


def seek_mismatches(path: str | Path, meta: VideoMetadata, positions: Iterable[int]) -> list[int]:
    """Frames at `positions` for which seeking does NOT return exactly what sequential decoding returns.

    Empty means seeking is frame-exact at those positions. OpenCV's seek is exact for ordinary constant-frame-rate
    files but can land a few frames off on variable-frame-rate video or odd containers, which would silently misalign
    every box, so seeking is only used after this check (see `FrameReader`).
    """
    wanted = sorted({int(p) for p in positions if 0 <= int(p) < meta.frame_count})
    truth = {i: _digest(f) for i, f in read_frames(path, wanted, meta)}
    bad: list[int] = []
    with FrameReader(path, meta, seek="always") as r:
        # a forward jump from the start, then back and forth, so backward seeks are checked too
        order = wanted[::-1][:1] + wanted[:1] + wanted[len(wanted) // 2:] + wanted[:len(wanted) // 2]
        for i in dict.fromkeys(order):
            try:
                ok = _digest(r.read(i)) == truth.get(i)
            except IndexError:
                ok = False
            if not ok:
                bad.append(i)
    return sorted(bad)


class FrameReader:
    """Keeps one video open and reads frames by index.

    Reading forwards only decodes what is in between. Asking for an earlier frame, or one far ahead, normally means
    decoding from the start of the file; `seek` makes that cheap by jumping to the nearest keyframe first:

    - `seek="auto"` (default): seek only if it has been verified for this file: not variable frame rate, and the frames
      returned after seeking are identical to sequential decoding at a few test positions (checked once per file per
      process, a short decode). Otherwise decode sequentially, with a warning.
    - `seek="off"`: never seek.
    - `seek="always"`: seek without checking (for tests and for `seek_mismatches`).

    For training prefer `VideoStream`, which reads each video front to back and never needs to seek.
    Not thread-safe; use one reader per process or thread.
    """

    def __init__(self, path: str | Path, meta: VideoMetadata, view: View | None = None, *, seek: str = "auto",
                 seek_threshold: int = 100):
        if seek not in ("auto", "off", "always"):
            raise ValueError("seek must be 'auto', 'off' or 'always'")
        self.path, self.meta, self.view = Path(path), meta, view
        self.seek, self.seek_threshold = seek, seek_threshold
        self._cap = None
        self._pos = 0  # index of the next frame the decoder will return
        self._cap_rotates = False

    @property
    def seek_enabled(self) -> bool:
        """Whether this reader will use seeking (runs the one-off check in "auto" mode)."""
        if self.seek != "auto":
            return self.seek == "always"
        if self.meta.is_vfr or self.meta.frame_count <= _SEEK_PROBES[0] + 20:
            return False   # variable frame rate is where seeking is known to be wrong; tiny files do not need it
        st = os.stat(self.path)
        key = (str(self.path.resolve()), st.st_size, st.st_mtime_ns, self.meta.frame_count)
        if key not in _SEEK_VERDICTS:
            bad = seek_mismatches(self.path, self.meta, _SEEK_PROBES)
            _SEEK_VERDICTS[key] = not bad
            if bad:
                warnings.warn(f"{self.path.name}: seeking is not frame-exact here (frames {bad} differ), so random access "
                              f"decodes from the start and is slow. Use VideoStream, or re-encode as constant frame rate.",
                              RuntimeWarning, stacklevel=3)
        return _SEEK_VERDICTS[key]

    def _reopen(self) -> None:
        self.close()
        self._cap = _open(self.path)
        self._cap_rotates = bool(self._cap.get(_cv2().CAP_PROP_ORIENTATION_AUTO))
        self._pos = 0

    def read(self, index: int) -> np.ndarray:
        """RGB frame `index` in displayed orientation, cropped to the view if there is one."""
        index = int(index)
        if index < 0:
            raise IndexError("frame indices must be >= 0")
        if self._cap is None:
            self._reopen()
        gap = index - self._pos
        if gap < 0 or (gap > self.seek_threshold and self.seek != "off"):
            if self.seek_enabled:
                self._cap.set(_cv2().CAP_PROP_POS_FRAMES, index)
                self._pos = index
            elif gap < 0:
                self._reopen()
        while self._pos < index:
            if not self._cap.grab():
                raise IndexError(f"frame {index} is past the end of the video (only {self._pos} frames could be decoded)")
            self._pos += 1
        ok, bgr = self._cap.read()
        if not ok:
            raise IndexError(f"frame {index} is past the end of the video (only {self._pos} frames could be decoded)")
        self._pos += 1
        return _display_frame(bgr, self.meta, self._cap_rotates, self.view)

    def close(self) -> None:
        """Release the open video file. Safe to call twice; the reader reopens it on the next `read`."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def read_frame(path: str | Path, index: int, meta: VideoMetadata, *, view: View | None = None) -> np.ndarray:
    """One frame, RGB, displayed orientation. Decodes from the start of the video: use `read_frames` or `FrameReader` for many."""
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
