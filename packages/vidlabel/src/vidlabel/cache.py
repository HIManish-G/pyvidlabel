"""An optional frame cache: the frames a training run will use, decoded once and saved as image files.

Random access into compressed video costs a decode from the previous keyframe (see docs/training.md). When you need
truly random sampling, decode each sampled frame once, in order and at full speed, and read it back as an image after
that: a file read instead of up to a keyframe interval of decoding.

The cache is *derived* data. It lives wherever you point it, outside the dataset, and can be deleted and rebuilt at any
time. Each video's entry records the content hash of the file it was decoded from, and a cache for a different file is
ignored. Frames are stored whole (rotation applied, no view crop), so one cache serves every view.

JPEG (the default) is lossy: pixels differ slightly from decoding the video. Use `format="png"` for exact pixels, at
several times the size and a slower read.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from vidlabel_spec import VideoMetadata, View

from .frames import FrameReader, _cv2, crop_to_view

_FORMATS = {"jpg": ".jpg", "png": ".png"}
CACHE_FILE = "cache.json"


@dataclass
class CacheStats:
    """What `FrameCache.build` did."""
    written: int      # frames decoded and saved in this call
    skipped: int      # frames that were already cached
    bytes: int        # size of the files written
    seconds: float

    @property
    def mean_kb(self) -> float:
        return self.bytes / 1024 / self.written if self.written else 0.0


class FrameCache:
    """Frames saved as `<root>/<video_id>/<frame>.jpg|png`, plus `<root>/cache.json` saying which video file each came from.

    Open an existing cache with `FrameCache(root)`; make or extend one with `FrameCache.build(samples, root)`.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        p = self.root / CACHE_FILE
        if not p.exists():
            raise FileNotFoundError(f"{self.root} is not a frame cache (no {CACHE_FILE}); build one with `vidlabel cache`")
        info = json.loads(p.read_text(encoding="utf-8"))
        if info.get("format") not in _FORMATS:
            raise ValueError(f"{p}: unknown format {info.get('format')!r}")
        self.format: str = info["format"]
        self.quality: int | None = info.get("quality")
        self.videos: dict[str, dict] = info.get("videos", {})
        self._warned: set[str] = set()

    def _file(self, video_id: str, frame: int) -> Path:
        return self.root / video_id / f"{int(frame):08d}{_FORMATS[self.format]}"

    def usable(self, video_id: str, meta: VideoMetadata) -> bool:
        """Whether this cache has frames for the video *as registered*: same file (hash) and same size."""
        entry = self.videos.get(video_id)
        if entry is None:
            return False
        h = entry.get("content_hash")
        if h and meta.content_hash and h != meta.content_hash:
            if video_id not in self._warned:
                self._warned.add(video_id)
                warnings.warn(f"frame cache for '{video_id}' was made from a different file; ignoring it "
                              f"(rebuild with `vidlabel cache ... --overwrite`)", RuntimeWarning, stacklevel=3)
            return False
        return True

    def has(self, video_id: str, frame: int) -> bool:
        return video_id in self.videos and self._file(video_id, frame).exists()

    def read(self, video_id: str, frame: int) -> np.ndarray:
        """The cached frame as an RGB uint8 array (whole frame, displayed orientation). KeyError if it is not cached."""
        p = self._file(video_id, frame)
        cv2 = _cv2()
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise KeyError(f"frame {frame} of '{video_id}' is not in the cache")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @classmethod
    def build(cls, samples, root: str | Path, *, format: str = "jpg", quality: int = 95,
              overwrite: bool = False) -> tuple["FrameCache", CacheStats]:
        """Decode every frame `samples` (a `FrameSamples`) will use, in order, and save it under `root`.

        Frames already cached are kept unless `overwrite=True`. Returns the cache and what was done. Needs OpenCV."""
        if format not in _FORMATS:
            raise ValueError(f"format must be one of {sorted(_FORMATS)}")
        if not 1 <= quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        cv2 = _cv2()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        cf = root / CACHE_FILE
        info = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else {"format": format, "quality": quality, "videos": {}}
        if info["format"] != format:
            raise ValueError(f"{root} already holds {info['format']} frames; use another folder or the same format")
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = _FORMATS[format]
        t0, written, skipped, size = time.time(), 0, 0, 0
        for vid in samples.video_ids():
            meta = samples.ds.video(vid)
            entry = info["videos"].get(vid)
            if entry and entry.get("content_hash") != meta.content_hash:
                overwrite_video = True            # cached from another file: start this video again
            else:
                overwrite_video = overwrite
            (root / vid).mkdir(exist_ok=True)
            todo = [f for f in samples.frames_of(vid)
                    if overwrite_video or not (root / vid / f"{f:08d}{ext}").exists()]
            skipped += len(samples.frames_of(vid)) - len(todo)
            if todo:
                with FrameReader(samples.video_path(vid), meta, None, index=samples.ds.frame_index(vid)) as reader:
                    for f in todo:
                        rgb = reader.read(f)
                        out = root / vid / f"{f:08d}{ext}"
                        if not cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params):
                            raise OSError(f"could not write {out}")
                        written += 1
                        size += out.stat().st_size
            info["videos"][vid] = {"content_hash": meta.content_hash, "frame_count": meta.frame_count}
            cf.write_text(json.dumps(info, indent=2), encoding="utf-8")   # after each video, so an interrupted build keeps its work
        return cls(root), CacheStats(written=written, skipped=skipped, bytes=size, seconds=time.time() - t0)


class CachedReader:
    """Same interface as `FrameReader` (`read`, `read_many`, `close`), but frames come from a `FrameCache` when it has them
    and from the video file otherwise, so a partial cache is fine. Crops to `view` like `FrameReader` does."""

    def __init__(self, cache: FrameCache, video_id: str, meta: VideoMetadata, view: View | None, fallback):
        self.cache, self.video_id, self.meta, self.view = cache, video_id, meta, view
        self._fallback = fallback        # a zero-argument function that makes the FrameReader to use for misses
        self._reader: FrameReader | None = None

    def read(self, index: int):
        index = int(index)
        if self.cache.has(self.video_id, index):
            img = self.cache.read(self.video_id, index)
            if (img.shape[1], img.shape[0]) != self.meta.display_size:
                raise ValueError(f"cached frame {index} of '{self.video_id}' is {img.shape[1]}x{img.shape[0]} but the "
                                 f"video is {self.meta.display_size[0]}x{self.meta.display_size[1]}: rebuild the cache")
            return crop_to_view(img, self.view)
        if self._reader is None:
            self._reader = self._fallback()
        return self._reader.read(index)

    def read_many(self, indices: Iterable[int]) -> list:
        """Frames for `indices` in the order given; frames missing from the cache are decoded together, in increasing order."""
        indices = [int(i) for i in indices]
        out: list = [None] * len(indices)
        misses = [k for k, i in enumerate(indices) if not self.cache.has(self.video_id, i)]
        missing = set(misses)
        for k, i in enumerate(indices):
            if k not in missing:
                out[k] = self.read(i)
        if misses:
            if self._reader is None:
                self._reader = self._fallback()
            for k, frame in zip(misses, self._reader.read_many([indices[k] for k in misses])):
                out[k] = frame
        return out

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
