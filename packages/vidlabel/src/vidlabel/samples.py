"""Turn a dataset (ideally a pinned compiled release) into training samples: a frame of a video plus the boxes,
masks, labels and frame-level markers that apply to it. This module needs no deep-learning framework; `torch_data`
wraps it for PyTorch, and the same class works for JAX, TensorFlow or plain NumPy.

A sample is a dict of NumPy arrays:
  image        (H, W, 3) uint8 RGB, displayed orientation, cropped/resized to the view if one is given
  boxes        (N, 4) float32 in `box_format` (pixels of `image`, or fractions with normalized=True)
  labels       (N,) int64 class index (position in `class_names`, plus `label_offset`)
  track_ids    (N,) int64, -1 when the annotation has no track
  annotation_ids (N,) int64
  masks        (N, H, W) uint8, only with with_masks=True
  frame_labels (C,) float32 multi-hot of the time markers active on this frame
  video_id, view_id, frame_index
"""
from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from vidlabel_spec import View

from . import masks as masklib
from .boxes import convert_boxes
from .build import Compiled
from .dataset import Dataset
from .cache import CachedReader, FrameCache
from .frames import FrameReader, crop_mask_to_view
from .readplan import FALLBACK_BLOCK, ReadPlan

_BASE_COLUMNS = ["annotation_id", "video_id", "start_frame", "end_frame", "kind", "label", "track_id",
                 "x0", "y0", "x1", "y1", "view_id", "origin", "confidence"]


class FrameSamples:
    """Which frames to train on, and what the targets for each are. Build it once, then read samples by (video, frame).

    Parameters (all keyword-only except `ds`):

    - `release`: `"latest"`, a release id such as `"r3fa91c0b2d44"` (pinned: the same rows and class indices every time), or
      `None` to read the live annotations (unreviewed and approved), which can change between runs.
    - `view_id`: train on that view's crop with only that view's annotations. `None`: full frames, all annotations.
    - `videos`: restrict to these video ids (default: all).
    - `kinds`: which kinds define the samples and targets, from `"box"`, `"mask"`, `"marker"`. Markers always fill
      `frame_labels`; add `"marker"` to also sample frames that only have a marker.
    - `labels`: keep only these labels; class indices then follow this order (default: the vocabulary's order).
    - `origins`: keep only rows with these origins, e.g. `["human"]`.
    - `min_confidence`: drop rows whose confidence is below this (rows with no confidence are kept).
    - `frames`: `"annotated"` (frames with a selected annotation) or `"all"` (every frame).
    - `stride`: keep every Nth frame of that list.
    - `with_masks`: decode masks into the sample (`masks`, aligned with `boxes`).
    - `box_format`: `"xyxy"`, `"xywh"` or `"cxcywh"`; `normalized=True` gives fractions of the image size.
    - `label_offset`: added to class indices (e.g. `1` for detectors that reserve 0 for background).
    - `roots`: folders to search for videos whose recorded path is stale.
    - `cache`: a `FrameCache` (or the folder of one made by `vidlabel cache`). Frames it holds are read as image files
      instead of decoded from video; anything missing falls back to the video. See docs/training.md.

    See docs/training.md for what a sample contains.
    """

    def __init__(self, ds: Dataset, *, release: str | None = "latest", view_id: str | None = None,
                 videos: Sequence[str] | None = None, kinds: Sequence[str] = ("box", "mask"),
                 labels: Sequence[str] | None = None, origins: Sequence[str] | None = None,
                 min_confidence: float | None = None, frames: str = "annotated", stride: int = 1,
                 with_masks: bool = False, box_format: str = "xyxy", normalized: bool = False, label_offset: int = 0,
                 roots: Sequence[str | Path] = (), cache: FrameCache | str | Path | None = None):
        if frames not in ("annotated", "all"):
            raise ValueError("frames must be 'annotated' or 'all'")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        bad = set(kinds) - {"box", "mask", "marker"}
        if bad:
            raise ValueError(f"unknown kinds {sorted(bad)}")
        self.ds, self.view_id, self.kinds = ds, view_id, tuple(kinds)
        self.with_masks, self.box_format, self.normalized, self.label_offset = with_masks, box_format, normalized, label_offset
        self.roots = list(roots)
        self.cache = FrameCache(cache) if isinstance(cache, (str, Path)) else cache
        self.view: View | None = ds.view(view_id) if view_id else None
        table, self.release_id, pinned_vocab = self._load(release, with_masks)
        # a pinned release carries the vocabulary it was built with, so class indices cannot drift if labels are added later
        vocab = [l.name for l in (pinned_vocab or ds.vocab).labels]
        if labels is not None:
            unknown = [l for l in labels if l not in vocab]
            if unknown:
                raise ValueError(f"labels not in the vocabulary: {unknown}")
        self.class_names: list[str] = list(labels) if labels is not None else vocab
        self._class_of = {n: i + label_offset for i, n in enumerate(self.class_names)}
        convert_boxes(np.zeros((0, 4)), "xyxy", box_format)  # validates box_format early

        keep = pc.is_in(table.column("label"), value_set=pa.array(self.class_names, pa.string()))
        if view_id:
            keep = pc.and_(keep, pc.fill_null(pc.equal(table.column("view_id"), view_id), False))
        if videos is not None:
            keep = pc.and_(keep, pc.is_in(table.column("video_id"), value_set=pa.array(list(videos), pa.string())))
        if origins is not None:
            keep = pc.and_(keep, pc.is_in(table.column("origin"), value_set=pa.array(list(origins), pa.string())))
        if min_confidence is not None:
            keep = pc.and_(keep, pc.fill_null(pc.greater_equal(table.column("confidence"), min_confidence), True))
        self._rows = table.filter(keep).to_pylist()

        self._points: dict[tuple[str, int], list[int]] = {}   # (video, frame) -> row indices, single-frame rows
        self._spans: dict[str, list[tuple[int, int, int]]] = {}  # video -> (start, end, row index) for multi-frame rows
        for i, r in enumerate(self._rows):
            if r["start_frame"] == r["end_frame"]:
                self._points.setdefault((r["video_id"], r["start_frame"]), []).append(i)
            else:
                self._spans.setdefault(r["video_id"], []).append((r["start_frame"], r["end_frame"], i))

        video_ids = list(videos) if videos is not None else ds.video_ids()
        self._videos = [v for v in video_ids if v in set(ds.video_ids())]
        self.index: list[tuple[str, int]] = []
        self._by_video: dict[str, list[int]] = {}
        for vid in self._videos:
            fr = self._frames_for(vid, frames)[::stride]
            if fr:
                self._by_video[vid] = fr
                self.index.extend((vid, f) for f in fr)

    # ---- loading ----
    def _load(self, release, with_masks):
        cols = list(_BASE_COLUMNS) + (["mask"] if with_masks else [])
        if release is None:
            t = self.ds.annotations(status=["unreviewed", "approved"], columns=cols)
            return t, None, None
        root = self.ds.path / "compiled"
        if not root.exists() or (release == "latest" and not (root / "LATEST").exists()):
            raise FileNotFoundError(f"no compiled release in {self.ds.path}. Run `vidlabel build {self.ds.path}` first "
                                    f"(or pass release=None to read the live annotations)")
        path = root if release == "latest" else root / release
        if not path.exists():
            raise FileNotFoundError(f"release '{release}' not found in {root}")
        c = Compiled(path)
        return c.read(columns=cols), c.release_id, c.vocab

    def _frames_for(self, vid: str, mode: str) -> list[int]:
        n = self.ds.video(vid).frame_count
        lo, hi = 0, n - 1
        if self.view is not None and self.view.frame_range is not None:
            lo, hi = max(lo, self.view.frame_range[0]), min(hi, self.view.frame_range[1])
        if mode == "all":
            return list(range(lo, hi + 1))
        wanted = set(self.kinds)
        out: set[int] = set()
        for (v, f), idx in self._points.items():
            if v == vid and any(self._rows[i]["kind"] in wanted for i in idx):
                out.add(f)
        for a, b, i in self._spans.get(vid, []):
            if self._rows[i]["kind"] in wanted:
                out.update(range(max(a, lo), min(b, hi) + 1))
        return sorted(f for f in out if lo <= f <= hi)

    # ---- access ----
    def __len__(self) -> int:
        return len(self.index)

    def video_ids(self) -> list[str]:
        """Videos that contribute at least one sample."""
        return list(self._by_video)

    def frames_of(self, video_id: str) -> list[int]:
        """The frame numbers sampled from one video, in order."""
        return list(self._by_video.get(video_id, []))

    def video_path(self, video_id: str) -> Path:
        """Where the video file is on this machine (recorded paths first, then `roots`). Raises FileNotFoundError."""
        p = self.ds.resolve_video_path(video_id, self.roots)
        if p is None:
            raise FileNotFoundError(f"cannot find the video for '{video_id}'; pass roots=[...] to FrameSamples")
        return Path(p)

    def reader(self, video_id: str):
        """A reader for one video (`read(frame)`, `read_many(frames)`, `close()`), already set up with this object's view.
        It uses the video's frame index when it has one, and the frame cache if there is one that matches the video.
        Close it when done (or use `with`)."""
        meta = self.ds.video(video_id)

        def from_video() -> FrameReader:
            return FrameReader(self.video_path(video_id), meta, self.view, index=self.ds.frame_index(video_id))

        if self.cache is not None and self.cache.usable(video_id, meta):
            return CachedReader(self.cache, video_id, meta, self.view, from_video)
        return from_video()

    def plan_reads(self, requests: Iterable, *, seed: int = 0, block: int = FALLBACK_BLOCK) -> ReadPlan:
        """Arrange the reading of a list of frames you chose, so each keyframe interval is decoded once (see `ReadPlan`).

        `requests` holds `(video_id, frame)` pairs or integer positions in `self.index` (what `numpy.random.Generator.choice(len(samples), ...)`
        returns), in any order, with repeats allowed. Which frames to request, and how likely each is, is entirely up to you.
        Videos without a frame index are cut into blocks of `block` frames. Run the plan with `iter_plan` or `PlannedStream`."""
        pairs = [self.index[int(r)] if isinstance(r, (int, np.integer)) else (str(r[0]), int(r[1])) for r in requests]
        videos = {v for v, _ in pairs}
        return ReadPlan(pairs, indexes={v: self.ds.frame_index(v) for v in videos},
                        frame_counts={v: self.ds.video(v).frame_count for v in videos}, seed=seed, block=block)

    def iter_plan(self, plan: ReadPlan, *, epoch: int = 0, shard: int = 0, num_shards: int = 1, shuffle_buffer: int = 0,
                  max_open: int = 4) -> Iterator[dict]:
        """Yield the samples of one shard of one epoch of `plan` (NumPy samples, like `sample`), reading each keyframe interval once.

        Chunks come in the plan's order for this `epoch`; frames inside a chunk are read in increasing order. A frame requested
        several times is decoded once and yields one sample per request. `shuffle_buffer` (samples held in memory, each a whole
        decoded frame) mixes neighbouring frames; 0 yields them as read. `max_open` videos are kept open at a time. For several
        workers or ranks, give each its own `shard` out of `num_shards`: together they yield every sample exactly once."""
        rng = random.Random(plan.seed * 1_000_003 + epoch * 9_973 + shard * 31 + 7)
        readers: OrderedDict[str, object] = OrderedDict()
        buf: list[dict] = []
        try:
            for chunk in plan.shard(epoch, shard, num_shards):
                reader = readers.pop(chunk.video_id, None) or self.reader(chunk.video_id)
                readers[chunk.video_id] = reader                      # most recently used last
                while len(readers) > max_open:
                    readers.popitem(last=False)[1].close()
                for frame, repeats in zip(chunk.frames, chunk.repeats):
                    image = reader.read(frame)
                    for k in range(repeats):
                        sample = self.sample(chunk.video_id, frame, image if k == repeats - 1 else image.copy())
                        if shuffle_buffer <= 0:
                            yield sample
                            continue
                        buf.append(sample)
                        if len(buf) >= shuffle_buffer:
                            j = rng.randrange(len(buf))
                            buf[j], buf[-1] = buf[-1], buf[j]
                            yield buf.pop()
            rng.shuffle(buf)
            yield from buf
        finally:
            for r in readers.values():
                r.close()

    def _active(self, video_id: str, frame: int) -> list[dict]:
        rows = [self._rows[i] for i in self._points.get((video_id, frame), [])]
        rows += [self._rows[i] for a, b, i in self._spans.get(video_id, []) if a <= frame <= b]
        return rows

    def targets(self, video_id: str, frame: int, image_size: tuple[int, int]) -> dict:
        """Boxes, labels, masks and frame labels for one frame. `image_size` is (width, height) of the image returned."""
        w, h = image_size
        frame_labels = np.zeros(len(self.class_names), np.float32)
        boxes, labels, tracks, ids, mks = [], [], [], [], []
        for r in self._active(video_id, frame):
            if r["kind"] == "marker":
                frame_labels[self._class_of[r["label"]] - self.label_offset] = 1.0
                continue
            if r["kind"] not in self.kinds or None in (r["x0"], r["y0"], r["x1"], r["y1"]):
                continue
            box = (r["x0"], r["y0"], r["x1"], r["y1"])
            boxes.append(self.view.to_view_box(box) if self.view else box)
            labels.append(self._class_of[r["label"]])
            tracks.append(-1 if r["track_id"] is None else r["track_id"])
            ids.append(r["annotation_id"])
            if self.with_masks:
                m = np.zeros((h, w), np.uint8)
                if r.get("mask") is not None:
                    full = crop_mask_to_view(masklib.decode(r["mask"]), self.view)
                    if full.shape[:2] == (h, w):
                        m = full.astype(np.uint8)
                mks.append(m)
        out = {
            "boxes": convert_boxes(np.array(boxes).reshape(-1, 4), "xyxy", self.box_format, size=(w, h),
                                   dst_normalized=self.normalized),
            "labels": np.array(labels, np.int64), "track_ids": np.array(tracks, np.int64),
            "annotation_ids": np.array(ids, np.int64), "frame_labels": frame_labels,
        }
        if self.with_masks:
            out["masks"] = np.stack(mks) if mks else np.zeros((0, h, w), np.uint8)
        return out

    def sample(self, video_id: str, frame: int, image: np.ndarray) -> dict:
        """The full sample for a frame whose pixels you already have."""
        t = self.targets(video_id, frame, (image.shape[1], image.shape[0]))
        return {"image": image, **t, "video_id": video_id, "view_id": self.view_id, "frame_index": int(frame)}

    def describe(self) -> dict:
        """What to log next to a training run so it can be reproduced."""
        return {"release_id": self.release_id, "view_id": self.view_id, "kinds": list(self.kinds),
                "class_names": self.class_names, "label_offset": self.label_offset, "box_format": self.box_format,
                "normalized": self.normalized, "n_samples": len(self), "n_videos": len(self._by_video)}
