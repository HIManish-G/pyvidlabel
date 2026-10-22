"""Plan how to read a list of frames from video cheaply. No video access and no deep-learning framework needed.

Reading frame N of a video costs a decode from the keyframe before it, so reading a list of frames one by one in random
order pays that cost for every frame (see docs/training.md). A `ReadPlan` takes *any* list of `(video, frame)` requests,
however you chose them (uniform, weighted, hand-picked, repeated), and arranges the reading:

- the frames of one video are grouped by keyframe interval into **chunks**; inside a chunk they are read in increasing
  order, so every interval is decoded once, however the requests were ordered;
- a frame requested several times is decoded once;
- the chunks, not the individual frames, are what gets shuffled each epoch and shared out between workers, so the order
  still changes from epoch to epoch and the work is spread evenly.

The plan never decides *which* frames to read or how likely each is: that is yours. Run a plan with
`FrameSamples.iter_plan` (NumPy) or `vidlabel.torch_data.PlannedStream` (PyTorch).
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from itertools import groupby
from typing import Iterable, Mapping, Sequence

from vidlabel_spec import FrameIndex

FALLBACK_BLOCK = 250   # frames per chunk for a video that has no frame index (a typical keyframe interval)


@dataclass(frozen=True)
class Chunk:
    """The requested frames of one video inside one keyframe interval."""
    video_id: str
    start: int                    # the keyframe the interval begins at (or the block start, without an index)
    frames: tuple[int, ...]       # distinct frames, increasing
    repeats: tuple[int, ...]      # how many samples were requested for each of them (at least 1)
    cost: int                     # estimated frames decoded to read them all: reach the first from the keyframe, read on to the last

    @property
    def n_samples(self) -> int:
        return sum(self.repeats)


class ReadPlan:
    """A reading plan for a list of requests. Build it with `FrameSamples.plan_reads(...)`, or directly from `(video_id, frame)` pairs.

    - `requests`: `(video_id, frame)` pairs, in any order, repeats allowed (each repeat gives one more sample).
    - `indexes`: video id -> its `FrameIndex` (or None). Videos without one are cut into blocks of `block` frames,
      which is only an approximation of the real keyframes, so costs for them are estimates.
    - `frame_counts`: video id -> number of frames; if given, requests outside the video raise ValueError.
    - `seed`: makes the chunk order and its sharding reproducible; together with the epoch it fixes the order of every epoch.
    """

    def __init__(self, requests: Iterable[tuple[str, int]], *, indexes: Mapping[str, FrameIndex | None] | None = None,
                 frame_counts: Mapping[str, int] | None = None, block: int = FALLBACK_BLOCK, seed: int = 0):
        if block < 1:
            raise ValueError("block must be >= 1")
        self.seed, self.block = int(seed), int(block)
        indexes, frame_counts = indexes or {}, frame_counts or {}
        wanted: dict[str, dict[int, int]] = {}
        self._one_by_one = 0
        n = 0
        for video_id, frame in requests:
            video_id, frame = str(video_id), int(frame)
            limit = frame_counts.get(video_id, indexes[video_id].frame_count if indexes.get(video_id) is not None else None)
            if frame < 0 or (limit is not None and frame >= limit):
                raise ValueError(f"frame {frame} is outside video '{video_id}'" + (f" ({limit} frames)" if limit is not None else ""))
            per = wanted.setdefault(video_id, {})
            per[frame] = per.get(frame, 0) + 1
            idx = indexes.get(video_id)
            self._one_by_one += idx.decode_cost(frame) if idx is not None else frame % self.block + 1
            n += 1
        self._n = n
        chunks: list[Chunk] = []
        for video_id in sorted(wanted):
            idx = indexes.get(video_id)
            frames = sorted(wanted[video_id])
            start_of = idx.keyframe_before if idx is not None else (lambda f: f - f % self.block)
            for start, group in groupby(frames, key=start_of):
                fs = tuple(group)
                head = idx.decode_cost(fs[0]) if idx is not None else fs[0] - start + 1
                chunks.append(Chunk(video_id, start, fs, tuple(wanted[video_id][f] for f in fs), head + (fs[-1] - fs[0])))
        self.chunks: list[Chunk] = chunks
        self._shards: dict[tuple[int, int], list[list[Chunk]]] = {}

    # ---- what is in the plan ----
    def __len__(self) -> int:
        """Number of samples the plan produces in one epoch (repeats counted)."""
        return self._n

    @property
    def video_ids(self) -> list[str]:
        return sorted({c.video_id for c in self.chunks})

    def summary(self) -> dict[str, int]:
        """`samples`, `distinct_frames`, `videos`, `chunks`, and the estimated frames decoded per epoch under this plan
        (`est_decoded_frames`) against reading the same requests one at a time (`est_decoded_frames_one_by_one`)."""
        return {"samples": self._n, "distinct_frames": sum(len(c.frames) for c in self.chunks), "videos": len(self.video_ids),
                "chunks": len(self.chunks), "est_decoded_frames": sum(c.cost for c in self.chunks),
                "est_decoded_frames_one_by_one": self._one_by_one}

    # ---- one epoch ----
    def order(self, epoch: int = 0) -> list[Chunk]:
        """All chunks in this epoch's order: the same for a given seed and epoch, different between epochs."""
        out = list(self.chunks)
        random.Random(self.seed * 1_000_003 + int(epoch) * 9_973 + 1).shuffle(out)
        return out

    def _split(self, epoch: int, num_shards: int) -> list[list[Chunk]]:
        key = (int(epoch), int(num_shards))
        if key not in self._shards:
            if len(self._shards) > 8:
                self._shards.clear()
            heap = [(0, s) for s in range(num_shards)]
            out: list[list[Chunk]] = [[] for _ in range(num_shards)]
            for chunk in self.order(epoch):            # each chunk goes to the shard with the fewest samples so far
                total, s = heapq.heappop(heap)
                out[s].append(chunk)
                heapq.heappush(heap, (total + chunk.n_samples, s))
            self._shards[key] = out
        return self._shards[key]

    def shard(self, epoch: int = 0, shard: int = 0, num_shards: int = 1) -> list[Chunk]:
        """The chunks one worker (or rank) reads in this epoch, in reading order. Every chunk belongs to exactly one shard,
        whatever `num_shards` is, and shards get nearly the same number of samples (they differ by at most one chunk's worth)."""
        if num_shards < 1 or not 0 <= shard < num_shards:
            raise ValueError("need 0 <= shard < num_shards")
        return self._split(epoch, num_shards)[shard]

    def shard_sizes(self, epoch: int = 0, num_shards: int = 1) -> list[int]:
        """Samples per shard, to see how even the split is (distributed training wants equal numbers)."""
        return [sum(c.n_samples for c in part) for part in self._split(epoch, num_shards)]
