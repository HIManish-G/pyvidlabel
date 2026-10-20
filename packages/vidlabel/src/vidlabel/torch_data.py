"""PyTorch datasets over a vidlabel dataset. Needs: pip install "vidlabel[torch]".

Two ways to read, because decoding video is the cost that matters:

  VideoStream  (recommended for training) iterates each video once, front to back, so every frame is decoded at most
               once per epoch. Whole videos are shared out between DataLoader workers (and distributed ranks), so use
               at least as many videos as workers. Randomness comes from shuffling the video order and a shuffle buffer.
  VideoFrames  map-style: `ds[i]` returns any sample, so samplers and random access work. Going backwards in a video
               re-decodes from its start, so this is for validation, small or short-clip datasets, and debugging.

Both yield dicts of tensors (see `vidlabel.samples`): `image` is uint8 CHW. Use `collate_frames` with a DataLoader.
"""
from __future__ import annotations

import random
from typing import Callable, Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    from torch.utils.data import IterableDataset as _IterableDataset
    from torch.utils.data import get_worker_info
except ImportError as e:  # pragma: no cover
    raise ImportError('PyTorch is needed for this module. Install it with: pip install "vidlabel[torch]" '
                      "(or follow https://pytorch.org/get-started for your GPU)") from e

from .samples import FrameSamples


def to_tensors(sample: dict) -> dict:
    """NumPy sample -> PyTorch sample: image becomes uint8 CHW; arrays become tensors; strings and ints stay as they are."""
    out = {}
    for k, v in sample.items():
        if k == "image":
            out[k] = torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1)))
        elif isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        else:
            out[k] = v
    return out


def collate_frames(batch: list[dict]) -> dict:
    """Batch samples: images are stacked when they share a size, everything else (boxes, masks...) stays a list per sample."""
    out = {k: [s[k] for s in batch] for k in batch[0]}
    shapes = {tuple(i.shape) for i in out["image"]}
    if len(shapes) == 1:
        out["image"] = torch.stack(out["image"])
        out["frame_labels"] = torch.stack(out["frame_labels"])
    return out


class VideoFrames(_Dataset):
    """Map-style dataset: one item per sampled frame. See the module docstring for the cost of random access."""

    def __init__(self, samples: FrameSamples, transform: Callable[[dict], dict] | None = None):
        self.samples, self.transform = samples, transform
        self._readers: dict[str, object] = {}
        self._pid = None

    def __len__(self) -> int:
        return len(self.samples)

    def _reader(self, video_id: str):
        import os
        if self._pid != os.getpid():  # a DataLoader worker was forked: open its own decoders, never share the parent's
            self._readers, self._pid = {}, os.getpid()
        if video_id not in self._readers:
            if len(self._readers) >= 4:  # keep a few videos open, not all of them
                self._readers.pop(next(iter(self._readers))).close()
            self._readers[video_id] = self.samples.reader(video_id)
        return self._readers[video_id]

    def __getitem__(self, i: int) -> dict:
        video_id, frame = self.samples.index[i]
        sample = to_tensors(self.samples.sample(video_id, frame, self._reader(video_id).read(frame)))
        return self.transform(sample) if self.transform else sample


class VideoStream(_IterableDataset):
    """Streaming dataset: decodes each video once, in order. Set `shuffle_videos` and `shuffle_buffer` for randomness,
    and call `set_epoch(n)` each epoch so the order changes (it is deterministic for a given seed and epoch)."""

    def __init__(self, samples: FrameSamples, transform: Callable[[dict], dict] | None = None, *,
                 shuffle_videos: bool = True, shuffle_buffer: int = 0, seed: int = 0,
                 rank: int | None = None, world_size: int | None = None):
        self.samples, self.transform = samples, transform
        self.shuffle_videos, self.shuffle_buffer, self.seed = shuffle_videos, shuffle_buffer, seed
        self.rank, self.world_size, self.epoch = rank, world_size, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:  # total over all workers; each worker sees its share
        return len(self.samples)

    def _shard(self) -> tuple[int, int]:
        rank, world = self.rank, self.world_size
        if rank is None or world is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
            else:
                rank, world = 0, 1
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        return rank * nw + wid, world * nw

    def __iter__(self) -> Iterator[dict]:
        shard, total = self._shard()
        videos = self.samples.video_ids()
        if self.shuffle_videos:
            random.Random(self.seed + self.epoch).shuffle(videos)  # same order in every worker, so shards line up
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 9_973 + shard)
        buf: list[dict] = []

        def emit(s):
            return self.transform(s) if self.transform else s

        for vid in videos[shard::total]:
            with self.samples.reader(vid) as reader:
                for f in self.samples.frames_of(vid):
                    s = to_tensors(self.samples.sample(vid, f, reader.read(f)))
                    if self.shuffle_buffer <= 0:
                        yield emit(s)
                        continue
                    buf.append(s)
                    if len(buf) >= self.shuffle_buffer:
                        j = rng.randrange(len(buf))
                        buf[j], buf[-1] = buf[-1], buf[j]
                        yield emit(buf.pop())
        rng.shuffle(buf)
        for s in buf:
            yield emit(s)
