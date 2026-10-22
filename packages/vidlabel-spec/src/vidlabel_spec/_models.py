"""Metadata models. Core fields are strict (unknown fields are rejected); anything custom goes in `extra`."""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPEC_VERSION = "0.5.0"

_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_HASH_RE = re.compile(r"^[a-z0-9]+:[0-9a-f]+$")


class _Strict(BaseModel):
    # Same rule as GEFF: custom data must live inside `extra`, so the core stays checkable.
    model_config = ConfigDict(extra="forbid")


def _check_id(v: str) -> str:
    if not _ID_RE.match(v):
        raise ValueError("ids may only contain letters, digits, '.', '_' and '-'")
    return v


class Rational(_Strict):
    num: int
    den: int = Field(gt=0)

    @property
    def value(self) -> float:
        return self.num / self.den


class VideoSource(_Strict):
    """One known location of the video file. Several may be listed; paths move, hashes don't."""
    uri: str
    note: str | None = None


class FrameIndex(_Strict):
    """Optional per-frame timing of one video, stored in videos/<video_id>/frame_index.json.

    Read from the file's packet table (no decoding), so it is cheap to make. It lets a reader seek by timestamp, including
    in variable-frame-rate video, check where a seek really landed, and plan reads so each keyframe interval is decoded once.

    `pts` holds one presentation timestamp per frame, in `timebase` ticks, in display order (so `pts[i]` belongs to frame
    `i`) and strictly increasing. `keyframes` lists the frame numbers where decoding can start without earlier frames.
    """
    content_hash: str | None = Field(default=None, description="Hash of the file this was read from; a different hash means the index is stale.")
    timebase: Rational
    pts: list[int] = Field(min_length=1)
    keyframes: list[int] = Field(default_factory=list)

    @field_validator("content_hash")
    @classmethod
    def _v_hash(cls, v: str | None) -> str | None:
        if v is not None and not _HASH_RE.match(v):
            raise ValueError("content_hash must look like '<algo>:<lowercase hex>'")
        return v

    @model_validator(mode="after")
    def _v_order(self) -> "FrameIndex":
        p, k = self.pts, self.keyframes
        if any(b <= a for a, b in zip(p, p[1:])):
            raise ValueError("pts must be strictly increasing (one timestamp per frame, in display order)")
        if any(b <= a for a, b in zip(k, k[1:])):
            raise ValueError("keyframes must be strictly increasing frame numbers")
        if k and (k[0] < 0 or k[-1] >= len(p)):
            raise ValueError("keyframes must be frame numbers inside the video")
        return self

    @property
    def frame_count(self) -> int:
        return len(self.pts)

    def seconds(self, frame: int) -> float:
        """Time of `frame` in seconds from the first frame."""
        return (self.pts[frame] - self.pts[0]) * self.timebase.num / self.timebase.den

    def frame_at(self, ticks: float, tolerance: float = 0.3) -> int | None:
        """The frame whose timestamp is within `tolerance` ticks of `ticks` (absolute, like `pts`), or None."""
        j = bisect_left(self.pts, ticks)
        best = min((c for c in (j - 1, j) if 0 <= c < len(self.pts)), key=lambda c: abs(self.pts[c] - ticks), default=None)
        return best if best is not None and abs(self.pts[best] - ticks) <= tolerance else None

    def keyframe_before(self, frame: int) -> int:
        """The last keyframe at or before `frame` (0 if the list is empty): where decoding for `frame` has to start."""
        j = bisect_right(self.keyframes, frame)
        return self.keyframes[j - 1] if j else 0

    def decode_cost(self, frame: int) -> int:
        """Frames that must be decoded to reach `frame` from a cold start: its distance from the previous keyframe, plus one."""
        return frame - self.keyframe_before(frame) + 1

    def interval_stats(self) -> dict[str, float]:
        """Keyframe spacing: `min`, `median`, `max` interval in frames, and `mean_cost`, the average frames decoded to reach a random frame."""
        k = self.keyframes or [0]
        gaps = sorted([b - a for a, b in zip(k, k[1:])] + [len(self.pts) - k[-1]])
        n = len(self.pts)
        # frames in an interval of length g cost 1..g, so the mean cost is the sum of g*(g+1)/2 over intervals, divided by n
        total = sum(g * (g + 1) // 2 for g in gaps)
        return {"min": gaps[0], "median": gaps[len(gaps) // 2], "max": gaps[-1], "mean_cost": total / n}


class VideoMetadata(_Strict):
    """Generic facts about one video. Lives in videos/<video_id>/metadata.json."""
    video_id: str
    content_hash: str | None = Field(default=None, description="'<algo>:<hex>', e.g. 'sha256:ab12...'. Lets anyone detect a re-encoded or wrong file.")
    duration_s: float | None = Field(default=None, ge=0)
    frame_count: int = Field(ge=0)
    fps: Rational
    timebase: Rational | None = None
    is_vfr: bool = False
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    rotation: int = Field(default=0, description="Clockwise degrees (0, 90, 180, 270) applied to the stored frame to display it upright. 90 and 270 swap width and height.")
    codec: str | None = None
    pix_fmt: str | None = None
    bit_rate: int | None = Field(default=None, ge=0, description="Bits per second as reported by the container/stream.")
    sources: list[VideoSource] = Field(default_factory=list)
    ingested_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    _v_id = field_validator("video_id")(_check_id)

    @property
    def display_size(self) -> tuple[int, int]:
        """(width, height) as displayed, i.e. after rotation. All annotation coordinates use this orientation."""
        return (self.height, self.width) if self.rotation in (90, 270) else (self.width, self.height)

    @field_validator("content_hash")
    @classmethod
    def _v_hash(cls, v: str | None) -> str | None:
        if v is not None and not _HASH_RE.match(v):
            raise ValueError("content_hash must look like '<algo>:<lowercase hex>'")
        return v

    @field_validator("rotation")
    @classmethod
    def _v_rot(cls, v: int) -> int:
        if v not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180 or 270")
        return v


class Region(_Strict):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class View(_Strict):
    """A named crop and/or time range of a video. No new video is made; annotations stay in source coordinates.

    `region` is in the video's displayed orientation. If the crop is resized before it reaches a pipeline, give
    `output_size` (width, height) so coordinates can be converted back. `allowed_labels` optionally limits which
    labels may be used in this view (e.g. a caption strip that only allows caption labels).
    """
    view_id: str
    video_id: str
    name: str | None = None
    region: Region | None = None
    output_size: tuple[int, int] | None = Field(
        default=None, description="(width, height) the region is resized to when shown to a pipeline. Omit if not resized.")
    frame_range: tuple[int, int] | None = Field(default=None, description="Inclusive [first, last] frame.")
    allowed_labels: list[str] | None = Field(default=None, description="If set, annotations in this view may only use these labels.")
    extra: dict[str, Any] = Field(default_factory=dict)

    _v_id = field_validator("view_id", "video_id")(_check_id)

    @model_validator(mode="after")
    def _v_checks(self) -> "View":
        if self.frame_range is not None:
            if self.frame_range[0] < 0 or self.frame_range[0] > self.frame_range[1]:
                raise ValueError("frame_range must be [first, last] with 0 <= first <= last")
        if self.output_size is not None:
            if self.region is None:
                raise ValueError("output_size needs a region")
            if self.output_size[0] <= 0 or self.output_size[1] <= 0:
                raise ValueError("output_size must be positive")
        return self

    @property
    def scale(self) -> tuple[float, float]:
        """(x, y) factor from source pixels to view pixels (1, 1 if the crop is not resized)."""
        if self.region is not None and self.output_size is not None:
            return (self.output_size[0] / self.region.width, self.output_size[1] / self.region.height)
        return (1.0, 1.0)

    def to_source_box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Box in this view's pixels -> box in full source-video pixels."""
        x0, y0, x1, y1 = (float(v) for v in box)
        if self.region is None:
            return (x0, y0, x1, y1)
        sx, sy = self.scale
        r = self.region
        return (r.x + x0 / sx, r.y + y0 / sy, r.x + x1 / sx, r.y + y1 / sy)

    def to_view_box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Box in full source-video pixels -> box in this view's pixels."""
        x0, y0, x1, y1 = (float(v) for v in box)
        if self.region is None:
            return (x0, y0, x1, y1)
        sx, sy = self.scale
        r = self.region
        return ((x0 - r.x) * sx, (y0 - r.y) * sy, (x1 - r.x) * sx, (y1 - r.y) * sy)


class Label(_Strict):
    name: str
    parent: str | None = None
    description: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Vocab(_Strict):
    """The one shared list of labels."""
    version: str = "0"
    labels: list[Label] = Field(default_factory=list)

    @model_validator(mode="after")
    def _v_labels(self) -> "Vocab":
        names = [l.name for l in self.labels]
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        missing = {l.parent for l in self.labels if l.parent is not None} - set(names)
        if missing:
            raise ValueError(f"unknown parent label(s): {sorted(missing)}")
        return self


class Manifest(_Strict):
    """Dataset-level index. Lives in manifest.json."""
    spec_version: str = SPEC_VERSION
    videos: list[str] = Field(default_factory=list)
    views: list[str] = Field(default_factory=list)
    vocab_version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Release(_Strict):
    """Describes one compiled (training) build. release_id is derived from content, so a training run can pin it."""
    release_id: str
    spec_version: str
    created_at: datetime
    n_rows: int
    n_videos: int
    row_group_size: int
    files: dict[str, str] = Field(description="file name -> 'sha256:<hex>'")
    extra: dict[str, Any] = Field(default_factory=dict, description="Build filters etc.")
