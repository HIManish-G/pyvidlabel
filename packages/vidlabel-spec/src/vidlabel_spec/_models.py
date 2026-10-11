"""Metadata models. Core fields are strict (unknown fields are rejected); anything custom goes in `extra`."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPEC_VERSION = "0.1.0"

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
    rotation: int = 0
    codec: str | None = None
    pix_fmt: str | None = None
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
    """A named crop and/or time range of a video. No new video is made; annotations stay in source coordinates."""
    view_id: str
    video_id: str
    name: str | None = None
    region: Region | None = None
    frame_range: tuple[int, int] | None = Field(default=None, description="Inclusive [first, last] frame.")
    extra: dict[str, Any] = Field(default_factory=dict)

    _v_id = field_validator("view_id", "video_id")(_check_id)

    @model_validator(mode="after")
    def _v_range(self) -> "View":
        if self.frame_range is not None and self.frame_range[0] > self.frame_range[1]:
            raise ValueError("frame_range must be [first, last] with first <= last")
        return self


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
