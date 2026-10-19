"""The annotation table: one row per annotation, the same columns for every kind of annotation.

Time markers, boxes and masks all live in this single table. Who or what made an
annotation is a column (`origin`, `tool`, ...), never a separate file or folder.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict


class Kind(str, Enum):
    MARKER = "marker"  # a point or span in time, no geometry
    BOX = "box"        # a bounding box on one frame
    MASK = "mask"      # a pixel mask on one frame (also carries its bounding box)


class Origin(str, Enum):
    HUMAN = "human"
    MODEL = "model"
    IMPORTED = "imported"  # converted from another tool or dataset; name it in `tool`
    OTHER = "other"


_CUSTOM_ORIGIN_RE = re.compile(r"^x-[a-z0-9][a-z0-9._-]{0,62}$")


def is_valid_origin(value) -> bool:
    """True for the built-in origins, or a custom one: 'x-' + lowercase letters, digits, '.', '_' or '-'."""
    v = value.value if isinstance(value, Enum) else value
    return isinstance(v, str) and (v in {o.value for o in Origin} or bool(_CUSTOM_ORIGIN_RE.match(v)))


class Status(str, Enum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


class MaskCodec(str, Enum):
    BITPACK_ZSTD = "bitpack_zstd"  # crop to bbox, bit-pack rows, zstd (see docs/SPEC.md)


class ColumnSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    dtype: str  # one of: int32, int64, float32, string, dict_string, binary
    nullable: bool
    description: str


ANNOTATION_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(name="annotation_id", dtype="int64", nullable=False,
               description="Stable unique id. Random 63-bit integer assigned on creation; never reused."),
    ColumnSpec(name="video_id", dtype="dict_string", nullable=False, description="Which video (matches videos/<video_id>/)."),
    ColumnSpec(name="start_frame", dtype="int32", nullable=False,
               description="First frame, 0-indexed in display order. A point in time has start_frame == end_frame."),
    ColumnSpec(name="end_frame", dtype="int32", nullable=False, description="Last frame, inclusive."),
    ColumnSpec(name="kind", dtype="dict_string", nullable=False, description="marker | box | mask"),
    ColumnSpec(name="label", dtype="dict_string", nullable=False, description="A label name from vocab.json."),
    ColumnSpec(name="track_id", dtype="int32", nullable=True,
               description="Ties boxes/masks of one object across frames. Unique within a video."),
    ColumnSpec(name="x0", dtype="float32", nullable=True, description="Box left edge, pixels, source-video coordinates."),
    ColumnSpec(name="y0", dtype="float32", nullable=True, description="Box top edge, pixels (origin top-left)."),
    ColumnSpec(name="x1", dtype="float32", nullable=True, description="Box right edge (exclusive)."),
    ColumnSpec(name="y1", dtype="float32", nullable=True, description="Box bottom edge (exclusive)."),
    ColumnSpec(name="mask", dtype="binary", nullable=True, description="Encoded mask bytes, interpreted by mask_codec."),
    ColumnSpec(name="mask_codec", dtype="dict_string", nullable=True, description="How `mask` is encoded."),
    ColumnSpec(name="view_id", dtype="dict_string", nullable=True,
               description="Optional: the view (crop / focused version) this was labelled in. Geometry is always in source-video coordinates."),
    ColumnSpec(name="origin", dtype="dict_string", nullable=False, description="human | model | imported | other, or a custom value starting with 'x-' (e.g. x-crowd)"),
    ColumnSpec(name="tool", dtype="dict_string", nullable=True, description="Tool or model name that produced it."),
    ColumnSpec(name="tool_version", dtype="dict_string", nullable=True, description="Tool or model version."),
    ColumnSpec(name="author", dtype="dict_string", nullable=True, description="Person responsible, if any."),
    ColumnSpec(name="confidence", dtype="float32", nullable=True, description="0..1, usually only for model output."),
    ColumnSpec(name="status", dtype="dict_string", nullable=False, description="unreviewed | approved | rejected"),
    ColumnSpec(name="extra", dtype="string", nullable=True,
               description="Optional JSON object (as text) for anything custom. Use your own top-level key, e.g. extra.mytool."),
)

COLUMN_NAMES: tuple[str, ...] = tuple(c.name for c in ANNOTATION_COLUMNS)
REQUIRED_COLUMNS: tuple[str, ...] = tuple(c.name for c in ANNOTATION_COLUMNS if not c.nullable)
