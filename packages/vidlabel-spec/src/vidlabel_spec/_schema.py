from __future__ import annotations

import json
from pathlib import Path

from ._columns import ANNOTATION_COLUMNS
from ._models import SPEC_VERSION, Manifest, Release, VideoMetadata, View, Vocab


def json_schema() -> dict:
    """JSON Schema for every metadata file plus the annotation columns, for implementers in other languages."""
    return {
        "spec_version": SPEC_VERSION,
        "manifest": Manifest.model_json_schema(),
        "video_metadata": VideoMetadata.model_json_schema(),
        "view": View.model_json_schema(),
        "vocab": Vocab.model_json_schema(),
        "release": Release.model_json_schema(),
        "annotation_columns": [c.model_dump() for c in ANNOTATION_COLUMNS],
    }


def write_json_schema(path: str | Path) -> None:
    Path(path).write_text(json.dumps(json_schema(), indent=2) + "\n")
