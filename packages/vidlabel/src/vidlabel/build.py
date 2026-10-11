"""Compile per-video sources into one sorted, immutable training table, pinned by a content-derived release id."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from vidlabel_spec import SPEC_VERSION, Release

from ._arrow import ARROW_SCHEMA, ROW_GROUP_SIZE, write_parquet
from .dataset import Dataset
from .validate import ValidationError, errors

SORT_KEYS = [("video_id", "ascending"), ("start_frame", "ascending"), ("annotation_id", "ascending")]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(ds: Dataset, *, statuses: Sequence[str] = ("unreviewed", "approved"), origins: Sequence[str] | None = None,
          kinds: Sequence[str] | None = None, video_ids: Sequence[str] | None = None,
          row_group_size: int = ROW_GROUP_SIZE, check: bool = True) -> tuple[Release, Path]:
    """Write compiled/<release_id>/{annotations.parquet, release.json}. Rejected annotations are excluded by default.

    The release id is the hash of the file contents: same data and filters give the same id, so a training run can
    record it and later reproduce exactly the same table.
    """
    if check:
        bad = errors(ds.validate())
        if bad:
            raise ValidationError(bad)
    ids = sorted(video_ids) if video_ids else sorted(ds.manifest.videos)
    parts = []
    for vid in ids:
        t = ds.annotations(vid, kind=kinds, origin=origins, status=list(statuses))
        if t.num_rows:
            parts.append(t.sort_by(SORT_KEYS))
    table = pa.concat_tables(parts) if parts else ARROW_SCHEMA.empty_table()
    out_root = ds.path / "compiled"
    out_root.mkdir(exist_ok=True)
    tmp = out_root / "_building.parquet"
    write_parquet(table, tmp, row_group_size=row_group_size)
    digest = _sha256(tmp)
    release_id = "r" + digest[:12]
    final = out_root / release_id
    if final.exists():
        tmp.unlink()
    else:
        final.mkdir()
        shutil.move(str(tmp), final / "annotations.parquet")
    release = Release(
        release_id=release_id, spec_version=SPEC_VERSION, created_at=datetime.now(timezone.utc),
        n_rows=table.num_rows, n_videos=len(parts), row_group_size=row_group_size,
        files={"annotations.parquet": f"sha256:{digest}"},
        extra={"statuses": list(statuses), "origins": list(origins) if origins else None,
               "kinds": list(kinds) if kinds else None, "video_ids": ids},
    )
    if not (final / "release.json").exists():
        (final / "release.json").write_text(release.model_dump_json(indent=2))
    else:
        release = Release.model_validate_json((final / "release.json").read_text())
    (out_root / "LATEST").write_text(release_id + "\n")
    return release, final


class Compiled:
    """Reader for a compiled release. Fast clip lookups use Parquet row-group statistics (the table is sorted)."""

    def __init__(self, path: str | Path):
        p = Path(path)
        if p.is_dir() and (p / "LATEST").exists():  # a compiled/ directory: use the newest release
            p = p / (p / "LATEST").read_text().strip()
        self.path = p
        self.release = Release.model_validate_json((p / "release.json").read_text())
        self._pf = pq.ParquetFile(p / "annotations.parquet")
        self._schema = self._pf.schema_arrow
        md = self._pf.metadata
        names = [md.schema.column(i).name for i in range(md.num_columns)]
        ci = {n: names.index(n) for n in ("video_id", "start_frame", "end_frame")}
        self._rg = []
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            st = {n: rg.column(i).statistics for n, i in ci.items()}
            self._rg.append((st["video_id"].min, st["video_id"].max, st["start_frame"].min, st["end_frame"].max)
                            if all(s is not None and s.has_min_max for s in st.values()) else None)
        self.last_row_groups_read: list[int] = []

    def __len__(self) -> int:
        return self.release.n_rows

    @property
    def release_id(self) -> str:
        return self.release.release_id

    def row_groups_for(self, video_id: str, start: int, end: int) -> list[int]:
        out = []
        for g, s in enumerate(self._rg):
            if s is None or (s[0] <= video_id <= s[1] and s[2] <= end and s[3] >= start):
                out.append(g)
        return out

    def clip(self, video_id: str, start: int, end: int, columns: Sequence[str] | None = None) -> pa.Table:
        """Annotations of one video overlapping frames [start, end] inclusive, reading only the row groups that can match."""
        groups = self.row_groups_for(video_id, start, end)
        self.last_row_groups_read = groups
        cols = list(columns) if columns else None
        read_cols = cols
        if cols is not None:  # need the filter columns even if not requested
            read_cols = list(dict.fromkeys([*cols, "video_id", "start_frame", "end_frame"]))
        if not groups:
            t = ARROW_SCHEMA.empty_table()
            return t.select(cols) if cols else t
        t = self._pf.read_row_groups(groups, columns=read_cols)
        keep = pc.and_(pc.equal(t.column("video_id"), video_id),
                       pc.and_(pc.less_equal(t.column("start_frame"), end), pc.greater_equal(t.column("end_frame"), start)))
        t = t.filter(keep)
        return t.select(cols) if cols else t

    def read(self, columns: Sequence[str] | None = None) -> pa.Table:
        return self._pf.read(columns=list(columns) if columns else None)
