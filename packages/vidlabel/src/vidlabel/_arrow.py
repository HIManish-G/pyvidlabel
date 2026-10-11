from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from vidlabel_spec import ANNOTATION_COLUMNS, SPEC_VERSION

ROW_GROUP_SIZE = 20_000

_TYPES = {
    "int32": pa.int32(), "int64": pa.int64(), "float32": pa.float32(),
    "string": pa.string(), "dict_string": pa.string(), "binary": pa.large_binary(),
}

ARROW_SCHEMA = pa.schema(
    [pa.field(c.name, _TYPES[c.dtype], nullable=c.nullable) for c in ANNOTATION_COLUMNS],
    metadata={b"vidlabel.spec_version": SPEC_VERSION.encode()},
)

_NO_DICT = {"mask"}


def write_parquet(table: pa.Table, path: Path, row_group_size: int = ROW_GROUP_SIZE) -> None:
    """Atomic write (temp file + rename) with the settings the spec recommends."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(
        table, tmp, compression="zstd", compression_level=3, row_group_size=row_group_size,
        use_dictionary=[n for n in table.column_names if n not in _NO_DICT],
        write_statistics=True, write_page_index=True,
    )
    os.replace(tmp, path)


def empty_table(columns: list[str] | None = None) -> pa.Table:
    t = ARROW_SCHEMA.empty_table()
    return t.select(columns) if columns else t
