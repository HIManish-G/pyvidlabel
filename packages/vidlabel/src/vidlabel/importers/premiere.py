"""Import Adobe Premiere Pro marker exports.

!! The CSV column layout below is written from memory of Premiere's Markers panel export
!! (tab-separated: Marker Name, Description, In, Out, Duration, Marker Type) and has NOT been checked against a real
!! export yet. Column names are matched case-insensitively and the delimiter is sniffed, but please run it on a real
!! file and report differences. Timecode maths (including drop-frame) is unit-tested.

Premiere exports *sequence* timecode, not source-video frames. Pass `start_tc` (the sequence's start timecode,
often 00:00:00:00 or 01:00:00:00) so frame 0 of the video maps correctly. Nested sequences are not handled.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

from ..dataset import Dataset

_TC = re.compile(r"^\s*(\d+)[:;](\d+)[:;](\d+)([:;])(\d+)\s*$")


def timecode_to_frames(tc: str, fps_num: int, fps_den: int = 1, drop_frame: bool | None = None) -> int:
    """'HH:MM:SS:FF' (or ';' before FF for drop-frame) -> frame count since 00:00:00:00.

    Drop-frame is only defined for 30000/1001 and 60000/1001; if `drop_frame` is None it is inferred from the ';'.
    """
    m = _TC.match(tc)
    if not m:
        raise ValueError(f"not a timecode: {tc!r}")
    h, mi, s, sep, f = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
    nominal = round(fps_num / fps_den)
    if drop_frame is None:
        drop_frame = sep == ";"
    if f >= nominal:
        raise ValueError(f"frame field {f} too large for {nominal} fps timecode {tc!r}")
    frames = (h * 3600 + mi * 60 + s) * nominal + f
    if drop_frame:
        if (fps_num, fps_den) not in ((30000, 1001), (60000, 1001)):
            raise ValueError("drop-frame timecode only applies to 29.97 and 59.94 fps")
        drop = 2 if nominal == 30 else 4
        total_minutes = h * 60 + mi
        frames -= drop * (total_minutes - total_minutes // 10)
    return frames


def read_markers(path: str | Path) -> list[dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text else ""
    delim = "\t" if first.count("\t") >= first.count(",") else ","
    rows = []
    for rec in csv.DictReader(text.splitlines(), delimiter=delim):
        rows.append({(k or "").strip().lower(): (v or "").strip() for k, v in rec.items()})
    return rows


def import_markers(ds: Dataset, video_id: str, path: str | Path, *, start_tc: str = "00:00:00:00",
                   label_map: Mapping[str, str] | None = None, label_from: str = "name", author: str | None = None,
                   default_label: str | None = None, create_labels: bool = False) -> list[int]:
    """Read a Premiere marker export and add each marker as a `kind='marker'` annotation with origin='premiere'.

    label_from: 'name' (marker name) or 'type' (Marker Type column). `label_map` renames them (e.g. {'Cut': 'cut'}).
    Markers whose label cannot be determined use `default_label`, or raise if that is None.
    """
    meta = ds.video(video_id)
    num, den = meta.fps.num, meta.fps.den
    start = timecode_to_frames(start_tc, num, den)
    out = []
    for i, rec in enumerate(read_markers(path)):
        raw_in = rec.get("in") or rec.get("marker in") or ""
        raw_out = rec.get("out") or rec.get("marker out") or ""
        if not raw_in:
            raise ValueError(f"marker row {i}: no 'In' timecode (columns: {sorted(rec)})")
        a = timecode_to_frames(raw_in, num, den) - start
        b = timecode_to_frames(raw_out, num, den) - start if raw_out else a
        end = max(a, b - 1) if b > a else a  # Out is exclusive: In + Duration
        key = rec.get("marker name", "") if label_from == "name" else rec.get("marker type", "")
        label = (label_map or {}).get(key, key) or default_label
        if not label:
            raise ValueError(f"marker row {i}: empty label ({label_from}); pass default_label or label_map")
        extra: dict[str, Any] = {"premiere": {k: v for k, v in {
            "name": rec.get("marker name"), "description": rec.get("description"), "type": rec.get("marker type"),
            "in": raw_in, "out": raw_out}.items() if v}}
        out.append({"kind": "marker", "label": label, "start_frame": a, "end_frame": end, "origin": "premiere",
                    "tool": "premiere-marker-export", "author": author, "status": "approved" if author else "unreviewed",
                    "extra": extra})
    return ds.add_annotations(out, video_id, create_labels=create_labels)
