"""Example converter: read a simple CSV of tracked boxes and add it to a vidlabel dataset.

vidlabel ships no importers because every source format differs; this is the pattern to copy. The made-up input is:

    frame,track,class,x,y,w,h,score
    0,1,Car,100,50,40,30,0.91

(frame is 0-based, x/y/w/h are pixels of the displayed frame, track and score may be empty.)

Run it:  python examples/convert_csv.py my_dataset clip detections.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from vidlabel import Dataset


def convert(ds: Dataset, csv_path: str | Path, video_id: str, *, tool: str = "csv-converter", tool_version: str = "1",
            replace: bool = True) -> list[int]:
    """Add the boxes in `csv_path` to `video_id`. Returns the new annotation ids.

    replace=True first removes what an earlier run of this converter added to the video, so re-running does not
    duplicate rows.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for line_no, rec in enumerate(csv.DictReader(f), start=2):        # start=2: line 1 is the header
            x, y, w, h = (float(rec[k]) for k in ("x", "y", "w", "h"))
            rows.append({
                "kind": "box",
                "label": rec["class"].strip().lower(),                    # map the source's names to your vocabulary here
                "start_frame": int(rec["frame"]),                         # add or subtract 1 here if the source counts from 1
                "box": (x, y, x + w, y + h),                              # vidlabel stores corners: x0, y0, x1, y1
                "track_id": int(rec["track"]) if rec.get("track") else None,
                "confidence": float(rec["score"]) if rec.get("score") else None,
                "origin": "imported",                                     # where the row came from; details go in tool/extra
                "tool": tool,
                "tool_version": tool_version,
                "extra": {"csv_line": line_no},                           # anything else worth keeping, under your own key
            })
    if replace:
        old = ds.annotations(video_id, origin="imported", columns=["annotation_id", "tool"])
        mine = [i for i, t in zip(old.column("annotation_id").to_pylist(), old.column("tool").to_pylist()) if t == tool]
        ds.delete_annotations(video_id, mine)
    # create_labels=True adds unseen labels to the vocabulary; leave it off to make unknown labels an error instead.
    # Nothing is written if any row is invalid: the ValidationError says which check failed and for how many rows.
    return ds.add_annotations(rows, video_id, create_labels=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ids = convert(Dataset.open(sys.argv[1]), sys.argv[3], sys.argv[2])
    print(f"added {len(ids)} annotations")
