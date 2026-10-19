"""Draw a dataset's annotations onto frames of the real video, to check by eye that they line up. Needs OpenCV."""
from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from vidlabel_spec import VideoMetadata, View

from . import masks
from .dataset import Dataset
from .frames import _cv2, read_frames


def label_color(label: str) -> tuple[int, int, int]:
    """A stable, distinct-ish RGB colour per label name."""
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def parse_frames(spec: str, frame_count: int) -> list[int]:
    """'10,20,30-40' -> [10, 20, 30..40]. A trailing '-' means to the end ('100-'). Clipped to the video."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a or 0), int(b) if b else frame_count - 1
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted({f for f in out if 0 <= f < frame_count})


def choose_frames(rows: Sequence[dict], frame_count: int, limit: int = 12) -> list[int]:
    """A useful default: frames that carry annotations, spread out; evenly spaced frames if there are none."""
    marks = sorted({f for r in rows for f in ((r["start_frame"],) if r["kind"] != "marker" else (r["start_frame"], r["end_frame"]))
                    if 0 <= f < frame_count})
    pool = marks or list(range(frame_count))
    if len(pool) <= limit:
        return pool
    idx = np.linspace(0, len(pool) - 1, limit).round().astype(int)
    return [pool[i] for i in sorted(set(idx.tolist()))]


def _mask_in_view(mask: np.ndarray, view: View) -> np.ndarray:
    r = view.region
    m = mask[r.y:r.y + r.height, r.x:r.x + r.width] if r else mask
    if r and view.output_size:
        m = masks.resize_nearest(m, view.output_size[1], view.output_size[0])
    return m


def draw_annotations(frame: np.ndarray, rows: Iterable[dict], frame_index: int, *, view: View | None = None) -> np.ndarray:
    """Return a copy of an RGB frame with the annotations active at `frame_index` drawn on it."""
    cv2 = _cv2()
    out = np.ascontiguousarray(frame.copy())
    banner: list[tuple[str, tuple[int, int, int]]] = []
    for r in rows:
        if not (r["start_frame"] <= frame_index <= r["end_frame"]):
            continue
        color = label_color(r["label"])
        text = r["label"] + (f"#{r['track_id']}" if r.get("track_id") is not None else "")
        if r.get("confidence") is not None:
            text += f" {r['confidence']:.2f}"
        if r["kind"] == "marker":
            span = f"{r['start_frame']}" if r["start_frame"] == r["end_frame"] else f"{r['start_frame']}-{r['end_frame']}"
            banner.append((f"{r['label']} [{span}]", color))
            continue
        if r["kind"] == "mask" and r.get("mask") is not None:
            m = masks.decode(r["mask"])
            if view is not None:
                m = _mask_in_view(m, view)
            if m.shape[:2] == out.shape[:2]:
                out[m] = (0.5 * out[m] + 0.5 * np.array(color)).astype(np.uint8)
        box = (r["x0"], r["y0"], r["x1"], r["y1"])
        if None in box:
            continue
        if view is not None:
            box = view.to_view_box(box)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        cv2.putText(out, text, (x0 + 2, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for i, (text, color) in enumerate(banner):
        y = 16 + 18 * i
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def render_preview(ds: Dataset, video_id: str, out_dir: str | Path, *, frames: Sequence[int] | None = None,
                   video_path: str | Path | None = None, view_id: str | None = None, mp4: bool = False,
                   roots: Sequence[str | Path] = ()) -> list[Path]:
    """Write annotated frames (PNG, or one MP4) to out_dir and return the paths.

    With `view_id`, shows that view's crop with only that view's annotations, as its pipeline would see them.
    """
    cv2 = _cv2()
    meta: VideoMetadata = ds.video(video_id)
    path = Path(video_path) if video_path else ds.resolve_video_path(video_id, roots)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"cannot find the video for '{video_id}'. Pass its path explicitly "
                                f"(recorded sources: {[s.uri for s in meta.sources]})")
    view = ds.view(view_id) if view_id else None
    rows = ds.annotations(video_id, view_id=view_id).to_pylist()
    chosen = list(frames) if frames is not None else choose_frames(rows, meta.frame_count)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    writer = None
    try:
        for f, rgb in read_frames(path, chosen, meta, view=view):
            img = draw_annotations(rgb, rows, f, view=view)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if mp4:
                if writer is None:
                    target = out / f"{video_id}{'_' + view_id if view_id else ''}_preview.mp4"
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), max(meta.fps.value, 1.0),
                                             (bgr.shape[1], bgr.shape[0]))
                    written.append(target)
                writer.write(bgr)
            else:
                target = out / f"{video_id}{'_' + view_id if view_id else ''}_f{f:06d}.png"
                cv2.imwrite(str(target), bgr)
                written.append(target)
    finally:
        if writer is not None:
            writer.release()
    return written
