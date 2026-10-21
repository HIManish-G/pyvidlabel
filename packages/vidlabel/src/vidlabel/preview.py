"""Draw a dataset's annotations onto frames of the real video, to check by eye that they line up. Needs OpenCV."""
from __future__ import annotations

import colorsys
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from vidlabel_spec import VideoMetadata, View

from . import masks
from .dataset import Dataset
from .frames import _cv2, crop_mask_to_view, read_frames


def label_color(label: str) -> tuple[int, int, int]:
    """A stable, distinct-ish RGB colour per label name."""
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


_NAMED = {"red": (255, 0, 0), "green": (0, 200, 0), "blue": (0, 90, 255), "yellow": (255, 220, 0), "orange": (255, 140, 0),
          "cyan": (0, 220, 220), "magenta": (255, 0, 255), "white": (255, 255, 255), "black": (0, 0, 0)}


def parse_color(text: str) -> tuple[int, int, int]:
    """'red', '#ff8800', 'ff8800' or '255,136,0' -> (r, g, b)."""
    t = text.strip().lower()
    if t in _NAMED:
        return _NAMED[t]
    if re.fullmatch(r"#?[0-9a-f]{6}", t):
        t = t.lstrip("#")
        return int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
    parts = t.split(",")
    if len(parts) == 3 and all(p.strip().isdigit() and int(p) < 256 for p in parts):
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    raise ValueError(f"cannot read colour '{text}': use a name ({', '.join(_NAMED)}), '#rrggbb' or 'r,g,b'")


def parse_colors(text: str) -> tuple[tuple[int, int, int] | None, dict[str, tuple[int, int, int]]]:
    """One colour for everything ('red'), per-label colours ('person=red;car=#00ff00'), or both (a bare colour is the default)."""
    default, per_label = None, {}
    for part in re.split(r"[;]", text):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            per_label[k.strip()] = parse_color(v)
        else:
            default = parse_color(part)
    return default, per_label


@dataclass
class PreviewStyle:
    """How annotations are drawn. Anything left as None is chosen automatically from the frame size."""
    color: tuple[int, int, int] | None = None                                # one colour for every label
    label_colors: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)  # overrides, by label name
    thickness: int | None = None        # box line width in pixels
    font_scale: float | None = None     # label text size
    show_labels: bool = True            # text next to boxes (track id, confidence); marker banners are always shown
    outline: bool = True                # dark border around lines and text so they stay visible on any background
    mask_opacity: float = 0.5

    def color_for(self, label: str) -> tuple[int, int, int]:
        """The colour used for a label: its entry in `label_colors`, else `color`, else a stable colour derived from the name."""
        return self.label_colors.get(label) or self.color or label_color(label)


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


def draw_annotations(frame: np.ndarray, rows: Iterable[dict], frame_index: int, *, view: View | None = None,
                     style: PreviewStyle | None = None) -> np.ndarray:
    """Return a copy of an RGB frame with the annotations active at `frame_index` drawn on it."""
    cv2 = _cv2()
    st = style or PreviewStyle()
    auto = max(0.55, frame.shape[1] / 1600)
    scale = st.font_scale or auto
    t = st.thickness or max(2, round(frame.shape[1] / 500))
    out = np.ascontiguousarray(frame.copy())
    banner: list[tuple[str, tuple[int, int, int]]] = []
    for r in rows:
        if not (r["start_frame"] <= frame_index <= r["end_frame"]):
            continue
        color = st.color_for(r["label"])
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
                m = crop_mask_to_view(m, view)
            if m.shape[:2] == out.shape[:2]:
                out[m] = ((1 - st.mask_opacity) * out[m] + st.mask_opacity * np.array(color)).astype(np.uint8)
        box = (r["x0"], r["y0"], r["x1"], r["y1"])
        if None in box:
            continue
        if view is not None:
            box = view.to_view_box(box)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if st.outline:
            cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), t + 2)  # dark outline so it shows on any colour
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), color, t)
        if st.show_labels:
            pos = (x0 + 4, y0 - 8 if y0 > 24 else y0 + int(24 * scale))
            if st.outline:
                cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 2, cv2.LINE_AA)
            cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, t - 1), cv2.LINE_AA)
    for i, (text, color) in enumerate(banner):
        th = max(1, round(scale * 1.8))
        y = int(24 * scale) + int(30 * scale) * i
        if st.outline:
            cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)
    return out


def count_drawn(rows: Sequence[dict], frames: Iterable[int]) -> tuple[int, int]:
    """(annotations active on at least one of `frames`, number of those frames that have any annotation)."""
    frames = list(frames)
    hit = {i for i, r in enumerate(rows) if any(r["start_frame"] <= f <= r["end_frame"] for f in frames)}
    busy = sum(any(r["start_frame"] <= f <= r["end_frame"] for r in rows) for f in frames)
    return len(hit), busy


def render_preview(ds: Dataset, video_id: str, out_dir: str | Path, *, frames: Sequence[int] | None = None,
                   video_path: str | Path | None = None, view_id: str | None = None, mp4: bool = False,
                   roots: Sequence[str | Path] = (), style: PreviewStyle | None = None) -> list[Path]:
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
            img = draw_annotations(rgb, rows, f, view=view, style=style)
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
