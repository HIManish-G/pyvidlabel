"""Build VideoMetadata from a real file using ffprobe (must be installed)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from vidlabel_spec import FrameIndex, Rational, VideoMetadata, VideoSource


def file_hash(path: str | Path) -> str:
    """`sha256:<hex>` of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _frac(s: str | None) -> Fraction | None:
    if not s or s == "0/0":
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _rotation(stream: dict) -> tuple[int, float | None]:
    """Clockwise degrees to apply to the stored frame for display (0, 90, 180 or 270).

    ffprobe's display-matrix `rotation` is counter-clockwise degrees, so it is negated. The older `rotate` tag is
    already clockwise. Angles that are not a multiple of 90 cannot be represented: returns (0, raw_angle) so the
    caller can record them instead of silently dropping them.
    """
    raw = None
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            raw = -float(sd["rotation"])
    if raw is None and (stream.get("tags") or {}).get("rotate"):
        raw = float(stream["tags"]["rotate"])
    if raw is None:
        return 0, None
    snapped = round(raw / 90) * 90
    if abs(raw - snapped) > 1.0:
        return 0, raw
    return int(snapped) % 360, None


def probe_video(path: str | Path, video_id: str | None = None, *, hash: bool = True) -> VideoMetadata:
    """Read a video's facts with ffprobe (size, fps as a fraction, frame count, rotation, codec, bit rate) and hash the file
    unless `hash=False`. Needs `ffprobe` on the PATH. `video_id` defaults to the file name without extension."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found; install ffmpeg or build VideoMetadata by hand")
    p = Path(path).resolve()
    raw = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                          "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout
    info = json.loads(raw)
    if not info.get("streams"):
        raise ValueError(f"{p} has no video stream")
    s = info["streams"][0]
    avg, real = _frac(s.get("avg_frame_rate")), _frac(s.get("r_frame_rate"))
    fps = avg or real
    if fps is None:
        raise ValueError("could not determine frame rate")
    duration = float(s.get("duration") or info.get("format", {}).get("duration") or 0) or None
    extra: dict = {}
    if str(s.get("nb_frames", "")).isdigit():
        frame_count = int(s["nb_frames"])
    else:
        frame_count = round(duration * float(fps)) if duration else 0
        extra["frame_count_estimated"] = True
    rotation, odd_rotation = _rotation(s)
    if odd_rotation is not None:
        extra["unsupported_rotation_degrees"] = odd_rotation
    tb = _frac(s.get("time_base"))
    br = s.get("bit_rate") or info.get("format", {}).get("bit_rate")
    bit_rate = int(br) if str(br or "").isdigit() else None
    return VideoMetadata(
        video_id=video_id or p.stem.replace(" ", "_"), content_hash=file_hash(p) if hash else None,
        duration_s=duration, frame_count=frame_count, fps=Rational(num=fps.numerator, den=fps.denominator),
        timebase=Rational(num=tb.numerator, den=tb.denominator) if tb else None,
        is_vfr=bool(avg and real and avg != real), width=int(s["width"]), height=int(s["height"]),
        rotation=rotation, codec=s.get("codec_name"), pix_fmt=s.get("pix_fmt"), bit_rate=bit_rate,
        sources=[VideoSource(uri=str(p))], ingested_at=datetime.now(timezone.utc), extra=extra,
    )


def probe_frame_index(path: str | Path, *, content_hash: str | None = None) -> FrameIndex:
    """Read a video's per-frame timestamps and keyframes from its packet table with ffprobe. Nothing is decoded, so it takes
    seconds even for long videos. Pass `content_hash` (from the metadata) to record which file the index belongs to.
    Raises ValueError if the container has no timestamps (some raw streams), and RuntimeError without ffprobe."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found; install ffmpeg")
    p = Path(path).resolve()
    tb_out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base",
                             "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    tb = _frac(tb_out.splitlines()[0]) if tb_out else None
    if tb is None:
        raise ValueError(f"{p.name}: could not read the video time base")
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,flags",
                          "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True).stdout
    packets: list[tuple[int, bool]] = []
    for line in out.splitlines():
        pts, _, flags = line.partition(",")
        if not pts.lstrip("-").isdigit():
            raise ValueError(f"{p.name}: packets have no timestamps, so a frame index cannot be made")
        packets.append((int(pts), "K" in flags))
    if not packets:
        raise ValueError(f"{p.name}: no video packets found")
    packets.sort()   # packets come in decoding order; sorting by timestamp gives display order, which is what frame numbers mean
    return FrameIndex(content_hash=content_hash, timebase=Rational(num=tb.numerator, den=tb.denominator),
                      pts=[a for a, _ in packets], keyframes=[i for i, (_, k) in enumerate(packets) if k])


def reconcile_frame_count(meta: VideoMetadata, index: FrameIndex) -> VideoMetadata:
    """Make `meta.frame_count` agree with `index`. If the count was only estimated when the video was probed, the index (an
    exact count) replaces it; a real disagreement raises ValueError, because one of the two describes a different file."""
    if meta.frame_count == index.frame_count:
        return meta
    if meta.extra.get("frame_count_estimated"):
        extra = {k: v for k, v in meta.extra.items() if k != "frame_count_estimated"}
        return meta.model_copy(update={"frame_count": index.frame_count, "extra": extra})
    raise ValueError(f"the file has {index.frame_count} frames but the metadata says {meta.frame_count}")
