"""Build VideoMetadata from a real file using ffprobe (must be installed)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from vidlabel_spec import Rational, VideoMetadata, VideoSource


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
