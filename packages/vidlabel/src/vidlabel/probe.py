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


def probe_video(path: str | Path, video_id: str | None = None, *, hash: bool = True) -> VideoMetadata:
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
    rotation = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(-float(sd["rotation"]))) % 360
    if not rotation and (s.get("tags") or {}).get("rotate"):
        rotation = int(float(s["tags"]["rotate"])) % 360
    tb = _frac(s.get("time_base"))
    return VideoMetadata(
        video_id=video_id or p.stem.replace(" ", "_"), content_hash=file_hash(p) if hash else None,
        duration_s=duration, frame_count=frame_count, fps=Rational(num=fps.numerator, den=fps.denominator),
        timebase=Rational(num=tb.numerator, den=tb.denominator) if tb else None,
        is_vfr=bool(avg and real and avg != real), width=int(s["width"]), height=int(s["height"]),
        rotation=rotation if rotation in (0, 90, 180, 270) else 0, codec=s.get("codec_name"), pix_fmt=s.get("pix_fmt"),
        sources=[VideoSource(uri=str(p))], ingested_at=datetime.now(timezone.utc), extra=extra,
    )
