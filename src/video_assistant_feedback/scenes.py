"""Scene-cut detection via ffmpeg's scene filter."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# ffmpeg's showinfo logs one line per selected (scene-change) frame to stderr,
# each containing 'pts_time:<seconds>'.
_PTS_RE = re.compile(r"pts_time:([0-9]+\.?[0-9]*)")


@dataclass
class Scene:
    index: int
    start: float
    end: float
    frame_timestamps: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_cuts(video_path: Path, threshold: float = 0.4) -> list[float]:
    """Return scene-change timestamps (seconds) detected by ffmpeg.

    Returns an empty list if ffmpeg is missing or no cuts are found (single-shot video).
    Note: this decodes the full stream, so cost scales with video length.
    """
    if shutil.which("ffmpeg") is None:
        return []
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(video_path),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # showinfo writes to stderr regardless of success.
    cuts = [float(m) for m in _PTS_RE.findall(proc.stderr)]
    return sorted(set(cuts))


def build_scenes(duration: float, cuts: list[float], min_len: float = 0.5) -> list[Scene]:
    """Turn cut timestamps into contiguous Scene segments spanning [0, duration]."""
    boundaries = [0.0]
    for c in cuts:
        if 0.0 < c < duration and (c - boundaries[-1]) >= min_len:
            boundaries.append(c)
    boundaries.append(duration)

    return [
        Scene(index=i, start=boundaries[i], end=boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]
