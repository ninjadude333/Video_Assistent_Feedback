"""Frame and audio extraction via ffmpeg/ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or fails."""


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FFmpegError(
            f"'{tool}' not found on PATH. Install ffmpeg (provides ffmpeg + ffprobe)."
        )


def _parse_fps(value: str) -> float:
    """Parse an ffprobe frame-rate string like '25/1' into a float."""
    try:
        num, den = value.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video_info(video_path: Path) -> dict:
    """Return {duration, width, height, fps} for the first video stream via ffprobe."""
    _require("ffprobe")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
            "-show_entries", "format=duration", "-of", "json", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {video_path}:\n{result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"Could not parse ffprobe output: {exc}") from exc

    duration = float(data.get("format", {}).get("duration", 0) or 0)
    streams = data.get("streams", [])
    width = height = 0
    fps = 0.0
    if streams:
        s = streams[0]
        width = int(s.get("width", 0) or 0)
        height = int(s.get("height", 0) or 0)
        fps = _parse_fps(s.get("avg_frame_rate", "0/0")) or _parse_fps(s.get("r_frame_rate", "0/0"))
    return {"duration": duration, "width": width, "height": height, "fps": fps}


def probe_duration(video_path: Path) -> float:
    """Return the video duration in seconds using ffprobe."""
    duration = probe_video_info(video_path)["duration"]
    if duration <= 0:
        raise FFmpegError(f"Could not read a valid duration from {video_path}.")
    return duration


def _scale_args(max_dim: int) -> list[str]:
    """ffmpeg -vf args to fit within max_dim x max_dim, preserving aspect, no upscale."""
    if max_dim and max_dim > 0:
        return [
            "-vf",
            f"scale='min({max_dim},iw)':'min({max_dim},ih)':force_original_aspect_ratio=decrease",
        ]
    return []


def extract_frames_at(
    video_path: Path,
    output_dir: Path,
    timestamps: list[float],
    max_dim: int = 0,
) -> list[tuple[float, Path]]:
    """Extract one JPEG per timestamp.

    ``max_dim`` caps the longest edge (px) without upscaling; 0 disables scaling.
    Returns a list of (timestamp, frame_path) for frames that were successfully written.
    """
    _require("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = _scale_args(max_dim)

    extracted: list[tuple[float, Path]] = []
    for i, ts in enumerate(timestamps):
        out = output_dir / f"frame_{i:04d}.jpg"
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
                "-frames:v", "1", *scale, "-q:v", "2", str(out),
            ],
            capture_output=True,
        )
        if proc.returncode == 0 and out.exists():
            extracted.append((ts, out))

    if not extracted:
        raise FFmpegError(f"No frames were extracted from {video_path}.")

    return extracted


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract a 16kHz mono WAV track for transcription. Returns False if no audio."""
    _require("ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path), "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path),
        ],
        capture_output=True,
    )
    return proc.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0
