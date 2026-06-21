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


def probe_duration(video_path: Path) -> float:
    """Return the video duration in seconds using ffprobe."""
    _require("ffprobe")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {video_path}:\n{result.stderr.strip()}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise FFmpegError(f"Could not read duration from ffprobe output: {exc}") from exc


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
