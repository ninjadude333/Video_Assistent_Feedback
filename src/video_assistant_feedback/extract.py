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


def extract_frames(video_path: Path, output_dir: Path, n_frames: int) -> tuple[list[Path], float]:
    """Extract ``n_frames`` evenly spaced JPEG frames.

    Returns the list of extracted frame paths and the video duration in seconds.
    """
    _require("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    if duration <= 0:
        raise FFmpegError(f"Video reports non-positive duration ({duration}s): {video_path}")

    interval = duration / n_frames
    frame_paths: list[Path] = []

    for i in range(n_frames):
        timestamp = i * interval
        out = output_dir / f"frame_{i:04d}.jpg"
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "2", str(out),
            ],
            capture_output=True,
        )
        if proc.returncode == 0 and out.exists():
            frame_paths.append(out)

    if not frame_paths:
        raise FFmpegError(f"No frames were extracted from {video_path}.")

    return frame_paths, duration


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
