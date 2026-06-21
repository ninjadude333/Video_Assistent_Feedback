"""Configuration for the video review pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Video container formats we will auto-discover in the input/ directory.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}

# Defaults chosen for the local DGX1 Ollama install.
#   vision   -> best locally available vision-language model
#   synthesis-> strongest locally available text model for writing the report
DEFAULT_VISION_MODEL = "qwen3-vl:30b"
DEFAULT_SYNTHESIS_MODEL = "qwen3.6:35b"
DEFAULT_NUM_FRAMES = 30
DEFAULT_WHISPER_MODEL = "base"


@dataclass
class Config:
    """Resolved settings for a single review run."""

    video_path: Path
    output_path: Path
    vision_model: str = DEFAULT_VISION_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    num_frames: int = DEFAULT_NUM_FRAMES
    use_whisper: bool = False
    whisper_model: str = DEFAULT_WHISPER_MODEL
    ollama_host: str | None = None
    keep_frames: bool = False
