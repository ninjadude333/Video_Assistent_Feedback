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

# Longest-edge cap (px) applied at frame extraction. Never upscales.
# A guardrail for high-res sources (1080p/4K) so vision-token counts stay sane;
# a no-op for footage already below this. 0 disables scaling entirely.
DEFAULT_MAX_FRAME_DIM = 1280

# Scene-cut sensitivity for ffmpeg's scene filter (0-1; lower = more cuts).
DEFAULT_SCENE_THRESHOLD = 0.4

# --fast preset: SAME vision model (model loading is the real bottleneck —
# swapping in a smaller model would force an extra load). Fast differs only in
# resolution + how many frames are sampled, plus a per-frame output cap.
FAST_FRAME_INTERVAL = 3.0          # one frame every 3 seconds
FAST_MAX_FRAME_DIM = 1024
FAST_PER_FRAME_TOKENS = 250


@dataclass
class Config:
    """Resolved settings for a single review run."""

    video_path: Path
    output_path: Path
    vision_model: str = DEFAULT_VISION_MODEL
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL
    num_frames: int = DEFAULT_NUM_FRAMES
    frame_interval: float | None = None   # seconds between frames; overrides num_frames
    max_frame_dim: int = DEFAULT_MAX_FRAME_DIM
    per_frame_tokens: int = 0             # cap per-frame output tokens; 0 = unlimited
    detect_scenes: bool = True
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    use_whisper: bool = False
    whisper_model: str = DEFAULT_WHISPER_MODEL
    ollama_host: str | None = None
    keep_frames: bool = False
