"""Command-line entry point for video-assistant-feedback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_MAX_FRAME_DIM,
    DEFAULT_NUM_FRAMES,
    DEFAULT_SYNTHESIS_MODEL,
    DEFAULT_VISION_MODEL,
    DEFAULT_WHISPER_MODEL,
    FAST_FRAME_INTERVAL,
    FAST_MAX_FRAME_DIM,
    FAST_PER_FRAME_TOKENS,
    FAST_VISION_MODEL,
    VIDEO_EXTENSIONS,
    Config,
)
from .pipeline import run


def discover_video(input_dir: Path) -> Path:
    """Find exactly one video file in ``input_dir``."""
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(
            f"No video files found in {input_dir} "
            f"(looked for: {', '.join(sorted(VIDEO_EXTENSIONS))})."
        )
    if len(videos) > 1:
        names = "\n  ".join(v.name for v in videos)
        raise SystemExit(
            f"Multiple videos found in {input_dir}:\n  {names}\n"
            "Pass one explicitly, e.g.: video-review input/clip.mp4"
        )
    return videos[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-review",
        description="Local, offline AI video review using Ollama vision models.",
    )
    p.add_argument(
        "video", nargs="?", type=Path,
        help="Path to the video file. If omitted, auto-discovers the single video in --input-dir.",
    )
    p.add_argument("--input-dir", type=Path, default=Path("input"),
                   help="Directory to auto-discover the input video (default: input/).")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                   help="Directory to write the report (default: output/).")
    p.add_argument("--output", type=Path, default=None,
                   help="Explicit report path (overrides --output-dir).")
    p.add_argument("--model", default=None,
                   help=f"Ollama vision model (default: {DEFAULT_VISION_MODEL}, or {FAST_VISION_MODEL} with --fast).")
    p.add_argument("--synth-model", default=DEFAULT_SYNTHESIS_MODEL,
                   help=f"Ollama text model for synthesis (default: {DEFAULT_SYNTHESIS_MODEL}).")
    p.add_argument("--frames", type=int, default=None,
                   help=f"Number of frames to sample (default: {DEFAULT_NUM_FRAMES}).")
    p.add_argument("--interval", type=float, default=None,
                   help="Sample one frame every N seconds (overrides --frames).")
    p.add_argument("--max-dim", type=int, default=None,
                   help=f"Cap longest frame edge in px, no upscale (default: {DEFAULT_MAX_FRAME_DIM}; 0 disables).")
    p.add_argument("--per-frame-tokens", type=int, default=None,
                   help="Cap output tokens per frame for faster analysis (default: unlimited).")
    p.add_argument("--fast", action="store_true",
                   help=(f"Quick-draft preset: {FAST_VISION_MODEL}, ~1 frame/{FAST_FRAME_INTERVAL:g}s, "
                         f"{FAST_MAX_FRAME_DIM}px, {FAST_PER_FRAME_TOKENS}-token cap. "
                         "Explicit flags still override."))
    p.add_argument("--whisper", action="store_true",
                   help="Transcribe the audio track with Whisper (requires the 'whisper' extra).")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                   help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL}).")
    p.add_argument("--ollama-host", default=None,
                   help="Ollama host URL (default: env OLLAMA_HOST or http://localhost:11434).")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep extracted frames on disk instead of cleaning them up.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    video_path = args.video if args.video else discover_video(args.input_dir)
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    if args.frames is not None and args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if args.interval is not None and args.interval <= 0:
        raise SystemExit("--interval must be > 0")

    # Resolve settings: explicit flags win; otherwise --fast preset or plain defaults.
    fast = args.fast
    vision_model = args.model or (FAST_VISION_MODEL if fast else DEFAULT_VISION_MODEL)

    frame_interval = args.interval
    num_frames = args.frames
    if frame_interval is None and num_frames is None:
        if fast:
            frame_interval = FAST_FRAME_INTERVAL
        else:
            num_frames = DEFAULT_NUM_FRAMES

    max_dim = args.max_dim
    if max_dim is None:
        max_dim = FAST_MAX_FRAME_DIM if fast else DEFAULT_MAX_FRAME_DIM

    per_frame_tokens = args.per_frame_tokens
    if per_frame_tokens is None:
        per_frame_tokens = FAST_PER_FRAME_TOKENS if fast else 0

    output_path = args.output or (args.output_dir / f"{video_path.stem}.md")

    config = Config(
        video_path=video_path,
        output_path=output_path,
        vision_model=vision_model,
        synthesis_model=args.synth_model,
        num_frames=num_frames or DEFAULT_NUM_FRAMES,
        frame_interval=frame_interval,
        max_frame_dim=max_dim,
        per_frame_tokens=per_frame_tokens,
        use_whisper=args.whisper,
        whisper_model=args.whisper_model,
        ollama_host=args.ollama_host,
        keep_frames=args.keep_frames,
    )

    try:
        run(config)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
