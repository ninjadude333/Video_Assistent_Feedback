"""Command-line entry point for video-assistant-feedback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_MAX_FRAME_DIM,
    DEFAULT_NUM_FRAMES,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_VISION_MODEL,
    DEFAULT_WHISPER_MODEL,
    FAST_FRAME_INTERVAL,
    FAST_MAX_FRAME_DIM,
    FAST_PER_FRAME_TOKENS,
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
            "Pass one explicitly (video-review input/clip.mp4) or use --batch."
        )
    return videos[0]


def discover_all_videos(input_dir: Path) -> list[Path]:
    """Return all video files in ``input_dir`` (sorted). For --batch."""
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
    return videos


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
                   help=f"Ollama vision model (default: {DEFAULT_VISION_MODEL}).")
    p.add_argument("--synth-model", default=None,
                   help="Ollama text model for synthesis + bridges (default: reuse --model, "
                        "so only one model stays loaded; e.g. pass qwen3.6:35b for higher quality).")
    p.add_argument("--frames", type=int, default=None,
                   help=f"Number of frames to sample (default: {DEFAULT_NUM_FRAMES}).")
    p.add_argument("--interval", type=float, default=None,
                   help="Sample one frame every N seconds (overrides --frames).")
    p.add_argument("--max-dim", type=int, default=None,
                   help=f"Cap longest frame edge in px, no upscale (default: {DEFAULT_MAX_FRAME_DIM}; 0 disables).")
    p.add_argument("--per-frame-tokens", type=int, default=None,
                   help="Cap output tokens per frame for faster analysis (default: unlimited).")
    p.add_argument("--fast", action="store_true",
                   help=(f"Quick-draft preset (same vision model): ~1 frame/{FAST_FRAME_INTERVAL:g}s, "
                         f"{FAST_MAX_FRAME_DIM}px, {FAST_PER_FRAME_TOKENS}-token cap. "
                         "Explicit flags still override."))
    p.add_argument("--full", action="store_true",
                   help=(f"Full-quality preset: native resolution (no downscale), "
                         f"{DEFAULT_NUM_FRAMES} frames, uncapped output. Explicit flags still override."))
    p.add_argument("--scene-threshold", type=float, default=DEFAULT_SCENE_THRESHOLD,
                   help=f"Scene-cut sensitivity, 0-1, lower=more cuts (default: {DEFAULT_SCENE_THRESHOLD:g}).")
    p.add_argument("--no-scenes", action="store_true",
                   help="Disable scene detection, the scene timeline table, and bridge suggestions.")
    p.add_argument("--whisper", action="store_true",
                   help="Transcribe the audio track with Whisper (requires the 'whisper' extra).")
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL,
                   help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL}).")
    p.add_argument("--ollama-host", default=None,
                   help="Ollama host URL (default: env OLLAMA_HOST or http://localhost:11434).")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep extracted frames on disk instead of cleaning them up.")
    p.add_argument("--batch", action="store_true",
                   help="Process EVERY video in --input-dir sequentially, writing "
                        "--output-dir/<name>.md for each. One failure doesn't abort the rest.")
    p.add_argument("--skip-existing", action="store_true",
                   help="In --batch, skip videos that already have a non-empty report.")
    return p


def _resolve_settings(args) -> dict:
    """Resolve the shared per-run settings (everything except video/output paths).

    Explicit flags win; otherwise the --fast/--full preset or plain defaults apply.
    --fast keeps the SAME vision model (model loading is the bottleneck) and differs only
    in resolution + frames sampled + per-frame output cap. --full uses native resolution.
    """
    vision_model = args.model or DEFAULT_VISION_MODEL
    # Reuse the vision model for the text passes unless explicitly overridden — keeps a
    # single model resident (model loading + larger models are the real bottleneck here).
    synthesis_model = args.synth_model or vision_model

    frame_interval = args.interval
    num_frames = args.frames
    if frame_interval is None and num_frames is None:
        if args.fast:
            frame_interval = FAST_FRAME_INTERVAL
        else:
            num_frames = DEFAULT_NUM_FRAMES

    max_dim = args.max_dim
    if max_dim is None:
        if args.fast:
            max_dim = FAST_MAX_FRAME_DIM
        elif args.full:
            max_dim = 0  # native resolution, no downscale
        else:
            max_dim = DEFAULT_MAX_FRAME_DIM

    per_frame_tokens = args.per_frame_tokens
    if per_frame_tokens is None:
        per_frame_tokens = FAST_PER_FRAME_TOKENS if args.fast else 0

    return dict(
        vision_model=vision_model,
        synthesis_model=synthesis_model,
        num_frames=num_frames or DEFAULT_NUM_FRAMES,
        frame_interval=frame_interval,
        max_frame_dim=max_dim,
        per_frame_tokens=per_frame_tokens,
        detect_scenes=not args.no_scenes,
        scene_threshold=args.scene_threshold,
        use_whisper=args.whisper,
        whisper_model=args.whisper_model,
        ollama_host=args.ollama_host,
        keep_frames=args.keep_frames,
    )


def _run_batch(videos: list[Path], output_dir: Path, settings: dict, skip_existing: bool) -> int:
    """Process every video sequentially, isolating per-file failures. Returns exit code."""
    total = len(videos)
    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for i, video_path in enumerate(videos, 1):
        out = output_dir / f"{video_path.stem}.md"
        print(f"\n{'=' * 60}\n[{i}/{total}] {video_path.name}\n{'=' * 60}")
        if skip_existing and out.exists() and out.stat().st_size > 0:
            print(f"⏭️  Skipping (report already exists): {out}")
            skipped.append(video_path.name)
            continue
        try:
            run(Config(video_path=video_path, output_path=out, **settings))
            succeeded.append(video_path.name)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001 - isolate one bad file from the batch
            print(f"❌ Failed: {video_path.name}: {exc}", file=sys.stderr)
            failed.append(video_path.name)

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {len(succeeded)} ok, {len(failed)} failed, {len(skipped)} skipped (of {total})")
    if failed:
        print("Failed:")
        for name in failed:
            print(f"  - {name}")
    print('=' * 60)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.frames is not None and args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if args.interval is not None and args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    if args.fast and args.full:
        raise SystemExit("--fast and --full are mutually exclusive.")

    settings = _resolve_settings(args)

    if args.batch:
        if args.video is not None:
            raise SystemExit("--batch processes the whole --input-dir; don't also pass a video path.")
        if args.output is not None:
            raise SystemExit("--batch writes one report per video; use --output-dir, not --output.")
        videos = discover_all_videos(args.input_dir)
        print(f"📦 Batch: {len(videos)} video(s) in {args.input_dir} → {args.output_dir}/")
        return _run_batch(videos, args.output_dir, settings, args.skip_existing)

    if args.skip_existing:
        raise SystemExit("--skip-existing only applies with --batch.")

    video_path = args.video if args.video else discover_video(args.input_dir)
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")
    output_path = args.output or (args.output_dir / f"{video_path.stem}.md")

    try:
        run(Config(video_path=video_path, output_path=output_path, **settings))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
