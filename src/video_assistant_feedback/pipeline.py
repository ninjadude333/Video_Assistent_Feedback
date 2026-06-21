"""End-to-end review pipeline orchestration."""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .analyze import analyze_frames, synthesize_report
from .config import Config
from .extract import extract_frames
from .transcribe import transcribe_audio


def run(config: Config) -> Path:
    """Run the full pipeline and write the Markdown report. Returns the report path."""
    print(f"🎬 Video:     {config.video_path}")
    print(f"👁️  Vision:    {config.vision_model}")
    print(f"📝 Synthesis: {config.synthesis_model}")
    print(f"🖼️  Frames:    {config.num_frames}")
    print(f"🎙️  Whisper:   {'on (' + config.whisper_model + ')' if config.use_whisper else 'off'}\n")

    work_dir = Path(tempfile.mkdtemp(prefix="vaf_"))
    try:
        # 1. Extract frames
        frame_paths, duration = extract_frames(config.video_path, work_dir / "frames", config.num_frames)
        print(f"✅ Extracted {len(frame_paths)} frames from {duration:.1f}s video\n")

        # 2. Optional audio transcription
        transcript = ""
        if config.use_whisper:
            print("🎙️  Transcribing audio...")
            transcript = transcribe_audio(config.video_path, work_dir, config.whisper_model)
            if transcript:
                print(f"✅ Transcript: {len(transcript)} chars\n")

        # 3. Per-frame vision analysis
        print(f"👁️  Running vision analysis on {len(frame_paths)} frames...")
        analyses = analyze_frames(frame_paths, duration, config.vision_model, config.ollama_host)

        # 4. Synthesize report
        print("\n📝 Synthesizing final report...")
        report_body = synthesize_report(analyses, transcript, config.synthesis_model, config.ollama_host)

        # 5. Write output
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        header = _header(config, duration, len(frame_paths), bool(transcript))
        config.output_path.write_text(header + report_body + "\n", encoding="utf-8")
        print(f"\n✅ Report saved to {config.output_path}")
        return config.output_path
    finally:
        if config.keep_frames:
            print(f"🗂️  Frames kept in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def _header(config: Config, duration: float, n_frames: int, has_audio: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"# Video Review: {config.video_path.name}\n\n"
        f"- **Generated:** {now}\n"
        f"- **Duration:** {duration:.1f}s\n"
        f"- **Frames analyzed:** {n_frames}\n"
        f"- **Vision model:** `{config.vision_model}`\n"
        f"- **Synthesis model:** `{config.synthesis_model}`\n"
        f"- **Audio transcript:** {'yes' if has_audio else 'no'}\n\n"
        f"---\n\n"
    )
