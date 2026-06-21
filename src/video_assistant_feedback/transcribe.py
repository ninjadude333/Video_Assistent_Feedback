"""Optional audio transcription via Whisper."""

from __future__ import annotations

from pathlib import Path

from .extract import extract_audio


def transcribe_audio(video_path: Path, work_dir: Path, whisper_model: str = "base") -> str:
    """Transcribe the video's audio track. Returns '' if Whisper or audio is unavailable."""
    try:
        import whisper  # type: ignore
    except ImportError:
        print("⚠️  Whisper not installed. Install with: uv sync --extra whisper")
        return ""

    audio_path = work_dir / "audio.wav"
    if not extract_audio(video_path, audio_path):
        print("⚠️  No audio track found (or extraction failed); skipping transcription.")
        return ""

    try:
        model = whisper.load_model(whisper_model)
        result = model.transcribe(str(audio_path))
        return result.get("text", "").strip()
    except Exception as exc:  # noqa: BLE001 - whisper raises a variety of errors
        print(f"⚠️  Whisper transcription failed: {exc}")
        return ""
