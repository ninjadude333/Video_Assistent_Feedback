"""Vision analysis and report synthesis via Ollama."""

from __future__ import annotations

from pathlib import Path

import ollama

# Per-frame prompt. Deliberately style-agnostic — works for animation,
# live-action, motion graphics, etc. Kept short and scoped to reduce hallucination.
FRAME_PROMPT = (
    "You are reviewing frame {idx} of {total} from a video (timestamp ~{ts:.1f}s). "
    "Be concise and factual. Report:\n"
    "1) What is happening visually (subjects, setting, composition, camera).\n"
    "2) Any technical or quality issues: flicker, temporal inconsistency, artifacts, "
    "style drift, compositing/masking errors, lighting mismatches, motion blur or "
    "interpolation problems, color banding/grading issues, clipping, or warping.\n"
    "3) The mood/tone of the shot.\n"
    "If a frame looks clean, say so — do not invent issues."
)

SYNTHESIS_PROMPT = """You are a professional video director and editor reviewing a video.

Below are per-frame analyses sampled at regular intervals across the full runtime{audio_note}.
{audio_section}
FRAME-BY-FRAME ANALYSES:
{frame_summary}

Produce a detailed, well-structured Markdown review with exactly these three sections:

## 1. SYNOPSIS
A 3-6 sentence description of the video's content, visual style, and tone. Do not
invent narrative details that the frames do not support.

## 2. ISSUES TO FIX
A prioritized list of technical and artistic issues. For each issue:
- Describe the problem clearly.
- Give the approximate timestamp(s) where it appears.
- Suggest a concrete fix.
Order from most to least important. If no real issues were found, say so explicitly.

## 3. GENERAL FEEDBACK
Overall impressions: what works, what doesn't, pacing, consistency, and one key
recommendation for the next iteration.
"""


def _client(host: str | None) -> ollama.Client:
    return ollama.Client(host=host) if host else ollama.Client()


def analyze_frames(
    frame_paths: list[Path],
    duration: float,
    model: str,
    host: str | None = None,
) -> list[dict]:
    """Run the vision model on each frame; return per-frame analyses."""
    client = _client(host)
    total = len(frame_paths)
    interval = duration / total if total else 0.0
    analyses: list[dict] = []

    for i, frame_path in enumerate(frame_paths):
        ts = i * interval
        print(f"  Analyzing frame {i + 1}/{total} (t={ts:.1f}s)...")
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": FRAME_PROMPT.format(idx=i + 1, total=total, ts=ts),
                "images": [str(frame_path)],
            }],
        )
        analyses.append({
            "frame": i,
            "timestamp": ts,
            "analysis": response["message"]["content"].strip(),
        })

    return analyses


def synthesize_report(
    analyses: list[dict],
    audio_transcript: str,
    model: str,
    host: str | None = None,
) -> str:
    """Synthesize the final Markdown report from all frame analyses."""
    client = _client(host)

    frame_summary = "\n\n".join(
        f"[Frame {a['frame'] + 1} @ {a['timestamp']:.1f}s]: {a['analysis']}"
        for a in analyses
    )
    if audio_transcript:
        audio_section = f"\nAUDIO TRANSCRIPT:\n{audio_transcript}\n"
        audio_note = ", along with the audio transcript"
    else:
        audio_section = ""
        audio_note = ""

    prompt = SYNTHESIS_PROMPT.format(
        audio_note=audio_note,
        audio_section=audio_section,
        frame_summary=frame_summary,
    )
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()
