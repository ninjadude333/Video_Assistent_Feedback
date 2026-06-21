"""Vision analysis, report synthesis, and bridge suggestions via Ollama."""

from __future__ import annotations

import json
from pathlib import Path

import ollama

# Shot-type vocabulary the model must classify into (kept stable for the table).
SHOT_TYPES = [
    "extreme wide", "wide", "full", "medium wide", "medium",
    "medium close-up", "close-up", "extreme close-up",
    "insert", "establishing", "over-the-shoulder", "pov", "unknown",
]

# Structured per-frame output schema (Ollama enforces this via `format`).
FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "shot_type": {"type": "string", "enum": SHOT_TYPES},
        "camera_angle": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "mood": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["problem", "severity"],
            },
        },
    },
    "required": ["description", "shot_type", "camera_angle", "mood", "issues"],
}

FRAME_PROMPT = (
    "You are reviewing frame {idx} of {total} from a video (timestamp ~{ts:.1f}s). "
    "Be concise and factual. Return JSON only.\n"
    "- description: what is visible (subjects, setting, composition).\n"
    f"- shot_type: one of {SHOT_TYPES}.\n"
    "- camera_angle: e.g. eye-level, low-angle, high-angle, overhead, dutch.\n"
    "- subjects: key subjects in frame.\n"
    "- mood: the tone of the shot.\n"
    "- issues: technical/quality problems (flicker, temporal inconsistency, artifacts, "
    "style drift, compositing/masking errors, lighting mismatch, motion-blur/interpolation, "
    "color banding, clipping, warping), each with severity low/medium/high. "
    "If the frame looks clean, return an empty issues list — do not invent issues."
)

SYNTHESIS_PROMPT = """You are a professional video director and editor reviewing a video.

Below are per-frame analyses sampled across the full runtime{audio_note}, grouped by detected scene.
{audio_section}
SCENE TIMELINE:
{scene_summary}

FRAME-BY-FRAME ANALYSES:
{frame_summary}

Produce a detailed, well-structured Markdown review with exactly these three sections:

## 1. SYNOPSIS
A 3-6 sentence description of the video's content, visual style, and tone. Do not
invent narrative details the frames do not support.

## 2. ISSUES TO FIX
A prioritized list of technical and artistic issues. For each: describe the problem,
give the approximate timestamp(s), and suggest a concrete fix. Order most to least
important. If no real issues were found, say so explicitly.

## 3. GENERAL FEEDBACK
Overall impressions: what works, what doesn't, pacing, consistency, and one key
recommendation for the next iteration.
"""

# Structured bridge-suggestion schema for the LTX-2.3 image-to-video workflow.
BRIDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "bridges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "ltx_prompt": {"type": "string"},
                    "length_seconds": {"type": "number"},
                    "first_frame_tc": {"type": ["number", "null"]},
                    "last_frame_tc": {"type": ["number", "null"]},
                },
                "required": ["title", "rationale", "ltx_prompt", "length_seconds"],
            },
        }
    },
    "required": ["bridges"],
}

BRIDGE_PROMPT = """You are a film editor planning transition/bridging shots for an
AI-generated video produced with an LTX-2.3 image-to-video pipeline.

The video is {duration:.1f}s long. Here is its scene timeline:
{scene_summary}

Review context (issues + feedback already written):
{review_context}

Suggest 0-4 bridging or transition clips that would improve flow, pacing, or coverage
(e.g. smoothing an abrupt cut, adding an establishing or reaction beat, varying shot
scale). Only suggest bridges that genuinely help — return an empty list if none are needed.

For each bridge return JSON:
- title: short name + where it goes (e.g. "Establishing beat before Scene 2").
- rationale: why it helps.
- ltx_prompt: a concrete LTX-2.3 image-to-video generation prompt describing subject,
  motion, camera movement, and style. Write it ready to paste into the i2v pipeline.
- length_seconds: suggested clip length (typically 1.5-5).
- first_frame_tc: timecode in SECONDS of an EXISTING frame to condition the FIRST image
  on (usually near the end of the preceding scene), or null.
- last_frame_tc: timecode in SECONDS of an EXISTING frame to condition the LAST image on
  (usually near the start of the following scene), or null.
All timecodes must be within 0 and {duration:.1f}. Return JSON only.
"""


def _client(host: str | None) -> ollama.Client:
    return ollama.Client(host=host) if host else ollama.Client()


def _chat_json(client, model, content, schema, images=None, num_predict=0):
    """Call ollama.chat with a JSON schema; return (parsed_obj_or_None, raw_text)."""
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    options = {"num_predict": num_predict} if num_predict and num_predict > 0 else None
    resp = client.chat(model=model, messages=[msg], format=schema, options=options)
    text = resp["message"]["content"].strip()
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        return None, text


def analyze_frames(
    frames: list[tuple[float, Path]],
    model: str,
    host: str | None = None,
    num_predict: int = 0,
) -> list[dict]:
    """Run structured vision analysis on each (timestamp, frame_path).

    Returns one dict per frame: {timestamp, frame_path, data, raw}.
    ``data`` is the parsed structured object (or None if the model returned invalid JSON).
    """
    client = _client(host)
    total = len(frames)
    results: list[dict] = []

    for i, (ts, path) in enumerate(frames):
        print(f"  Analyzing frame {i + 1}/{total} (t={ts:.1f}s)...")
        data, raw = _chat_json(
            client, model,
            FRAME_PROMPT.format(idx=i + 1, total=total, ts=ts),
            schema=FRAME_SCHEMA, images=[str(path)], num_predict=num_predict,
        )
        results.append({"timestamp": ts, "frame_path": str(path), "data": data, "raw": raw})

    return results


def _render_frame_summary(analyses: list[dict]) -> str:
    lines = []
    for a in analyses:
        ts = a["timestamp"]
        d = a["data"]
        if d:
            issues = "; ".join(f"{i['problem']} ({i['severity']})" for i in d.get("issues", [])) or "none"
            lines.append(
                f"[{ts:.1f}s | {d.get('shot_type', '?')} | {d.get('camera_angle', '?')}] "
                f"{d.get('description', '').strip()} | mood: {d.get('mood', '?')} | issues: {issues}"
            )
        else:
            lines.append(f"[{ts:.1f}s] {a['raw'][:300]}")
    return "\n".join(lines)


def synthesize_report(
    analyses: list[dict],
    scene_summary: str,
    audio_transcript: str,
    model: str,
    host: str | None = None,
) -> str:
    """Synthesize the prose review (3 sections) from structured frame analyses."""
    client = _client(host)
    if audio_transcript:
        audio_section = f"\nAUDIO TRANSCRIPT:\n{audio_transcript}\n"
        audio_note = ", along with the audio transcript"
    else:
        audio_section = ""
        audio_note = ""

    prompt = SYNTHESIS_PROMPT.format(
        audio_note=audio_note,
        audio_section=audio_section,
        scene_summary=scene_summary,
        frame_summary=_render_frame_summary(analyses),
    )
    response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


def suggest_bridges(
    scene_summary: str,
    duration: float,
    review_context: str,
    model: str,
    host: str | None = None,
) -> list[dict]:
    """Ask the text model for structured LTX-2.3 bridge-clip suggestions."""
    client = _client(host)
    data, _ = _chat_json(
        client, model,
        BRIDGE_PROMPT.format(
            duration=duration, scene_summary=scene_summary, review_context=review_context,
        ),
        schema=BRIDGE_SCHEMA,
    )
    if not data:
        return []
    bridges = data.get("bridges", [])
    # Clamp any timecodes into range defensively.
    for b in bridges:
        for key in ("first_frame_tc", "last_frame_tc"):
            tc = b.get(key)
            if isinstance(tc, (int, float)):
                b[key] = min(max(0.0, float(tc)), duration)
    return bridges
