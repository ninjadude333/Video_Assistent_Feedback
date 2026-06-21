"""End-to-end review pipeline orchestration."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .analyze import analyze_frames, suggest_bridges, synthesize_report
from .config import Config
from .extract import extract_frames_at, probe_video_info
from .scenes import Scene, build_scenes, detect_cuts
from .transcribe import transcribe_audio


def fmt_tc(seconds: float) -> str:
    """Format seconds as a compact timecode (m:ss.s or h:mm:ss.s)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}:{m:02d}:{s:04.1f}"
    return f"{m}:{s:04.1f}"


def _sampling_timestamps(duration: float, scenes: list[Scene], config: Config) -> list[float]:
    """Evenly sample the timeline, ensuring every scene gets at least one frame."""
    if config.frame_interval and config.frame_interval > 0:
        n = max(1, round(duration / config.frame_interval))
    else:
        n = max(1, config.num_frames)
    step = duration / n
    # Center samples within each step (start at step/2) so we never grab the t=0
    # frame, which is frequently a black fade-in and misrepresents the opening scene.
    ts = [(i + 0.5) * step for i in range(n)]

    covered = {next((sc.index for sc in scenes if sc.start <= t < sc.end), None) for t in ts}
    for sc in scenes:
        if sc.index not in covered:
            ts.append(sc.start + sc.duration / 2)

    return sorted({round(t, 3) for t in ts})


def _assign_to_scenes(scenes: list[Scene], analyses: list[dict]) -> None:
    for a in analyses:
        ts = a["timestamp"]
        sc = next((s for s in scenes if s.start <= ts < s.end), scenes[-1])
        a["scene"] = sc.index
        sc.frame_timestamps.append(ts)


def _scene_rows(scenes: list[Scene], analyses: list[dict]) -> list[dict]:
    """Aggregate structured frame data into one row per scene."""
    by_scene: dict[int, list[dict]] = {sc.index: [] for sc in scenes}
    for a in analyses:
        by_scene[a["scene"]].append(a)

    rows = []
    for sc in scenes:
        frames = [a for a in by_scene[sc.index] if a["data"]]
        shots = [a["data"]["shot_type"] for a in frames]
        angles = [a["data"]["camera_angle"] for a in frames if a["data"].get("camera_angle")]
        shot = Counter(shots).most_common(1)[0][0] if shots else "unknown"
        angle = Counter(angles).most_common(1)[0][0] if angles else "—"
        # Summarize from the frame nearest the scene midpoint — the most representative
        # shot, rather than the first frame (which may sit on a cut/fade boundary).
        if frames:
            mid = (sc.start + sc.end) / 2
            rep = min(frames, key=lambda a: abs(a["timestamp"] - mid))
            summary = rep["data"]["description"].strip()
        else:
            summary = "(no analysis)"
        if len(summary) > 120:
            summary = summary[:117].rstrip() + "…"
        rows.append({
            "index": sc.index + 1, "start": sc.start, "duration": sc.duration,
            "shot": shot, "angle": angle, "summary": summary,
        })
    return rows


def _scene_table_md(rows: list[dict], n_cuts: int, threshold: float) -> str:
    note = (
        f"_{len(rows)} scene(s); {n_cuts} hard cut(s) detected (threshold {threshold:g})._\n\n"
        if n_cuts else
        f"_No hard cuts detected (threshold {threshold:g}); treated as a single continuous shot._\n\n"
    )
    lines = [
        "## Scene Timeline", "", note.rstrip(), "",
        "| # | Start | Dur | Shot | Angle | Summary |",
        "|---|-------|-----|------|-------|---------|",
    ]
    for r in rows:
        summary = r["summary"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['index']} | {fmt_tc(r['start'])} | {r['duration']:.1f}s "
            f"| {r['shot']} | {r['angle']} | {summary} |"
        )
    return "\n".join(lines) + "\n"


def _scene_summary_text(rows: list[dict]) -> str:
    """Plain-text scene timeline for feeding into the model prompts."""
    return "\n".join(
        f"Scene {r['index']} [{fmt_tc(r['start'])}, +{r['duration']:.1f}s]: "
        f"{r['shot']}, {r['angle']} — {r['summary']}"
        for r in rows
    )


def _bridges_md(bridges: list[dict], video_path: Path, output_dir: Path, max_dim: int) -> str:
    """Render bridge suggestions and export the referenced conditioning frames full-res."""
    if not bridges:
        return "## 4. Bridge / Transition Suggestions\n\n_No bridging clips suggested._\n"

    lines = ["## 4. Bridge / Transition Suggestions", ""]
    stem = video_path.stem
    for i, b in enumerate(bridges, 1):
        lines.append(f"### Bridge {i} — {b.get('title', 'untitled')}")
        lines.append(f"- **Why:** {b.get('rationale', '').strip()}")
        lines.append(f"- **Suggested length:** {float(b.get('length_seconds', 0)):.1f}s")
        lines.append(f"- **LTX-2.3 i2v prompt:** {b.get('ltx_prompt', '').strip()}")

        for role, key in (("First", "first_frame_tc"), ("Last", "last_frame_tc")):
            tc = b.get(key)
            if isinstance(tc, (int, float)):
                out = output_dir / f"{stem}_bridge{i}_{role.lower()}_{tc:.1f}s.jpg"
                try:
                    extract_frames_at(video_path, output_dir, [float(tc)], max_dim=0)
                    # extract_frames_at names frame_0000.jpg; rename to a descriptive name.
                    produced = output_dir / "frame_0000.jpg"
                    if produced.exists():
                        produced.replace(out)
                    rel = out.name
                    lines.append(f"- **{role} conditioning frame:** {fmt_tc(tc)} → `output/{rel}`")
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"- **{role} conditioning frame:** {fmt_tc(tc)} (extraction failed: {exc})")
        lines.append("")
    return "\n".join(lines)


def run(config: Config) -> Path:
    """Run the full pipeline and write the Markdown report. Returns the report path."""
    info = probe_video_info(config.video_path)
    duration = info["duration"]
    if duration <= 0:
        from .extract import FFmpegError
        raise FFmpegError(f"Could not read a valid duration from {config.video_path}.")
    res = f"{info['width']}x{info['height']}" if info["width"] else "unknown"

    if config.frame_interval and config.frame_interval > 0:
        sampling = f"every {config.frame_interval:g}s"
    else:
        sampling = f"{config.num_frames} frames"

    print(f"🎬 Video:     {config.video_path} ({duration:.1f}s, {res}, {info['fps']:.3g}fps)")
    print(f"👁️  Vision:    {config.vision_model}")
    print(f"📝 Synthesis: {config.synthesis_model}")
    print(f"🖼️  Sampling:  {sampling}")
    if config.max_frame_dim:
        print(f"📐 Max dim:   {config.max_frame_dim}px (longest edge, no upscale)")
    else:
        print(f"📐 Max dim:   native ({res})")
    print(f"🎞️  Scenes:    {'on (threshold ' + format(config.scene_threshold, 'g') + ')' if config.detect_scenes else 'off'}")
    print(f"🎙️  Whisper:   {'on (' + config.whisper_model + ')' if config.use_whisper else 'off'}\n")

    work_dir = Path(tempfile.mkdtemp(prefix="vaf_"))
    try:
        # 1. Scene detection
        cuts = detect_cuts(config.video_path, config.scene_threshold) if config.detect_scenes else []
        scenes = build_scenes(duration, cuts)
        print(f"🎞️  {len(scenes)} scene(s), {len(cuts)} cut(s) detected")

        # 2. Sample + extract frames
        timestamps = _sampling_timestamps(duration, scenes, config)
        frames = extract_frames_at(config.video_path, work_dir / "frames", timestamps, config.max_frame_dim)
        print(f"✅ Extracted {len(frames)} frames from {duration:.1f}s video\n")

        # 3. Optional audio transcription
        transcript = ""
        if config.use_whisper:
            print("🎙️  Transcribing audio...")
            transcript = transcribe_audio(config.video_path, work_dir, config.whisper_model)

        # 4. Structured per-frame vision analysis
        print(f"👁️  Running vision analysis on {len(frames)} frames...")
        analyses = analyze_frames(frames, config.vision_model, config.ollama_host, config.per_frame_tokens)

        # 5. Build scene timeline from structured data
        _assign_to_scenes(scenes, analyses)
        rows = _scene_rows(scenes, analyses)
        scene_summary = _scene_summary_text(rows)
        scene_table = _scene_table_md(rows, len(cuts), config.scene_threshold) if config.detect_scenes else ""

        # 6. Prose synthesis
        print("📝 Synthesizing review...")
        review = synthesize_report(analyses, scene_summary, transcript, config.synthesis_model, config.ollama_host)

        # 7. Bridge suggestions (+ export conditioning frames)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        bridges_md = ""
        if config.detect_scenes:
            print("🌉 Suggesting bridge clips...")
            bridges = suggest_bridges(
                scene_summary, duration, review, config.synthesis_model, config.ollama_host
            )
            bridges_md = "\n" + _bridges_md(bridges, config.video_path, config.output_path.parent, config.max_frame_dim)

        # 8. Assemble + write report
        header = _header(config, info, len(frames), len(scenes), bool(transcript))
        parts = [header]
        if scene_table:
            parts.append(scene_table + "\n")
        parts.append(review + "\n")
        if bridges_md:
            parts.append(bridges_md)
        report = "".join(parts)
        config.output_path.write_text(report.rstrip() + "\n", encoding="utf-8")
        print(f"\n✅ Report saved to {config.output_path}")
        return config.output_path
    finally:
        if config.keep_frames:
            print(f"🗂️  Frames kept in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def _header(config: Config, info: dict, n_frames: int, n_scenes: int, has_audio: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    res = f"{info['width']}x{info['height']}" if info["width"] else "unknown"
    return (
        f"# Video Review: {config.video_path.name}\n\n"
        f"- **Generated:** {now}\n"
        f"- **Duration:** {info['duration']:.1f}s\n"
        f"- **Resolution:** {res}\n"
        f"- **Frame rate:** {info['fps']:.3g} fps\n"
        f"- **Scenes:** {n_scenes}\n"
        f"- **Frames analyzed:** {n_frames}\n"
        f"- **Vision model:** `{config.vision_model}`\n"
        f"- **Synthesis model:** `{config.synthesis_model}`\n"
        f"- **Audio transcript:** {'yes' if has_audio else 'no'}\n\n"
        f"---\n\n"
    )
