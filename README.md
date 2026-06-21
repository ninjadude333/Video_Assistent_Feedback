# video-assistant-feedback

A local, fully offline pipeline for reviewing **video of any style** — AI-generated animation, live-action, motion graphics — using Ollama vision models. Drop a video in `input/`, run one command, and get a detailed Markdown report in `output/` covering a **scene timeline**, synopsis, technical issues (timestamped + prioritized), directorial feedback, and **LTX-2.3 bridge-clip suggestions**.

Built to run on a DGX1 alongside a local Ollama install.

---

## Quickstart

```bash
# 1. Install (uv-managed)
uv sync

# 2. Drop a video into input/
cp my_clip.mp4 input/

# 3. Run — auto-discovers the single video in input/, writes output/my_clip.md
uv run video-review
```

That's the MVP: one command in, one detailed `.md` report out.

---

## How it works

Two-pass design for coherent output:

```
input/video.mp4
    │
    ├─► ffmpeg scene detect ────► scene cut timestamps → scene segments
    │
    ├─► ffmpeg ─────────────────► frames sampled across the timeline
    │                               (≥1 per scene; downscaled if --max-dim)
    │
    ├─► ffmpeg + Whisper ───────► audio transcript (optional, --whisper)
    │
    ├─► Ollama vision model ────► structured per-frame analysis (JSON)
    │   (qwen3-vl:30b)              shot type, angle, subjects, mood, issues
    │
    ├─► Ollama text model ──────► synthesized Markdown review
    │   (reuses qwen3-vl:30b)       1. Synopsis  2. Issues  3. Feedback
    │
    └─► Ollama text model ──────► LTX-2.3 bridge-clip suggestions
        (reuses qwen3-vl:30b)       4. Bridges (prompt + length + frame TCs)
                                       → conditioning frames exported to output/
        → output/<video>.md
```

1. **Scene detection** — ffmpeg's scene filter finds hard cuts; the timeline is split into scene segments.
2. **Structured frame analysis** — each sampled frame is classified into JSON (shot type, camera angle, subjects, mood, issues w/ severity). This builds a deterministic, verifiable **scene reference table**.
3. **Synthesis pass** — the structured analyses (plus transcript, if any) feed a text model that writes the prose review.
4. **Bridge pass** — the model proposes optional transition clips for an **LTX-2.3 image-to-video** workflow: a ready-to-paste i2v prompt, clip length, and the timecodes of existing frames to use as first/last conditioning images. Those frames are exported full-res to `output/`.

---

## Models

Defaults target the locally available Ollama models on the DGX1:

| Role | Default | Notes |
|---|---|---|
| Vision | `qwen3-vl:30b` | Best local vision-language model (~19 GB). Override with `--model`. |
| Synthesis | *reuses `--model`* | Same model writes the report + bridges, so only one model stays loaded. On these V100s a separate 35B text pass is very slow and forces a second model load. Pass `--synth-model qwen3.6:35b` if you want max report quality and can spare the time. |

Lighter vision options if you need faster iteration: `qwen3-vl:8b`, `qwen2.5vl:7b`.

Pull whatever you intend to use:

```bash
ollama pull qwen3-vl:30b
ollama pull qwen3.6:35b
```

---

## Prerequisites

- **ffmpeg** (provides `ffmpeg` + `ffprobe`) on `PATH`
- **uv** for environment management
- **Ollama** running locally with the chosen models pulled (`ollama serve`)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Usage

```bash
# Auto-discover the single video in input/
uv run video-review

# Explicit video path
uv run video-review input/other.mp4

# More frames + audio transcription + custom output
uv run video-review input/clip.mp4 --frames 60 --whisper --output output/review_v3.md

# Fast draft: lighter model, coarser sampling, capped per-frame output
uv run video-review --fast

# Sample one frame every 4 seconds instead of a fixed count
uv run video-review --interval 4
```

### Options

| Flag | Default | Description |
|---|---|---|
| `video` (positional) | auto-discover `input/` | Path to the video file |
| `--input-dir` | `input` | Where to auto-discover the video |
| `--output-dir` | `output` | Where to write the report |
| `--output` | `output/<video>.md` | Explicit report path |
| `--model` | `qwen3-vl:30b` | Ollama vision model |
| `--synth-model` | reuses `--model` | Text model for synthesis + bridges |
| `--frames` | `30` | Frames to sample |
| `--interval` | — | Sample one frame every N seconds (overrides `--frames`) |
| `--max-dim` | `1280` | Cap longest frame edge in px, never upscales (`0` disables) |
| `--per-frame-tokens` | unlimited | Cap output tokens per frame (faster analysis) |
| `--fast` | off | Draft preset (same vision model): ~1 frame/3s, 1024px, 250-token cap |
| `--scene-threshold` | `0.4` | Scene-cut sensitivity, 0–1, lower = more cuts |
| `--no-scenes` | off | Disable scene timeline + bridge suggestions |
| `--whisper` | off | Transcribe audio (needs the `whisper` extra) |
| `--whisper-model` | `base` | Whisper size: tiny/base/small/medium/large |
| `--ollama-host` | env `OLLAMA_HOST` | Ollama server URL |
| `--keep-frames` | off | Keep extracted frames for debugging |

### Audio transcription (optional)

Whisper is off by default and not installed by default (it pulls `torch`). Enable it:

```bash
uv sync --extra whisper
uv run video-review --whisper
```

### Frame count guide

| `--frames` | Coverage (5-min video) | Use case |
|---|---|---|
| 15 | ~1 per 20s | Rapid story-level pass |
| 30 | ~1 per 10s | **Default — recommended** |
| 60 | ~1 per 5s | Detailed artifact hunting |
| 150 | ~1 per 2s | Near-frame-accurate QC |

### Speed & resolution

The vision pass is the bottleneck (each frame is a separate model call). Levers, fastest payoff first:

- **Fewer frames** — `--interval`/`--frames`. The dominant cost is per-frame model calls.
- **Cap per-frame output** — `--per-frame-tokens`. Much of the time is the model *writing* the description; capping it helps a lot at low resolution.
- **`--max-dim`** — only helps for **high-res sources** (1080p/4K). Vision models tokenize by pixel count, so capping the longest edge to ~1024–1280px cuts tokens with negligible QC loss. For footage already ≤720–1080p this is a no-op — and downscaling *below* the source resolution will start erasing the fine artifacts (flicker edges, compositing halos) the QC pass exists to catch, so don't force it lower than needed.
- **Lighter model** — `--model qwen3-vl:8b` for quick drafts; keep `qwen3-vl:30b` for final QC. Note: switching models forces an Ollama model load, which on this box is itself a large cost — so `--fast` deliberately keeps the same vision model and economizes on resolution + frame count instead.

---

## Project structure

```
video-assistant-feedback/
├── pyproject.toml                  # uv project + console-script entry point
├── requirements.txt                # mirror of core deps (uv sync is canonical)
├── input/                          # drop your video here (contents gitignored)
├── output/                         # reports land here (contents gitignored)
└── src/video_assistant_feedback/
    ├── cli.py                      # argument parsing + input discovery
    ├── config.py                   # Config dataclass + defaults
    ├── extract.py                  # ffmpeg/ffprobe frame & audio extraction
    ├── transcribe.py               # optional Whisper transcription
    ├── analyze.py                  # Ollama vision + synthesis calls
    └── pipeline.py                 # end-to-end orchestration
```

---

## Output

A Markdown report with a metadata header, a scene timeline table, the prose review, and bridge suggestions:

```markdown
# Video Review: my_clip.mp4

- Generated: 2026-06-21 11:45 UTC
- Duration: 47.1s
- Scenes: 3
- Frames analyzed: 18
...

## Scene Timeline
| # | Start | Dur  | Shot   | Angle      | Summary                       |
|---|-------|------|--------|------------|-------------------------------|
| 1 | 0:00  | 12.3s| medium | eye-level  | Two boys walk along a side... |
| 2 | 0:12  | 18.0s| wide   | low-angle  | Dog runs across the lawn...   |

## 1. SYNOPSIS
...3-6 sentence description of content, visual style, tone...

## 2. ISSUES TO FIX
- **[~0:42] Flickering edge artifact** — ... Suggest a fix.

## 3. GENERAL FEEDBACK
...what works, what doesn't, one key recommendation...

## 4. Bridge / Transition Suggestions
### Bridge 1 — Establishing beat before Scene 2
- Why: smooths the abrupt cut from sidewalk to lawn.
- Suggested length: 3.0s
- LTX-2.3 i2v prompt: "Slow push-in on a sunlit suburban lawn, dog trotting..."
- First conditioning frame: 0:11.5 → output/my_clip_bridge1_first_11.5s.jpg
- Last conditioning frame: 0:12.5 → output/my_clip_bridge1_last_12.5s.jpg
```

The **scene timeline** is built from the model's structured output, so it's a quick reference for verifying the model read each shot correctly. The exported bridge frames are full-resolution (ignoring `--max-dim`) so they're usable directly as i2v conditioning images.

---

## Known limitations

- **No true temporal understanding** — frames are analyzed independently, so per-frame artifacts are caught but subtle motion issues (micro-jitter, easing) may not be.
- **Hallucination risk on sparse frames** — use ≥ 30 frames for a reliable synopsis.
- **No spatial localization** — issues are timestamped but not located within the frame (Ollama vision API has no bounding-box output).

---

## Roadmap

- [x] Scene-cut-aware frame extraction (ffmpeg scene detection)
- [x] Scene timeline table (shot type, angle, duration) for verification
- [x] LTX-2.3 bridge-clip suggestions with exported conditioning frames
- [ ] Per-scene sub-reports
- [ ] HTML report with embedded frame thumbnails per issue
- [ ] Comparison mode (`--compare v1.mp4 v2.mp4`) to diff two renders
- [ ] DaVinci Resolve `.edl` marker export for timestamped issues
