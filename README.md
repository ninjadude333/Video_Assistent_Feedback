# video-assistant-feedback

A local, fully offline pipeline for reviewing **video of any style** — AI-generated animation, live-action, motion graphics — using Ollama vision models. Drop a video in `input/`, run one command, and get a detailed Markdown report in `output/` covering synopsis, technical issues (timestamped + prioritized), and directorial feedback.

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
    ├─► ffmpeg ─────────────────► N evenly-spaced JPEG frames (default 30)
    │
    ├─► ffmpeg + Whisper ───────► audio transcript (optional, --whisper)
    │
    ├─► Ollama vision model ────► per-frame analysis (description + issues + tone)
    │   (qwen3-vl:30b)
    │
    └─► Ollama text model ──────► synthesized Markdown report
        (qwen3.6:35b)               1. Synopsis
                                    2. Issues to Fix (timestamped, prioritized)
                                    3. General Feedback
        → output/<video>.md
```

1. **Frame analysis pass** — each frame gets a short, style-agnostic prompt (visual description, quality-issue detection, mood). Keeping it scoped reduces hallucination.
2. **Synthesis pass** — all per-frame analyses (plus transcript, if any) are concatenated into one large-context text prompt. A stronger text model writes the final report.

---

## Models

Defaults target the locally available Ollama models on the DGX1:

| Role | Default | Notes |
|---|---|---|
| Vision | `qwen3-vl:30b` | Best local vision-language model (~19 GB). Override with `--model`. |
| Synthesis | `qwen3.6:35b` | Strongest local text model for report writing. Override with `--synth-model`. |

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
| `--synth-model` | `qwen3.6:35b` | Ollama text model for synthesis |
| `--frames` | `30` | Frames to sample |
| `--interval` | — | Sample one frame every N seconds (overrides `--frames`) |
| `--max-dim` | `1280` | Cap longest frame edge in px, never upscales (`0` disables) |
| `--per-frame-tokens` | unlimited | Cap output tokens per frame (faster analysis) |
| `--fast` | off | Draft preset: `qwen3-vl:8b`, ~1 frame/3s, 1024px, 250-token cap |
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
- **Lighter model** — `--model qwen3-vl:8b` (or `--fast`) for quick drafts; keep `qwen3-vl:30b` for final QC.

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

A Markdown report with a metadata header followed by three sections:

```markdown
# Video Review: my_clip.mp4

- Generated: 2026-06-21 11:45 UTC
- Duration: 300.0s
- Frames analyzed: 30
- Vision model: qwen3-vl:30b
...

## 1. SYNOPSIS
...3-6 sentence description of content, visual style, tone...

## 2. ISSUES TO FIX
- **[~0:42] Flickering edge artifact** — ... Suggest a fix.

## 3. GENERAL FEEDBACK
...what works, what doesn't, one key recommendation...
```

---

## Known limitations

- **No true temporal understanding** — frames are analyzed independently, so per-frame artifacts are caught but subtle motion issues (micro-jitter, easing) may not be.
- **Hallucination risk on sparse frames** — use ≥ 30 frames for a reliable synopsis.
- **No spatial localization** — issues are timestamped but not located within the frame (Ollama vision API has no bounding-box output).

---

## Roadmap

- [ ] Scene-cut-aware frame extraction (ffmpeg scene detection) for better coverage at cuts
- [ ] Per-scene sub-reports
- [ ] HTML report with embedded frame thumbnails per issue
- [ ] Comparison mode (`--compare v1.mp4 v2.mp4`) to diff two renders
- [ ] DaVinci Resolve `.edl` marker export for timestamped issues
