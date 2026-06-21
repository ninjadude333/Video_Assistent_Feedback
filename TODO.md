# TODO / Feature Backlog

Candidate features for video-assistant-feedback, grouped by priority. Each item notes
**why** it matters for the AI-generated-animation QC workflow, a rough **how**, and **effort**.

---

## 🥇 Quick wins (low effort, high daily value — do these first)

- [ ] **Machine-readable JSON sidecar** — write `output/<name>.json` next to the `.md`
  (issues with severity, timestamp, scene, shot type).
  - *Why:* unlocks automation — CI quality gates, dashboards, and the "Claude Code drives
    the next ComfyUI iteration" loop from the README. Data is already structured internally.
  - *How:* serialize the structured analyses + issues. **Effort: low.**

- [ ] **Analysis caching + resume / re-synthesize-only** — cache per-frame structured
  results keyed by (video hash, timestamp, model); add `--resynth` to regenerate the
  report from cache without re-running the vision pass.
  - *Why:* full runs are ~24 min; re-running everything just to re-word the report (or
    after a synthesis failure) is painful.
  - *How:* JSON cache under a cache dir; skip vision calls on hit. **Effort: low.**

- [ ] **HTML report with embedded frame thumbnails** — emit an `.html` with the scene
  table and each issue's frame inlined (base64).
  - *Why:* issues are timestamped but not visual; seeing the frame makes verification
    instant (the stated "verify the model got it right" goal).
  - *How:* template + base64-embedded JPEGs. **Effort: low–medium.**

- [ ] **Cheap classical-CV pre-pass (no LLM)** — detect pure-black/frozen frames, color
  banding, blown exposure, blur/sharpness, letterboxing.
  - *Why:* some defects are caught better deterministically and for free (we hit a black
    t=0 frame); also primes the vision pass.
  - *How:* small OpenCV/numpy pass over extracted frames. **Effort: low.**

---

## 🎯 High-value accuracy upgrades

- [ ] **Temporal-aware analysis (frame *windows*, not single frames)** — TOP accuracy pick.
  - *Why:* the #1 limitation; single-frame judging caused the animation→"live-action"
    misread. Flicker, jitter, morphing, PuLID face-popping are *between-frame* artifacts
    the current pipeline structurally cannot see.
  - *How:* send 2–3 consecutive frames as one multi-image message ("what changes between
    these; any flicker/jitter/identity drift?"). **Effort: medium.**

- [ ] **Character & style consistency tracking (embeddings)** — quantify identity/palette/
    style drift across frames.
  - *Why:* consistency is the dominant failure mode for AI animation (identity drift,
    PuLID ghosting). Nothing measures it today.
  - *How:* CLIP/image embeddings per frame (or per face crop), flag outliers + drift; add a
    "Consistency" section. **Effort: medium.**

- [ ] **Spatial localization (bounding boxes)** — "...top-left edge of the dog" instead of
    just a timestamp.
  - *Why:* makes artifact fixes precise. Roadmap item.
  - *How:* Qwen-VL grounding via the Transformers API (bypass Ollama), or ask for
    normalized coords. **Effort: high (separate inference path).**

- [ ] **Animation-context prompt hint** — configurable note (e.g. "this is AI-generated
    animation") to steer style classification.
  - *Why:* locks in correct style reads even on sparse runs (the misread we saw).
  - *How:* optional `--context` string injected into the per-frame prompt. **Effort: low.**

---

## 🔁 Iteration & workflow

- [ ] **Version comparison — `--compare v1.mp4 v2.mp4`** — diff two renders.
  - *Why:* the core loop is render → review → re-render; the question is always "did my fix
    work / did I regress?" Roadmap item.
  - *How:* align scenes by timecode, diff structured issue lists → fixed / still present /
    new. **Effort: medium.**

- [ ] **Adaptive scene-aware sampling** — denser sampling in high-motion/high-change
    scenes, sparser in static ones.
  - *Why:* spends the expensive inference budget where change happens; better coverage at
    the same cost.
  - *How:* weight density by ffmpeg per-frame scene scores (already invoked). **Effort: medium.**

- [ ] **ComfyUI / LTX feedback hooks** — per-issue regen guidance (which nodes/params to
    adjust: inpainting mask, sampler, PuLID weight) or a ready fix-prompt.
  - *Why:* closes the generation loop end-to-end.
  - *How:* map issue types → actionable regen suggestions; optionally emit ComfyUI-ready
    params. **Effort: high / exploratory.**

- [ ] **Audio analysis + A/V sync (Whisper)** — validate the `--whisper` path; detect
    silence/gaps and audio-visual sync issues (you use MMAudio / ACE-Step).
  - *Why:* audio is currently untested and unanalyzed beyond transcription.
  - *How:* exercise whisper extra; add sync/silence checks. **Effort: medium.**

---

## 📤 Output & NLE integration

- [ ] **NLE marker export (Resolve / Premiere)** — EDL / FCPXML / CSV markers from issues.
  - *Why:* jump straight to each flagged timecode in the editor instead of eyeballing.
    Roadmap item.
  - *How:* map issues → timeline markers, colour-by-severity; FCPXML or Resolve-CSV are the
    easiest targets. **Effort: low–medium.**

- [ ] **Per-scene sub-reports** — optional per-scene detail breakouts. *(Roadmap)*
- [ ] **First/last frame export for ALL scenes** — not just bridges; handy as i2v
    conditioning anchors across the whole cut. **Effort: low.**

---

## ✅ Already shipped

- [x] uv-managed package + one-command CLI (`video-review`), auto-discover `input/` → `output/<name>.md`
- [x] Two-pass pipeline (structured per-frame vision + synthesis), style-agnostic
- [x] Scene-cut detection + scene timeline table (shot type, angle, duration)
- [x] LTX-2.3 bridge-clip suggestions with exported full-res conditioning frames
- [x] Single resident model for all passes (synthesis reuses vision model)
- [x] Presets & controls: `--fast`, `--full`, `--interval`, `--max-dim`, `--per-frame-tokens`, `--scene-threshold`, `--no-scenes`, `--whisper`
- [x] Dynamic resolution + fps probing in the report header
