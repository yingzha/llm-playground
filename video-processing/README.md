# video-processing

VSS-inspired **video summarization + multi-turn Q&A**, sized for one machine.
All model inference is a **Gemini** call; **video decode + frame selection run
locally** — on CPU anywhere, or on the RTX 5090 (**NVDEC decode + TransNetV2
shot detection**, see the GPU section). Every run prints a **per-stage
timing/cost profile**, and a **benchmark** command compares approaches across
video sizes/lengths.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how this maps to NVIDIA's VSS blueprint.

---

## Setup (uv)

```bash
cd video-processing
uv sync                                   # creates .venv, installs CPU deps
cp .env.template .env                     # then add GOOGLE_API_KEY (repo-root .env is also read)
```

Only `GOOGLE_API_KEY` is required — the whole thing runs today, no GPU needed.
No system ffmpeg required (OpenCV/PyAV bundle their own codecs).

Run commands with `uv run python ...` (or `source .venv/bin/activate` first).

---

## One-liners

```bash
# make a synthetic test clip (OpenCV, no ffmpeg)
uv run python make_test_video.py --seconds 20 --scenes 3 --out outputs/test.mp4

# summarize — Arm A (Gemini reads the whole video)
uv run python vss.py summarize --video outputs/test.mp4

# summarize — Arm B (local frame extraction → Gemini captions → map-reduce)
uv run python vss.py summarize --video outputs/test.mp4 --arm local --sampling scene

# multi-turn Q&A (interactive; ':quit' / ':reset')
uv run python vss.py qa --video outputs/test.mp4

# batch Q&A → JSON
uv run python vss.py qa --video outputs/test.mp4 --questions qs.txt --answers-out outputs/answers.json

# offline plumbing test (no API, no key)
uv run python vss.py summarize --video outputs/test.mp4 --engine mock

# benchmark: arms × samplers across your clips → CSV
uv run python vss.py benchmark --videos "outputs/*.mp4" --arms native,local \
    --tasks summarize --grid sampling=uniform,scene chunk_duration=10,20

# decoder-only latency (frame extraction, no API calls / key needed)
uv run python vss.py decode --video outputs/test.mp4 --decoder gpu --sampling uniform
uv run python vss.py benchmark --videos "outputs/*.mp4" --arms local \
    --decoders cpu,gpu --tasks decode --grid sampling=uniform
```

### Key flags
`--arm {native,local}` · `--engine {gemini,mock}` · `--decoder {cpu,gpu}` ·
`--sampling {uniform,iframe,scene,transnet}` · `--chunk-duration` ·
`--frames-per-chunk` · `--max-frames-per-chunk` · `--scene-threshold` ·
`--transnet-threshold` · `--no-dedup` / `--dedup-hamming` · `--top-k` ·
`--gemini-model <id>` · `--no-cache` · `--output-dir`.

> Valid (decoder, sampling) combos — **cpu:** `uniform` / `iframe` / `scene` ·
> **gpu:** `uniform` / `transnet`. Anything else raises a `ValueError`:
> `transnet` needs the GPU (dense every-frame decode + CUDA TransNetV2), while
> `iframe`/`scene` are bound to CPU libraries (PyAV / PySceneDetect).

---

## Outputs

- `outputs/profiles/<video>__<arm>_<task>__<ts>.json` — per-run stage profile
- `outputs/benchmark/bench_<ts>.csv` / `.json` — benchmark matrix
- `outputs/cache/<key>/` — captions + FAISS index, reused across runs (`--no-cache` to bypass)

The console profile table breaks out every stage (decode, frame_select,
vlm_caption, embed, summarize_map/reduce, retrieve, qa_generate) with wall time,
tokens, and estimated cost. **Verify the pricing table in `src/config.py`
(`PRICING`) against current Gemini pricing** — the cost column depends on it.

---

## Which arm / sampler?

- **Arm A (`native`)** — best whole-video understanding, fewest moving parts.
  Frame sampling is internal to Gemini (not separately profilable).
- **Arm B (`local`)** — profilable frame extraction, controllable cost, retrieval
  Q&A, scales to long video. After per-chunk captioning it runs **two independent
  paths** (see [ARCHITECTURE.md](ARCHITECTURE.md#end-to-end-pipeline)):
  `summarize` = map-reduce over the captions (no embeddings); `qa` = embed
  captions → FAISS → retrieve top-k per question. `scene` sampling sends far
  fewer, more-informative frames than `uniform` (lower token cost); `iframe` is
  cheapest to decode. Trade-off: aggressive sampling can thin out Q&A context —
  use the benchmark to find the sweet spot for your videos.

---

## GPU (RTX 5090 / Blackwell sm_120)

Two GPU paths, both via **torchcodec NVDEC** (frames decode straight to CUDA
tensors):

- `--decoder gpu --sampling uniform` — NVDEC decode with the same timestamp
  math as the CPU path. With identical sampling, `benchmark --decoders cpu,gpu
  --tasks decode` isolates pure decode speedup (no API calls; see also the
  `decode` subcommand for one-off runs).
- `--decoder gpu --sampling transnet` — the full GPU pipeline: **NVDEC dense
  decode** → 48×27 GPU resize → **TransNetV2** per-frame shot-transition
  probabilities (windows of 100 frames, 25-frame overlap, cut at
  `--transnet-threshold`, default 0.5) → sharpest-of-3 candidates per shot via
  **GPU Laplacian variance** → CPU JPEG.

Design notes: the originally sketched *histogram gate* was dropped (TransNetV2
on 48×27 frames is negligible next to decode, which is paid either way), and
JPEG encode deliberately **stays on CPU** — only ~1 frame per shot survives
selection, and keeping dedup+encode identical across decoders makes the
cpu-vs-gpu `decode` profile rows directly comparable. GPU runs populate the
`gpu_ms` / `vram_gb` profile columns.

Setup on the 5090 box (after `uv sync`):

```bash
uv pip install torch torchcodec --index-url https://download.pytorch.org/whl/cu130  # matched pair; driver >= 580
uv pip install transnetv2-pytorch
uv run python gpu_check.py outputs/test.mp4  # capability (12,0), real NVDEC decode, TransNetV2 forward
```

Use **torchcodec >= 0.14 (cu130)**. The older cu128 pairing (torchcodec 0.11 +
`nvidia-npp-cu12`) *silently returns garbage frames* (constant solid color) from
`device="cuda"` on this setup — pipeline runs, shot detection finds nothing.
`gpu_check.py`'s NVDEC content test exists precisely to catch this.

torchcodec also dlopens **FFmpeg shared libs** (versions 4–8) and uses FFmpeg's
CUDA hwaccel for NVDEC. **Ubuntu 24.04's system FFmpeg is built *without*
NVDEC/CUDA support** (no `--enable-nvdec` — `apt install ffmpeg` does not help),
which also yields garbage GPU frames. Fix without sudo: stage an NVDEC-enabled
shared build (e.g. [BtbN's](https://github.com/BtbN/FFmpeg-Builds/releases)
`linux64-gpl-shared`) into `.venv/lib/ffmpeg/`:

```bash
curl -sL -o ff.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1.tar.xz
mkdir -p ffx .venv/lib/ffmpeg && tar -xf ff.tar.xz -C ffx --strip-components=1
cp -a ffx/lib/*.so* .venv/lib/ffmpeg/ && rm -rf ffx ff.tar.xz
```

`_preload_gpu_libs()` in `src/frames.py` preloads that directory (plus NPP from
a pip wheel, if present, for older torchcodec) automatically at `--decoder gpu`
startup — no `LD_LIBRARY_PATH` needed.

> **Gotchas:** (1) a plain `uv sync` *removes* these manually-installed wheels —
> re-install after any sync, or use `uv sync --inexact` (`uv run` alone is safe;
> the staged `.venv/lib/ffmpeg/` survives either way). (2) Generate test clips
> with `--codec h264` (`make_test_video.py --codec h264`) — H.264 is the
> safest NVDEC codec (mp4v also worked on the cu130 stack, but is unverified
> elsewhere).

---

## Status — validated

Smoke-tested on macOS (CPU) against a synthetic clip:

- **Mock engine:** native + local (`uniform`/`iframe`/`scene`), batch Q&A, benchmark — all pass.
- **Real Gemini (`gemini-3.1-flash-lite` + `gemini-embedding-2-preview`):**
  native summarize, local summarize, and local multi-turn Q&A (embeddings +
  FAISS retrieval + question condensing) all produce correct results with full
  per-stage cost profiles. Example: on the 20s clip, `scene` sampling sent 4
  frames vs `uniform`'s ~16–33 — a large token-cost difference the profile makes
  visible.

GPU paths validated on the RTX 5090 (Ubuntu 24.04, driver 595.71, torch
2.12.1+cu130 + torchcodec 0.14.0+cu130 + transnetv2-pytorch 1.0.5, BtbN FFmpeg
7.1 staged in `.venv/lib/ffmpeg/`), mock engine, synthetic clips:

- **`gpu`+`transnet`** finds exactly the right shot count (6/6 and 8/8 scene
  cuts on 60s/180s clips), one sharp frame per shot; `decode` rows report
  `gpu_ms` + `vram_gb` (e.g. 2.69 s / 3.6 GB for a dense 4500-frame 720p decode
  + TransNetV2 — faster than CPU PySceneDetect's 3.22 s on the same clip).
- **`gpu`+`uniform`** matches CPU selection; GPU decode wins once clips get
  long/large (720p 3 min: 2.43 s vs 2.90 s total) and loses on tiny SD clips
  (NVDEC init + sparse-seek overhead dominates).
- Invalid combos (`gpu`+`scene`, `cpu`+`transnet`) fail fast with clear errors;
  mp4v decoded correctly on NVDEC with the cu130 stack.
