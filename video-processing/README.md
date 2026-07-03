# video-processing

VSS-inspired **video summarization + multi-turn Q&A**, sized for one machine.
All model inference is a **Gemini** call; **video decode + frame selection run
locally** (CPU today; GPU/NVDEC on the RTX 5090 is a *planned* Phase-2 path, not
yet implemented — see below). Every run prints a **per-stage timing/cost
profile**, and a **benchmark** command compares approaches across video
sizes/lengths.

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
```

### Key flags
`--arm {native,local}` · `--engine {gemini,mock}` · `--decoder {cpu,gpu}` ·
`--sampling {uniform,iframe,scene,transnet}` · `--chunk-duration` ·
`--frames-per-chunk` · `--max-frames-per-chunk` · `--scene-threshold` ·
`--no-dedup` / `--dedup-hamming` · `--top-k` · `--gemini-model <id>` ·
`--no-cache` · `--output-dir`.

> `--decoder gpu` and `--sampling transnet` are Phase-2 placeholders and currently
> raise `NotImplementedError` (see the GPU section). Use `cpu` + `scene`/`iframe`/`uniform`.

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

## GPU (RTX 5090 / Blackwell sm_120) — Phase 2 (NOT yet implemented)

**Current state:** everything runs on **CPU decode**. The GPU paths are
*scaffolded only* — the `--decoder gpu` flag, the `transnet` sampler choice, and
`gpu_check.py` exist, but **`--decoder gpu` and `--sampling transnet` currently
raise a clear `NotImplementedError`**. They are the first task to build (and
validate) when the 5090 is available.

**Intended Phase-2 design:** decode on **NVDEC** (torchcodec / PyNvVideoCodec,
frames stay as CUDA tensors) → cheap **GPU histogram gate** → **TransNetV2** shot
cuts on 48×27 frames → **nvJPEG** encode — keeping frames on-GPU until the final
JPEGs. Planned setup on the 5090:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # sm_120 needs cu128 (cu124 → "no kernel image")
uv pip install torchcodec transnetv2-pytorch
uv run python gpu_check.py                 # expect capability (12, 0), NVDEC + TransNetV2 OK
# then implement the NVDEC decode + transnet selection in src/frames.py
```

Until then use `--decoder cpu` with `--sampling scene` / `iframe` / `uniform`.

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

**Not yet implemented** (Phase 2, to build on the 5090): `--decoder gpu` and
`--sampling transnet` — both currently raise a clear `NotImplementedError`.
