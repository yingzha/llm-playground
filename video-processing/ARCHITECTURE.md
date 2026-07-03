# Architecture — VSS-inspired video summarization + multi-turn Q&A

This folder re-implements the *useful core* of NVIDIA's
[Video Search & Summarization (VSS) blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
as a lean, single-machine, **profiled** pipeline. All model inference is a
**Gemini API** call; the only heavy local work is **video decode + frame
selection**, which runs on CPU today (GPU/NVDEC acceleration on the RTX 5090 is a
planned Phase-2 path, **not yet implemented**).

This doc is both the **study** (how VSS works, stage by stage, mapped to real
source files) and the **build spec** (what each stage became here).

---

## Why not run the real blueprint?

VSS is a Docker/NIM microservice stack (DeepStream/NVDEC decode, Milvus, Neo4j,
a reranker, Nemotron 49B, NeMo Agent Toolkit, a Next.js UI) validated only on
**80 GB HBM data-center GPUs**. That 80 GB comes almost entirely from (a)
DeepStream parallel live-stream ingest and (b) co-locating VLM + LLM + embed +
rerank + Milvus + Neo4j at once. For **file-based summary + Q&A** we need none of
it. Both VSS repos are **Apache-2.0**, so we vendor their prompts + numeric
defaults and drop the infrastructure.

---

## End-to-end pipeline

```
[1] decode → [2] chunk → [3] select frames → [4] VLM caption ─┬─► SUMMARY PATH
                                                              │   [6] map-reduce over captions → summary
                                                              │
                                                              └─► Q&A PATH
                                                                  [5] embed captions → FAISS index
                                                                  [7] (at question time) embed question
                                                                      → retrieve top-k → answer + history
```

Key idea inherited from VSS: the **VLM turns video into text captions once**
(steps 1–4); everything downstream runs on captions, never pixels again.

**The captions then feed TWO independent downstream paths — they do not chain:**

- **Summary path** ([6], `carag.map_reduce_summarize`): batch all captions
  (`batch_size=6`) → **MAP** each batch through the LLM → **REDUCE** the partials
  into one summary. Uses the captions *directly*; **no embeddings, no FAISS**.
  Map-reduce is what lets it scale to long video (hundreds of captions never fit
  one prompt).
- **Q&A path** ([5] index build, [7] query): embed each caption with
  `gemini-embedding` into a **FAISS** vector index. Then at question time, embed
  the *question*, retrieve the `top_k` most similar captions, and answer from only
  those (+ chat history). This is retrieval-augmented Q&A — cheap, focused, and
  scalable. Used **only** by the `qa` command; `summarize` never touches it.

(Arm A `native` skips both paths — Gemini's long context reads the whole video
directly for summary and Q&A.)

---

## Stage → NVIDIA source → our replacement

| # | Stage | NVIDIA source (studied) | Our replacement |
|---|-------|-------------------------|-----------------|
| 1 | Decode | `services/rtvi/rt-embed/.../video_file_frame_getter.py` (GStreamer + DeepStream/NVDEC) | `src/frames.py` — CPU: OpenCV/PyAV (implemented). GPU/NVDEC via torchcodec = Phase 2, **not yet implemented** |
| 2 | Chunk | `chunk_duration`/`chunk_overlap` in `deploy/.../config.yml` | `frames.chunk_ranges()` — same knobs, defaults `chunk_duration=10`, `overlap=0` |
| 3 | Frame select | `DefaultFrameSelector` — "N equally spaced frames per chunk" (**uniform only**) | `src/frames.py` `FrameExtractor` — `uniform` (parity) **+ new** `iframe`, `scene`, `transnet` + phash dedup |
| 4 | VLM caption | `tools/video_understanding.py` (OpenAI multi-image request) + `via_stream_handler._create_vlm_prompt` | `pipeline_local.py` → `gemini.py` — Gemini multi-image call; prompt `prompts/caption.txt` |
| 5 | Embed + store | NeMo Retriever `llama-3.2-nv-embedqa` → **Milvus** | `carag.QAEngine.build_index` → `gemini-embedding` → **FAISS** (`vectorstore.py`, numpy fallback) |
| 6 | Summary (map-reduce) | `vss-ctx-rag/functions/summarization/batch.py` + `Batcher` | `carag.map_reduce_summarize` — same batch→MAP→REDUCE, `batch_size=6` |
| 7 | Multi-turn Q&A | `vss-ctx-rag/utils/prompts.py` (chat + question-transform) + retrieve/rerank | `carag.QAEngine.answer` — condense → FAISS retrieve `top_k=5` → answer + history |
| — | Orchestration | NeMo Agent Toolkit + REST + UI | `vss.py` CLI (`summarize`/`qa`/`benchmark`) |

**Vendored prompts** (`prompts/`, adapted from NVIDIA's Apache-2.0 text, warehouse
domain generalized; attributed in [`prompts/NOTICE.md`](prompts/NOTICE.md)):
`caption.txt`, `map_summarize.txt` (`caption_summarization`),
`reduce_summarize.txt` (`summary_aggregation`), `qa_answer.txt`
(`CHAT_SYSTEM_TEMPLATE`), `question_transform.txt` (`QUESTION_TRANSFORM_TEMPLATE`).
Note: the map-reduce control flow itself is **reimplemented** in `carag.py`, not
copied from NVIDIA's `Batcher`.

**Skipped from VSS:** GStreamer/DeepStream/`pyds` decode, Milvus/Neo4j/Elastic,
the reranker, NeMo Agent Toolkit, Docker/Helm, and the UI.

---

## Two comparison arms

Both use Gemini for every model call; they differ in *who decomposes the video*.

**Arm A — `native`** (`pipeline_native.py`): upload the whole video once →
Gemini samples + summarizes it in one call; multi-turn Q&A reuses the uploaded
file handle + chat history. No local frame extraction, no FAISS. Simplest;
strongest whole-video understanding.

**Arm B — `local`** (`pipeline_local.py`): we decode → chunk → **select frames**
→ send curated frames to Gemini for a per-chunk caption. Those captions then feed
the **two independent paths** described above — map-reduce → summary (for
`summarize`), and embed → FAISS → retrieval (for `qa`). This is the arm with a
**profilable frame-extraction stage**, controllable cost (fewer, better frames),
and the NVIDIA-parity RAG architecture.

The `benchmark` command runs both arms (and, in Arm B, the decode backends and
frame-selection strategies) across a video size×length matrix.

---

## Frame-selection strategies (net-new vs NVIDIA)

VSS ships **uniform sampling only**. Selection is **decode-bound** (the detection
math is nearly free), so efficiency comes from GPU decode + on-GPU selection +
decoding fewer frames — not a fancier algorithm.

| Strategy | Phase | How | Notes |
|----------|-------|-----|-------|
| `uniform` | 1 (CPU) | N equally-spaced frames/chunk | NVIDIA-parity baseline |
| `iframe` | 1 (CPU) | PyAV `skip_frame=NONKEY` (I-frames only) | cheapest; encoder-driven |
| `scene` | 1 (CPU) | PySceneDetect `AdaptiveDetector`, downscaled → sharpest frame per shot | **default**; best quality/cost |
| `transnet` | 2 (GPU) | NVDEC → GPU histogram gate → TransNetV2 shot cuts → nvJPEG | **Phase 2 — not yet implemented** (stub raises `NotImplementedError`) |

All strategies feed an optional **phash dedup** post-filter
(`imagehash.phash`, Hamming ≤ `dedup_hamming`) to drop near-duplicate frames.

Deferred (not built): embedding-based diversity (DINOv2/CLIP + k-means), optical
flow, the stale Katna library.

---

## Profiling (`src/profiling.py`)

`ProfileReport.stage(name, gpu=?)` is a context manager recording per stage:

- **wall time** always (`time.perf_counter`)
- **GPU time + peak VRAM** when `gpu=True` and torch+CUDA are present
  (`cuda.Event`, `reset_peak_memory_stats` on enter, `synchronize` before read)
- **tokens + estimated cost** when the caller sets them (`st.set_tokens(...)`)

Repeated stages accumulate (`vlm_caption` over N chunks = one row, calls=N).
Outputs: a console table, per-run JSON (`outputs/profiles/`), and an aggregated
benchmark CSV (`outputs/benchmark/`). Cost uses the editable `PRICING` table in
`config.py` — **verify against current Gemini pricing**.

Stages: Arm A = `upload`, `vlm_summarize`, `qa_generate`. Arm B = `decode`,
`frame_select`, `vlm_caption`, `embed`, `index_build`, `summarize_map`,
`summarize_reduce`, `retrieve`, `condense`, `qa_generate`. Both wrapped in `total`.

Note on Arm B naming: the **`decode`** stage also includes frame *selection*
(uniform/iframe/scene detection interleave with decoding), and **`frame_select`**
is dedup + JPEG encode — sum the two for total frame-extraction cost. The `embed`
stage's tokens/cost are an **estimate** (chars/4); `embed_content` doesn't
reliably return usage. `condense` only appears on multi-turn Q&A (turn ≥ 2).

---

## Module map

```
vss.py                CLI (summarize | qa | benchmark)
gpu_check.py          5090 preflight (torch cap, NVDEC, TransNetV2)
make_test_video.py    OpenCV synthetic clip (no ffmpeg needed)
src/
  config.py           PipelineConfig dataclass + pricing + cache keys
  profiling.py        StageTimer + ProfileReport
  gemini.py           GeminiClient (google-genai) + MockClient
  frames.py           decode backends × frame-selection strategies + dedup + chunker
  vectorstore.py      FAISS store (numpy fallback)
  carag.py            map-reduce summarize + retrieval multi-turn Q&A
  pipeline_native.py  Arm A
  pipeline_local.py   Arm B (+ caption/index caching per video+config)
  runner.py           shared run_summarize / run_qa_batch
  benchmark.py        matrix runner → CSV/JSON
prompts/              vendored NVIDIA prompts (Apache-2.0)
```
