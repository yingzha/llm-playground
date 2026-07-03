"""Context-Aware RAG (Arm B): map-reduce summarization + retrieval Q&A.

Mirrors NVIDIA's vss-ctx-rag control flow with vendored prompts:
  - summarize: batch per-chunk captions (batch_size) -> MAP each batch -> REDUCE
  - Q&A:       (optional) condense question -> embed -> retrieve top_k -> answer
Everything runs against a Gemini/Mock client; the vector store is local FAISS.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import price_for
from .vectorstore import VectorStore


def _embed_cost(model: str, approx_tokens: int) -> float:
    # embed_content doesn't reliably return usage, so tokens (and thus cost) are an
    # estimate from chars/4 — not exact. Better than showing $0 for a paid call.
    return approx_tokens / 1_000_000.0 * price_for(model)[0]


@lru_cache(maxsize=None)
def _load_prompt(prompts_dir: str, name: str) -> str:
    return (Path(prompts_dir) / name).read_text().strip()


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _timestamped(captions: list[dict]) -> list[str]:
    """Render captions as 'start:end: text' lines the NVIDIA prompts expect."""
    return [f"{_fmt_time(c['start'])}:{_fmt_time(c['end'])}: {c['caption']}" for c in captions]


# --------------------------------------------------------------------------- #
def map_reduce_summarize(client, cfg, captions: list[dict], report) -> str:
    """captions: list of {idx, start, end, caption}."""
    lines = _timestamped(captions)
    map_prompt = _load_prompt(cfg.prompts_dir, "map_summarize.txt")
    reduce_prompt = _load_prompt(cfg.prompts_dir, "reduce_summarize.txt")

    bs = max(cfg.batch_size, 1)
    batches = [lines[i:i + bs] for i in range(0, len(lines), bs)]

    partials: list[str] = []
    for batch in batches:
        with report.stage("summarize_map") as st:
            r = client.generate(cfg.llm_model, ["\n".join(batch)], system=map_prompt,
                                temperature=cfg.temperature, top_p=cfg.top_p,
                                max_output_tokens=cfg.max_output_tokens)
            st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)
        partials.append(r.text)

    if len(partials) == 1:
        # a single batch: the map output already is the summary body; still run
        # reduce to produce the overall-summary paragraph.
        pass

    with report.stage("summarize_reduce") as st:
        r = client.generate(cfg.llm_model, ["\n".join(partials)], system=reduce_prompt,
                            temperature=cfg.temperature, top_p=cfg.top_p,
                            max_output_tokens=cfg.max_output_tokens)
        st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)
    return r.text


# --------------------------------------------------------------------------- #
class QAEngine:
    """Retrieval-augmented multi-turn Q&A over per-chunk captions."""

    def __init__(self, client, cfg, summary: str = ""):
        self.client = client
        self.cfg = cfg
        self.summary = summary
        self.store: VectorStore | None = None
        self.history: list[tuple[str, str]] = []

    # ---- index build ----
    def build_index(self, captions: list[dict], report) -> VectorStore:
        texts = _timestamped(captions)
        with report.stage("embed") as st:
            vecs = self.client.embed(self.cfg.embed_model, texts)
            # embeddings tokens aren't always reported; approximate by chars/4
            approx = sum(len(t) for t in texts) // 4
            st.set_tokens(approx, 0, _embed_cost(self.cfg.embed_model, approx))
        with report.stage("index_build"):
            store = VectorStore(vecs.shape[1])
            metas = [{"start": c["start"], "end": c["end"], "caption": c["caption"]}
                     for c in captions]
            store.add(vecs, metas)
        self.store = store
        return store

    def set_index(self, store: VectorStore):
        self.store = store

    # ---- one turn ----
    def answer(self, question: str, report) -> str:
        query = question
        if self.cfg.condense_question and self.history:
            qt = _load_prompt(self.cfg.prompts_dir, "question_transform.txt")
            hist = "\n".join(f"User: {q}\nAI: {a}" for q, a in self.history)
            with report.stage("condense") as st:
                r = self.client.generate(self.cfg.llm_model,
                                         [qt.format(history=hist, question=question)],
                                         temperature=0.0, top_p=1.0, max_output_tokens=128)
                st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)
            query = (r.text or question).strip() or question

        with report.stage("retrieve") as st:
            qv = self.client.embed(self.cfg.embed_model, [query])[0]
            hits = self.store.search(qv, self.cfg.top_k) if self.store else []
            approx = len(query) // 4
            st.set_tokens(approx, 0, _embed_cost(self.cfg.embed_model, approx))

        ctx_parts = []
        if self.summary:
            ctx_parts.append("Video summary:\n" + self.summary)
        if hits:
            ctx_parts.append("Retrieved captions:\n" + "\n".join(
                f"- {_fmt_time(m['start'])}-{_fmt_time(m['end'])}: {m['caption']}"
                for _, m in hits))
        if self.history:
            ctx_parts.append("Chat history:\n" + "\n".join(
                f"User: {q}\nAI: {a}" for q, a in self.history[-4:]))
        context = "\n\n".join(ctx_parts) if ctx_parts else "(no context)"

        qa_prompt = _load_prompt(self.cfg.prompts_dir, "qa_answer.txt")
        with report.stage("qa_generate") as st:
            r = self.client.generate(
                self.cfg.llm_model,
                [qa_prompt.format(context=context, input=question)],
                temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                max_output_tokens=self.cfg.max_output_tokens)
            st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)

        answer = r.text or ""
        self.history.append((question, answer))
        return answer
