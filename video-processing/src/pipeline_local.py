"""Arm B — local preprocessing: decode -> select frames -> Gemini captions each
chunk -> embed -> FAISS -> map-reduce summary + retrieval Q&A.

Per-chunk captions + the FAISS index are cached per (video, config) so `qa` can
reuse `summarize`'s work (and vice-versa) without re-captioning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .frames import FrameExtractor
from .carag import map_reduce_summarize, QAEngine, _load_prompt
from .vectorstore import VectorStore


def _video_hash(path: str) -> str:
    h = hashlib.sha1()
    p = Path(path)
    h.update(str(p.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))  # first 1 MB is plenty to fingerprint
    return h.hexdigest()[:16]


class LocalPipeline:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self._captions: list[dict] | None = None
        self._cache_dir = self._resolve_cache_dir()

    def _resolve_cache_dir(self) -> Path:
        key = self.cfg.cache_key(_video_hash(self.cfg.video))
        return self.cfg.out / "cache" / key

    # ---- caption phase (cache-aware) ----
    def captions(self, report) -> list[dict]:
        if self._captions is not None:
            return self._captions

        cap_file = self._cache_dir / "captions.json"
        if self.cfg.cache and cap_file.exists():
            self._captions = json.loads(cap_file.read_text())
            report.counter("captions_cache", "hit")
            report.counter("num_chunks", len(self._captions))
            return self._captions

        report.counter("captions_cache", "miss")
        extractor = FrameExtractor(self.cfg)
        chunks, info = extractor.extract(self.cfg.video, report)

        caption_prompt = _load_prompt(self.cfg.prompts_dir, "caption.txt")
        captions: list[dict] = []
        for ch in chunks:
            if not ch.frames:
                continue
            contents = [caption_prompt] + [("image", jp) for jp in ch.frames]
            with report.stage("vlm_caption") as st:
                r = self.client.generate(
                    self.cfg.vlm_model, contents,
                    temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                    max_output_tokens=1024)
                st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)
            captions.append({"idx": ch.idx, "start": ch.start, "end": ch.end,
                             "caption": r.text.strip()})

        if self.cfg.cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cap_file.write_text(json.dumps(captions, indent=2))
        self._captions = captions
        return captions

    # ---- summarize ----
    def summarize(self, report) -> str:
        captions = self.captions(report)
        if not captions:
            return "(no captions produced)"
        return map_reduce_summarize(self.client, self.cfg, captions, report)

    # ---- Q&A engine (builds/loads the index) ----
    def qa_engine(self, report, summary: str = "") -> QAEngine:
        captions = self.captions(report)
        engine = QAEngine(self.client, self.cfg, summary=summary)
        idx_path = self._cache_dir / "index"
        if self.cfg.cache and VectorStore.exists(idx_path):
            engine.set_index(VectorStore.load(idx_path))
        else:
            store = engine.build_index(captions, report)
            if self.cfg.cache:
                store.persist(idx_path)
        return engine
