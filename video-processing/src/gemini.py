"""Gemini inference client (VLM + LLM + embeddings) with a no-API mock engine.

Both clients expose the same tiny surface used by the pipelines:
  - upload_video(path)        -> opaque handle (Arm A)
  - generate(model, contents, system=...) -> GenResult(text, tokens, cost)
  - embed(model, texts)       -> np.ndarray [n, d]

`contents` is a list whose items are one of:
  - str                     (text)
  - ("image", jpeg_bytes)   (inline image part)
  - a video handle from upload_video()
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from .config import price_for


@dataclass
class GenResult:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = price_for(model)
    return (in_tok * pin + out_tok * pout) / 1_000_000.0


def make_client(engine: str):
    if engine == "mock":
        return MockClient()
    return GeminiClient()


# ---------------------------------------------------------------------------
class GeminiClient:
    def __init__(self):
        from google import genai  # imported lazily so mock mode needs no SDK/key
        from google.genai import types
        self._genai = genai
        self._types = types
        self.client = genai.Client()  # reads GOOGLE_API_KEY from env

    # ---- video upload (Arm A) ----
    def upload_video(self, path: str):
        f = self.client.files.upload(file=path)
        # poll until the file is ACTIVE (large videos take a few seconds)
        for _ in range(120):
            state = getattr(f.state, "name", str(f.state))
            if state == "ACTIVE":
                return f
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {path}")
            time.sleep(1.0)
            f = self.client.files.get(name=f.name)
        raise TimeoutError(f"Gemini file not ACTIVE after upload: {path}")

    def delete_file(self, handle):
        try:
            self.client.files.delete(name=handle.name)
        except Exception:
            pass

    def _to_parts(self, contents: list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(self._types.Part.from_text(text=item))
            elif isinstance(item, tuple) and item and item[0] == "image":
                parts.append(self._types.Part.from_bytes(data=item[1], mime_type="image/jpeg"))
            else:  # assume an uploaded file handle
                parts.append(item)
        return parts

    def generate(self, model: str, contents: list, system: str | None = None,
                 temperature: float = 0.2, top_p: float = 0.7,
                 max_output_tokens: int = 4096) -> GenResult:
        cfg = self._types.GenerateContentConfig(
            temperature=temperature, top_p=top_p,
            max_output_tokens=max_output_tokens,
            system_instruction=system,
        )
        resp = self.client.models.generate_content(
            model=model,
            contents=self._to_parts(contents),
            config=cfg,
        )
        um = getattr(resp, "usage_metadata", None)
        in_tok = getattr(um, "prompt_token_count", 0) or 0
        out_tok = getattr(um, "candidates_token_count", 0) or 0
        return GenResult(text=resp.text or "", in_tokens=in_tok, out_tokens=out_tok,
                         cost_usd=_cost(model, in_tok, out_tok))

    def embed(self, model: str, texts: list[str]) -> np.ndarray:
        resp = self.client.models.embed_content(model=model, contents=texts)
        vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# ---------------------------------------------------------------------------
class MockClient:
    """Deterministic, offline stand-in — exercises all plumbing without an API."""

    EMBED_DIM = 256

    def upload_video(self, path: str):
        return {"mock_file": path}

    def delete_file(self, handle):
        pass

    def generate(self, model, contents, system=None, temperature=0.2,
                 top_p=0.7, max_output_tokens=4096) -> GenResult:
        n_img = sum(1 for c in contents if isinstance(c, tuple) and c and c[0] == "image")
        texts = [c for c in contents if isinstance(c, str)]
        head = (texts[0][:60].replace("\n", " ") if texts else "").strip()
        time.sleep(0.005)  # tiny sleep so timers register nonzero
        body = f"[mock:{model}] images={n_img} prompt='{head}...'"
        if system and "aggregate" in system.lower():
            body = "00:00:00.0:00:00:05.0: mock aggregated event.\nOverall summary: A mock video."
        elif system and "summarize" in system.lower():
            body = "00:00:00.0:00:00:05.0: mock event description."
        in_tok = 50 + n_img * 258 + sum(len(t) // 4 for t in texts)
        out_tok = 40
        return GenResult(text=body, in_tokens=in_tok, out_tokens=out_tok,
                         cost_usd=_cost(model, in_tok, out_tok))

    def embed(self, model, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.EMBED_DIM).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) or 1.0)
        return out
