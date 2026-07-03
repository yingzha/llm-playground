"""Pipeline configuration.

A single dataclass holds every knob for both arms so the CLI, pipelines, and
benchmark harness share one source of truth. Model ids default to Gemini and can
be overridden via env vars or CLI flags.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- approximate Gemini pricing (USD per 1M tokens) -------------------------
# VERIFY against current https://ai.google.dev/pricing — these are editable
# placeholders used only for the cost column in the profile. Unknown models
# fall back to DEFAULT_PRICE.
DEFAULT_PRICE = (0.10, 0.40)  # (input, output) per 1M tokens
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3-pro-preview": (1.25, 10.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-embedding-2-preview": (0.15, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    return PRICING.get(model, DEFAULT_PRICE)


@dataclass
class PipelineConfig:
    # --- input ---
    video: str = ""

    # --- arm / engine selection ---
    arm: str = "native"            # native | local
    engine: str = "gemini"         # gemini | mock
    decoder: str = "cpu"           # cpu | gpu   (Arm B only)
    sampling: str = "scene"        # uniform | iframe | scene | transnet  (Arm B only)

    # --- Gemini model ids ---
    vlm_model: str = field(default_factory=lambda: os.environ.get("VSS_VLM_MODEL", "gemini-3.1-flash-lite"))
    llm_model: str = field(default_factory=lambda: os.environ.get("VSS_LLM_MODEL", "gemini-3.1-flash-lite"))
    embed_model: str = field(default_factory=lambda: os.environ.get("VSS_EMBED_MODEL", "gemini-embedding-2-preview"))

    # --- chunking / sampling knobs (NVIDIA-parity defaults) ---
    chunk_duration: float = 10.0   # seconds per chunk
    chunk_overlap: float = 0.0
    frames_per_chunk: int = 10     # uniform sampling count
    max_frames_per_chunk: int = 20 # cap for scene/keyframe selection
    scene_threshold: float = 3.0   # PySceneDetect AdaptiveDetector adaptive_threshold
    detect_downscale: int = 2      # downscale factor for scene detection (speed)
    dedup: bool = True             # phash near-duplicate dedup
    dedup_hamming: int = 10        # max Hamming distance to treat frames as duplicates
    jpeg_max_side: int = 768       # resize longest side before sending to Gemini

    # --- summarization / retrieval ---
    batch_size: int = 6            # map-reduce batch size (NVIDIA default)
    top_k: int = 5                 # retrieval count for Q&A
    condense_question: bool = True # multi-turn: rewrite Q to standalone query
    temperature: float = 0.2
    top_p: float = 0.7
    max_output_tokens: int = 4096

    # --- io / cache ---
    output_dir: str = "outputs"
    cache: bool = True             # reuse captions/index per (video, config)
    no_cache: bool = False

    # --- prompts dir (resolved relative to package root by default) ---
    prompts_dir: str = ""

    def __post_init__(self) -> None:
        if not self.prompts_dir:
            self.prompts_dir = str(Path(__file__).resolve().parent.parent / "prompts")
        if self.no_cache:
            self.cache = False

    # config fingerprint used in cache keys so different knobs don't collide
    def cache_key(self, video_hash: str) -> str:
        relevant = {
            k: getattr(self, k)
            for k in (
                "arm", "engine", "decoder", "sampling", "vlm_model", "embed_model",
                "chunk_duration", "chunk_overlap", "frames_per_chunk",
                "max_frames_per_chunk", "scene_threshold", "dedup", "dedup_hamming",
                "jpeg_max_side",
            )
        }
        blob = video_hash + repr(sorted(relevant.items()))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
