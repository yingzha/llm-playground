"""Shared run helpers used by the CLI and the benchmark harness."""

from __future__ import annotations

import time
from pathlib import Path

from .config import PipelineConfig
from .profiling import ProfileReport
from .gemini import make_client
from .pipeline_native import NativePipeline
from .pipeline_local import LocalPipeline


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _save_profile(cfg: PipelineConfig, report: ProfileReport, task: str,
                  tag: str | None = None) -> Path:
    stem = Path(cfg.video).stem
    path = cfg.out / "profiles" / f"{stem}__{tag or cfg.arm}_{task}__{_stamp()}.json"
    report.save_json(path)
    return path


def run_decode(cfg: PipelineConfig):
    """Frame extraction only (decode -> select -> dedup -> JPEG). No API client,
    so it isolates decoder/sampling latency — use for cpu-vs-gpu comparisons."""
    from .frames import FrameExtractor

    report = ProfileReport(
        label=f"decode · {cfg.decoder}/{cfg.sampling} · {Path(cfg.video).name}")
    with report.stage("total"):
        chunks, _ = FrameExtractor(cfg).extract(cfg.video, report)
    profile_path = _save_profile(cfg, report, "decode",
                                 tag=f"{cfg.decoder}-{cfg.sampling}")
    return chunks, report, profile_path


def run_summarize(cfg: PipelineConfig):
    report = ProfileReport(label=f"summarize · {cfg.arm} · {Path(cfg.video).name}")
    report.counter("arm", cfg.arm)
    report.counter("engine", cfg.engine)
    report.counter("vlm_model", cfg.vlm_model)
    client = make_client(cfg.engine)
    with report.stage("total"):
        if cfg.arm == "native":
            summary = NativePipeline(client, cfg).summarize(report)
        else:
            summary = LocalPipeline(client, cfg).summarize(report)
    profile_path = _save_profile(cfg, report, "summarize")
    return summary, report, profile_path


def run_qa_batch(cfg: PipelineConfig, questions: list[str]):
    report = ProfileReport(label=f"qa · {cfg.arm} · {Path(cfg.video).name}")
    report.counter("arm", cfg.arm)
    report.counter("engine", cfg.engine)
    client = make_client(cfg.engine)
    qas: list[tuple[str, str]] = []
    with report.stage("total"):
        if cfg.arm == "native":
            pipe = NativePipeline(client, cfg)
            pipe.prepare(report)
            history: list[tuple[str, str]] = []
            for q in questions:
                a = pipe.answer(q, history, report)
                history.append((q, a))
                qas.append((q, a))
        else:
            engine = LocalPipeline(client, cfg).qa_engine(report)
            for q in questions:
                a = engine.answer(q, report)
                qas.append((q, a))
    profile_path = _save_profile(cfg, report, "qa")
    return qas, report, profile_path
