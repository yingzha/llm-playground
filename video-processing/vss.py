#!/usr/bin/env python3
"""vss.py — one-liner CLI for VSS-inspired video summarization + multi-turn Q&A.

  python vss.py summarize --video foo.mp4
  python vss.py qa        --video foo.mp4
  python vss.py qa        --video foo.mp4 --questions qs.txt
  python vss.py benchmark --videos "clips/*.mp4" --arms native,local

See README.md for the full flag list and the 5090 (GPU) setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# load .env from repo root and this folder (repo root first, local overrides)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

from src.config import PipelineConfig
from src.runner import run_summarize, run_qa_batch
from src.pipeline_native import NativePipeline
from src.pipeline_local import LocalPipeline
from src.gemini import make_client
from src.profiling import ProfileReport


def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--video", help="path to a video file")
    p.add_argument("--arm", choices=["native", "local"], default="native")
    p.add_argument("--engine", choices=["gemini", "mock"], default="gemini")
    p.add_argument("--decoder", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--sampling", choices=["uniform", "iframe", "scene", "transnet"],
                   default="scene")
    p.add_argument("--chunk-duration", type=float, default=10.0)
    p.add_argument("--chunk-overlap", type=float, default=0.0)
    p.add_argument("--frames-per-chunk", type=int, default=10)
    p.add_argument("--max-frames-per-chunk", type=int, default=20)
    p.add_argument("--scene-threshold", type=float, default=3.0)
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--dedup-hamming", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--gemini-model", default=None, help="override VLM+LLM model id")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output-dir", default="outputs")


def _cfg_from_args(a) -> PipelineConfig:
    cfg = PipelineConfig(
        video=a.video or "",
        arm=a.arm, engine=a.engine, decoder=a.decoder, sampling=a.sampling,
        chunk_duration=a.chunk_duration, chunk_overlap=a.chunk_overlap,
        frames_per_chunk=a.frames_per_chunk, max_frames_per_chunk=a.max_frames_per_chunk,
        scene_threshold=a.scene_threshold, dedup=not a.no_dedup,
        dedup_hamming=a.dedup_hamming, top_k=a.top_k,
        no_cache=a.no_cache, output_dir=a.output_dir,
    )
    if a.gemini_model:
        cfg.vlm_model = cfg.llm_model = a.gemini_model
    return cfg


def _check_video(cfg):
    if not cfg.video or not Path(cfg.video).exists():
        sys.exit(f"error: video not found: {cfg.video!r}")


# --------------------------------------------------------------------------- #
def cmd_summarize(a):
    cfg = _cfg_from_args(a)
    _check_video(cfg)
    summary, report, path = run_summarize(cfg)
    print("\n===== SUMMARY =====\n")
    print(summary)
    print("\n" + report.render())
    print(f"[profile saved: {path}]")
    if a.out:
        Path(a.out).write_text(summary)
        print(f"[summary saved: {a.out}]")


def cmd_qa(a):
    cfg = _cfg_from_args(a)
    _check_video(cfg)
    if a.questions:  # batch mode
        questions = [q.strip() for q in Path(a.questions).read_text().splitlines() if q.strip()]
        qas, report, path = run_qa_batch(cfg, questions)
        for q, ans in qas:
            print(f"\nQ: {q}\nA: {ans}")
        print("\n" + report.render())
        if a.answers_out:
            Path(a.answers_out).write_text(json.dumps(
                [{"question": q, "answer": ans} for q, ans in qas], indent=2))
            print(f"[answers saved: {a.answers_out}]")
        print(f"[profile saved: {path}]")
        return

    # interactive multi-turn loop
    report = ProfileReport(label=f"qa · {cfg.arm} · {Path(cfg.video).name}")
    client = make_client(cfg.engine)
    print(f"Preparing Q&A ({cfg.arm}) — ':quit' to exit, ':reset' to clear history\n")
    with report.stage("total"):
        if cfg.arm == "native":
            pipe = NativePipeline(client, cfg)
            pipe.prepare(report)
            history: list[tuple[str, str]] = []
            while True:
                try:
                    q = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q in (":quit", ":q", ""):
                    break
                if q == ":reset":
                    history.clear(); print("(history cleared)"); continue
                ans = pipe.answer(q, history, report)
                history.append((q, ans))
                print(f"vss> {ans}\n")
        else:
            engine = LocalPipeline(client, cfg).qa_engine(report)
            while True:
                try:
                    q = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q in (":quit", ":q", ""):
                    break
                if q == ":reset":
                    engine.history.clear(); print("(history cleared)"); continue
                ans = engine.answer(q, report)
                print(f"vss> {ans}\n")
    print(report.render())


def cmd_benchmark(a):
    from src.benchmark import run_benchmark
    run_benchmark(a)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("summarize", help="summarize a video")
    _add_common(ps)
    ps.add_argument("--out", default=None, help="write the summary text to a file")
    ps.set_defaults(func=cmd_summarize)

    pq = sub.add_parser("qa", help="multi-turn Q&A over a video")
    _add_common(pq)
    pq.add_argument("--questions", default=None, help="file with one question per line (batch)")
    pq.add_argument("--answers-out", default=None, help="write batch answers to JSON")
    pq.set_defaults(func=cmd_qa)

    pb = sub.add_parser("benchmark", help="run a video × arm × config matrix -> CSV")
    pb.add_argument("--videos", required=True, help="glob of video files")
    pb.add_argument("--arms", default="native,local")
    pb.add_argument("--decoders", default="cpu")
    pb.add_argument("--tasks", default="summarize")
    pb.add_argument("--grid", nargs="*", default=[], help="e.g. sampling=uniform,scene chunk_duration=10,20")
    pb.add_argument("--questions", default=None, help="questions file for qa task")
    pb.add_argument("--engine", choices=["gemini", "mock"], default="gemini")
    pb.add_argument("--gemini-model", default=None)
    pb.add_argument("--output-dir", default="outputs")
    pb.add_argument("--csv-out", default=None)
    pb.set_defaults(func=cmd_benchmark)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
