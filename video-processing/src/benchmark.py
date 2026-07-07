"""Benchmark harness: run a video × arm × decoder × grid matrix, emit a
comparison CSV/JSON so we can see how each stage scales with video size/length.
"""

from __future__ import annotations

import csv
import glob
import itertools
import json
import time
import traceback
from pathlib import Path

from .config import PipelineConfig
from .runner import run_decode, run_summarize, run_qa_batch


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v


def _parse_grid(tokens: list[str]) -> dict[str, list]:
    grid: dict[str, list] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        key, vals = tok.split("=", 1)
        grid[key.strip()] = [_coerce(v) for v in vals.split(",")]
    return grid


def _flatten(report_dict: dict, base: dict) -> dict:
    row = dict(base)
    row.update({f"counter.{k}": v for k, v in report_dict["counters"].items()})
    for st in report_dict["stages"]:
        n = st["name"]
        if n == "total":
            continue
        row[f"{n}.wall_s"] = round(st["wall_s"], 4)
        if st["gpu_ms"]:
            row[f"{n}.gpu_ms"] = round(st["gpu_ms"], 2)
        if st["peak_vram_gb"]:
            row[f"{n}.vram_gb"] = round(st["peak_vram_gb"], 3)
        if st["in_tokens"] or st["out_tokens"]:
            row[f"{n}.tokens"] = st["in_tokens"] + st["out_tokens"]
    row["total.wall_s"] = report_dict["totals"]["wall_s"]
    row["total.cost_usd"] = report_dict["totals"]["cost_usd"]
    row["total.tokens"] = report_dict["totals"]["tokens"]
    return row


def run_benchmark(a):
    videos = sorted(glob.glob(a.videos))
    if not videos:
        raise SystemExit(f"no videos matched: {a.videos!r}")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    decoders = [x.strip() for x in a.decoders.split(",") if x.strip()]
    tasks = [x.strip() for x in a.tasks.split(",") if x.strip()]
    grid = _parse_grid(a.grid)
    grid_keys = list(grid.keys())
    questions = None
    if a.questions and "qa" in tasks:
        questions = [q.strip() for q in Path(a.questions).read_text().splitlines() if q.strip()]

    rows: list[dict] = []
    for video in videos:
        for arm in arms:
            # native ignores decoder/sampling — collapse those axes to one run
            decs = ["cpu"] if arm == "native" else decoders
            local_keys = grid_keys
            if arm == "native":
                local_keys = [k for k in grid_keys if k not in ("sampling", "decoder")]
            combos = list(itertools.product(*[grid[k] for k in local_keys])) or [()]
            for dec in decs:
                for combo in combos:
                    cfg = PipelineConfig(video=video, arm=arm, engine=a.engine,
                                         decoder=dec, output_dir=a.output_dir)
                    if a.gemini_model:
                        cfg.vlm_model = cfg.llm_model = a.gemini_model
                    for k, v in zip(local_keys, combo):
                        setattr(cfg, k, v)
                    cfg.__post_init__()
                    label = f"{Path(video).name} | {arm} | {dec} | " + \
                            ", ".join(f"{k}={v}" for k, v in zip(local_keys, combo))
                    for task in tasks:
                        if task == "decode" and arm == "native":
                            continue  # native sends the video to Gemini; no local decode
                        base = {"video": Path(video).name, "arm": arm, "decoder": dec,
                                "task": task, "engine": a.engine}
                        base.update({k: v for k, v in zip(local_keys, combo)})
                        print(f"▶ {task}: {label}")
                        try:
                            if task == "decode":
                                _, report, _ = run_decode(cfg)
                            elif task == "summarize":
                                _, report, _ = run_summarize(cfg)
                            else:
                                qs = questions or ["What happens in the video?"]
                                _, report, _ = run_qa_batch(cfg, qs)
                            rows.append(_flatten(report.to_dict(), base))
                        except Exception as e:
                            traceback.print_exc()
                            rows.append({**base, "error": str(e)})

    # write CSV + JSON
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(a.output_dir) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(a.csv_out) if a.csv_out else out_dir / f"bench_{stamp}.csv"
    json_path = out_dir / f"bench_{stamp}.json"
    json_path.write_text(json.dumps(rows, indent=2))

    header: list[str] = []
    for r in rows:
        for k in r:
            if k not in header:
                header.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    _print_summary(rows)
    print(f"\n[benchmark CSV: {csv_path}]\n[benchmark JSON: {json_path}]")


def _print_summary(rows: list[dict]):
    try:
        from rich.console import Console
        from rich.table import Table
        cols = ["video", "arm", "decoder", "sampling", "task",
                "decode.wall_s", "counter.frames_decoded_per_sec",
                "total.wall_s", "total.tokens", "total.cost_usd", "counter.frames_selected"]
        present = [c for c in cols if any(c in r for r in rows)]
        table = Table(title="Benchmark results")
        for c in present:
            table.add_column(c, justify="left" if c in ("video", "arm") else "right")
        for r in sorted(rows, key=lambda x: x.get("total.wall_s", 0)):
            table.add_row(*[str(r.get(c, "")) for c in present])
        Console().print(table)
    except Exception:
        for r in rows:
            print(r)
