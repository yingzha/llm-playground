"""Per-stage profiling.

`ProfileReport.stage(name, gpu=False)` returns a context manager that records:
  - wall time (always, via time.perf_counter)
  - GPU time + peak VRAM (when gpu=True and torch+CUDA are available)
  - token usage + estimated cost (when the caller sets them inside the block)

Repeated stage names accumulate (calls/total/mean) — e.g. a per-chunk
`vlm_caption` stage called N times shows up as one row with calls=N.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:  # torch is only present on the GPU box; profiling degrades gracefully without it
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    _HAS_CUDA = False


@dataclass
class StageStat:
    name: str
    calls: int = 0
    wall_s: float = 0.0
    gpu_ms: float = 0.0
    peak_vram_gb: float = 0.0    # max across calls
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0

    def merge(self, wall_s, gpu_ms, vram, in_tok, out_tok, cost):
        self.calls += 1
        self.wall_s += wall_s
        self.gpu_ms += gpu_ms or 0.0
        self.peak_vram_gb = max(self.peak_vram_gb, vram or 0.0)
        self.in_tokens += in_tok or 0
        self.out_tokens += out_tok or 0
        self.cost_usd += cost or 0.0


class StageTimer:
    def __init__(self, report: "ProfileReport", name: str, gpu: bool):
        self.report = report
        self.name = name
        self.gpu = gpu and _HAS_CUDA
        self.in_tokens = 0
        self.out_tokens = 0
        self.cost_usd = 0.0

    # callers set token usage inside the block
    def set_tokens(self, in_tokens: int, out_tokens: int, cost_usd: float = 0.0):
        self.in_tokens += int(in_tokens or 0)
        self.out_tokens += int(out_tokens or 0)
        self.cost_usd += float(cost_usd or 0.0)

    def __enter__(self):
        if self.gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._e0 = torch.cuda.Event(enable_timing=True)
            self._e1 = torch.cuda.Event(enable_timing=True)
            self._e0.record()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter() - self._t0
        gpu_ms = None
        vram = None
        if self.gpu:
            self._e1.record()
            torch.cuda.synchronize()  # essential before reading events/peak memory
            gpu_ms = self._e0.elapsed_time(self._e1)
            vram = torch.cuda.max_memory_allocated() / 1e9
        self.report._record(self.name, wall, gpu_ms, vram,
                            self.in_tokens, self.out_tokens, self.cost_usd)
        return False


@dataclass
class ProfileReport:
    label: str = ""
    stages: dict = field(default_factory=dict)   # name -> StageStat (insertion-ordered)
    counters: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def __post_init__(self):
        self.environment.setdefault("has_cuda", _HAS_CUDA)
        if _HAS_CUDA:
            try:
                self.environment.setdefault("gpu_name", torch.cuda.get_device_name(0))
                self.environment.setdefault("cuda_capability",
                                            list(torch.cuda.get_device_capability(0)))
            except Exception:
                pass

    def stage(self, name: str, gpu: bool = False) -> StageTimer:
        return StageTimer(self, name, gpu)

    def counter(self, key: str, value):
        self.counters[key] = value

    def _record(self, name, wall, gpu_ms, vram, in_tok, out_tok, cost):
        st = self.stages.get(name)
        if st is None:
            st = StageStat(name=name)
            self.stages[name] = st
        st.merge(wall, gpu_ms, vram, in_tok, out_tok, cost)

    # ---- derived ----
    def total_wall(self) -> float:
        # prefer an explicit "total" stage if present, else sum of top-level stages
        if "total" in self.stages:
            return self.stages["total"].wall_s
        return sum(s.wall_s for s in self.stages.values())

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.stages.values())

    def total_tokens(self) -> int:
        return sum(s.in_tokens + s.out_tokens for s in self.stages.values())

    # ---- output ----
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "environment": self.environment,
            "counters": self.counters,
            "stages": [asdict(s) for s in self.stages.values()],
            "totals": {
                "wall_s": round(self.total_wall(), 4),
                "cost_usd": round(self.total_cost(), 6),
                "tokens": self.total_tokens(),
            },
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def render(self) -> str:
        total = self.total_wall() or 1e-9
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title=f"Profile: {self.label}", show_lines=False)
            for col in ("stage", "calls", "wall_s", "% wall", "gpu_ms",
                        "vram_gb", "in_tok", "out_tok", "cost_$"):
                table.add_column(col, justify="right" if col != "stage" else "left")
            for st in self.stages.values():
                if st.name == "total":
                    continue
                table.add_row(
                    st.name, str(st.calls), f"{st.wall_s:.3f}",
                    f"{100 * st.wall_s / total:.1f}",
                    f"{st.gpu_ms:.1f}" if st.gpu_ms else "-",
                    f"{st.peak_vram_gb:.2f}" if st.peak_vram_gb else "-",
                    str(st.in_tokens) if st.in_tokens else "-",
                    str(st.out_tokens) if st.out_tokens else "-",
                    f"{st.cost_usd:.5f}" if st.cost_usd else "-",
                )
            table.add_row("TOTAL", "", f"{self.total_wall():.3f}", "100.0",
                          "", "", "", str(self.total_tokens()) or "-",
                          f"{self.total_cost():.5f}")
            import io
            con = Console(record=True, width=140, file=io.StringIO())  # capture, don't print
            con.print(table)
            if self.counters:
                con.print("[dim]counters:[/dim] " +
                          "  ".join(f"{k}={v}" for k, v in self.counters.items()))
            return con.export_text()
        except Exception:
            # plain-text fallback if rich is unavailable
            lines = [f"Profile: {self.label}"]
            lines.append(f"{'stage':22} {'calls':>5} {'wall_s':>9} {'%':>6} "
                        f"{'gpu_ms':>8} {'vram':>6} {'in_tok':>8} {'out_tok':>8} {'cost$':>9}")
            for st in self.stages.values():
                if st.name == "total":
                    continue
                lines.append(
                    f"{st.name:22} {st.calls:>5} {st.wall_s:>9.3f} "
                    f"{100*st.wall_s/total:>6.1f} {st.gpu_ms:>8.1f} "
                    f"{st.peak_vram_gb:>6.2f} {st.in_tokens:>8} {st.out_tokens:>8} "
                    f"{st.cost_usd:>9.5f}")
            lines.append(f"{'TOTAL':22} {'':>5} {self.total_wall():>9.3f} {100.0:>6.1f}"
                        f"  tokens={self.total_tokens()} cost=${self.total_cost():.5f}")
            if self.counters:
                lines.append("counters: " + "  ".join(f"{k}={v}" for k, v in self.counters.items()))
            return "\n".join(lines)
