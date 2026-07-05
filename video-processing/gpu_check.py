#!/usr/bin/env python3
"""Preflight for the RTX 5090 (Blackwell sm_120) GPU paths.

Verifies: torch CUDA capability (12, 0), a real NVDEC decode via torchcodec
(when a video is available), and a TransNetV2 forward pass on dummy input.
Safe to run on the Mac — it just reports what's missing.

  python gpu_check.py [video]      # video defaults to outputs/test.mp4
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_torch():
    try:
        import torch
    except Exception as e:
        return False, f"torch not installed ({e}). Install: pip install torch --index-url https://download.pytorch.org/whl/cu128"
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__} installed but CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    ok = cap == (12, 0)
    msg = (f"{name}, capability {cap} (expected (12, 0) for 5090), "
           f"torch {torch.__version__}, cuda {torch.version.cuda}")
    return ok, msg


def check_torchcodec(video: Path):
    try:
        from src.frames import _preload_gpu_libs
        _preload_gpu_libs()
        from torchcodec.decoders import VideoDecoder
    except Exception as e:
        return False, f"torchcodec missing ({e}). Install: pip install torchcodec --index-url https://download.pytorch.org/whl/cu128"
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        return True, "torchcodec import ok (no CUDA here, NVDEC untested)"
    if not video.exists():
        return True, f"torchcodec import ok (no test video at {video}, NVDEC untested)"
    # real NVDEC smoke test: decode on the GPU and CHECK CONTENT — a broken
    # NVDEC setup (NPP-era torchcodec, or FFmpeg without --enable-nvdec, e.g.
    # Ubuntu's system build) silently returns constant solid-color frames.
    try:
        dec = VideoDecoder(str(video), device="cuda")
        n = dec.metadata.num_frames
        fb = dec.get_frames_at([0, n // 2, n - 1]).data.float()
        per_frame_std = float(fb.std(dim=(1, 2, 3)).mean())
        across = float((fb[0] - fb[1]).abs().mean() + (fb[1] - fb[2]).abs().mean())
        if per_frame_std < 1.0 and across < 1.0:
            return False, (f"NVDEC returns near-constant frames (std={per_frame_std:.3f}) — "
                           "broken GPU decode: use torchcodec>=0.14 (cu130) and an "
                           "NVDEC-enabled FFmpeg in .venv/lib/ffmpeg (see README)")
        return True, (f"NVDEC decode ok: {video.name} -> {tuple(fb.shape)} on cuda, "
                      f"content varies (std={per_frame_std:.1f})")
    except Exception as e:
        return False, f"torchcodec imports but NVDEC decode of {video} failed: {e}"


def check_transnet():
    try:
        from transnetv2_pytorch import TransNetV2
    except Exception as e:
        return False, f"transnetv2-pytorch missing ({e}). Install: pip install transnetv2-pytorch"
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        return True, "transnetv2-pytorch import ok (no CUDA here, forward untested)"
    # instantiate + dummy forward: resolves weight loading and the forward
    # signature (expected: logits [1,100,1] + dict of extra outputs)
    try:
        import torch
        model = TransNetV2().eval().cuda()
        with torch.inference_mode():
            out = model(torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8, device="cuda"))
        desc = ", ".join(str(tuple(o.shape)) if hasattr(o, "shape")
                         else f"{type(o).__name__}({list(o.keys())})" if isinstance(o, dict)
                         else type(o).__name__
                         for o in (out if isinstance(out, (tuple, list)) else (out,)))
        return True, f"TransNetV2 forward ok on dummy [1,100,27,48,3]; outputs: {desc}"
    except Exception as e:
        return False, f"transnetv2-pytorch imports but forward failed: {e}"


def main():
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/test.mp4")
    print("== GPU preflight (5090 / sm_120) ==")
    all_ok = True
    for name, fn in (("torch/CUDA", check_torch),
                     ("torchcodec (NVDEC)", lambda: check_torchcodec(video)),
                     ("TransNetV2", check_transnet)):
        ok, msg = fn()
        all_ok &= ok
        print(f"[{'OK ' if ok else 'XX '}] {name}: {msg}")
    print("\n" + ("All GPU paths ready — use --decoder gpu --sampling transnet"
                  if all_ok else
                  "Some GPU paths unavailable; CPU paths (--decoder cpu) still work."))


if __name__ == "__main__":
    main()
