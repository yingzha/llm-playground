#!/usr/bin/env python3
"""Preflight for the RTX 5090 (Blackwell sm_120) Phase-2 GPU paths.

Verifies: torch CUDA capability (12, 0), NVDEC decode via torchcodec, and that
TransNetV2 loads. Safe to run on the Mac — it just reports what's missing.

  python gpu_check.py
"""

from __future__ import annotations


def check_torch():
    try:
        import torch
    except Exception as e:
        return False, f"torch not installed ({e}). Install: pip install torch --index-url https://download.pytorch.org/whl/cu128"
    if not torch.cuda.is_available():
        return False, "torch installed but CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    ok = cap == (12, 0)
    msg = f"{name}, capability {cap} (expected (12, 0) for 5090)"
    return ok, msg


def check_torchcodec():
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        return True, "torchcodec import ok (NVDEC via device='cuda')"
    except Exception as e:
        return False, f"torchcodec missing ({e}). Install: pip install torchcodec"


def check_transnet():
    try:
        from transnetv2_pytorch import TransNetV2  # noqa: F401
        return True, "transnetv2-pytorch import ok"
    except Exception as e:
        return False, f"transnetv2-pytorch missing ({e}). Install: pip install transnetv2-pytorch"


def main():
    print("== GPU preflight (5090 / sm_120) ==")
    all_ok = True
    for name, fn in (("torch/CUDA", check_torch),
                     ("torchcodec (NVDEC)", check_torchcodec),
                     ("TransNetV2", check_transnet)):
        ok, msg = fn()
        all_ok &= ok
        print(f"[{'OK ' if ok else 'XX '}] {name}: {msg}")
    print("\n" + ("All GPU paths ready — use --decoder gpu --sampling transnet"
                  if all_ok else
                  "Some GPU paths unavailable; CPU paths (--decoder cpu) still work."))


if __name__ == "__main__":
    main()
