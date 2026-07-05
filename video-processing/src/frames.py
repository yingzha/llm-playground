"""Local video preprocessing (Arm B): decode -> chunk -> select frames -> JPEG.

Decode backends:  cpu (OpenCV/PyAV)  |  gpu (NVDEC via torchcodec)
Selection:        uniform | iframe | scene (PySceneDetect) | transnet (TransNetV2, GPU)
Dedup:            perceptual-hash (imagehash.phash) near-duplicate removal

Valid (decoder, sampling) combos:
    cpu: uniform | iframe | scene        gpu: uniform | transnet
transnet requires the GPU (dense every-frame decode + CUDA TransNetV2); iframe and
scene are bound to CPU libraries (PyAV / PySceneDetect). GPU deps (torch cu128,
torchcodec, transnetv2-pytorch) are imported lazily, so the CPU paths run on a
machine without them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VideoInfo:
    path: str
    duration: float
    width: int
    height: int
    fps: float
    size_mb: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class Chunk:
    idx: int
    start: float
    end: float
    frames: list[bytes] = field(default_factory=list)  # JPEG bytes
    n_decoded: int = 0                                  # candidates considered


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def probe_video(path: str) -> VideoInfo:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = (nframes / fps) if fps > 0 else 0.0
    size_mb = os.path.getsize(path) / 1e6
    return VideoInfo(path, duration, w, h, fps, size_mb)


def chunk_ranges(duration: float, chunk_duration: float, overlap: float):
    ranges, idx, start = [], 0, 0.0
    step = max(chunk_duration - overlap, 0.1)
    while start < duration - 1e-3:
        end = min(start + chunk_duration, duration)
        ranges.append((idx, start, end))
        idx += 1
        start += step
    if not ranges:  # degenerate / very short clip
        ranges.append((0, 0.0, max(duration, 0.1)))
    return ranges


def _encode_jpeg(bgr: np.ndarray, max_side: int) -> bytes:
    import cv2
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _sharpness(bgr: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _phash(bgr: np.ndarray):
    import cv2
    import imagehash
    from PIL import Image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def _dedup(frames_bgr: list[np.ndarray], max_hamming: int) -> list[np.ndarray]:
    kept, hashes = [], []
    for f in frames_bgr:
        h = _phash(f)
        if any((h - kh) <= max_hamming for kh in hashes):
            continue
        kept.append(f)
        hashes.append(h)
    return kept


# --------------------------------------------------------------------------- #
# GPU helpers (torch/torchcodec/transnetv2 imported lazily — GPU box only)
# --------------------------------------------------------------------------- #
_GPU_DECODE_BATCH = 256  # frames per NVDEC batch; bounds VRAM (~1.6 GB at 1080p)


def _preload_gpu_libs():
    """Best-effort preload of shared libs torchcodec needs but the dynamic
    loader can't find on its own:
      - NPP from the pip `nvidia-npp-cu12` wheel (torch cu128 doesn't ship it,
        and site-packages/nvidia/npp/lib is outside the loader's search path)
      - an NVDEC-enabled FFmpeg staged in `.venv/lib/ffmpeg/` (e.g. a BtbN
        gpl-shared build) — needed when the system FFmpeg lacks CUDA/NVDEC
        (Ubuntu 24.04's build has no --enable-nvdec, which makes torchcodec's
        device="cuda" silently return garbage frames). See README GPU section.
    Loads with RTLD_GLOBAL so subsequently-loaded libs resolve these sonames.
    A fixed-point loop handles load order (deps must register first)."""
    import ctypes
    import sys
    from pathlib import Path

    dirs = []
    try:
        import nvidia
        dirs += [Path(p) / "npp" / "lib" for p in nvidia.__path__]
    except ImportError:
        pass
    dirs.append(Path(sys.prefix) / "lib" / "ffmpeg")

    pending = [p for d in dirs if d.is_dir() for p in sorted(d.glob("lib*.so*"))]
    for _ in range(8):
        failed = []
        for p in pending:
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(p)
        if not failed or len(failed) == len(pending):  # done, or no progress
            break
        pending = failed


def _require_gpu_stack():
    try:
        import torch
        _preload_gpu_libs()
        from torchcodec.decoders import VideoDecoder
    except ImportError as e:
        raise RuntimeError(
            "--decoder gpu needs torch + torchcodec (>=0.14). On the GPU box:\n"
            "  uv pip install torch torchcodec --index-url https://download.pytorch.org/whl/cu130\n"
            "  uv pip install transnetv2-pytorch   # only for --sampling transnet\n"
            "See the README GPU section (NVDEC-enabled FFmpeg in .venv/lib/ffmpeg)."
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError("--decoder gpu: torch imported but CUDA is not available")
    return torch, VideoDecoder


def _gpu_frames_to_bgr(data) -> list[np.ndarray]:
    # data: uint8 CUDA tensor [N, C, H, W], RGB -> list of BGR HWC ndarrays
    arr = data.permute(0, 2, 3, 1).cpu().numpy()
    return [np.ascontiguousarray(a[..., ::-1]) for a in arr]


def _laplacian_var_gpu(torch, frames_u8):
    # frames_u8: uint8 CUDA [N, 3, H, W] RGB -> per-frame Laplacian variance [N]
    # (GPU analog of _sharpness)
    import torch.nn.functional as F
    f = frames_u8.float()
    gray = (0.299 * f[:, 0] + 0.587 * f[:, 1] + 0.114 * f[:, 2]).unsqueeze(1)
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                     device=f.device).view(1, 1, 3, 3)
    return F.conv2d(gray, k).var(dim=(1, 2, 3))


def _transnet_probs(torch, model, lowres) -> np.ndarray:
    # lowres: uint8 CUDA [T, 27, 48, 3] RGB. Standard soCzech sliding window:
    # pad 25 head / 25+align tail with edge frames, windows of 100 step 50,
    # keep predictions [25:75] of each window -> per-frame transition probs [T].
    T = lowres.shape[0]
    tail_pad = 25 + (-(T + 50) % 50)
    frames = torch.cat([lowres[:1].expand(25, -1, -1, -1),
                        lowres,
                        lowres[-1:].expand(tail_pad, -1, -1, -1)])
    out = []
    with torch.inference_mode():
        for i in range(0, frames.shape[0] - 50, 50):
            window = frames[i:i + 100].unsqueeze(0)   # [1, 100, 27, 48, 3]
            single_frame_pred, _ = model(window)
            out.append(torch.sigmoid(single_frame_pred)[0, 25:75, 0])
    return torch.cat(out)[:T].cpu().numpy()


def _probs_to_shots(probs: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    # soCzech predictions_to_scenes: shots are maximal runs of below-threshold
    # frames between above-threshold transition frames. Inclusive index pairs.
    pred = (probs > threshold).astype(np.uint8)
    shots, start, prev = [], 0, 0
    for i, p in enumerate(pred):
        if prev == 1 and p == 0:
            start = i
        if prev == 0 and p == 1 and i != 0:
            shots.append((start, i))
        prev = int(p)
    if prev == 0:
        shots.append((start, len(pred) - 1))
    return shots or [(0, max(len(pred) - 1, 0))]


# --------------------------------------------------------------------------- #
# main extractor
# --------------------------------------------------------------------------- #
class FrameExtractor:
    def __init__(self, config):
        self.cfg = config
        self._last_shots: int | None = None  # transnet shot count, for a counter

    def extract(self, path: str, report) -> tuple[list[Chunk], VideoInfo]:
        if self.cfg.sampling == "transnet" and self.cfg.decoder != "gpu":
            raise ValueError(
                "--sampling transnet requires --decoder gpu (NVDEC + CUDA "
                "TransNetV2); use --sampling scene on CPU")
        if self.cfg.decoder == "gpu" and self.cfg.sampling not in ("uniform", "transnet"):
            raise ValueError(
                f"--decoder gpu supports --sampling uniform|transnet "
                f"(got {self.cfg.sampling!r}); iframe/scene are CPU-only paths")

        info = probe_video(path)
        report.counter("video_duration_s", round(info.duration, 2))
        report.counter("video_resolution", info.resolution)
        report.counter("file_size_mb", round(info.size_mb, 2))
        report.counter("decoder", self.cfg.decoder)
        report.counter("sampling", self.cfg.sampling)

        ranges = chunk_ranges(info.duration, self.cfg.chunk_duration, self.cfg.chunk_overlap)

        # --- decode + select (per-chunk BGR candidate frames) ---
        # NOTE: this "decode" stage also does frame SELECTION (uniform/iframe/scene),
        # since selection interleaves with decoding. The separate "frame_select" stage
        # below is dedup + JPEG encode. Sum both for total frame-extraction cost.
        gpu = self.cfg.decoder == "gpu"
        with report.stage("decode", gpu=gpu):
            if self.cfg.sampling == "iframe":
                per_chunk = self._select_iframe(path, ranges, info)
            elif self.cfg.sampling == "scene":
                per_chunk = self._select_scene(path, ranges, info)
            elif self.cfg.sampling == "transnet":
                per_chunk = self._select_transnet(path, ranges, info)
            elif gpu:
                per_chunk = self._select_uniform_gpu(path, ranges, info)
            else:
                per_chunk = self._select_uniform(path, ranges, info)
        if self._last_shots is not None:
            report.counter("transnet_shots", self._last_shots)

        total_decoded = sum(len(v) for v in per_chunk.values())

        # --- dedup + JPEG encode ---
        chunks: list[Chunk] = []
        with report.stage("frame_select"):
            for idx, start, end in ranges:
                cand = per_chunk.get(idx, [])
                if self.cfg.dedup and cand:
                    cand = _dedup(cand, self.cfg.dedup_hamming)
                cand = cand[: self.cfg.max_frames_per_chunk]
                jpegs = [_encode_jpeg(f, self.cfg.jpeg_max_side) for f in cand]
                chunks.append(Chunk(idx=idx, start=start, end=end,
                                    frames=jpegs, n_decoded=len(per_chunk.get(idx, []))))

        total_selected = sum(len(c.frames) for c in chunks)
        report.counter("num_chunks", len(chunks))
        report.counter("frames_per_chunk", self.cfg.frames_per_chunk)
        report.counter("total_frames", total_decoded)
        report.counter("frames_selected", total_selected)
        decode_wall = report.stages["decode"].wall_s or 1e-9
        report.counter("frames_decoded_per_sec", round(total_decoded / decode_wall, 1))
        return chunks, info

    # ---- CPU: uniform ----
    def _select_uniform(self, path, ranges, info):
        import cv2
        cap = cv2.VideoCapture(path)
        out: dict[int, list] = {}
        for idx, start, end in ranges:
            n = max(self.cfg.frames_per_chunk, 1)
            ts = np.linspace(start, end, n, endpoint=False) + (end - start) / (2 * n)
            frames = []
            for t in ts:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
            out[idx] = frames
        cap.release()
        return out

    # ---- CPU: I-frames only (PyAV skip_frame=NONKEY) ----
    def _select_iframe(self, path, ranges, info):
        import av
        out: dict[int, list] = {idx: [] for idx, _, _ in ranges}
        container = av.open(path)
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        tb = stream.time_base
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            idx = self._chunk_of(t, ranges)
            if idx is None:
                continue
            if len(out[idx]) >= self.cfg.max_frames_per_chunk:
                continue
            bgr = frame.to_ndarray(format="bgr24")
            out[idx].append(bgr)
        container.close()
        # guarantee at least one frame per chunk (fall back to midpoint)
        self._backfill_midpoints(path, ranges, out)
        return out

    # ---- CPU: scene detection (PySceneDetect Adaptive) ----
    def _select_scene(self, path, ranges, info):
        import cv2
        from scenedetect import open_video, SceneManager, AdaptiveDetector

        video = open_video(path)
        sm = SceneManager()
        sm.add_detector(AdaptiveDetector(adaptive_threshold=self.cfg.scene_threshold))
        if self.cfg.detect_downscale > 1:
            sm.auto_downscale = False
            sm.downscale = self.cfg.detect_downscale
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list()  # list[(start_tc, end_tc)]

        out: dict[int, list] = {idx: [] for idx, _, _ in ranges}
        cap = cv2.VideoCapture(path)
        for s_tc, e_tc in scenes:
            s, e = s_tc.get_seconds(), e_tc.get_seconds()
            # sharpest of a few candidates within the scene
            cands_t = [s + (e - s) * f for f in (0.15, 0.5, 0.85)]
            best, best_sharp, best_t = None, -1.0, s
            for t in cands_t:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if ok and frame is not None:
                    sh = _sharpness(frame)
                    if sh > best_sharp:
                        best, best_sharp, best_t = frame, sh, t
            if best is not None:
                idx = self._chunk_of(best_t, ranges)
                if idx is not None and len(out[idx]) < self.cfg.max_frames_per_chunk:
                    out[idx].append(best)
        cap.release()
        self._backfill_midpoints(path, ranges, out)
        return out

    # ---- GPU: NVDEC uniform (same timestamp math as _select_uniform) ----
    def _select_uniform_gpu(self, path, ranges, info):
        torch, VideoDecoder = _require_gpu_stack()
        dec = VideoDecoder(path, device="cuda")
        end_s = dec.metadata.end_stream_seconds or info.duration
        out: dict[int, list] = {}
        for idx, start, end in ranges:
            n = max(self.cfg.frames_per_chunk, 1)
            ts = np.linspace(start, end, n, endpoint=False) + (end - start) / (2 * n)
            ts = np.clip(ts, 0.0, max(end_s - 1e-3, 0.0))  # torchcodec raises on OOB pts
            fb = dec.get_frames_played_at(ts.tolist())
            out[idx] = _gpu_frames_to_bgr(fb.data)
        return out

    # ---- GPU: NVDEC dense decode + TransNetV2 shot cuts + GPU sharpness ----
    def _select_transnet(self, path, ranges, info):
        torch, VideoDecoder = _require_gpu_stack()
        try:
            from transnetv2_pytorch import TransNetV2
        except ImportError as e:
            raise RuntimeError(
                "--sampling transnet needs transnetv2-pytorch: "
                "uv pip install transnetv2-pytorch") from e
        import torch.nn.functional as F

        dec = VideoDecoder(path, device="cuda")
        n = dec.metadata.num_frames

        # 1) dense decode -> 48x27 lowres on GPU, keeping per-frame pts
        lowres = torch.empty((n, 27, 48, 3), dtype=torch.uint8, device="cuda")
        pts = np.empty(n, dtype=np.float64)
        for i0 in range(0, n, _GPU_DECODE_BATCH):
            fb = dec.get_frames_in_range(i0, min(i0 + _GPU_DECODE_BATCH, n))
            x = F.interpolate(fb.data.float(), size=(27, 48), mode="area")
            got = x.shape[0]
            lowres[i0:i0 + got] = x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
            p = fb.pts_seconds
            pts[i0:i0 + got] = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)

        # 2) shot boundaries from per-frame transition probabilities
        model = TransNetV2().eval().cuda()
        probs = _transnet_probs(torch, model, lowres)
        shots = _probs_to_shots(probs, self.cfg.transnet_threshold)
        self._last_shots = len(shots)
        del lowres, model

        # 3) sharpest of 3 candidates per shot (parity with _select_scene):
        #    one batched full-res fetch, GPU Laplacian variance, argmax per triple
        cand_ts = [float(pts[min(s + int((e - s) * f), n - 1)])
                   for s, e in shots for f in (0.15, 0.5, 0.85)]
        fb = dec.get_frames_played_at(cand_ts)
        sharp = _laplacian_var_gpu(torch, fb.data).view(-1, 3)
        win = sharp.argmax(dim=1).cpu().numpy()
        cand_bgr = _gpu_frames_to_bgr(fb.data)

        out: dict[int, list] = {idx: [] for idx, _, _ in ranges}
        for k in range(len(shots)):
            j = 3 * k + int(win[k])
            idx = self._chunk_of(cand_ts[j], ranges)
            if idx is not None and len(out[idx]) < self.cfg.max_frames_per_chunk:
                out[idx].append(cand_bgr[j])
        self._backfill_midpoints(path, ranges, out)
        return out

    # ---- utilities ----
    @staticmethod
    def _chunk_of(t: float, ranges):
        for idx, start, end in ranges:
            if start <= t < end:
                return idx
        # tail frame -> last chunk
        return ranges[-1][0] if ranges and t >= ranges[-1][1] else None

    def _backfill_midpoints(self, path, ranges, out):
        import cv2
        empty = [idx for idx, _, _ in ranges if not out.get(idx)]
        if not empty:
            return
        cap = cv2.VideoCapture(path)
        for idx, start, end in ranges:
            if out.get(idx):
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + end) / 2 * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                out[idx].append(frame)
        cap.release()
