"""Local video preprocessing (Arm B): decode -> chunk -> select frames -> JPEG.

Decode backends:  cpu (OpenCV/PyAV)  |  gpu (NVDEC, Phase-2 on the 5090)
Selection:        uniform | iframe | scene (PySceneDetect) | transnet (GPU, Phase-2)
Dedup:            perceptual-hash (imagehash.phash) near-duplicate removal

Everything except the `gpu`/`transnet` paths runs on a CPU-only machine today.
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
# main extractor
# --------------------------------------------------------------------------- #
class FrameExtractor:
    def __init__(self, config):
        self.cfg = config

    def extract(self, path: str, report) -> tuple[list[Chunk], VideoInfo]:
        # NOT IMPLEMENTED YET (Phase 2): GPU/NVDEC decode. Only CPU decode exists
        # today, so fail loudly rather than silently run CPU work under a GPU label.
        if self.cfg.decoder == "gpu":
            raise NotImplementedError(
                "GPU/NVDEC decode (--decoder gpu) is not implemented yet — it is the "
                "Phase-2 task to build on the RTX 5090. Use --decoder cpu for now.")

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
        with report.stage("decode"):
            if self.cfg.sampling == "iframe":
                per_chunk = self._select_iframe(path, ranges, info)
            elif self.cfg.sampling == "scene":
                per_chunk = self._select_scene(path, ranges, info)
            elif self.cfg.sampling == "transnet":
                per_chunk = self._select_transnet(path, ranges, info)
            else:
                per_chunk = self._select_uniform(path, ranges, info)

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

    # ---- GPU: NVDEC + TransNetV2 — NOT IMPLEMENTED YET (Phase 2) ----
    def _select_transnet(self, path, ranges, info):
        # Intended Phase-2 design (build + validate on the 5090): decode with NVDEC
        # (torchcodec/PyNvVideoCodec) to CUDA tensors -> cheap GPU histogram gate ->
        # TransNetV2 shot cuts on 48x27 frames -> one sharp frame per shot -> nvJPEG.
        # None of that exists yet; this is a placeholder, not a working path.
        raise NotImplementedError(
            "--sampling transnet is not implemented yet — it is the Phase-2 task to "
            "build on the RTX 5090. Use --sampling scene / iframe / uniform for now.")

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
