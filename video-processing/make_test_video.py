#!/usr/bin/env python3
"""Generate a synthetic test clip (moving shape + frame counter + scene cuts).

Uses OpenCV (already a dependency) so it needs NO system ffmpeg.

  python make_test_video.py --seconds 30 --out outputs/test.mp4
  python make_test_video.py --seconds 60 --resolution 1280x720 --scenes 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# distinct background colors (BGR) to create hard scene cuts
SCENE_COLORS = [(40, 40, 40), (20, 20, 120), (20, 110, 20), (110, 60, 20),
                (90, 20, 110), (20, 90, 110)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--scenes", type=int, default=3, help="number of distinct scenes (hard cuts)")
    ap.add_argument("--out", default="outputs/test.mp4")
    a = ap.parse_args()

    w, h = (int(x) for x in a.resolution.lower().split("x"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = a.seconds * a.fps
    scenes = max(a.scenes, 1)
    seg = max(total // scenes, 1)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("OpenCV VideoWriter failed to open (mp4v codec missing?)")

    for i in range(total):
        scene = min(i // seg, scenes - 1)
        frame = np.full((h, w, 3), SCENE_COLORS[scene % len(SCENE_COLORS)], dtype=np.uint8)
        # moving box
        x = int((i / total) * (w - 80))
        y = int(h / 2 + (h / 4) * np.sin(i / 8.0))
        cv2.rectangle(frame, (x, y - 30), (x + 80, y + 30), (255, 255, 255), -1)
        # burned-in timestamp + scene label
        t = i / a.fps
        cv2.putText(frame, f"t={t:5.1f}s  scene={scene}  frame={i}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({a.seconds}s, {a.resolution}, {scenes} scenes, {size:.2f} MB)")


if __name__ == "__main__":
    main()
