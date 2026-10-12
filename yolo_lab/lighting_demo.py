"""Show how detection holds up under bad lighting.

For each image, saves one strip: clean | dark | very_dark | glare | overexposed,
with detections drawn on each version.

Usage (from yolo_lab/):
  python lighting_demo.py                                   # 3 sample hazard photos
  python lighting_demo.py --source my_photo.jpg --preset max
  python lighting_demo.py --source test_images/hazard        # whole folder
Output: results/lighting_demo/<name>_lighting.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from benchmark import corrupt
from detect_pro import HAZARD_CLASSES, IMAGE_EXTS, ROOT, ProDetector

CONDITIONS = ["clean", "dark", "very_dark", "glare", "overexposed"]
DEFAULT_SOURCES = ["test_images/hazard/extinguisher_hallway2.jpg",
                   "test_images/hazard/hardhat_amsterdam.jpg",
                   "test_images/hazard/cones_car.jpg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", nargs="+", default=DEFAULT_SOURCES, help="images or folders")
    p.add_argument("--preset", default="balanced")
    p.add_argument("--out", default="results/lighting_demo")
    args = p.parse_args()

    files = []
    for s in args.source:
        s = Path(s) if Path(s).is_absolute() else ROOT / s
        files += sorted(f for f in s.rglob("*") if f.suffix.lower() in IMAGE_EXTS) if s.is_dir() else [s]
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    det = ProDetector(preset=args.preset, verbose=False)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip (unreadable): {f}")
            continue
        tiles, summary = [], []
        for kind in CONDITIONS:
            x = corrupt(img, kind, np.random.default_rng(3))
            dets = det.detect(x)
            hazards = sum(d.name in HAZARD_CLASSES for d in dets)
            vis = det.draw(x, dets)
            h, w = vis.shape[:2]
            vis = cv2.resize(vis, (480, int(h * 480 / w)))
            cv2.rectangle(vis, (0, 0), (480, 34), (0, 0, 0), -1)
            cv2.putText(vis, f"{kind}: {len(dets)} obj, {hazards} hazard", (6, 24), 0, 0.65, (0, 255, 255), 2)
            tiles.append(vis)
            summary.append(f"{kind} {len(dets)}/{hazards}")
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
        path = out_dir / f"{f.stem}_lighting.jpg"
        cv2.imwrite(str(path), np.hstack(tiles))
        print(f"{f.name}: " + ", ".join(summary) + "   (objects/hazards)")
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
