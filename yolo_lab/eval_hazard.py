"""Pick the best YOLO-World text prompts for each hazard class, using hand-checked test images.

Ground truth is IMAGE-LEVEL (does the photo contain the object or not), checked by eye.
For every candidate prompt we take the max confidence in each image and measure how well
that separates positive from negative images (best F1 over thresholds, and AUC).

Usage (from yolo_lab/):
  python eval_hazard.py                 # rank prompts
  python eval_hazard.py --pipeline      # evaluate detect_pro balanced preset end-to-end
"""
import argparse

import numpy as np
from ultralytics import YOLOWorld

from detect_pro import ROOT, ProDetector, resolve_weights, pick_device

IMG_DIR = ROOT / "test_images"

# image stem -> set of hazard classes present. Anything not listed = negative.
# IGNORE marks classes that are ambiguous in that photo (not counted either way).
GROUND_TRUTH = {
    "cones_blue": {"traffic_cone"},
    "cones_car": {"traffic_cone"},
    "cones_parking_garage": {"traffic_cone"},
    "cones_road": {"traffic_cone"},
    "extinguisher_bluewall": {"fire_extinguisher"},
    "extinguisher_boxes": {"fire_extinguisher"},
    "extinguisher_estintori": {"fire_extinguisher"},
    "extinguisher_floor": {"fire_extinguisher"},
    "extinguisher_hallway2": {"fire_extinguisher"},
    "extinguisher_wall": {"fire_extinguisher"},
    "first_aid_emergency_kit": {"first_aid_kit"},
    "gas_cylinders": {"gas_cylinder"},
    "oxygen_cylinders": {"gas_cylinder"},
    "oxygen_hospital": {"gas_cylinder"},
    "hardhat_amsterdam": {"safety_helmet", "safety_vest"},
    "hardhat_hongkong": {"safety_helmet", "safety_vest"},
    "hardhat_welder": {"safety_helmet"},
    "hardhat_white_gloves": {"safety_helmet"},
    "hardhat_worker": {"safety_helmet"},
}
IGNORE = {
    "oxygen_hospital": {"fire_extinguisher"},   # tiny extinguisher on the truck, barely visible
    "propane_tanks": {"gas_cylinder"},          # pile of small gas cartridges
    "cones_car": {"warning_sign"},
}

CANDIDATES = {
    "fire_extinguisher": ["fire extinguisher", "extinguisher", "red fire extinguisher",
                          "red extinguisher", "fire extinguisher cylinder"],
    "safety_helmet": ["safety helmet", "hard hat", "helmet", "construction helmet", "hardhat"],
    "traffic_cone": ["traffic cone", "cone", "safety cone", "orange safety cone"],
    "gas_cylinder": ["gas cylinder", "gas cylinders", "oxygen cylinder", "compressed gas cylinder",
                     "gas bottle", "gas tank"],
    "first_aid_kit": ["first aid kit", "first aid box", "emergency kit", "medical kit"],
    "safety_vest": ["safety vest", "high visibility vest", "reflective vest", "hi-vis vest"],
}


def images():
    return sorted(p for p in IMG_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def labels_for(stem, cls):
    """1 = positive, 0 = negative, None = ignore."""
    if cls in IGNORE.get(stem, set()):
        return None
    return int(cls in GROUND_TRUTH.get(stem, set()))


def best_f1(scores, labels):
    best = (0.0, 0.0, 0, 0, 0)
    for t in sorted(set(scores)) + [1.01]:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, t, tp, fp, fn)
    return best


def auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def rank_prompts(imgsz_list):
    prompts = sorted({p for ps in CANDIDATES.values() for p in ps})
    model = YOLOWorld(resolve_weights("yolov8x-worldv2.pt"))
    model.set_classes(prompts)
    files = images()
    for imgsz in imgsz_list:
        scores = np.zeros((len(files), len(prompts)))
        for i, f in enumerate(files):
            r = model.predict(str(f), conf=0.01, imgsz=imgsz, device=pick_device(), verbose=False)[0]
            for c, k in zip(r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                scores[i, int(k)] = max(scores[i, int(k)], c)
        print(f"\n===== imgsz {imgsz} ({len(files)} images) =====")
        for cls, cands in CANDIDATES.items():
            lab = [labels_for(f.stem, cls) for f in files]
            keep = np.array([l is not None for l in lab])
            y = np.array([l for l in lab if l is not None])
            print(f"\n{cls}  (positives={int(y.sum())}, negatives={int((y == 0).sum())})")
            rows = []
            for p in cands:
                s = scores[keep, prompts.index(p)]
                f1, t, tp, fp, fn = best_f1(s, y)
                rows.append((f1, auc(s, y), p, t, tp, fp, fn))
            ens = scores[keep][:, [prompts.index(p) for p in cands]].max(1)
            f1, t, tp, fp, fn = best_f1(ens, y)
            rows.append((f1, auc(ens, y), "<max of all>", t, tp, fp, fn))
            for f1, a, p, t, tp, fp, fn in sorted(rows, key=lambda r: (-r[0], -r[1])):
                print(f"  {p:26s} F1={f1:.2f} AUC={a:.2f} thr={t:.3f}  TP={tp} FP={fp} FN={fn}")


def eval_pipeline(preset, conditions=("clean",), low_light=None, verbose=True):
    """Image-level hits per hazard class, optionally under simulated lighting problems."""
    import cv2
    from benchmark import corrupt
    det = ProDetector(preset=preset, low_light=low_light, verbose=False)
    files = images()
    table = {}
    for kind in conditions:
        rng = np.random.default_rng(0)
        hits = {c: [0, 0, 0] for c in CANDIDATES}  # tp, fp, fn
        for f in files:
            found = {d.name for d in det.detect(corrupt(cv2.imread(str(f)), kind, rng))}
            for c in CANDIDATES:
                l = labels_for(f.stem, c)
                if l is None:
                    continue
                if c in found and l:
                    hits[c][0] += 1
                elif c in found:
                    hits[c][1] += 1
                    if verbose:
                        print(f"  [{kind}] false alarm: {c} in {f.name}")
                elif l:
                    hits[c][2] += 1
                    if verbose:
                        print(f"  [{kind}] missed:      {c} in {f.name}")
        table[kind] = hits
    print(f"\n{preset} pipeline (low-light boost {'on' if det.low_light else 'off'}), image-level found/total (false alarms):")
    print(f"  {'class':18s}" + "".join(f"{k:>14s}" for k in conditions))
    for c in CANDIDATES:
        cells = []
        for k in conditions:
            tp, fp, fn = table[k][c]
            cells.append(f"{tp}/{tp + fn} ({fp})")
        print(f"  {c:18s}" + "".join(f"{x:>14s}" for x in cells))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument("--corrupt", nargs="+", default=["clean"],
                    help="clean dark very_dark glare overexposed blur (see benchmark.py)")
    ap.add_argument("--low-light", action=argparse.BooleanOptionalAction, default=None,
                    help="override the preset's dark-frame boost")
    a = ap.parse_args()
    if a.pipeline:
        eval_pipeline(a.preset, a.corrupt, a.low_light, verbose=False)
    else:
        rank_prompts(a.imgsz)
