"""Measure detection accuracy against ground-truth labels (COCO-format YOLO dataset).

Compares the team's current setup (yolov8n @ 320) with detect_pro configurations,
on clean images and on degraded images (dark / blur / small) that mimic a robot camera.

Usage (from yolo_lab/):
  python benchmark.py                                  # default suite on datasets/coco128
  python benchmark.py --configs team fast pro_x --corrupt clean dark
  python benchmark.py --limit 32                       # quick run
Writes results/benchmark_<time>/benchmark.md and .json
"""
import argparse
import json
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics.utils.metrics import ap_per_class, box_iou
import torch

from detect_pro import ProDetector, ROOT


CONFIGS = {
    # name: (description, ProDetector kwargs)
    "team":      ("Team now: yolov8n @320 (code/YOLO_detector.py)",
                  dict(preset="fast", models=["yolov8n.pt"], imgsz=320, low_light=False)),
    "fast":      ("fast preset: yolo26n @640", dict(preset="fast")),
    "x640":      ("yolo26x @640 (balanced preset, no hazard model)", dict(preset="balanced", world="none", low_light=False)),
    "ens":       ("yolo26x + yolo11x, WBF", dict(preset="max", tta=False, world="none", low_light=False)),
    "ens_tile":  ("ensemble + tiling", dict(preset="max", tta=False, tile=True, world="none", low_light=False)),
    "ens_tta":   ("ensemble + TTA (max preset, no hazard model)", dict(preset="max", world="none", low_light=False)),
    "x640_tta":  ("yolo26x @640 + TTA", dict(preset="balanced", tta=True, world="none", low_light=False)),
    "x640_light": ("balanced preset as shipped: yolo26x + low-light boost", dict(preset="balanced", world="none")),
    "max_light": ("max preset as shipped: ensemble + TTA + low-light boost", dict(preset="max", world="none")),
}
DEFAULT_SUITE = ["team", "fast", "x640", "ens", "ens_tta"]


def corrupt(img, kind, rng):
    if kind == "clean":
        return img
    if kind == "dark":  # underexposed + sensor noise
        out = img.astype(np.float32) * 0.22
        out += rng.normal(0, 4, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "blur":  # motion blur from a moving robot
        k = np.zeros((15, 15), np.float32)
        k[7, :] = 1 / 15
        return cv2.filter2D(img, -1, k)
    if kind == "small":  # objects far away: shrink, pad back to the original canvas
        h, w = img.shape[:2]
        canvas = np.full_like(img, 114)
        s = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        canvas[: h // 2, : w // 2] = s
        return canvas
    if kind == "very_dark":  # night / unlit room: 8% brightness, strong sensor noise
        out = img.astype(np.float32) * 0.08
        out += rng.normal(0, 3, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "glare":  # specular reflections / light bloom: 1-3 bright blobs + haze veil
        h, w = img.shape[:2]
        out = img.astype(np.float32) + 25  # veiling glare lowers contrast everywhere
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(rng.integers(1, 4)):
            cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
            r = rng.uniform(0.12, 0.30) * min(h, w)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
            out += (blob * rng.uniform(140, 240))[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "overexposed":  # camera auto-exposure blown out (window, flashlight)
        out = img.astype(np.float32) * 1.8 + 40
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(kind)


def corrupt_labels(gt, kind):
    if kind == "small":
        gt = gt.copy()
        gt[:, 1:] /= 2
    return gt


def load_labels(label_path, w, h):
    if not label_path.is_file():
        return np.zeros((0, 5), np.float32)
    rows = np.loadtxt(label_path, ndmin=2).astype(np.float32)
    cls, xc, yc, bw, bh = rows.T
    return np.stack([cls, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], 1)


def match(pred_boxes, pred_cls, gt, iouv):
    """Ultralytics-style greedy matching -> (n_pred, n_iou) bool TP matrix."""
    tp = np.zeros((len(pred_boxes), len(iouv)), bool)
    if len(pred_boxes) == 0 or len(gt) == 0:
        return tp
    iou = box_iou(torch.tensor(gt[:, 1:]), torch.tensor(pred_boxes)).numpy()
    correct_class = gt[:, :1] == pred_cls[None]
    iou = iou * correct_class
    for i, thr in enumerate(iouv):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1].astype(int), i] = True
    return tp


def evaluate(det, images, labels_dir, kind, seed=0):
    rng = np.random.default_rng(seed)
    iouv = np.linspace(0.5, 0.95, 10)
    name_to_id = {v: k for k, v in det.models[0][1].names.items()}
    stats = dict(tp=[], conf=[], pcls=[], tcls=[])
    op_tp = op_fp = n_gt = 0
    times = []
    for f in images:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        gt = corrupt_labels(load_labels(labels_dir / f"{f.stem}.txt", w, h), kind)
        x = corrupt(img, kind, rng)
        t0 = time.perf_counter()
        dets = [d for d in det.detect_raw(x) if d.name in name_to_id]
        times.append((time.perf_counter() - t0) * 1000)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], np.float32).reshape(-1, 4)
        conf = np.array([d.conf for d in dets], np.float32)
        pcls = np.array([name_to_id[d.name] for d in dets], np.float32)
        tp = match(boxes, pcls, gt, iouv)
        stats["tp"].append(tp); stats["conf"].append(conf); stats["pcls"].append(pcls)
        stats["tcls"].append(gt[:, 0])
        # operating point = what the robot would actually show
        keep = conf >= det.cfg["conf"]
        op_tp += int(tp[keep, 0].sum()); op_fp += int((~tp[keep, 0]).sum()); n_gt += len(gt)
    tp = np.concatenate(stats["tp"]); conf = np.concatenate(stats["conf"])
    pcls = np.concatenate(stats["pcls"]); tcls = np.concatenate(stats["tcls"])
    res = ap_per_class(tp, conf, pcls, tcls)
    ap = res[5]  # (n_cls, 10)
    return dict(
        mAP50=float(ap[:, 0].mean()), mAP50_95=float(ap.mean()),
        precision=op_tp / max(op_tp + op_fp, 1), recall=op_tp / max(n_gt, 1),
        found=op_tp, false_alarms=op_fp, objects=n_gt,
        ms=float(np.median(times)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/coco128")
    p.add_argument("--configs", nargs="+", default=DEFAULT_SUITE, choices=CONFIGS)
    p.add_argument("--corrupt", nargs="+", default=["clean", "dark", "blur", "small"])
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    data = ROOT / args.data
    images = sorted((data / "images").rglob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    labels_dir = next((data / "labels").iterdir())
    print(f"{len(images)} images, configs={args.configs}, conditions={args.corrupt}")

    out_dir = ROOT / "results" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.configs:
        desc, kw = CONFIGS[name]
        det = ProDetector(verbose=False, **kw)
        for img in images[:2]:  # warm-up (model compile / MPS init)
            det.detect(cv2.imread(str(img)))
        results[name] = {}
        for kind in args.corrupt:
            r = evaluate(det, images, labels_dir, kind)
            results[name][kind] = r
            print(f"{name:9s} {kind:6s} mAP50={r['mAP50']:.3f} mAP50-95={r['mAP50_95']:.3f} "
                  f"P={r['precision']:.2f} R={r['recall']:.2f} {r['ms']:.0f}ms", flush=True)
        del det

    lines = [f"# Benchmark ({len(images)} images, {args.data})", "",
             "mAP50-95 = COCO standard accuracy. Recall/Precision measured at the preset's display threshold.", ""]
    for kind in args.corrupt:
        lines += [f"## {kind}", "", "| config | description | mAP50 | mAP50-95 | recall | precision | ms/img |",
                  "|---|---|---|---|---|---|---|"]
        for name in args.configs:
            r = results[name][kind]
            lines.append(f"| {name} | {CONFIGS[name][0]} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | "
                         f"{r['recall']:.2f} | {r['precision']:.2f} | {r['ms']:.0f} |")
        lines.append("")
    (out_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'benchmark.md'}")


if __name__ == "__main__":
    main()
