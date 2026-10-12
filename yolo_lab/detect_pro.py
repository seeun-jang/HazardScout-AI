"""High-accuracy object detection pipeline for HazardScout-AI.

On top of plain YOLO (numbers = COCO128 mAP50-95 measured with benchmark.py):
  1. Bigger model         yolo26x instead of yolov8n@320          0.346 -> 0.618
  2. Model ensemble       yolo26x + yolo11x, Weighted Box Fusion  0.618 -> 0.633
  3. TTA                  flip + multi-scale at test time         0.633 -> 0.654
  4. Tiled inference      SAHI-style crops, opt-in (--tile)       far objects 0.565 -> 0.592
  5. Open vocabulary      YOLO-World with tuned prompts adds hazard classes COCO lacks
                          (fire extinguisher, helmet, traffic cone, ...; see eval_hazard.py)
  6. Low-light boost      gain + bilateral denoise on dark frames   very dark recall 43% -> 50%

Presets:
  fast      yolo26n                       ~30 ms   Raspberry Pi / real time
  balanced  yolo26x + hazard model        ~0.2 s   laptop, live camera
  max       yolo26x + yolo11x + TTA + hazard ~0.6 s  offline photos/video, best accuracy

Usage (from yolo_lab/):
  python detect_pro.py --source path/to/image.jpg
  python detect_pro.py --source path/to/folder --preset max
  python detect_pro.py --source 0 --preset fast --show    # webcam
  python detect_pro.py --source test.mp4 --preset balanced --save

Library usage:
  from detect_pro import ProDetector
  det = ProDetector(preset="balanced")
  detections = det.detect(frame_bgr)
  annotated = det.draw(frame_bgr, detections)
"""
import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOWorld

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Hazard vocabulary for YOLO-World. key = prompt text, value = class name we report.
# Chosen with eval_hazard.py on hand-checked photos. Findings:
#  - YOLO-World keeps only the TOP prompt per box, so near-synonyms steal votes from each
#    other ("fire extinguisher" scored 0 while "extinguisher" scored 0.45).
#  - Short prompts beat long ones ("helmet" >> "safety helmet", "hard hat").
#  - Distractor prompts (value None) soak up look-alikes (bottle vs gas cylinder,
#    hat vs helmet) and are dropped from the output. They raised mean F1 0.615 -> 0.701.
HAZARD_PROMPTS = {
    "extinguisher": "fire_extinguisher",
    "red extinguisher": "fire_extinguisher",
    "helmet": "safety_helmet",
    "traffic cone": "traffic_cone",
    "gas cylinders": "gas_cylinder",
    "gas tank": "gas_cylinder",
    "first aid kit": "first_aid_kit",
    "hi-vis vest": "safety_vest",
    "fire": "fire",
    "smoke": "smoke",
    # distractors
    "person": None, "bottle": None, "chair": None, "car": None, "book": None, "sign": None,
    "box": None, "door": None, "hat": None, "shirt": None, "jacket": None, "pole": None,
    "trash can": None,
}
HAZARD_CLASSES = sorted({v for v in HAZARD_PROMPTS.values() if v})

# Minimum confidence per hazard class (YOLO-World scores are on a different scale than COCO
# models, so these replace the preset's global conf). Tuned on test_images/ (small set!).
CLASS_CONF = {
    "fire_extinguisher": 0.20, "safety_helmet": 0.08, "traffic_cone": 0.45,
    "gas_cylinder": 0.15, "safety_vest": 0.05,
    # no positive test photos for these -> conservative, verify before relying on them
    "first_aid_kit": 0.30, "fire": 0.40, "smoke": 0.40,
}

# imgsz 640 everywhere: upscaling past native resolution hurt accuracy in benchmarks.
PRESETS = {
    "fast": dict(models=["yolo26n.pt"], imgsz=640, tile=False, tta=False, world=None,
                 conf=0.25, low_light=False),
    "balanced": dict(models=["yolo26x.pt"], imgsz=640, tile=False, tta=False,
                     world="yolov8x-worldv2.pt", conf=0.25, low_light=True),
    "max": dict(models=["yolo26x.pt", "yolo11x.pt"], imgsz=640, tile=False, tta=True,
                world="yolov8x-worldv2.pt", conf=0.25, low_light=True),
}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    name: str
    conf: float
    sources: list = field(default_factory=list)
    tile: int = -1  # index of the input image/tile this box came from

    @property
    def box(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


# ----------------------------------------------------------------------------- helpers
def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(name):
    path = Path(name)
    if path.is_file():
        return str(path)
    if (MODELS / name).is_file():
        return str(MODELS / name)
    MODELS.mkdir(parents=True, exist_ok=True)
    return str(MODELS / name)  # ultralytics downloads official weights to this path


def enhance_low_light(img, threshold=45, target=110):
    """Brighten dark frames, then smooth the amplified sensor noise. Returns (image, applied).

    Picked by experiment on dark COCO images (yolo26x, mAP50-95 dark / very dark):
      off 0.542 / 0.433, CLAHE 0.526 / 0.429, gamma 0.458 / 0.216, gain 0.539 / 0.384,
      gain + bilateral 0.562 / 0.436  <- this one
    """
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    if mean >= threshold:
        return img, False
    gain = min(target / max(mean, 1.0), 12.0)
    out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(out, 7, 40, 7), True


def make_tiles(h, w, tile, overlap):
    """Overlapping tile windows covering the image (x1, y1, x2, y2)."""
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]


def iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def weighted_boxes_fusion(dets, n_sources, iou_thr=0.55, skip_penalty=0.5):
    """Class-wise WBF. Each cluster -> one box, coords averaged by confidence.

    Score = mean of the best score per source, scaled down when only some
    sources agree:  score * ((1 - p) + p * agreeing / n_sources).
    """
    fused = []
    by_class = {}
    for d in dets:
        by_class.setdefault(d.name, []).append(d)
    for name, items in by_class.items():
        items.sort(key=lambda d: d.conf, reverse=True)
        clusters = []  # each: list of Detection
        fused_boxes = np.zeros((0, 4), dtype=np.float32)
        for d in items:
            if len(clusters):
                ious = iou_one_to_many(d.box, fused_boxes)
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    clusters[j].append(d)
                    w = np.array([c.conf for c in clusters[j]])
                    fused_boxes[j] = (np.stack([c.box for c in clusters[j]]) * w[:, None]).sum(0) / w.sum()
                    continue
            clusters.append([d])
            fused_boxes = np.vstack([fused_boxes, d.box[None]])
        for cl, fb in zip(clusters, fused_boxes):
            best = {}
            for c in cl:
                for s in c.sources:
                    best[s] = max(best.get(s, 0.0), c.conf)
            agree = len(best)
            score = sum(best.values()) / agree
            score *= (1 - skip_penalty) + skip_penalty * min(agree, n_sources) / n_sources
            fused.append(Detection(*fb.tolist(), name=name, conf=float(score), sources=sorted(best)))
    return fused


def suppress_fragments(dets, contain_thr=0.85):
    """Drop boxes mostly contained in a bigger, more confident box of the same class.
    Removes half-objects produced by tiles cutting through an object."""
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep = []
    for d in dets:
        a = (d.x2 - d.x1) * (d.y2 - d.y1)
        drop = False
        for k in keep:
            if k.name != d.name:
                continue
            ix = max(0, min(d.x2, k.x2) - max(d.x1, k.x1))
            iy = max(0, min(d.y2, k.y2) - max(d.y1, k.y1))
            ka = (k.x2 - k.x1) * (k.y2 - k.y1)
            if ka > a and ix * iy / max(a, 1e-9) > contain_thr:
                drop = True
                break
        if not drop:
            keep.append(d)
    return keep


# ----------------------------------------------------------------------------- detector
class ProDetector:
    def __init__(self, preset="balanced", models=None, imgsz=None, tile=None, tta=None,
                 world=None, conf=None, low_light=None, tile_size=640, tile_overlap=0.25,
                 candidate_conf=0.05, wbf_iou=0.55, tile_weight=0.9, skip_penalty=0.5, device=None,
                 world_prompts=None, world_imgsz=640, verbose=True):
        cfg = dict(PRESETS[preset])
        for k, v in dict(models=models, imgsz=imgsz, tile=tile, tta=tta, world=world, conf=conf,
                         low_light=low_light).items():
            if v is not None:
                cfg[k] = v
        if cfg["world"] in ("", "none", False):
            cfg["world"] = None
        self.cfg = cfg
        self.preset = preset
        self.low_light = cfg["low_light"]
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.candidate_conf = candidate_conf
        self.wbf_iou = wbf_iou
        self.world_imgsz = world_imgsz
        self.tile_weight = tile_weight
        self.skip_penalty = skip_penalty
        self.device = device or pick_device()
        self.log = print if verbose else (lambda *a, **k: None)

        self.models = []
        for name in cfg["models"]:
            self.log(f"[ProDetector] loading {name}")
            self.models.append((Path(name).stem, YOLO(resolve_weights(name))))

        self.world = None
        self.world_names = []
        if cfg["world"]:
            prompts = world_prompts or HAZARD_PROMPTS
            self.log(f"[ProDetector] loading {cfg['world']} with {len(prompts)} hazard prompts")
            self.world = YOLOWorld(resolve_weights(cfg["world"]))
            self.world.set_classes(list(prompts))
            self.world_names = [prompts[p] for p in prompts]

        # A "source" is one independent opinion that WBF counts votes from.
        self.n_coco_sources = len(self.models)

    # -- raw predictions -------------------------------------------------------
    def _predict(self, model, imgs, imgsz, offsets, source, names_override=None):
        out = []
        results = model.predict(imgs, imgsz=imgsz, conf=self.candidate_conf, iou=0.6,
                                augment=self.cfg["tta"], device=self.device, verbose=False,
                                max_det=300)
        for i, (r, (ox, oy)) in enumerate(zip(results, offsets)):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                name = names_override[k] if names_override else r.names[k]
                out.append(Detection(x1 + ox, y1 + oy, x2 + ox, y2 + oy, name, float(c), [source], i))
        return out

    def _tiles_for(self, img):
        h, w = img.shape[:2]
        # Tile only when the image is noticeably larger than one tile;
        # upscale small images so tiles still zoom in.
        scale = 1.0
        if max(h, w) < self.tile_size * 1.5:
            scale = self.tile_size * 2 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        wins = make_tiles(h, w, self.tile_size, self.tile_overlap)
        crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]
        return crops, wins, scale, (w, h)

    def _tile_detections(self, tag, model, img):
        """Small objects found in zoomed tiles. Boxes cut by a tile border and
        large boxes are dropped: the full-image pass handles those better."""
        crops, wins, scale, (w, h) = self._tiles_for(img)
        dets = self._predict(model, crops, self.tile_size, [(x1, y1) for x1, y1, _, _ in wins], tag)
        kept = []
        m = 4
        max_side = self.tile_size * 0.5
        for d in dets:
            x1, y1, x2, y2 = wins[d.tile]
            cut = ((d.x1 <= x1 + m and x1 > 0) or (d.y1 <= y1 + m and y1 > 0) or
                   (d.x2 >= x2 - m and x2 < w) or (d.y2 >= y2 - m and y2 < h))
            if cut or max(d.x2 - d.x1, d.y2 - d.y1) > max_side:
                continue
            d.x1, d.y1, d.x2, d.y2 = d.x1 / scale, d.y1 / scale, d.x2 / scale, d.y2 / scale
            kept.append(d)
        return kept

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img = img_bgr
        self.last_enhanced = False
        if self.low_light:
            img, self.last_enhanced = enhance_low_light(img_bgr)

        # Never upscale past the native resolution: enlarging a 640px frame to 1280
        # made boxes less precise in benchmarks (mAP50-95 0.618 -> 0.566).
        full_sz = min(self.cfg["imgsz"], int(np.ceil(max(h, w) / 32) * 32))
        coco, tiles = [], []
        for tag, model in self.models:
            coco += self._predict(model, [img], full_sz, [(0, 0)], tag)
            if self.cfg["tile"]:
                tiles += self._tile_detections(tag, model, img)
        fused = weighted_boxes_fusion(coco, self.n_coco_sources, self.wbf_iou, self.skip_penalty)
        if tiles:
            # Tiles only ADD objects the full-image pass missed (small / far away).
            tiles = weighted_boxes_fusion(tiles, self.n_coco_sources, self.wbf_iou, self.skip_penalty)
            strong = [d for d in fused if d.conf >= self.candidate_conf * 2]
            boxes = np.stack([d.box for d in strong]) if strong else np.zeros((0, 4), np.float32)
            for t in tiles:
                if len(boxes) and iou_one_to_many(t.box, boxes).max() > 0.3:
                    continue
                t.conf *= self.tile_weight
                t.sources = [s + ":tile" for s in t.sources]
                fused.append(t)

        if self.world is not None:
            wd = self._predict(self.world, [img], self.world_imgsz, [(0, 0)], "world",
                               names_override=self.world_names)
            hazards = [h for h in weighted_boxes_fusion([d for d in wd if d.name], 1, self.wbf_iou, 0.0)
                       if h.conf >= CLASS_CONF.get(h.name, self.cfg["conf"])]
            if hazards:
                # A hazard box is more specific than the COCO box on the same object
                # (extinguisher vs "bottle", gas cylinder vs "vase"): drop the COCO one.
                hb = np.stack([h.box for h in hazards])
                fused = [d for d in fused if iou_one_to_many(d.box, hb).max() < 0.5]
            fused += hazards

        fused = suppress_fragments(fused)
        out = []
        for d in fused:
            if d.conf < CLASS_CONF.get(d.name, self.cfg["conf"]):
                continue
            d.x1, d.x2 = float(np.clip(d.x1, 0, w)), float(np.clip(d.x2, 0, w))
            d.y1, d.y2 = float(np.clip(d.y1, 0, h)), float(np.clip(d.y2, 0, h))
            if d.x2 - d.x1 >= 2 and d.y2 - d.y1 >= 2:
                out.append(d)
        return sorted(out, key=lambda d: d.conf, reverse=True)

    def detect_raw(self, img_bgr):
        """All fused candidates without the final threshold (for mAP evaluation)."""
        saved = self.cfg["conf"], dict(CLASS_CONF)
        self.cfg["conf"] = self.candidate_conf
        CLASS_CONF.clear()
        try:
            return self.detect(img_bgr)
        finally:
            self.cfg["conf"] = saved[0]
            CLASS_CONF.update(saved[1])

    # -- drawing ---------------------------------------------------------------
    @staticmethod
    def draw(img_bgr, dets, header=None):
        out = img_bgr.copy()
        thick = max(2, round(max(out.shape[:2]) / 600))
        fs = max(0.5, max(out.shape[:2]) / 1600)
        for d in dets:
            hazard = d.name in HAZARD_CLASSES
            color = (0, 0, 255) if hazard else (0, 200, 0)
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, color, thick)
            label = f"{d.name} {d.conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            ty = max(p1[1], th + bl + 4)
            cv2.rectangle(out, (p1[0], ty - th - bl - 4), (p1[0] + tw + 6, ty), color, -1)
            cv2.putText(out, label, (p1[0] + 3, ty - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
        if header:
            cv2.putText(out, header, (10, int(30 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 255, 255), thick, cv2.LINE_AA)
        return out


class ProYOLODetector:
    """Drop-in replacement for code/YOLO_detector.py's YOLODetector.

    Same detect_and_draw(frame_bgr) interface and same frame-skipping idea, so a server
    or GUI can switch with one line:
        from detect_pro import ProYOLODetector as YOLODetector
    Heavy presets (balanced/max) need a laptop GPU; on a Raspberry Pi 4 use preset="fast".
    self.last_detections keeps the old tuple format (x1, y1, x2, y2, class_name, confidence).
    """

    def __init__(self, preset="balanced", detect_every_n_frames=3, **kwargs):
        self.detector = ProDetector(preset=preset, **kwargs)
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.frame_count = 0
        self.last = []
        self.last_detections = []

    def detect_and_draw(self, frame_bgr):
        self.frame_count += 1
        if self.frame_count == 1 or self.frame_count % self.detect_every_n_frames == 0:
            try:
                self.last = self.detector.detect(frame_bgr)
                self.last_detections = [(int(d.x1), int(d.y1), int(d.x2), int(d.y2), d.name, d.conf)
                                        for d in self.last]
            except Exception as e:
                print("[ProYOLO] detection error:", repr(e))
        return ProDetector.draw(frame_bgr, self.last, f"YOLO Objects: {len(self.last)}")


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="HazardScout high-accuracy detector")
    p.add_argument("--source", required=True, help="image, folder, video, or camera index (0)")
    p.add_argument("--preset", choices=PRESETS, default="balanced")
    p.add_argument("--models", nargs="+", help="override COCO models, e.g. yolo26x.pt yolo11x.pt")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--conf", type=float)
    p.add_argument("--tile", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no-world", action="store_true", help="disable hazard open-vocabulary model")
    p.add_argument("--low-light", action=argparse.BooleanOptionalAction, default=None,
                   help="dark-frame boost (on by default in balanced/max)")
    p.add_argument("--out", help="output folder (default results/pro_<time>)")
    p.add_argument("--show", action="store_true", help="open a preview window")
    p.add_argument("--save", action="store_true", help="save annotated video (video/camera)")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    det = ProDetector(preset=args.preset, models=args.models, imgsz=args.imgsz, conf=args.conf,
                      tile=args.tile, tta=args.tta, world="none" if args.no_world else None,
                      low_light=args.low_light)
    out_dir = Path(args.out) if args.out else ROOT / "results" / f"pro_{args.preset}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    if src.is_dir():
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    elif src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        files = [src]
    else:
        files = None

    rows = []
    summary = {}
    if files is not None:
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"skip (unreadable): {f}")
                continue
            t0 = time.perf_counter()
            dets = det.detect(img)
            ms = (time.perf_counter() - t0) * 1000
            header = f"{args.preset} | {len(dets)} objects | {ms:.0f} ms" + (" | low-light" if det.last_enhanced else "")
            cv2.imwrite(str(out_dir / f"{f.stem}_pro.jpg"), det.draw(img, dets, header))
            counts = {}
            for d in dets:
                counts[d.name] = counts.get(d.name, 0) + 1
                rows.append([f.name, d.name, round(d.conf, 4), *[round(v) for v in (d.x1, d.y1, d.x2, d.y2)],
                             "+".join(d.sources), round(ms, 1)])
            summary[f.name] = dict(objects=len(dets), ms=round(ms, 1), low_light=det.last_enhanced, counts=counts)
            print(f"{f.name}: {len(dets)} objects in {ms:.0f} ms -> {counts}")
            if args.show:
                cv2.imshow("detect_pro", det.draw(img, dets, header))
                cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        if not cap.isOpened():
            p.error(f"cannot open source: {args.source}")
        writer, n = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            t0 = time.perf_counter()
            dets = det.detect(frame)
            ms = (time.perf_counter() - t0) * 1000
            vis = det.draw(frame, dets, f"{args.preset} | frame {n} | {len(dets)} obj | {ms:.0f} ms")
            for d in dets:
                rows.append([n, d.name, round(d.conf, 4), *[round(v) for v in (d.x1, d.y1, d.x2, d.y2)],
                             "+".join(d.sources), round(ms, 1)])
            if args.save:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    writer = cv2.VideoWriter(str(out_dir / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps if 1 <= fps <= 120 else 15.0, vis.shape[1::-1])
                writer.write(vis)
            if args.show:
                cv2.imshow("detect_pro (q=quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and n >= args.max_frames:
                break
        cap.release()
        if writer is not None:
            writer.release()
        print(f"processed {n} frames")
    if args.show:
        cv2.destroyAllWindows()

    with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_or_frame", "class", "confidence", "x1", "y1", "x2", "y2", "sources", "ms"])
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(dict(preset=args.preset, config=det.cfg, images=summary),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
