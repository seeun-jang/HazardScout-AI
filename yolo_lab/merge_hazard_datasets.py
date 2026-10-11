#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine 3 Roboflow YOLOv8 ZIP exports for HazardScout object detection.

Read the original ZIP files directly; neither the ZIPs nor existing training
folders are modified. Converts cone segmentation polygons to detection boxes.
Run from yolo_lab: python merge_hazard_datasets.py
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dataset" / "hazard_3class"
SOURCES = (
    ("fire", "fire extinguisher.v1i.yolov8.zip", {0: 0}, "fire_extinguisher"),
    ("helmet", "Hard-Hat.v1i.yolov8.zip", {1: 1}, "safety_helmet"),
    ("cone", "traffic cone.v3i.yolov8.zip", {0: 2}, "traffic_cone"),
)
SPLITS = ("train", "valid", "test")
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def make_bbox(values: list[str]) -> tuple[float, float, float, float, bool]:
    """Accept YOLO xywh (4 values) or polygon vertices (6+ values)."""
    coords = [float(v) for v in values]
    is_polygon = len(coords) >= 6 and len(coords) % 2 == 0
    if len(coords) == 4:
        cx, cy, w, h = coords
        xmin, xmax = cx - w / 2, cx + w / 2
        ymin, ymax = cy - h / 2, cy + h / 2
    elif is_polygon:
        xmin, xmax = min(coords[::2]), max(coords[::2])
        ymin, ymax = min(coords[1::2]), max(coords[1::2])
    else:
        raise ValueError(f"Invalid YOLO label, coordinate count={len(coords)}")
    xmin, xmax = max(0.0, xmin), min(1.0, xmax)
    ymin, ymax = max(0.0, ymin), min(1.0, ymax)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Degenerate/out-of-image box")
    return ((xmin + xmax) / 2, (ymin + ymax) / 2,
            xmax - xmin, ymax - ymin, is_polygon)


def build_one(zippath: Path, key: str, idmap: dict[int, int], output: Path,
              counts: Counter) -> None:
    with zipfile.ZipFile(zippath, "r") as z:
        names = set(z.namelist())
        if "data.yaml" not in names:
            raise ValueError(f"Missing data.yaml in {zippath.name}")
        for split in SPLITS:
            prefix = f"{split}/images/"
            images = sorted(n for n in names if n.startswith(prefix) and
                            n.lower().endswith(EXTENSIONS))
            if not images:
                raise ValueError(f"No {split} images in {zippath.name}")
            for image_name in images:
                original = Path(image_name).name
                stem = Path(original).stem
                label_name = f"{split}/labels/{stem}.txt"
                if label_name not in names:
                    raise ValueError(f"Missing {label_name} in {zippath.name}")
                new_stem = f"{key}_{stem}"
                img_target = output / split / "images" / f"{new_stem}{Path(original).suffix}"
                label_target = output / split / "labels" / f"{new_stem}.txt"
                with z.open(image_name) as source, img_target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                converted = []
                for num, raw_line in enumerate(z.read(label_name).decode("utf-8-sig").splitlines(), 1):
                    parts = raw_line.split()
                    if not parts:
                        continue
                    try:
                        old_class = int(parts[0])
                    except ValueError as e:
                        raise ValueError(f"Bad class ID: {label_name}:{num}") from e
                    if old_class not in idmap:
                        counts[f"ignored_{key}"] += 1
                        continue
                    try:
                        cx, cy, w, h, is_poly = make_bbox(parts[1:])
                    except ValueError as e:
                        raise ValueError(f"Bad coordinates: {label_name}:{num}: {e}") from e
                    converted.append(f"{idmap[old_class]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    counts[f"boxes_{key}_{split}"] += 1
                    if is_poly:
                        counts["cone_polygons_converted"] += 1
                label_target.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
                counts[f"images_{key}_{split}"] += 1
                if not converted:
                    counts[f"empty_{key}_{split}"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge three HazardScout YOLOv8 ZIP datasets")
    parser.add_argument("--zip-dir", type=Path, default=Path.home() / "Downloads",
                        help="Directory containing original ZIPs (default: ~/Downloads)")
    parser.add_argument("--fire", type=Path, help="Full path to fire-extinguisher ZIP")
    parser.add_argument("--helmet", type=Path, help="Full path to hard-hat ZIP")
    parser.add_argument("--cone", type=Path, help="Full path to traffic-cone ZIP")
    args = parser.parse_args()
    if OUT.exists():
        parser.error(f"Output already exists: {OUT}\nRename it before re-running; existing files are never overwritten.")
    inputs = []
    for key, basename, ids, desc in SOURCES:
        location = getattr(args, key) or (args.zip_dir / basename)
        if not location.is_file():
            parser.error(f"Cannot find {key} ZIP: {location}\nCheck Downloads or provide --{key} <full-path>.")
        inputs.append((key, location, ids, desc))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="hazard_merge_", dir=OUT.parent) as tmpdir:
        tmp = Path(tmpdir)
        for split in SPLITS:
            (tmp / split / "images").mkdir(parents=True)
            (tmp / split / "labels").mkdir(parents=True)
        for key, zippath, ids, desc in inputs:
            print(f"[MERGE] {desc}: {zippath.name}", flush=True)
            build_one(zippath, key, ids, tmp, counts)
        (tmp / "data.yaml").write_text(
            "train: train/images\nval: valid/images\ntest: test/images\n\n"
            "nc: 3\nnames:\n"
            "  0: fire_extinguisher\n"
            "  1: safety_helmet\n"
            "  2: traffic_cone\n", encoding="utf-8")
        (tmp / "SOURCES.txt").write_text(
            "Source ZIP datasets (CC BY 4.0):\n"
            "fire extinguisher: https://universe.roboflow.com/cv-projects-ab4ye/fire-extinguisher-8kk5k/dataset/1\n"
            "hard hat: https://universe.roboflow.com/tunyong/hard-hat-emnvz/dataset/1\n"
            "traffic cone: https://universe.roboflow.com/project-k84yn/traffic-cone-ry08d/dataset/3\n\n"
            "Original source ZIPs were not modified. Helmet class 'head' was excluded.\n"
            "Traffic-cone polygons were converted to detection bounding boxes.\n"
            "Keep these source links/licenses with project documentation.\n"
            "WARNING: evaluate on independent robot-camera images; source splits may include related photos.\n",
            encoding="utf-8")
        with (tmp / "merge_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "count"])
            writer.writerows(sorted(counts.items()))
        tmp.rename(OUT)
    print("\n[OK] Merged dataset:", OUT)
    for split in SPLITS:
        total = sum(counts[f"images_{k}_{split}"] for k, *_ in SOURCES)
        print(f"  {split}: {total} images")
    print("  classes: 0 fire_extinguisher, 1 safety_helmet, 2 traffic_cone")
    print(f"  converted cone polygons: {counts['cone_polygons_converted']}")
    print(f"  excluded head annotations: {counts['ignored_helmet']}")
    print("  settings: dataset/hazard_3class/data.yaml")
    print("  Source datasets and original folders remain unchanged.")


if __name__ == "__main__":
    main()
