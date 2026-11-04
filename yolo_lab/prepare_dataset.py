"""검토한 YOLO Bounding Box 라벨만 복사합니다. 자동 라벨링/다운로드 없음."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from lab_utils import ROOT, NAMES, IMAGE_SUFFIXES, local_path, output_path


def read_boxes(path):
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue  # 명시적으로 검토한 빈 라벨만 배경 이미지로 허용
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{number}: class_id cx cy w h 형식만 허용")
        cls = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if not 0 <= cls < len(NAMES) or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"{path}:{number}: 잘못된 클래스/좌표")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{number}: 정규화 좌표 범위 오류")
        if min(x-w/2, y-h/2) < -0.001 or max(x+w/2, y+h/2) > 1.001:
            raise ValueError(f"{path}:{number}: 이미지 밖 Bounding Box")
        boxes.append((cls, x, y, w, h))
    return boxes


def image_hashes(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image.load()
        gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
        pixels = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
    dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return hashlib.sha256(path.read_bytes()).hexdigest(), dhash


def check_dataset(folder, minimum=30):
    """학습/평가 전에 라벨, 7종 수량, split 간 중복을 다시 검증합니다."""
    import yaml
    cfg = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
    expected = {i: name for i, name in enumerate(NAMES)}
    if cfg.get("names") != expected:
        raise ValueError("data.yaml의 클래스 ID/이름이 safety7 순서와 다릅니다.")
    counts = {split: Counter() for split in ("train", "val", "test")}
    seen = []; seen_sha = set()
    for split in counts:
        if cfg.get(split) != f"images/{split}":
            raise ValueError(f"{split} 경로는 images/{split}이어야 합니다.")
        images = sorted(p for p in (folder / "images" / split).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"{split} 이미지가 없습니다.")
        for image in images:
            boxes = read_boxes(folder / "labels" / split / f"{image.stem}.txt")
            sha, dhash = image_hashes(image)
            if sha in seen_sha:
                raise ValueError(f"중복 이미지: {image}")
            if any((dhash ^ old).bit_count() <= 5 for old, _ in seen):
                raise ValueError(f"유사 이미지 재검토 필요: {image}")
            seen.append((dhash, split)); seen_sha.add(sha)
            counts[split].update({box[0] for box in boxes})
    missing = []
    for cls, name in enumerate(NAMES):
        total = sum(c[cls] for c in counts.values())
        if total < minimum or any(c[cls] == 0 for c in counts.values()):
            missing.append(f"{name}: 총 {total}, train/val/test=" + "/".join(str(c[cls]) for c in counts.values()))
    if missing:
        raise ValueError("7종 학습 준비 부족 (각 클래스 총 30장 이상 + 모든 split 양성 필요):\n" + "\n".join(missing))
    return counts


def build(manifest, folder):
    import yaml
    catalog = json.loads((ROOT / "dataset/sources.json").read_text(encoding="utf-8"))
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("manifest.csv에 검토한 이미지가 없습니다.")
    if folder.exists():
        raise ValueError(f"덮어쓰기 금지: {folder}; --output으로 새 폴더를 지정하세요.")
    accepted = []; skipped = []; groups = {}; originals = set(); hashes = []; sha_seen = set()
    for number, row in enumerate(rows, 2):
        if row["reviewed"] != "yes":
            skipped.append({"row": number, "reason": "not_reviewed"}); continue
        source = catalog[row["source"]]
        if not source.get("license") or not source.get("url") or not source.get("allow_local_use"):
            raise ValueError(f"출처/사용 권한 확인 필요: {row['source']}")
        split, group = row["split"], row["group"]
        if split not in ("train", "val", "test") or not group:
            raise ValueError(f"행 {number}: split/group 오류")
        if group in groups and groups[group] != split:
            raise ValueError(f"동일 촬영 그룹의 split 누출: {group}")
        groups[group] = split
        image, label = local_path(row["image"]), local_path(row["label"])
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES or not label.is_file():
            raise ValueError(f"행 {number}: 이미지/라벨 없음")
        boxes = read_boxes(label)
        sha, dhash = image_hashes(image)
        original = (row["source"], image.name.split(".rf.")[0]) if ".rf." in image.name else str(image)
        reason = "same_original" if original in originals else "same_sha256" if sha in sha_seen else "near_duplicate" if any((dhash ^ h).bit_count() <= 5 for h in hashes) else None
        if reason:
            skipped.append({"row": number, "image": row["image"], "reason": reason}); continue
        originals.add(original); sha_seen.add(sha); hashes.append(dhash)
        accepted.append((row, image, label, boxes, sha))
    # 모든 입력을 먼저 검사했으므로 잘못된 라벨로 반쯤 만들어진 데이터셋을 남기지 않습니다.
    counts = {split: Counter() for split in ("train", "val", "test")}
    records = []
    for split in counts:
        (folder / "images" / split).mkdir(parents=True)
        (folder / "labels" / split).mkdir(parents=True)
    for row, image, label, boxes, sha in accepted:
        stem = sha[:20]; split = row["split"]
        target = folder / "images" / split / (stem + image.suffix.lower())
        shutil.copy2(image, target)
        shutil.copy2(label, folder / "labels" / split / f"{stem}.txt")
        counts[split].update({box[0] for box in boxes})
        records.append({**row, "output": target.relative_to(folder).as_posix(), "sha256": sha})
    cfg = {"train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(NAMES))}
    (folder / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    report = {"images": len(accepted), "counts": {s: {n: c[i] for i,n in enumerate(NAMES)} for s,c in counts.items()}, "skipped": skipped, "records": records}
    (folder / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("records", "skipped")}, ensure_ascii=False, indent=2))
    print(f"제외 {len(skipped)}개. 상세: {folder / 'build_report.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/manifest.csv")
    parser.add_argument("--output", default="dataset/safety7")
    parser.add_argument("--check", action="store_true", help="기존 폴더의 학습 준비 검사; 파일 변경 없음")
    args = parser.parse_args()
    try:
        folder = output_path(args.output)
        if args.check:
            print(check_dataset(folder)); return
        build(local_path(args.manifest), folder)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
