"""safety7 복구 검증/패키징 (표준 라이브러리만 사용, 원본 파일은 읽기만 함).

사용 예 (yolo_lab 폴더에서):
  python recover_safety7.py scan                       # 검사만. results/recovery_<시각>/ 에 보고서 작성
  python recover_safety7.py scan --extra D:\\USB폴더     # 다른 위치도 함께 검색
  python recover_safety7.py zip                        # 213장 + 7종 라벨이 모두 검증될 때만 새 ZIP 생성
  python recover_safety7.py zip --allow-partial        # 일부만 있어도 '부분' ZIP 생성 (이름에 partial 표시)
  python recover_safety7.py verify <ZIP경로>           # 전달 후 ZIP CRC + 이미지/라벨 해시 재검증
  python recover_safety7.py install <ZIP경로>          # dataset/safety7/images, labels 가 없을 때만 풀기

원칙
- 기존 파일을 수정/삭제/덮어쓰기하지 않는다. 새 파일은 yolo_lab/results/recovery_<시각>/ 와
  (install 시) 아직 없는 dataset/safety7/images, dataset/safety7/labels 에만 만든다.
- 이미지 일치는 build_report.json 의 SHA-256 으로만 판단한다.
- 7종 라벨로 인정하는 것은 (a) dataset/safety7/labels/<split>/<sha앞20>.txt 또는
  (b) build_report 에 기록된 raw 라벨 경로 뿐이다. 그 외(백업 사본 등)는 '검토 필요 후보'로만 보고한다.
- hazard_3class 등 3종 라벨은 절대 7종 라벨로 인정하지 않는다.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ["traffic_cone", "fire_extinguisher", "safety_helmet", "safety_vest",
         "safety_gloves", "earmuffs", "safety_goggles"]
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", ".config"}
REPORT = ROOT / "dataset/safety7/build_report.json"
PROVENANCE = [
    "dataset/safety7/data.yaml", "dataset/safety7/build_report.json", "dataset/manifest.csv",
    "dataset/sources.json", "dataset/SOURCES.md", "dataset/reference_manifest.json",
    "dataset/raw/curated/sh17/PROVENANCE.json", "dataset/raw/curated/sh17/ORIGINAL_IMAGE_URLS.json",
    "dataset/raw/local_reference/ANNOTATIONS.json",
    "dataset/raw/curated/sh17/ORIGINAL_README.md", "dataset/raw/curated/sh17/ORIGINAL_sh17.yaml",
    "results/training_preflight.json", "results/verification_final.json",
]


def relpath(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_label(data):
    """prepare_dataset.read_boxes 와 같은 규칙. (boxes, error) 반환."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, "utf-8 아님"
    boxes = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split()
        if len(f) != 5:
            return None, f"{n}행: 5열 아님"
        try:
            cls = int(f[0]); x, y, w, h = map(float, f[1:])
        except ValueError:
            return None, f"{n}행: 숫자 아님"
        if not 0 <= cls < len(NAMES) or not all(math.isfinite(v) for v in (x, y, w, h)):
            return None, f"{n}행: 클래스/좌표 오류"
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None, f"{n}행: 정규화 범위 오류"
        if min(x - w / 2, y - h / 2) < -0.001 or max(x + w / 2, y + h / 2) > 1.001:
            return None, f"{n}행: 이미지 밖 bbox"
        boxes.append(cls)
    return boxes, None


def walk_images(roots, exclude):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if Path(dirpath).resolve() == exclude:
                dirnames[:] = []; continue
            for name in filenames:
                if os.path.splitext(name)[1].lower() in IMG_EXT:
                    yield Path(dirpath) / name


def list_weights_and_runs():
    out = {"weights": [], "runs_results_csv": []}
    for p in ROOT.rglob("*.pt"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        st = p.stat()
        out["weights"].append({"path": str(p), "bytes": st.st_size,
                               "modified": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                               "sha256": sha256(p)})
    for p in ROOT.rglob("results.csv"):
        lines = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        out["runs_results_csv"].append({"path": str(p), "rows": max(len(lines) - 1, 0),
                                        "header": lines[0] if lines else "", "last": lines[-1] if len(lines) > 1 else ""})
    return out


def scan(extra_roots, outdir):
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    records = report["records"]
    want = {r["sha256"]: r for r in records}
    roots = [ROOT] + [Path(p) for p in extra_roots]
    found = defaultdict(list)
    n = 0
    for p in walk_images(roots, outdir.resolve()):
        n += 1
        if n % 500 == 0:
            print(f"  이미지 {n}개 해시 계산 중...", flush=True)
        try:
            h = sha256(p)
        except OSError as e:
            print(f"  읽기 실패: {p} ({e})"); continue
        if h in want:
            found[h].append(str(p))
    print(f"  검색한 이미지 {n}개")

    rows = []
    pos = {s: Counter() for s in SPLITS}; box = {s: Counter() for s in SPLITS}
    for r in records:
        h, split = r["sha256"], r["split"]
        stem = h[:20]
        cand = {
            "safety7_label": ROOT / "dataset/safety7/labels" / split / f"{stem}.txt",
            "raw_label": ROOT / r["label"],
        }
        labels = {}
        for kind, p in cand.items():
            if p.is_file():
                data = p.read_bytes()
                boxes, err = parse_label(data)
                labels[kind] = {"path": str(p), "sha256": hashlib.sha256(data).hexdigest(),
                                "boxes": len(boxes) if boxes is not None else None, "error": err,
                                "_classes": boxes}
        # 보조 후보: 찾은 이미지 옆 labels/ 폴더 (3종 데이터 경로는 제외), 검토 필요로만 보고
        secondary = []
        for img in found.get(h, []):
            ip = Path(img)
            if "hazard_3class" in ip.parts:
                continue
            sib = ip.parent.parent / "labels" / (ip.stem + ".txt")
            if sib.is_file() and str(sib) not in {v["path"] for v in labels.values()}:
                secondary.append({"path": str(sib), "sha256": sha256(sib)})
        primary = [v for v in labels.values() if v["error"] is None]
        status_label = "missing"
        if primary:
            hashes = {v["sha256"] for v in primary}
            status_label = "ok" if len(hashes) == 1 else "conflict(safety7≠raw)"
        if any(v["error"] for v in labels.values()):
            status_label = "invalid" if not primary else status_label + "+invalid_copy"
        chosen = primary[0] if status_label == "ok" else None
        if chosen:
            cls = chosen["_classes"]
            pos[split].update(set(cls)); box[split].update(cls)
        for v in labels.values():
            v.pop("_classes", None)
        rows.append({"output": r["output"], "source": r["source"], "split": split, "sha256": h,
                     "image_found": found.get(h, []), "labels": labels,
                     "label_status": status_label, "secondary_label_candidates": secondary})

    expected = report["counts"]
    pos_table = {s: {NAMES[i]: pos[s][i] for i in range(7)} for s in SPLITS}
    summary = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "yolo_lab": str(ROOT), "extra_roots": [str(p) for p in extra_roots],
        "records": len(records),
        "image_found": sum(1 for x in rows if x["image_found"]),
        "image_found_by_source": dict(Counter(x["source"] for x in rows if x["image_found"])),
        "label_ok": sum(1 for x in rows if x["label_status"] == "ok"),
        "complete_pairs": sum(1 for x in rows if x["image_found"] and x["label_status"] == "ok"),
        "label_status_counts": dict(Counter(x["label_status"] for x in rows)),
        "images_with_multiple_copies": sum(1 for x in rows if len(x["image_found"]) > 1),
        "positive_images_verified_labels": pos_table,
        "boxes_verified_labels": {s: {NAMES[i]: box[s][i] for i in range(7)} for s in SPLITS},
        "positive_images_expected": expected,
        "positive_counts_match_expected": pos_table == expected,
        "missing": [{"output": x["output"], "source": x["source"], "split": x["split"],
                     "image": not x["image_found"], "label": x["label_status"]}
                    for x in rows if not x["image_found"] or x["label_status"] != "ok"],
        **list_weights_and_runs(),
    }
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "scan_detail.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (outdir / "scan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    short = {k: v for k, v in summary.items() if k not in ("missing",)}
    short["missing_count"] = len(summary["missing"])
    print(json.dumps(short, ensure_ascii=False, indent=1))
    print(f"보고서: {outdir}")
    return rows, summary


def make_zip(rows, summary, outdir, allow_partial):
    complete = [x for x in rows if x["image_found"] and x["label_status"] == "ok"]
    full = len(complete) == summary["records"] and summary["positive_counts_match_expected"]
    if not full and not allow_partial:
        print(f"ZIP 미생성: 완전한 쌍 {len(complete)}/{summary['records']}. "
              f"일부라도 묶으려면 --allow-partial."); return None
    stamp = outdir.name.replace("recovery_", "")
    zpath = outdir / (f"safety7_{'full' if full else 'partial'}_{stamp}.zip")
    manifest = {"kind": "full" if full else "partial", "pairs": len(complete),
                "records": summary["records"], "files": {}}
    with zipfile.ZipFile(zpath, "x", zipfile.ZIP_DEFLATED) as z:
        for x in complete:
            img = Path(x["image_found"][0])
            lab = Path(next(v for v in x["labels"].values() if v["error"] is None)["path"])
            stem = x["sha256"][:20]
            ai = f"safety7/images/{x['split']}/{stem}.jpg"
            al = f"safety7/labels/{x['split']}/{stem}.txt"
            z.write(img, ai); z.write(lab, al)
            manifest["files"][ai] = {"sha256": x["sha256"], "from": relpath(img)}
            manifest["files"][al] = {"sha256": sha256(lab), "from": relpath(lab)}
        for rel in PROVENANCE:
            p = ROOT / rel
            if p.is_file():
                z.write(p, "records/" + rel)
                manifest["files"]["records/" + rel] = {"sha256": sha256(p), "from": relpath(p)}
        z.write(Path(__file__), "tools/recover_safety7.py")
        manifest["files"]["tools/recover_safety7.py"] = {"sha256": sha256(Path(__file__)), "from": "recover_safety7.py"}
        z.writestr("scan_summary.json", json.dumps(summary, ensure_ascii=False, indent=1))
        z.writestr("RECOVERY_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"ZIP 생성: {zpath} ({zpath.stat().st_size:,} bytes)")
    verify(zpath)
    return zpath


def verify(zpath):
    zpath = Path(zpath)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        man = json.loads(z.read("RECOVERY_MANIFEST.json"))
        mismatch = []
        for n, m in man["files"].items():
            try:
                ok = hashlib.sha256(z.read(n)).hexdigest() == m["sha256"]
            except (zipfile.BadZipFile, KeyError, OSError):
                ok = False
            if not ok:
                mismatch.append(n)
    result = {"zip": str(zpath), "bytes": zpath.stat().st_size, "zip_sha256": sha256(zpath),
              "crc_ok": bad is None, "files_checked": len(man["files"]), "hash_mismatch": mismatch,
              "kind": man["kind"], "pairs": man["pairs"]}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def install(zpath):
    zpath = Path(zpath)
    res = verify(zpath)
    if not res["crc_ok"] or res["hash_mismatch"]:
        sys.exit("검증 실패: 설치하지 않습니다.")
    target = ROOT / "dataset/safety7"
    for d in ("images", "labels"):
        if (target / d).exists():
            sys.exit(f"이미 존재: {target / d} — 덮어쓰지 않습니다. 비교 후 따로 결정하세요.")
    with zipfile.ZipFile(zpath) as z:
        for rel in ("dataset/safety7/data.yaml", "dataset/safety7/build_report.json"):
            local = ROOT / rel
            if local.is_file() and ("records/" + rel) in z.namelist():
                a, b = z.read("records/" + rel), local.read_bytes()
                if a == b:
                    msg = "ZIP과 동일"
                elif a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"):
                    msg = "줄바꿈(CRLF/LF)만 다르고 내용 동일 (로컬 파일 유지)"
                else:
                    msg = "내용이 ZIP과 다름 (로컬 파일 유지, 덮어쓰지 않음 — 확인 필요)"
                print(f"{rel}: {msg}")
        names = [n for n in z.namelist() if n.startswith(("safety7/images/", "safety7/labels/"))]
        for n in names:
            dest = target / n[len("safety7/"):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "xb") as f:
                f.write(z.read(n))
    print(f"설치 완료: {len(names)}개 파일 -> {target}. 다음: python prepare_dataset.py --check")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scan", "zip"):
        p = sub.add_parser(name)
        p.add_argument("--extra", action="append", default=[], help="추가 검색 폴더 (읽기 전용)")
        if name == "zip":
            p.add_argument("--allow-partial", action="store_true")
    sub.add_parser("verify").add_argument("zip")
    sub.add_parser("install").add_argument("zip")
    a = ap.parse_args()
    if a.cmd in ("verify", "install"):
        (verify if a.cmd == "verify" else install)(a.zip); return
    base = "recovery_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir, k = ROOT / "results" / base, 1
    while outdir.exists():
        k += 1; outdir = ROOT / "results" / f"{base}_{k}"
    rows, summary = scan(a.extra, outdir)
    if a.cmd == "zip":
        make_zip(rows, summary, outdir, a.allow_partial)


if __name__ == "__main__":
    main()
