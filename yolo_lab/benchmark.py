"""정답 라벨과 비교해서 탐지 정확도를 숫자로 재는 스크립트 (benchmark.py)

팀 현재 설정(yolov8n @320)과 detect_pro의 여러 구성을 같은 사진으로 비교합니다.
깨끗한 사진뿐 아니라, 로봇 카메라에서 생길 법한 문제(어두움, 흔들림, 원거리, 난반사 등)를
사진에 인위적으로 만들어서 그 상황에서도 얼마나 버티는지 봅니다.

데이터: datasets/coco128 (ultralytics 공식 테스트셋 128장, 사람이 그린 정답 박스 포함)
  받는 법은 README_DETECT_PRO.md의 "재현" 참고

실행 예 (yolo_lab/ 폴더에서):
  python benchmark.py                                         # 기본 구성 전부
  python benchmark.py --configs team fast --corrupt clean dark  # 원하는 것만
  python benchmark.py --limit 32                              # 32장만 빠르게
결과: results/benchmark_<시각>/benchmark.md (표), benchmark.json

출력 보는 법:
  mAP50     = 박스가 정답과 50% 이상 겹치면 맞은 걸로 칠 때의 정확도
  mAP50-95  = 겹침 기준을 50%~95%로 바꿔가며 평균낸 정확도 (COCO 표준, 가장 엄격)
  P (정밀도) = 화면에 띄운 박스 중 실제로 맞은 비율 (낮으면 오탐이 많음)
  R (재현율) = 실제 물체 중 찾아낸 비율 (낮으면 놓치는 게 많음)
  ms        = 사진 한 장 처리 시간 (중앙값)
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


# 비교할 구성 목록. 이름: (설명, ProDetector에 넘길 인자)
# world="none": 위험물 모델은 COCO 정답에 없는 클래스라 정확도 측정에서는 끔
CONFIGS = {
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
DEFAULT_SUITE = ["team", "fast", "x640", "ens", "ens_tta"]  # 옵션 없이 실행하면 돌리는 구성


def corrupt(img, kind, rng):
    """사진에 인위적인 문제를 만듦. kind 종류:
      clean       원본 그대로
      dark        밝기 22% + 센서 노이즈 (어두운 실내)
      very_dark   밝기 8% + 노이즈 (불 꺼진 방)
      blur        가로 15픽셀 모션 블러 (로봇이 움직이는 중)
      small       사진을 1/2로 줄여 왼쪽 위에 붙임 (물체가 멀리 있음)
      glare       밝은 빛 번짐 1~3개 + 전체가 뿌옇게 (난반사, 조명 반사)
      overexposed 밝기 1.8배 (창문이나 손전등을 정면으로 비출 때)
    rng: 난수 생성기. 같은 seed면 매번 같은 결과가 나와서 비교가 공정함."""
    if kind == "clean":
        return img
    if kind == "dark":
        out = img.astype(np.float32) * 0.22
        out += rng.normal(0, 4, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "blur":
        k = np.zeros((15, 15), np.float32)
        k[7, :] = 1 / 15  # 가로 한 줄로 평균 = 옆으로 움직일 때 생기는 번짐
        return cv2.filter2D(img, -1, k)
    if kind == "small":
        h, w = img.shape[:2]
        canvas = np.full_like(img, 114)  # 나머지는 회색으로 채움
        s = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        canvas[: h // 2, : w // 2] = s
        return canvas
    if kind == "very_dark":
        out = img.astype(np.float32) * 0.08
        out += rng.normal(0, 3, img.shape)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "glare":
        h, w = img.shape[:2]
        out = img.astype(np.float32) + 25  # 전체가 뿌옇게 (대비 감소)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(rng.integers(1, 4)):
            # 무작위 위치에 둥근 빛 번짐 (가운데가 가장 밝고 바깥으로 갈수록 흐려짐)
            cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
            r = rng.uniform(0.12, 0.30) * min(h, w)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
            out += (blob * rng.uniform(140, 240))[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "overexposed":
        out = img.astype(np.float32) * 1.8 + 40
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(kind)


def corrupt_labels(gt, kind):
    """사진을 변형한 만큼 정답 박스도 맞춰 바꿈 (small은 좌표가 절반이 됨)."""
    if kind == "small":
        gt = gt.copy()
        gt[:, 1:] /= 2
    return gt


def load_labels(label_path, w, h):
    """YOLO 라벨 파일(클래스 x중심 y중심 너비 높이, 0~1 비율)을 픽셀 좌표 [클래스, x1, y1, x2, y2]로 변환."""
    if not label_path.is_file():
        return np.zeros((0, 5), np.float32)
    rows = np.loadtxt(label_path, ndmin=2).astype(np.float32)
    cls, xc, yc, bw, bh = rows.T
    return np.stack([cls, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h], 1)


def match(pred_boxes, pred_cls, gt, iouv):
    """예측 박스가 정답과 맞았는지 판정 (ultralytics 공식 val과 같은 방식).
    결과: (예측 수, 겹침 기준 10개) 크기의 True/False 표.
    정답 하나에는 예측 하나만 맞을 수 있음 (같은 물체에 박스 두 개면 하나는 오탐)."""
    tp = np.zeros((len(pred_boxes), len(iouv)), bool)
    if len(pred_boxes) == 0 or len(gt) == 0:
        return tp
    iou = box_iou(torch.tensor(gt[:, 1:]), torch.tensor(pred_boxes)).numpy()
    correct_class = gt[:, :1] == pred_cls[None]
    iou = iou * correct_class  # 클래스가 다르면 겹쳐도 틀린 것
    for i, thr in enumerate(iouv):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                # 많이 겹치는 쌍부터, 정답과 예측이 한 번씩만 짝지어지게
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1].astype(int), i] = True
    return tp


def evaluate(det, images, labels_dir, kind, seed=0):
    """탐지기 하나를 사진 전체에 돌려서 정확도 지표를 계산."""
    rng = np.random.default_rng(seed)
    iouv = np.linspace(0.5, 0.95, 10)  # 겹침 기준 0.50, 0.55, ..., 0.95
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
        # mAP 계산에는 점수 낮은 후보까지 전부 필요해서 detect_raw 사용
        dets = [d for d in det.detect_raw(x) if d.name in name_to_id]
        times.append((time.perf_counter() - t0) * 1000)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], np.float32).reshape(-1, 4)
        conf = np.array([d.conf for d in dets], np.float32)
        pcls = np.array([name_to_id[d.name] for d in dets], np.float32)
        tp = match(boxes, pcls, gt, iouv)
        stats["tp"].append(tp); stats["conf"].append(conf); stats["pcls"].append(pcls)
        stats["tcls"].append(gt[:, 0])
        # 정밀도/재현율은 실제 화면에 띄우는 기준(conf)으로 계산 = 로봇이 실제로 보여줄 결과
        keep = conf >= det.cfg["conf"]
        op_tp += int(tp[keep, 0].sum()); op_fp += int((~tp[keep, 0]).sum()); n_gt += len(gt)
    tp = np.concatenate(stats["tp"]); conf = np.concatenate(stats["conf"])
    pcls = np.concatenate(stats["pcls"]); tcls = np.concatenate(stats["tcls"])
    res = ap_per_class(tp, conf, pcls, tcls)
    ap = res[5]  # 클래스별, 겹침 기준별 AP (클래스 수 x 10)
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
    p.add_argument("--limit", type=int, default=0)  # 0이면 전부, 숫자면 그 장수만
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
        for img in images[:2]:  # 예열: 첫 실행은 GPU 초기화 때문에 느려서 시간 측정에서 뺌
            det.detect(cv2.imread(str(img)))
        results[name] = {}
        for kind in args.corrupt:
            r = evaluate(det, images, labels_dir, kind)
            results[name][kind] = r
            print(f"{name:9s} {kind:6s} mAP50={r['mAP50']:.3f} mAP50-95={r['mAP50_95']:.3f} "
                  f"P={r['precision']:.2f} R={r['recall']:.2f} {r['ms']:.0f}ms", flush=True)
        del det  # 다음 구성 전에 메모리 비우기

    # 결과를 마크다운 표로 저장
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
