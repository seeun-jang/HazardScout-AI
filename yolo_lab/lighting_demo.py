"""조명이 나쁠 때 탐지가 얼마나 버티는지 눈으로 확인하는 스크립트 (lighting_demo.py)

사진 한 장을 5가지 조건으로 바꿔서 각각 탐지하고, 결과를 가로로 이어 붙인 이미지 한 장으로 저장합니다.
  왼쪽부터: 정상 | 어두움 | 매우 어두움 | 난반사 | 과노출
  (각 조건을 어떻게 만드는지는 benchmark.py의 corrupt() 참고)
각 칸 위에 "찾은 물체 수, 위험물 수"가 적힙니다.

실행 예 (yolo_lab/ 폴더에서):
  python lighting_demo.py                                    # 예시 사진 3장 (소화기, 안전모, 콘)
  python lighting_demo.py --source 내사진.jpg --preset max
  python lighting_demo.py --source test_images/hazard         # 폴더 전체
결과: results/lighting_demo/<사진이름>_lighting.jpg

로봇 카메라로 찍은 사진으로 돌려보는 게 인터넷 사진보다 훨씬 의미 있습니다.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from benchmark import corrupt
from detect_pro import HAZARD_CLASSES, IMAGE_EXTS, ROOT, ProDetector

CONDITIONS = ["clean", "dark", "very_dark", "glare", "overexposed"]  # 이 순서대로 왼쪽부터 붙음
DEFAULT_SOURCES = ["test_images/hazard/extinguisher_hallway2.jpg",
                   "test_images/hazard/hardhat_amsterdam.jpg",
                   "test_images/hazard/cones_car.jpg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", nargs="+", default=DEFAULT_SOURCES, help="images or folders")
    p.add_argument("--preset", default="balanced")
    p.add_argument("--out", default="results/lighting_demo")
    args = p.parse_args()

    # 입력 경로들을 사진 파일 목록으로 펼침 (폴더면 안의 사진 전부)
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
            # seed를 고정해서 몇 번을 돌려도 같은 빛 번짐/노이즈가 나오게 함
            x = corrupt(img, kind, np.random.default_rng(3))
            dets = det.detect(x)
            hazards = sum(d.name in HAZARD_CLASSES for d in dets)
            vis = det.draw(x, dets)
            # 칸 하나를 가로 480픽셀로 맞추고 위에 검은 띠 + 조건 이름/개수 표시
            h, w = vis.shape[:2]
            vis = cv2.resize(vis, (480, int(h * 480 / w)))
            cv2.rectangle(vis, (0, 0), (480, 34), (0, 0, 0), -1)
            cv2.putText(vis, f"{kind}: {len(dets)} obj, {hazards} hazard", (6, 24), 0, 0.65, (0, 255, 255), 2)
            tiles.append(vis)
            summary.append(f"{kind} {len(dets)}/{hazards}")
        # 칸 높이를 맞춘 뒤(아래를 검게 채움) 가로로 이어 붙임
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
        path = out_dir / f"{f.stem}_lighting.jpg"
        cv2.imwrite(str(path), np.hstack(tiles))
        print(f"{f.name}: " + ", ".join(summary) + "   (objects/hazards)")
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
