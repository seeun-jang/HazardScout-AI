"""위험물 탐지 평가와 YOLO-World 문구 고르기 (eval_hazard.py)

test_images/ 사진마다 "이 사진에 소화기가 있다/없다"를 사람이 직접 보고 정답으로 적어뒀습니다
(아래 GROUND_TRUTH). 박스 위치까지는 안 적었고, 사진 단위로 있다/없다만 봅니다.

두 가지 모드가 있습니다:
  1) 문구 순위 (기본): 후보 문구마다 사진별 최고 점수를 구하고, 그 점수로 정답 사진과
     오답 사진을 얼마나 잘 가르는지(F1, AUC) 비교합니다. detect_pro.py의 HAZARD_PROMPTS를
     이렇게 골랐습니다.
  2) --pipeline: detect_pro 탐지기 전체를 돌려서 클래스별로 몇 장 찾았는지/오탐 몇 장인지 셉니다.
     --corrupt로 어두움, 난반사 같은 조건도 같이 볼 수 있습니다.

실행 예 (yolo_lab/ 폴더에서):
  python eval_hazard.py                                    # 문구 순위
  python eval_hazard.py --pipeline --preset max            # 탐지기 평가
  python eval_hazard.py --pipeline --corrupt clean dark very_dark glare overexposed

새 사진을 추가하면 GROUND_TRUTH에 정답을 꼭 적어야 합니다. 안 적으면 "위험물 없음"으로 취급됩니다.
"""
import argparse

import numpy as np
from ultralytics import YOLOWorld

from detect_pro import ROOT, ProDetector, resolve_weights, pick_device

IMG_DIR = ROOT / "test_images"

# 정답표: 사진 파일 이름(확장자 빼고) -> 그 사진에 있는 위험물 클래스들.
# 여기 없는 사진/클래스는 전부 "없음"으로 봅니다 (예: test_images/general/ 사진들).
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
    # 참고: extinguisher_warehouse는 소화기 '표지판'만 있고 소화기는 없음 -> 일부러 "없음" 사진
}
# 애매해서 채점에서 빼는 경우 (맞혀도 틀려도 안 셈)
IGNORE = {
    "oxygen_hospital": {"fire_extinguisher"},   # 트럭에 작은 소화기가 있긴 한데 거의 안 보임
    "propane_tanks": {"gas_cylinder"},          # 작은 가스 캔 더미라 가스통으로 볼지 애매
    "cones_car": {"warning_sign"},
}

# 클래스별로 시험해 본 후보 문구들
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
    """test_images/ 아래 모든 사진 (하위 폴더 포함)."""
    return sorted(p for p in IMG_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def labels_for(stem, cls):
    """정답 조회: 1 = 있음, 0 = 없음, None = 채점 제외."""
    if cls in IGNORE.get(stem, set()):
        return None
    return int(cls in GROUND_TRUTH.get(stem, set()))


def best_f1(scores, labels):
    """기준 점수를 바꿔가며 F1이 가장 높은 지점을 찾음.
    F1 = 놓친 것(FN)과 오탐(FP)을 같이 따지는 점수, 1이면 완벽.
    반환: (F1, 그때의 기준 점수, 맞힘 TP, 오탐 FP, 놓침 FN)"""
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
    """AUC: 정답 사진 점수가 오답 사진 점수보다 높을 확률. 1이면 완벽하게 구분, 0.5면 찍기 수준."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])).mean())


def rank_prompts(imgsz_list):
    """후보 문구마다 성능을 계산해서 클래스별로 순위를 출력.
    주의: YOLO-World는 박스마다 1등 문구만 남기므로, 비슷한 문구끼리 표를 뺏어 0점이 나올 수 있음.
    그래서 실제 선택은 문구 '세트' 전체를 비교해서 했고, "<max of all>" 줄이 세트 성능에 가까움."""
    prompts = sorted({p for ps in CANDIDATES.values() for p in ps})
    model = YOLOWorld(resolve_weights("yolov8x-worldv2.pt"))
    model.set_classes(prompts)
    files = images()
    for imgsz in imgsz_list:
        # scores[사진 번호, 문구 번호] = 그 사진에서 그 문구의 최고 점수
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
            # 후보 문구들 중 최고 점수를 쓰는 경우 (문구 여러 개를 같이 쓸 때)
            ens = scores[keep][:, [prompts.index(p) for p in cands]].max(1)
            f1, t, tp, fp, fn = best_f1(ens, y)
            rows.append((f1, auc(ens, y), "<max of all>", t, tp, fp, fn))
            for f1, a, p, t, tp, fp, fn in sorted(rows, key=lambda r: (-r[0], -r[1])):
                print(f"  {p:26s} F1={f1:.2f} AUC={a:.2f} thr={t:.3f}  TP={tp} FP={fp} FN={fn}")


def eval_pipeline(preset, conditions=("clean",), low_light=None, verbose=True):
    """detect_pro 탐지기 전체를 돌려서 클래스별 결과표를 출력.
    표의 "4/5 (1)"은 위험물이 있는 사진 5장 중 4장에서 찾았고, 없는 사진 1장에서 잘못 잡았다는 뜻.
    low_light=None이면 프리셋 기본값을 따름 (balanced/max는 켜짐)."""
    import cv2
    from benchmark import corrupt
    det = ProDetector(preset=preset, low_light=low_light, verbose=False)
    files = images()
    table = {}
    for kind in conditions:
        rng = np.random.default_rng(0)
        hits = {c: [0, 0, 0] for c in CANDIDATES}  # 클래스별 [맞힘, 오탐, 놓침]
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
    ap.add_argument("--pipeline", action="store_true", help="탐지기 전체 평가 (없으면 문구 순위)")
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960], help="문구 순위 모드의 입력 크기들")
    ap.add_argument("--corrupt", nargs="+", default=["clean"],
                    help="clean dark very_dark glare overexposed blur (see benchmark.py)")
    ap.add_argument("--low-light", action=argparse.BooleanOptionalAction, default=None,
                    help="override the preset's dark-frame boost")
    a = ap.parse_args()
    if a.pipeline:
        eval_pipeline(a.preset, a.corrupt, a.low_light, verbose=False)
    else:
        rank_prompts(a.imgsz)
