"""HazardScout-AI 고정확도 물체 탐지 파이프라인 (detect_pro.py)

기본 YOLO 위에 아래 기법들을 얹어서 정확도를 올렸습니다.
괄호 안 숫자는 benchmark.py로 COCO128(128장)에서 잰 mAP50-95(정확도 지표, 높을수록 좋음)입니다.

  1. 큰 모델 사용        yolov8n@320 대신 yolo26x                        0.346 -> 0.618
  2. 모델 앙상블         yolo26x + yolo11x 결과를 합침(WBF)               0.618 -> 0.633
  3. TTA                 사진을 뒤집고 크기를 바꿔 여러 번 본 뒤 합침     0.633 -> 0.654
  4. 타일 분할 탐지      사진을 조각내 확대해서 봄 (--tile, 기본 꺼짐)   원거리 0.565 -> 0.592
  5. 위험물 탐지         YOLO-World에 문구를 주고 COCO에 없는 물체를 찾음
                         (소화기, 안전모, 교통 콘 등. 문구 고른 과정은 eval_hazard.py 참고)
  6. 어두운 화면 보정    어두운 프레임을 밝히고 노이즈 제거             매우 어두움 재현율 43% -> 50%

프리셋 (상황별 추천 설정):
  fast      yolo26n 하나만                    프레임당 ~0.01초 (맥 기준)   라즈베리파이, 실시간
  balanced  yolo26x + 위험물 모델 + 보정      프레임당 ~0.16초             노트북 실시간 카메라
  max       yolo26x + yolo11x + TTA + 위험물  프레임당 ~0.45초             사진/녹화 영상 정밀 분석

실행 방법 (yolo_lab/ 폴더에서):
  python detect_pro.py --source 사진.jpg
  python detect_pro.py --source 폴더 --preset max
  python detect_pro.py --source 0 --preset balanced --show     # 웹캠 (q로 종료)
  python detect_pro.py --source 영상.mp4 --preset balanced --save

다른 코드에서 가져다 쓰기:
  from detect_pro import ProDetector
  det = ProDetector(preset="balanced")
  detections = det.detect(frame_bgr)           # Detection 객체 리스트
  annotated = det.draw(frame_bgr, detections)  # 박스 그린 이미지

팀 서버/GUI에 붙일 때는 맨 아래 ProYOLODetector를 쓰면 됩니다 (기존 YOLODetector와 사용법 같음).
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

ROOT = Path(__file__).resolve().parent  # yolo_lab/ 폴더
MODELS = ROOT / "models"                # 모델 파일(.pt) 저장 위치 (git에는 안 올라감)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================================================
# 위험물 문구 설정 (YOLO-World용)
# =============================================================================
# 키 = YOLO-World에 넣는 영어 문구, 값 = 결과에 표시할 클래스 이름.
# 값이 None인 문구는 "미끼"라서 결과에서 버립니다.
#
# eval_hazard.py로 사진을 직접 보고 매긴 정답과 비교해서 골랐습니다. 알게 된 점:
#  - YOLO-World는 박스 하나에 점수가 가장 높은 문구 하나만 남깁니다.
#    그래서 비슷한 문구끼리 표를 뺏습니다. ("fire extinguisher"는 0점, "extinguisher"는 0.45점)
#  - 짧은 문구가 더 잘 됩니다. ("helmet"이 "safety helmet", "hard hat"보다 훨씬 좋음)
#  - 헷갈리기 쉬운 일반 물체(병, 모자 등)를 미끼로 넣으면, 가스통으로 착각할 병이나
#    안전모로 착각할 모자를 미끼가 대신 가져갑니다. 평균 F1 0.615 -> 0.701.
HAZARD_PROMPTS = {
    "extinguisher": "fire_extinguisher",     # 소화기
    "red extinguisher": "fire_extinguisher",
    "helmet": "safety_helmet",               # 안전모
    "traffic cone": "traffic_cone",          # 교통 콘
    "gas cylinders": "gas_cylinder",         # 가스통 (성능 낮음, 학습 필요)
    "gas tank": "gas_cylinder",
    "first aid kit": "first_aid_kit",        # 구급상자 (성능 낮음, 학습 필요)
    "hi-vis vest": "safety_vest",            # 안전조끼 (성능 낮음, 학습 필요)
    "fire": "fire",                          # 불 (테스트 사진 없음, 검증 안 됨)
    "smoke": "smoke",                        # 연기 (테스트 사진 없음, 검증 안 됨)
    # 미끼 문구: 헷갈리는 물체를 흡수하고 결과에서는 버림
    "person": None, "bottle": None, "chair": None, "car": None, "book": None, "sign": None,
    "box": None, "door": None, "hat": None, "shirt": None, "jacket": None, "pole": None,
    "trash can": None,
}
HAZARD_CLASSES = sorted({v for v in HAZARD_PROMPTS.values() if v})  # 위험물 클래스 이름 목록

# 위험물 클래스별 최소 신뢰도. 이 점수보다 낮으면 화면에 안 띄웁니다.
# YOLO-World 점수는 일반 YOLO와 기준이 달라서(대체로 낮음) 클래스마다 따로 정했습니다.
# test_images/ 사진(28장)으로 맞춘 값이라 표본이 작습니다. 실제 환경에서 다시 확인 필요.
CLASS_CONF = {
    "fire_extinguisher": 0.20, "safety_helmet": 0.08, "traffic_cone": 0.45,
    "gas_cylinder": 0.15, "safety_vest": 0.05,
    # 정답 사진이 없거나 부족한 클래스: 오탐을 막으려고 보수적으로 높게 잡음
    "first_aid_kit": 0.30, "fire": 0.40, "smoke": 0.40,
}

# =============================================================================
# 프리셋
# =============================================================================
# models   : 일반 물체(COCO 80종)용 모델 목록. 여러 개면 앙상블
# imgsz    : 모델 입력 크기. 원본보다 키우면 오히려 정확도가 떨어져서 전부 640
# tile     : 타일 분할 탐지 (작은 물체용, 9배 느림)
# tta      : 테스트 시 증강 (정확도 조금 상승, 2~3배 느림)
# world    : 위험물 모델. None이면 위험물 탐지 안 함
# conf     : 화면에 띄울 최소 신뢰도 (일반 물체 기준)
# low_light: 어두운 화면 보정
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
    """탐지 결과 하나 (박스 1개)."""
    x1: float               # 왼쪽 위 x (픽셀)
    y1: float               # 왼쪽 위 y
    x2: float               # 오른쪽 아래 x
    y2: float               # 오른쪽 아래 y
    name: str               # 클래스 이름 (person, fire_extinguisher 등)
    conf: float             # 신뢰도 0~1
    sources: list = field(default_factory=list)  # 이 박스를 찾은 모델 이름들 (앙상블 투표용)
    tile: int = -1          # 몇 번째 입력 이미지/타일에서 나왔는지 (타일 처리용)

    @property
    def box(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


# =============================================================================
# 보조 함수들
# =============================================================================
def pick_device():
    """사용할 장치 자동 선택: NVIDIA GPU > 맥 GPU(mps) > CPU."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(name):
    """모델 파일 경로 찾기. 없으면 models/ 경로를 돌려주고, ultralytics가 거기로 자동 다운로드함."""
    path = Path(name)
    if path.is_file():
        return str(path)
    if (MODELS / name).is_file():
        return str(MODELS / name)
    MODELS.mkdir(parents=True, exist_ok=True)
    return str(MODELS / name)


def enhance_low_light(img, threshold=45, target=110):
    """어두운 화면 보정. (보정된 이미지, 보정했는지 여부)를 돌려줌.

    평균 밝기가 threshold(45)보다 낮을 때만 작동합니다.
    1) 평균 밝기가 target(110)이 되도록 전체를 밝게 하고
    2) 밝히면서 같이 커진 센서 노이즈를 bilateral 필터로 줄입니다 (물체 경계는 유지됨).

    어두운 COCO 사진으로 여러 방법을 비교해서 골랐습니다 (mAP50-95, 어두움 / 매우 어두움):
      보정 없음 0.542 / 0.433, CLAHE 0.526 / 0.429, 감마 0.458 / 0.216,
      밝기만 0.539 / 0.384, 밝기 + bilateral 0.562 / 0.436  <- 채택
    기준 45: 처음엔 80이었는데, 배경만 어두운 정상 사진(평균 49)에서도 작동해서
    안전모를 놓쳤습니다. 진짜 어두운 화면에서만 켜지도록 낮췄습니다.
    """
    mean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    if mean >= threshold:
        return img, False
    gain = min(target / max(mean, 1.0), 12.0)  # 최대 12배까지만 밝힘 (너무 키우면 노이즈만 남음)
    out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(out, 7, 40, 7), True


def make_tiles(h, w, tile, overlap):
    """이미지를 겹치게 자른 타일 좌표 목록 (x1, y1, x2, y2).
    overlap=0.25면 이웃 타일과 25%씩 겹침 (경계에 걸린 물체를 놓치지 않으려고)."""
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    # 마지막 타일이 이미지 끝까지 안 닿으면 끝에 맞춘 타일을 하나 더 추가
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]


def iou_one_to_many(box, boxes):
    """박스 하나와 여러 박스 사이의 IoU(겹친 넓이 / 합친 넓이, 0~1). 1이면 완전히 같은 박스."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def weighted_boxes_fusion(dets, n_sources, iou_thr=0.55, skip_penalty=0.5):
    """WBF(Weighted Box Fusion): 여러 모델이 낸 박스를 물체당 하나로 합칩니다.

    같은 클래스끼리 IoU가 iou_thr보다 크면 같은 물체로 보고 한 묶음으로 만든 뒤,
    좌표는 신뢰도로 가중평균합니다 (점수 높은 모델의 박스 위치를 더 믿음).

    점수 = 각 모델의 최고 점수 평균 x (일부 모델만 찾았으면 깎음)
         = score * ((1 - p) + p * 찾은모델수 / 전체모델수),  p = skip_penalty
    예) 모델 2개 중 1개만 찾았으면 점수 x 0.75. 두 모델이 다 찾은 물체를 더 믿는다는 뜻.
    """
    fused = []
    by_class = {}
    for d in dets:
        by_class.setdefault(d.name, []).append(d)
    for name, items in by_class.items():
        items.sort(key=lambda d: d.conf, reverse=True)  # 점수 높은 박스부터 묶음의 기준이 됨
        clusters = []  # 묶음 목록. 묶음 하나 = 같은 물체로 판단한 박스들
        fused_boxes = np.zeros((0, 4), dtype=np.float32)  # 각 묶음의 현재 합친 박스
        for d in items:
            if len(clusters):
                ious = iou_one_to_many(d.box, fused_boxes)
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    # 기존 묶음에 추가하고, 합친 박스 좌표를 다시 계산
                    clusters[j].append(d)
                    w = np.array([c.conf for c in clusters[j]])
                    fused_boxes[j] = (np.stack([c.box for c in clusters[j]]) * w[:, None]).sum(0) / w.sum()
                    continue
            # 겹치는 묶음이 없으면 새 물체
            clusters.append([d])
            fused_boxes = np.vstack([fused_boxes, d.box[None]])
        for cl, fb in zip(clusters, fused_boxes):
            best = {}  # 모델별 최고 점수
            for c in cl:
                for s in c.sources:
                    best[s] = max(best.get(s, 0.0), c.conf)
            agree = len(best)  # 이 물체를 찾은 모델 수
            score = sum(best.values()) / agree
            score *= (1 - skip_penalty) + skip_penalty * min(agree, n_sources) / n_sources
            fused.append(Detection(*fb.tolist(), name=name, conf=float(score), sources=sorted(best)))
    return fused


def suppress_fragments(dets, contain_thr=0.85):
    """같은 클래스의 더 큰 박스 안에 85% 이상 들어가 있는 작은 박스를 지웁니다.
    타일 경계에서 물체가 잘려 생긴 '반쪽 박스'를 정리하는 용도."""
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


# =============================================================================
# 탐지기 본체
# =============================================================================
class ProDetector:
    """고정확도 탐지기. 프리셋을 고르고, 필요한 설정만 인자로 덮어쓰면 됩니다.

    예) ProDetector(preset="max", tile=True)         # max에 타일 분할 추가
        ProDetector(preset="balanced", world="none")  # 위험물 탐지 끄기

    주요 인자:
      candidate_conf: 모델에서 일단 받아오는 최소 점수. 낮게 받아서 합친 다음 최종 기준으로 거름
      wbf_iou      : 이 IoU보다 많이 겹치면 같은 물체로 봄
      tile_weight  : 타일에서만 찾은 박스의 점수 배율 (전체 이미지보다 덜 믿음)
      skip_penalty : 앙상블에서 일부 모델만 찾았을 때 깎는 정도
      world_imgsz  : 위험물 모델 입력 크기 (실험상 640이 960보다 좋았음)
    """

    def __init__(self, preset="balanced", models=None, imgsz=None, tile=None, tta=None,
                 world=None, conf=None, low_light=None, tile_size=640, tile_overlap=0.25,
                 candidate_conf=0.05, wbf_iou=0.55, tile_weight=0.9, skip_penalty=0.5, device=None,
                 world_prompts=None, world_imgsz=640, verbose=True):
        # 프리셋 값을 기본으로 두고, None이 아닌 인자만 덮어씀
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

        # 일반 물체 모델 로딩 (처음 실행이면 자동 다운로드)
        self.models = []
        for name in cfg["models"]:
            self.log(f"[ProDetector] loading {name}")
            self.models.append((Path(name).stem, YOLO(resolve_weights(name))))

        # 위험물 모델 로딩. set_classes()로 문구를 등록함 (이때 CLIP 텍스트 모델이 필요)
        self.world = None
        self.world_names = []
        if cfg["world"]:
            prompts = world_prompts or HAZARD_PROMPTS
            self.log(f"[ProDetector] loading {cfg['world']} with {len(prompts)} hazard prompts")
            self.world = YOLOWorld(resolve_weights(cfg["world"]))
            self.world.set_classes(list(prompts))
            self.world_names = [prompts[p] for p in prompts]  # 문구 번호 -> 클래스 이름 (미끼는 None)

        # WBF 투표에서 "전체 모델 수". 모델 하나가 한 표.
        self.n_coco_sources = len(self.models)

    # -- 모델 실행 ---------------------------------------------------------------
    def _predict(self, model, imgs, imgsz, offsets, source, names_override=None):
        """모델 하나를 이미지(들)에 돌려서 Detection 리스트로 바꿈.
        offsets: 타일일 때 원본 이미지 기준 좌표로 되돌리기 위한 (x, y) 이동량."""
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
        """타일 분할용 조각들을 만듦."""
        h, w = img.shape[:2]
        # 이미지가 타일보다 별로 안 크면, 2배로 키운 뒤 자름 (조각이 '확대' 효과를 내도록)
        scale = 1.0
        if max(h, w) < self.tile_size * 1.5:
            scale = self.tile_size * 2 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        wins = make_tiles(h, w, self.tile_size, self.tile_overlap)
        crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]
        return crops, wins, scale, (w, h)

    def _tile_detections(self, tag, model, img):
        """타일에서 찾은 '작은 물체'만 남김.
        타일 경계에 걸려 잘린 박스와 큰 박스는 버림 (큰 물체는 전체 이미지에서 더 잘 찾음).
        처음엔 이 필터가 없어서 반쪽 박스와 중복이 생겨 정확도가 오히려 떨어졌음."""
        crops, wins, scale, (w, h) = self._tiles_for(img)
        dets = self._predict(model, crops, self.tile_size, [(x1, y1) for x1, y1, _, _ in wins], tag)
        kept = []
        m = 4  # 경계에서 4픽셀 이내면 '잘린 박스'로 봄
        max_side = self.tile_size * 0.5
        for d in dets:
            x1, y1, x2, y2 = wins[d.tile]
            # 이미지 진짜 가장자리가 아닌 타일 경계에 닿아 있으면 잘린 것
            cut = ((d.x1 <= x1 + m and x1 > 0) or (d.y1 <= y1 + m and y1 > 0) or
                   (d.x2 >= x2 - m and x2 < w) or (d.y2 >= y2 - m and y2 < h))
            if cut or max(d.x2 - d.x1, d.y2 - d.y1) > max_side:
                continue
            # 확대했던 만큼 원래 좌표로 되돌림
            d.x1, d.y1, d.x2, d.y2 = d.x1 / scale, d.y1 / scale, d.x2 / scale, d.y2 / scale
            kept.append(d)
        return kept

    def detect(self, img_bgr):
        """이미지 한 장(OpenCV BGR)에서 물체를 찾아 Detection 리스트를 돌려줌 (점수 높은 순)."""
        h, w = img_bgr.shape[:2]
        img = img_bgr

        # 1) 어두우면 보정
        self.last_enhanced = False
        if self.low_light:
            img, self.last_enhanced = enhance_low_light(img_bgr)

        # 2) 일반 물체 탐지 (모델마다 실행 -> WBF로 합침)
        # 원본보다 크게 확대하지 않음: 640 사진을 1280으로 키웠더니 박스가 부정확해짐
        # (mAP50-95 0.618 -> 0.451)
        full_sz = min(self.cfg["imgsz"], int(np.ceil(max(h, w) / 32) * 32))
        coco, tiles = [], []
        for tag, model in self.models:
            coco += self._predict(model, [img], full_sz, [(0, 0)], tag)
            if self.cfg["tile"]:
                tiles += self._tile_detections(tag, model, img)
        fused = weighted_boxes_fusion(coco, self.n_coco_sources, self.wbf_iou, self.skip_penalty)

        # 3) 타일 결과는 전체 이미지에서 '놓친' 물체만 추가 (작은 물체 / 먼 물체)
        if tiles:
            tiles = weighted_boxes_fusion(tiles, self.n_coco_sources, self.wbf_iou, self.skip_penalty)
            strong = [d for d in fused if d.conf >= self.candidate_conf * 2]
            boxes = np.stack([d.box for d in strong]) if strong else np.zeros((0, 4), np.float32)
            for t in tiles:
                if len(boxes) and iou_one_to_many(t.box, boxes).max() > 0.3:
                    continue  # 이미 전체 이미지에서 찾은 물체
                t.conf *= self.tile_weight
                t.sources = [s + ":tile" for s in t.sources]
                fused.append(t)

        # 4) 위험물 탐지 (YOLO-World)
        if self.world is not None:
            wd = self._predict(self.world, [img], self.world_imgsz, [(0, 0)], "world",
                               names_override=self.world_names)
            # 미끼 문구(name=None)는 버리고, 클래스별 기준 점수를 넘는 것만 남김
            hazards = [h for h in weighted_boxes_fusion([d for d in wd if d.name], 1, self.wbf_iou, 0.0)
                       if h.conf >= CLASS_CONF.get(h.name, self.cfg["conf"])]
            if hazards:
                # 같은 물체에 일반 박스와 위험물 박스가 겹치면 위험물 쪽을 남김
                # (소화기를 일반 모델은 "bottle", 가스통을 "vase"로 잡는 경우)
                hb = np.stack([h.box for h in hazards])
                fused = [d for d in fused if iou_one_to_many(d.box, hb).max() < 0.5]
            fused += hazards

        # 5) 정리: 반쪽 박스 제거 -> 기준 점수 미달 제거 -> 이미지 밖 좌표 자르기
        fused = suppress_fragments(fused)
        out = []
        for d in fused:
            if d.conf < CLASS_CONF.get(d.name, self.cfg["conf"]):
                continue
            d.x1, d.x2 = float(np.clip(d.x1, 0, w)), float(np.clip(d.x2, 0, w))
            d.y1, d.y2 = float(np.clip(d.y1, 0, h)), float(np.clip(d.y2, 0, h))
            if d.x2 - d.x1 >= 2 and d.y2 - d.y1 >= 2:  # 너무 작은(2px 미만) 박스는 버림
                out.append(d)
        return sorted(out, key=lambda d: d.conf, reverse=True)

    def detect_raw(self, img_bgr):
        """최종 기준 점수로 거르지 않은 모든 후보를 돌려줌.
        정확도(mAP) 측정은 낮은 점수 후보까지 다 있어야 해서 benchmark.py에서만 씀."""
        saved = self.cfg["conf"], dict(CLASS_CONF)
        self.cfg["conf"] = self.candidate_conf
        CLASS_CONF.clear()
        try:
            return self.detect(img_bgr)
        finally:
            # 측정이 끝나면 원래 기준으로 복구
            self.cfg["conf"] = saved[0]
            CLASS_CONF.update(saved[1])

    # -- 그리기 ------------------------------------------------------------------
    @staticmethod
    def draw(img_bgr, dets, header=None):
        """박스를 그린 새 이미지를 돌려줌. 위험물은 빨간색, 일반 물체는 초록색.
        header를 주면 왼쪽 위에 노란 글씨로 표시."""
        out = img_bgr.copy()
        # 이미지 크기에 맞춰 선 두께와 글자 크기 조절
        thick = max(2, round(max(out.shape[:2]) / 600))
        fs = max(0.5, max(out.shape[:2]) / 1600)
        for d in dets:
            hazard = d.name in HAZARD_CLASSES
            color = (0, 0, 255) if hazard else (0, 200, 0)  # OpenCV는 BGR 순서
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, color, thick)
            label = f"{d.name} {d.conf:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            ty = max(p1[1], th + bl + 4)  # 박스가 맨 위에 붙어 있으면 라벨을 안쪽으로
            cv2.rectangle(out, (p1[0], ty - th - bl - 4), (p1[0] + tw + 6, ty), color, -1)
            cv2.putText(out, label, (p1[0] + 3, ty - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
        if header:
            cv2.putText(out, header, (10, int(30 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 255, 255), thick, cv2.LINE_AA)
        return out


class ProYOLODetector:
    """팀 코드(code/YOLO_detector.py)의 YOLODetector를 그대로 대체하는 클래스.

    사용법이 똑같아서 서버/GUI에서 import 한 줄만 바꾸면 됩니다:
        from detect_pro import ProYOLODetector as YOLODetector
        detector = YOLODetector(preset="balanced", detect_every_n_frames=3)
        frame = detector.detect_and_draw(frame)

    - detect_every_n_frames: N프레임마다 한 번만 탐지하고, 그 사이에는 마지막 박스를 다시 그림
    - self.last_detections: 기존과 같은 형식 [(x1, y1, x2, y2, 클래스이름, 신뢰도), ...]
    - balanced/max는 노트북 GPU용입니다. 라즈베리파이 4에서는 preset="fast"를 쓰세요.
    """

    def __init__(self, preset="balanced", detect_every_n_frames=3, **kwargs):
        self.detector = ProDetector(preset=preset, **kwargs)
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))
        self.frame_count = 0
        self.last = []             # 마지막 탐지 결과 (Detection 객체)
        self.last_detections = []  # 기존 YOLODetector와 같은 튜플 형식

    def detect_and_draw(self, frame_bgr):
        self.frame_count += 1
        if self.frame_count == 1 or self.frame_count % self.detect_every_n_frames == 0:
            try:
                self.last = self.detector.detect(frame_bgr)
                self.last_detections = [(int(d.x1), int(d.y1), int(d.x2), int(d.y2), d.name, d.conf)
                                        for d in self.last]
            except Exception as e:
                # 탐지가 실패해도 영상은 계속 나가도록 에러만 출력
                print("[ProYOLO] detection error:", repr(e))
        return ProDetector.draw(frame_bgr, self.last, f"YOLO Objects: {len(self.last)}")


# =============================================================================
# 명령줄 실행 (python detect_pro.py ...)
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="HazardScout 고정확도 물체 탐지")
    p.add_argument("--source", required=True, help="사진, 폴더, 영상 파일, 또는 웹캠 번호(0)")
    p.add_argument("--preset", choices=PRESETS, default="balanced")
    p.add_argument("--models", nargs="+", help="일반 물체 모델 직접 지정 (예: yolo26x.pt yolo11x.pt)")
    p.add_argument("--imgsz", type=int, help="모델 입력 크기")
    p.add_argument("--conf", type=float, help="화면에 띄울 최소 신뢰도 (낮추면 더 많이 잡고 오탐도 늘어남)")
    p.add_argument("--tile", action=argparse.BooleanOptionalAction, default=None,
                   help="타일 분할 탐지 (먼 물체용, 느림)")
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None, help="테스트 시 증강")
    p.add_argument("--no-world", action="store_true", help="위험물 탐지 끄기")
    p.add_argument("--low-light", action=argparse.BooleanOptionalAction, default=None,
                   help="어두운 화면 보정 (balanced/max는 기본으로 켜짐)")
    p.add_argument("--out", help="결과 폴더 (기본: results/pro_<프리셋>_<시각>)")
    p.add_argument("--show", action="store_true", help="결과 창 띄우기")
    p.add_argument("--save", action="store_true", help="영상/웹캠 결과를 mp4로 저장")
    p.add_argument("--max-frames", type=int, default=0, help="영상에서 처리할 최대 프레임 수 (0=전부)")
    args = p.parse_args()

    det = ProDetector(preset=args.preset, models=args.models, imgsz=args.imgsz, conf=args.conf,
                      tile=args.tile, tta=args.tta, world="none" if args.no_world else None,
                      low_light=args.low_light)
    out_dir = Path(args.out) if args.out else ROOT / "results" / f"pro_{args.preset}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 입력 종류 판단: 폴더 / 사진 한 장 / 그 외(영상 파일 또는 웹캠)
    src = Path(args.source)
    if src.is_dir():
        files = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
    elif src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        files = [src]
    else:
        files = None

    rows = []      # detections.csv에 쓸 줄들
    summary = {}   # summary.json에 쓸 사진별 요약
    if files is not None:
        # ---- 사진 처리 ----
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
                cv2.waitKey(0)  # 아무 키나 누르면 다음 사진
    else:
        # ---- 영상 / 웹캠 처리 ----
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

    # 결과 파일 저장: 박스 목록(CSV, 엑셀로 열림)과 요약(JSON)
    with (out_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_or_frame", "class", "confidence", "x1", "y1", "x2", "y2", "sources", "ms"])
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(dict(preset=args.preset, config=det.cfg, images=summary),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results: {out_dir}")


if __name__ == "__main__":
    main()
