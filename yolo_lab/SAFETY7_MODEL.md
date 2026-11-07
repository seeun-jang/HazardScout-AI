# safety7.pt — 안전 장비 7종 모델 (2026-11-07)

`models/safety7.pt`는 실제로 학습한 YOLOv8n 7종 탐지 모델입니다. COCO 80종은 포함하지 않습니다(`yolov8n.pt`와 별도 운영).

| 항목 | 값 |
|---|---|
| 클래스(순서 고정) | 0 traffic_cone, 1 fire_extinguisher, 2 safety_helmet, 3 safety_vest, 4 safety_gloves, 5 earmuffs, 6 safety_goggles |
| 파일 | `models/safety7.pt` 6,258,794 bytes |
| SHA-256 | `91e388a76c09f24141ea4a522466ef81bc92a071e7898dfdee0789746928d8b9` |
| 학습 | yolov8n.pt(COCO 사전학습) → 100 epoch, imgsz 640, batch 8, seed 42, Ultralytics 8.4.174(Pi와 동일) |
| 장비 | 노트북 CPU i7-12700H (CUDA 없음, `train_yolo.py --device cpu --allow-cpu`), 1.22시간 |
| 데이터 | `dataset/safety7` 213장(train 145 / val 40 / test 28), build_report.json SHA-256 213/213 일치 |
| 기록 | `models/training_info.json`, `models/safety7_eval.json` |

## 성능 (mAP50, Ultralytics 기본 평가)

| 클래스 | val@640 | test@640 | val@320 | test@320 |
|---|---:|---:|---:|---:|
| traffic_cone | 0.87 | 0.86 | 0.75 | 0.84 |
| fire_extinguisher | 0.95 | 0.99 | 0.78 | 0.99 |
| safety_helmet | 0.76 | 0.74 | 0.54 | 0.69 |
| safety_vest | 0.80 | 0.56 | 0.65 | 0.55 |
| safety_gloves | 0.32 | 0.08 | 0.41 | 0.19 |
| earmuffs | 0.03 | 0.58 | 0.00 | 0.39 |
| safety_goggles | 0.31 | 0.12 | 0.39 | 0.21 |
| **전체 mAP50** | **0.58** | **0.56** | **0.50** | **0.55** |
| 전체 mAP50-95 | 0.37 | 0.33 | 0.29 | 0.32 |
| 전체 P / R | 0.84 / 0.54 | 0.61 / 0.55 | 0.61 / 0.45 | 0.84 / 0.48 |

- 모델 선택은 **val 기준**입니다. test는 선택에 쓰지 않았습니다.
  50 epoch 후보(SHA `389dbef7…`, 노트북 `backups/20261107_safety7_e50/`)와 val 종합점수가 같아서,
  val 정밀도와 장갑 성능이 더 나은 100 epoch를 골랐습니다.
- **잘 되는 클래스:** 교통콘·소화기·안전모·조끼.
  **약한 클래스:** 장갑·보안경·귀덮개. 귀덮개는 val 6개, test 4개뿐이라 수치 편차가 큽니다.
- 실물 사진 8장(전부 train)에 대한 탐지(conf 0.25): 640에서는 장갑만 미탐, 320에서는 8장 모두 해당 장비를 탐지했습니다.
  보안경 사진에서는 귀덮개가 함께 잡혔습니다. 단, 학습에 쓴 사진이라 **일반화 성능의 근거가 아닙니다.**
- Pi 카메라 실측(속도, 거리, 조명)은 아직 하지 않았습니다.

## 다음 주 테스트 (Pi, 기존 `~/picar-yolo`)

1) 모델만 받기 — Pi의 팀 코드·브랜치를 건드리지 않는 방법:

```bash
mkdir -p ~/safety7 && cd ~/safety7
wget -O safety7.pt https://github.com/seeun-jang/HazardScout-AI/raw/YOLO-Seeun/yolo_lab/models/safety7.pt
sha256sum safety7.pt   # 91e388a76c09f24141ea4a522466ef81bc92a071e7898dfdee0789746928d8b9 확인
source ~/picar-yolo/bin/activate
python -c "from ultralytics import YOLO; m=YOLO('safety7.pt'); print(m.names)"
python -c "from ultralytics import YOLO; r=YOLO('safety7.pt').predict('사진.jpg', imgsz=320, conf=0.35, device='cpu')[0]; print([(r.names[int(b.cls)], round(float(b.conf),2)) for b in r.boxes])"
```

2) 이 저장소의 독립 시험 코드로 카메라 시험(카메라를 쓰는 서버를 먼저 종료):

```bash
# 저장소를 YOLO-Seeun으로 받은 폴더에서
cd HazardScout-AI/yolo_lab
python -c "from lab_utils import load_model; print(load_model('custom').names)"
python test_yolo.py --mode custom --source 사진.jpg --no-show
python test_pi_camera.py --mode custom --no-show --max-frames 120   # results/custom_*/summary.json에 속도 기록
```

3) 팀 GUI 서버에 연결할 때는 서버 담당자가 `YOLO_MODEL_PATH`에 `safety7.pt`의 절대 경로를 지정합니다([SERVER_REFERENCE.md](SERVER_REFERENCE.md)).
   이 모델은 7종만 출력하므로 사람·의자 같은 COCO 물체는 나오지 않습니다.

Windows에서 다시 평가하기(데이터셋이 있는 PC):

```powershell
python test_yolo.py --mode custom --evaluate val --imgsz 320
python test_yolo.py --mode custom --evaluate test --imgsz 320
```

## 테스트할 때 확인할 것

- 7종을 각각 1~2 m, 정면과 측면, 실내 조명에서 촬영해 미탐·오탐을 기록하세요. 장갑·보안경·귀덮개를 특히 확인합니다.
- 기본값 conf 0.35, imgsz 320에서 약한 클래스가 안 잡히면 `--conf 0.25` 또는 `--imgsz 416/640`과 비교하세요(속도와 맞바꿈).
- 로봇 카메라로 찍은 실물 사진을 **val/test용으로 따로** 모으면, 다음 학습에서 가장 큰 개선이 됩니다.
