# HazardScout-AI — YOLOv8n 경량 실험실

**7종 데이터셋 213장을 준비했고 0장 클래스는 없습니다.** 기본 COCO 모델의 CPU 추론도 검증했습니다.

> **2026-11-07 업데이트:** 7종 모델 `models/safety7.pt`를 실제로 학습했습니다(노트북 CPU, 100 epoch).
> 성능·한계·다음 주 Pi 테스트 방법은 **[SAFETY7_MODEL.md](SAFETY7_MODEL.md)** 를 보세요.
> CUDA가 없는 PC에서는 `python train_yolo.py --device cpu --allow-cpu`로 학습할 수 있습니다(느림).

모든 작업 파일은 `yolo_lab/` 안에 있으며 팀 서버·GUI·모터·OLED 코드는 수정하지 않았습니다.
기존 자료는 `backups/`에 보존했습니다. 사용자 승인으로 YOLO-Seeun에 로컬 커밋하며 Push/PR/Merge는 수행하지 않습니다.

## 최종 구조

```text
yolo_lab/
  README.md / REPORT.md / SERVER_REFERENCE.md
  prepare_models.py / prepare_dataset.py / train_yolo.py
  test_yolo.py / test_pi_camera.py / package_pi.py
  lab_utils.py / inference.py / requirements-laptop.txt
  tests/test_workflow.py
  dataset/
    SOURCES.md / sources.json / manifest.csv / manifest.example.csv
    reference_manifest.json
    raw/curated/          # 선정 공개 이미지·원본 라벨·출처 (Git 제외)
    raw/local_reference/  # 실제 사진8장 + 수동 bbox (Git 제외)
    references/           # 제공 원본 참고 사진, 변경 없음 (Git 제외)
    safety7/
      data.yaml / build_report.json
      images/{train,val,test}/  # Git 제외
      labels/{train,val,test}/  # Git 제외
  models/yolov8n.pt       # COCO80, 준비/로딩/추론 검증
  models/safety7.pt       # 7종 학습 완료(2026-11-07), Git 포함 — SAFETY7_MODEL.md 참고
  deploy/ / results/ / runs/ / .config/ / backups/  # Git 제외
```

## 모델 운영

| 방식 | 판단 |
|---|---|
| COCO80+장비7 단일 모델 | 공동 데이터셋과 양쪽 정확도 검증 필요. 이번에는 구현하지 않음 |
| baseline/custom 중 하나 | **현재 구현**. Pi에서 한 모델만 로드. 종료 후 다른 모드로 재실행 |
| 두 모델 동시 추론 | Pi의 메모리·지연 실측 전이므로 미구현·미검증 |

`baseline`은 `models/yolov8n.pt`의 COCO80(사람·의자·병·가방 등),
`custom`은 `models/safety7.pt`의 아래7종입니다. 7종만 fine-tuning해서 COCO80이 유지된다고 가정하지 않습니다.
모델이 없거나 클래스 순서가 다르면 중지하고 다른 모델로 대체하지 않습니다.

| ID | 클래스 | train/val/test | 합계 |
|---:|---|---|---:|
| 0 | traffic_cone | 29/6/6 | 41 |
| 1 | fire_extinguisher | 30/6/6 | 42 |
| 2 | safety_helmet | 37/8/4 | 49 |
| 3 | safety_vest | 20/6/4 | 30 |
| 4 | safety_gloves | 37/6/7 | 50 |
| 5 | earmuffs | 32/6/4 | 42 |
| 6 | safety_goggles | 28/13/5 | 46 |

총213장(train145/val40/test28)이며 한 이미지에 여러 장비가 있습니다.
실물8장도 수동 bbox를 검토해 모두 한 train 세션에 포함했습니다.
원본 라벨·일반 안경/음악 헤드셋 제외·중복/세션 분할·촬영 안내는 [dataset/SOURCES.md](dataset/SOURCES.md)에 있습니다.
**SH17 데이터는 원저자 CC BY-NC-SA 4.0/Pexels 조건을 유지한 학교 비상업 학습용입니다.**
공개 사진과 실물1~2장만으로 실제 모형의 인식 정확도를 보장하지 않습니다. 실물의 별도 촬영 val/test가 필요합니다.

## Windows 학교 PC — Python3.12 / RTX3080

모든 명령은 yolo_lab에서 실행합니다. 새 환경이 필요하면 이 폴더 안에 만드세요.

```powershell
cd C:\Users\dnflw\Documents\HazardScout-AI\yolo_lab
py -3.12 -m venv .venv-school
.\.venv-school\Scripts\Activate.ps1
```

먼저 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
Windows/Pip/Python/드라이버에 맞는 CUDA 명령을 실행한 다음:

```powershell
python -m pip install -r requirements-laptop.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python prepare_models.py
python prepare_dataset.py --check
python train_yolo.py --epochs 100 --batch 8 --device 0 --output models/safety7_school.pt
# CUDA 없는 PC: python train_yolo.py --epochs 100 --batch 8 --device cpu --allow-cpu
```

현재 데이터 검사는 통과합니다. CUDA가 없으면 학습은 명확한 오류와 함께 중지합니다.
학습 시작은 YOLOv8n, imgsz640, workers0, seed42, AMP off입니다.
AMP 점검의 추가 모델 자동 다운로드를 피하고 CUDA FP32로 먼저 확인합니다.
완료 모델은 `runs/safety7_<시간>/weights/best.pt` 및 `models/safety7.pt`입니다.
기존 safety7.pt는 덮어쓰지 않습니다. 먼저 내부 backups에 보존하세요.

```powershell
python test_yolo.py --mode baseline --source dataset/raw/local_reference/images/traffic_cone_1.jpg --no-show
python test_yolo.py --mode baseline --source 0 --max-frames 120
# 학습 후:
python test_yolo.py --mode custom --evaluate val --imgsz 320
# 모델/설정을 확정한 뒤 독립 test 평가:
python test_yolo.py --mode custom --evaluate test --imgsz 320
python test_yolo.py --mode custom --source C:/test/robot_scene.jpg --no-show
```

실시간 기본값은 CPU/imgsz320/conf0.35/every3입니다. 정량 평가의 PR/mAP confidence는 Ultralytics 기본 평가값을 사용합니다.
카메라/영상은 q로 종료하고 SSH는 --no-show를 사용합니다.
실제 추론 프레임만 CSV에 쓰며 건너뛴 프레임에는 오래된 상자를 그리지 않습니다.

## 데이터 추가·재생성

`dataset/manifest.csv`의 image/label은 yolo_lab 기준 경로, source는 sources.json 키,
group은 촬영 세션, split은 train/val/test, reviewed는 yes입니다.
라벨 한 줄은 `class_id cx cy width height`(0~1 정규화)입니다.
같은 세션은 한 split에만 넣고 모든 보이는 대상 장비를 검토해 라벨링하세요.
미라벨 사진을 빈 배경 라벨로 바꾸거나 임의 bbox를 만들지 않습니다.

```powershell
# 기존 출력은 덮어쓰지 않으므로 새 폴더로 먼저 시험:
python prepare_dataset.py --output dataset/safety7_next
python prepare_dataset.py --check --output dataset/safety7_next
```

검토 후 기존 safety7을 내부 backups의 새 이름으로 보존하고 새 출력을 safety7으로 이동합니다.
학습 전에는 각 클래스 총30장 이상과 모든 split의 양성 이미지, 라벨·중복 검사를 다시 요구합니다.

## Pi 전달 — 데이터·학습 코드 제외

```powershell
python package_pi.py --mode baseline
# 실제 7종 학습·평가 완료 후:
python package_pi.py --mode custom
```

출력된 deploy/<모드_시간> 폴더만 전송합니다. 선택한 models/*.pt 한 개와
lab_utils.py/inference.py/test_yolo.py/test_pi_camera.py/README.md/MODEL_INFO.json/SHA256.json을 포함합니다.
현재 baseline 패키지는 약6.3MB입니다. 학습 코드·데이터·백업·과거 가중치는 포함하지 않습니다.
--evaluate는 Windows 전체 프로젝트에서만 사용할 수 있습니다.

WinSCP 또는 scp로 전송하세요. 원격 부모 폴더가 있어야 하며 기존 runtime은 새 이름으로 보존하세요.

```powershell
scp -r "deploy/baseline_출력시간" 사용자@PI_IP:~/HazardScout-AI/yolo_lab/runtime_baseline
```

실제 원격 접속/전송은 수행하지 않았습니다.

## PuTTY — 기존 ~/picar-yolo 환경

Pi의 Debian12/Python3.11.2/CPU PyTorch/Ultralytics8.4.174 환경을 유지합니다.
먼저 카메라를 사용하는 기존 서버를 정상 종료하세요. 독립 시험 코드는 모터·Servo·OLED 모듈을 import하지 않습니다.

```bash
source ~/picar-yolo/bin/activate
cd ~/HazardScout-AI/yolo_lab/runtime_baseline
python -c "from lab_utils import load_model; import torch,ultralytics; m=load_model('baseline'); print(torch.__version__,ultralytics.__version__); print(m.names)"
python -c "from picamera2 import Picamera2; print('Picamera2 OK')"
python test_pi_camera.py --mode baseline --no-show --max-frames 120
```

custom 패키지에서는 해당 폴더로 이동해 load_model('custom')과 --mode custom을 사용합니다.
Picamera2 import 실패 시 기존 환경의 system-site-packages/시스템 카메라 패키지를 담당자와 확인하세요.
기존 가상환경을 재생성하거나 pip로 libcamera를 무작정 설치하지 않습니다.
Picamera2 RGB888은 BGR 배열이므로 추가 R/B 변환 없이 전달합니다.

## Pi 속도·서버 연동

results/<실행>/summary.json에 워밍업1회 제외 평균/p50/p95 추론 시간, 실제 프레임·추론 수,
처리 FPS/탐지 회수, Linux peak RSS·종료 온도를 기록합니다. 3프레임마다 탐지해도 한 번 추론 시간이 1/3로 줄지는 않습니다.
동일 조건에서 --every 1과 --every 3을 각각 비교하고 냉각·전원·작은 장비의 거리별 미검출을 확인하세요.
ONNX/NCNN/양자화는 기본 적용하지 않았습니다. 별도 복사본으로 변환 후 ARM 호환성/동일 test 정확도/Pi 지연·메모리를 비교해야 합니다.

서버 연결은 [SERVER_REFERENCE.md](SERVER_REFERENCE.md)에 읽기 전용으로 정리했습니다.
현재 서버는 YOLO_detector.py로 한 모델을 로딩하며 GUI는 상자가 그려진 JPEG를 표시합니다.
서버 담당자가 검증한 가중치 절대 경로를 GUIServer_custom.py:114 또는 GUIServer_mobile.py:140의 YOLO_MODEL_PATH에 지정하고 재시작하는 방법만 안내합니다.
코드 변경은 수행하지 않았습니다. custom으로 바꾸면 표시 형식은 같아도 COCO 일반 객체 대신7종만 나옵니다.
현재 GUI에 두 모델 모드 선택/동시 탐지 기능은 없습니다.
두 서버의 카메라 색상 변환과 이전 상자 재사용은 독립 시험과 차이가 있으므로 실제 적용 전에 담당자가 확인해야 합니다.

## 검증 범위

```powershell
python -B -m unittest discover -s tests -v
```

테스트8개, 7종 데이터 검사, 중복/세션 검사, 기본 모델 실제 사진 추론과 최소 패키지 실행을 검증했습니다.
기존 환경 Python3.12.15/torch2.14.1+cpu/Ultralytics8.4.175는 유지했습니다.
별도로 공식8.4.174를 .config 안에 분리해 기본 모델의 실제 사진 CPU 추론도 확인했습니다.
실제 Pi/Python3.11 호환성·학교 CUDA 학습·custom 정확도·GUI 통신·동시 모델 성능은 미검증입니다.
세부 변경·출처·Git 범위와 백업 내역은 [REPORT.md](REPORT.md)에 있습니다.

활동기록용 상세 정리는 [ACTIVITY_LOG.md](ACTIVITY_LOG.md), 실기 배포 기준 실행 명령은 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)에 있습니다.
