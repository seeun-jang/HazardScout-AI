# HazardScout-AI — YOLOv8n 실험실 (yolo_lab)

로봇 카메라 영상에서 **안전 장비 7종**과 **일반 물체(COCO 80종)** 를 인식하기 위한 YOLO 작업 폴더입니다.
모든 YOLO 작업은 이 `yolo_lab/` 안에서만 하며, 팀 서버·GUI·모터·OLED 코드(`code/`)는 수정하지 않습니다.

## 현재 상태 (2026-11-07)

| 항목 | 상태 |
|---|---|
| 7종 데이터셋 `dataset/safety7` | 213장 (train 145 / val 40 / test 28), SHA-256 검증 완료. 이미지·라벨은 Git 제외 |
| 7종 모델 `models/safety7.pt` | **학습 완료, Git 포함.** YOLOv8n, 100 epoch, 640, Ultralytics 8.4.174 |
| COCO 모델 `models/yolov8n.pt` | 로딩·CPU 추론 확인. Git 제외(`prepare_models.py`로 준비) |
| Pi 카메라 실측 | **아직 안 함.** 다음 주 테스트 예정 |

`safety7.pt` 정보:
- SHA-256 `91e388a76c09f24141ea4a522466ef81bc92a071e7898dfdee0789746928d8b9`
- 6,258,794 bytes
- 자세한 성능, 한계, 테스트 방법은 **[SAFETY7_MODEL.md](SAFETY7_MODEL.md)** 에 있습니다.

## 탐지 클래스

| ID | 클래스 | 실제 장비 | test mAP50 @640 / @320 |
|---:|---|---|---|
| 0 | traffic_cone | 주황색 교통콘 | 0.86 / 0.84 |
| 1 | fire_extinguisher | 빨간색 소화기 | 0.99 / 0.99 |
| 2 | safety_helmet | 노란색 안전모 | 0.74 / 0.69 |
| 3 | safety_vest | 형광 안전조끼 | 0.56 / 0.55 |
| 4 | safety_gloves | 작업장갑 | 0.08 / 0.19 |
| 5 | earmuffs | 귀덮개형 청력보호구 | 0.58 / 0.39 |
| 6 | safety_goggles | 보안경 | 0.12 / 0.21 |

- 교통콘·소화기·안전모·조끼는 잘 잡힙니다.
- **장갑·보안경·귀덮개는 아직 약합니다.** test가 28장뿐이라 수치 편차도 큽니다.
- `safety7.pt`는 7종만 출력합니다. 사람·의자 같은 COCO 물체는 `yolov8n.pt`(baseline)로 따로 인식합니다.

## 모델 운영 방식

| 방식 | 판단 |
|---|---|
| baseline(COCO80) / custom(7종) 중 하나 선택 | **현재 구현.** Pi에서 한 모델만 로드하고, 모드를 바꾸려면 종료 후 재실행 |
| COCO80+7종 단일 모델 | 공동 데이터셋과 양쪽 정확도 검증이 필요해 구현하지 않음 |
| 두 모델 동시 추론 | Pi 메모리·지연 실측 전이라 미구현 |

모델 파일이 없거나 클래스 순서가 다르면 `lab_utils.load_model()`이 중지합니다. 다른 모델로 대체하지 않습니다.

## 폴더 구조

```text
yolo_lab/
  README.md / SAFETY7_MODEL.md / REPORT.md / ACTIVITY_LOG.md
  SERVER_REFERENCE.md / SERVER_RUNBOOK.md
  prepare_models.py / prepare_dataset.py / train_yolo.py / recover_safety7.py
  test_yolo.py / test_pi_camera.py / package_pi.py / lab_utils.py / inference.py
  requirements-laptop.txt / tests/test_workflow.py
  models/
    safety7.pt / training_info.json / safety7_eval.json   # Git 포함
    yolov8n.pt                                            # Git 제외
  dataset/
    SOURCES.md / sources.json / manifest.csv / reference_manifest.json
    safety7/data.yaml / build_report.json                 # Git 포함
    safety7/images, labels                                # Git 제외
    raw/ / references/                                    # Git 제외 (원본·출처)
  backups/ / results/ / runs/ / deploy/ / .config/        # Git 제외
```

## 다음 주 Pi 테스트 — 빠른 시작

`~/picar-yolo` 환경(Python 3.11, CPU PyTorch, Ultralytics 8.4.174)을 그대로 씁니다.
카메라를 쓰는 기존 서버는 먼저 종료하세요.

**방법 A — 모델 파일만 받기.** Pi의 팀 코드와 브랜치를 건드리지 않습니다.

```bash
mkdir -p ~/safety7 && cd ~/safety7
wget -O safety7.pt https://github.com/seeun-jang/HazardScout-AI/raw/YOLO-Seeun/yolo_lab/models/safety7.pt
sha256sum safety7.pt   # 91e388a7...28d8b9 와 같은지 확인
source ~/picar-yolo/bin/activate
python -c "from ultralytics import YOLO; print(YOLO('safety7.pt').names)"
python -c "from ultralytics import YOLO; r=YOLO('safety7.pt').predict('사진.jpg', imgsz=320, conf=0.35, device='cpu')[0]; print([(r.names[int(b.cls)], round(float(b.conf),2)) for b in r.boxes])"
```

**방법 B — 이 폴더의 시험 코드로 카메라 시험.** 속도와 탐지 결과가 `results/`에 저장됩니다.

```bash
cd HazardScout-AI/yolo_lab      # YOLO-Seeun 브랜치를 받은 위치
python -c "from lab_utils import load_model; print(load_model('custom').names)"
python test_yolo.py --mode custom --source 사진.jpg --no-show
python test_pi_camera.py --mode custom --no-show --max-frames 120
```

실시간 기본값은 CPU, imgsz 320, conf 0.35, 3프레임마다 1회 추론입니다.
약한 클래스가 안 잡히면 `--conf 0.25` 또는 `--imgsz 416`/`640`과 비교해 보세요(그만큼 느려집니다).

**팀 GUI 서버에 연결할 때**는 서버 담당자가 `YOLO_MODEL_PATH`에 `safety7.pt`의 절대 경로를 지정하고 재시작합니다
(`GUIServer_custom.py:114` 또는 `GUIServer_mobile.py:140`, [SERVER_REFERENCE.md](SERVER_REFERENCE.md) 참고).
서버 코드는 여기서 수정하지 않습니다.

## Windows PC — 평가·재학습

모든 명령은 `yolo_lab`에서 실행합니다. 노트북은 프로젝트 최상위 `.venv`(Python 3.12, CPU PyTorch)를 씁니다.

```powershell
cd C:\Users\dnflw\Documents\HazardScout-AI\yolo_lab
..\.venv\Scripts\Activate.ps1
python prepare_dataset.py --check                       # 데이터셋 검사
python test_yolo.py --mode custom --evaluate val --imgsz 320
python test_yolo.py --mode custom --evaluate test --imgsz 320
python test_yolo.py --mode custom --source C:/test/robot_scene.jpg --no-show
```

재학습할 때는 기존 `safety7.pt`를 덮어쓰지 않도록 `--output`으로 새 파일을 지정합니다.

```powershell
# 학교 RTX 3080 (CUDA PyTorch 설치 후)
python train_yolo.py --epochs 100 --batch 8 --device 0 --output models/safety7_school.pt
# CUDA 없는 PC (노트북 i7-12700H에서 100 epoch 약 1.2시간)
python train_yolo.py --epochs 100 --batch 8 --device cpu --allow-cpu --output models/safety7_cpu.pt
```

노트북의 현재 `safety7.pt`는 Pi와 같은 버전으로 학습하려고 `$env:PYTHONPATH="$PWD\.config\compat_ultralytics_8_4_174"`를 지정해 Ultralytics 8.4.174로 실행했습니다(`.venv` 기본은 8.4.175).
학습 결과는 `runs/safety7_<시간>/`에 남고, 가중치는 `--output` 경로에 저장됩니다.
새 모델을 쓰려면 val 기준으로 비교한 뒤 기존 `safety7.pt`를 `backups/`에 보존하고 교체하세요.

학교 PC처럼 데이터셋 이미지가 없는 곳에서는 복구 ZIP을 먼저 설치합니다.
ZIP 위치는 노트북 `results/recovery_20261107_195020/`입니다.

```powershell
python recover_safety7.py verify <ZIP경로>    # CRC·해시 확인
python recover_safety7.py install <ZIP경로>   # safety7/images, labels가 없을 때만 풀림
python prepare_dataset.py --check
```

## 데이터셋

- 출처: 공개 데이터(교통콘 40, 소화기 40, RF100 20, SH17 105)와 실물 사진 8장.
  라이선스와 라벨 검토 내역은 [dataset/SOURCES.md](dataset/SOURCES.md)에 있습니다.
- **SH17은 CC BY-NC-SA 4.0(비상업)이고, 실물 사진은 재배포 허락이 없습니다.**
  그래서 이미지·라벨은 공개 저장소에 올리지 않습니다.
- 라벨 형식은 `class_id cx cy w h`(0~1 정규화)입니다. 같은 촬영 세션은 한 split에만 넣습니다.
- 데이터를 추가할 때는 `dataset/manifest.csv`에 행을 추가하고 새 폴더로 먼저 만들어 검사합니다.

```powershell
python prepare_dataset.py --output dataset/safety7_next
python prepare_dataset.py --check --output dataset/safety7_next
```

**정확도를 가장 크게 올리는 방법:** 로봇 카메라로 7종(특히 장갑·보안경·귀덮개)을 거리·각도·조명을 바꿔 찍은 사진을 모으고,
학습용과 별도의 val/test 세션으로 나누어 추가하는 것입니다.

## Pi 전달 패키지 (선택)

데이터·학습 코드 없이 추론에 필요한 파일만 묶습니다.

```powershell
python package_pi.py --mode custom      # deploy/custom_<시간>/ 생성
scp -r "deploy/custom_<시간>" 사용자@PI_IP:~/HazardScout-AI/yolo_lab/runtime_custom
```

Picamera2 RGB888 배열은 BGR이므로 추가 색상 변환 없이 전달합니다.
Picamera2 import가 실패하면 기존 환경을 다시 만들지 말고 담당자와 확인하세요.

## 검증 범위

```powershell
python -B -m unittest discover -s tests -v
```

완료:
- 기능 테스트 8개 통과
- 7종 데이터 검사(라벨, 중복, 세션 분할)
- `safety7.pt` 클래스 순서 확인
- val·test 정량 평가(640·320)와 샘플 탐지
- COCO baseline 실제 사진 CPU 추론

미검증:
- Pi 4(Python 3.11)에서의 custom 모델 로딩·속도·발열
- 실제 로봇 카메라에서의 정확도
- GUI 서버 연동
- 두 모델 동시 추론, ONNX/양자화

변경 기록과 백업 내역은 [REPORT.md](REPORT.md)와 [ACTIVITY_LOG.md](ACTIVITY_LOG.md),
실기 배포 명령은 [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md)에 있습니다.
