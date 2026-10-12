# detect_pro: 고정확도 물체 탐지 (강민재)

`code/YOLO_detector.py`(yolov8n @320)보다 물체를 훨씬 잘 찾도록 만든 탐지 파이프라인입니다.
팀 코드(`code/`)와 기존 `yolo_lab` 스크립트는 **수정하지 않았습니다.**

| 파일 | 역할 |
|---|---|
| `detect_pro.py` | 탐지기 본체 + CLI. `ProDetector`, 기존 코드 대체용 `ProYOLODetector` |
| `benchmark.py` | 정답 라벨(COCO128)과 비교해서 정확도(mAP)를 측정하고, 어두움·흐림·원거리 조건도 테스트 |
| `eval_hazard.py` | 위험물 프롬프트 튜닝과 평가 (`test_images/`의 사진을 직접 보고 매긴 정답 사용) |
| `test_images/` | 테스트 사진 28장 (위키미디어 공용 자유 라이선스, 출처는 각 폴더의 `SOURCES.txt`) |

## 설치

```bash
cd yolo_lab
python3 -m venv .venv
.venv/bin/pip install ultralytics opencv-python PyYAML
```

모델(`models/`)은 처음 실행할 때 자동으로 받습니다(약 750MB, git에는 올라가지 않음).
YOLO-World가 쓰는 CLIP 다운로드에서 SSL 오류가 나면 아래처럼 받으면 됩니다.

```bash
mkdir -p models/clip && curl -L -o models/clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

ultralytics 설정의 `weights_dir`을 `yolo_lab/models`로 맞춰야 이 경로를 찾습니다.

```bash
.venv/bin/yolo settings weights_dir="$PWD/models"
```

## 로봇(라즈베리파이)에 넣기

**[ROBOT_DEPLOY.md](ROBOT_DEPLOY.md)를 보세요.** 노트북에서 `prepare_robot.py`로 zip을 만들고, 파이에 복사해서
`pi_check.py`로 확인한 뒤 팀 서버에 몇 줄 추가하는 순서입니다. 파이용 `pi` 프리셋은 CLIP 없이 동작합니다.

## 사용법

```bash
.venv/bin/python detect_pro.py --source test_images --preset max
.venv/bin/python detect_pro.py --source 0 --preset balanced --show
.venv/bin/python detect_pro.py --source video.mp4 --preset balanced --save
```

결과는 `results/pro_<preset>_<시각>/`에 저장됩니다. 박스를 그린 이미지, `detections.csv`, `summary.json`이 생깁니다.
위험물은 빨간 박스, 일반 물체는 초록 박스로 표시됩니다.

| 프리셋 | 구성 | 속도 (M5 Mac) | 용도 |
|---|---|---|---|
| `fast` | yolo26n | ~30ms | 라즈베리파이, 실시간 |
| `balanced` | yolo26x + 위험물 모델 + 어두운 화면 보정 | ~0.2초 | 노트북 실시간 카메라 |
| `max` | yolo26x + yolo11x 앙상블 + TTA + 위험물 모델 + 어두운 화면 보정 | ~0.6초~1초 | 사진·녹화 영상 정밀 분석 |
| `pi` | yolo26n(416) + 작은 위험물 모델 + 어두운 화면 보정 | 파이에서 측정 필요 | **로봇 탑재용** ([ROBOT_DEPLOY.md](ROBOT_DEPLOY.md)) |

옵션: `--tile`(멀리 있는 작은 물체, 9배 느림), `--no-low-light`(어두운 화면 보정 끄기, balanced/max는 기본으로 켜져 있음), `--conf 0.15`(더 많이 잡지만 오탐 증가), `--no-world`(위험물 모델 끔).

### 팀 코드에 붙이기

`ProYOLODetector`는 기존 `YOLODetector`와 사용법(`detect_and_draw(frame)`, `last_detections`)이 같습니다.

```python
from detect_pro import ProYOLODetector as YOLODetector
detector = YOLODetector(preset="balanced", detect_every_n_frames=3)
frame = detector.detect_and_draw(frame)
```

**주의:** `balanced`와 `max`는 라즈베리파이 4에서 돌리기에 너무 무겁습니다.
파이는 이미 PC로 영상을 보내고 있으니(`robot_gui.py`), **PC 쪽에서 탐지하는 구조**를 추천합니다. 서버와 GUI 담당자와 상의가 필요합니다.

## 측정 결과

### 일반 물체 (COCO128 128장, 정답 라벨과 비교)

mAP50-95가 COCO 표준 정확도 지표입니다. 재현율(recall)은 실제 물체 중 찾아낸 비율이고, 화면 표시 기준(conf 0.25)에서 측정했습니다.

| 구성 | 정상 | 어두움 | 흔들림(blur) | 원거리(small) | 재현율(정상) | 속도 |
|---|---|---|---|---|---|---|
| 팀 현재 (yolov8n @320) | 0.346 | 0.257 | 0.183 | 0.179 | 36% | 23ms |
| `fast` (yolo26n) | 0.439 | 0.348 | 0.140 | 0.356 | 48% | 33ms |
| yolo26x 단독 (보정 없이) | 0.618 | 0.539 | 0.342 | 0.531 | 69% | 140ms |
| 앙상블 (26x+11x) | 0.633 | 0.558 | 0.376 | 0.565 | 69% | 340ms |
| **앙상블+TTA (보정 없이)** | **0.654** | **0.574** | **0.398** | 0.559 | **72%** | 555ms |
| 앙상블 + `--tile` | | | | **0.592** | | 2.0초 |

정리하면 `max`는 팀 현재 설정보다 **정확도가 약 1.9배**이고, **찾아내는 물체가 36%에서 72%로 2배**입니다.
어두움은 밝기를 22%로 낮추고 노이즈를 넣었고, 흔들림은 15px 모션 블러, 원거리는 이미지를 1/2로 축소해서 만들었습니다.

### 조명 조건 (어두움, 난반사, 과노출)

`benchmark.py`의 `--corrupt` 옵션으로 사진을 인위적으로 변형해서 측정했습니다. `results/lighting_demo/`에서 예시 이미지를 볼 수 있습니다.

- **어두움**: 밝기 22% + 센서 노이즈
- **매우 어두움**: 밝기 8% + 노이즈 (불 꺼진 방 수준)
- **난반사**: 강한 빛 번짐 1~3개 + 전체가 뿌옇게
- **과노출**: 밝기 1.8배 (창문이나 손전등을 정면으로 비출 때)

일반 물체 정확도 (mAP50-95, 128장):

| 구성 | 정상 | 어두움 | 매우 어두움 | 난반사 | 과노출 |
|---|---|---|---|---|---|
| 팀 현재 (yolov8n @320) | 0.346 | 0.257 | 0.084 | 0.206 | 0.279 |
| `balanced` (보정 없이) | 0.618 | 0.539 | 0.412 | 0.466 | 0.514 |
| `balanced` | 0.615 | 0.561 | 0.424 | 0.466 | 0.514 |
| **`max`** | **0.653** | **0.604** | **0.503** | **0.491** | **0.561** |

매우 어두운 화면에서 팀 현재 설정은 물체를 12%만 찾지만, `max`는 57%를 찾습니다.
어두운 화면 보정은 난반사와 과노출에는 효과가 없습니다(어두운 화면에서만 작동).

위험물, `max` 프리셋 (찾음/전체, 괄호는 오탐):

| 클래스 | 정상 | 어두움 | 매우 어두움 | 난반사 | 과노출 |
|---|---|---|---|---|---|
| 소화기 | 6/6 (1) | 6/6 (1) | 5/6 (1) | 5/6 (0) | 6/6 (1) |
| 안전모 | 4/5 | 5/5 | **2/5** | **2/5** | 5/5 |
| 교통 콘 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |

**약점은 매우 어두운 화면과 난반사 속의 안전모입니다.** 난반사는 소프트웨어로 복구하기 어려우니,
카메라 노출 고정이나 편광 필터 같은 하드웨어 쪽 대책을 같이 고려하는 게 좋습니다.

### 위험물 (사진 28장, 사진 단위로 있다/없다 판정)

| 클래스 | 찾음 | 오탐 | 판정 |
|---|---|---|---|
| 소화기 | 6/6 | 1 (물병을 소화기로) | 사용 가능 |
| 안전모 | 4/5 | 0 | 사용 가능 |
| 교통 콘 | 4/4 | 0 | 사용 가능 |
| 가스통 | 0/3 | 0 | **학습 필요** |
| 안전조끼 | 0/2 | 0 | **학습 필요** |
| 구급상자 | 0/1 | 0 | **학습 필요** |
| 불, 연기 | 테스트 사진 없음 | 0 | 검증 안 됨 |

사진 수가 적어서 대략적인 경향으로만 봐야 합니다. 가스통, 조끼, 구급상자는 YOLO-World 점수가 0.01~0.03 수준이라
기준을 낮추면 오탐이 쏟아집니다. 이 클래스들은 라벨링한 데이터로 `train_yolo.py` 학습을 해야 합니다.

## 실험하면서 알게 된 것

- **모델 크기가 압도적이다.** yolov8n에서 yolo26x로 바꾸는 것만으로 정확도가 0.35에서 0.62가 됐다. 나머지 기법을 다 합쳐도 추가 이득은 +0.04 정도다.
- **입력을 원본보다 크게 키우면 오히려 나빠진다.** 640px 이미지를 1280으로 키우니 0.618에서 0.451로 떨어졌다. 그래서 원본 해상도 이상으로는 확대하지 않게 막았다.
- **흔들림이 가장 치명적이다.** 모든 구성에서 가장 크게 떨어진다. 로봇이 이동 중일 때는 탐지를 믿기 어려우니, 멈춘 상태에서 판단하는 로직이 안전하다.
- `fast`(yolo26n @640)는 흔들림에서 팀 현재 설정(@320)보다도 나쁘다. 작게 줄이면 블러가 덜 보이기 때문으로 보인다.
- **흔히 쓰는 저조도 보정(CLAHE, 감마 보정)은 오히려 나빴다.** 노이즈까지 키우기 때문이다. 밝기를 올린 뒤 경계를 살리는 노이즈 제거(bilateral)를 하는 방식만 효과가 있었다.
- 보정이 어두운 배경의 정상 사진(평균 밝기 49)에서도 작동해서 안전모를 놓치는 일이 있었다. 기준을 평균 밝기 45 미만으로 낮춰 해결했다. 다만 이 기준은 테스트 사진을 보고 정한 것이라, 실제 로봇 카메라로 다시 확인해야 한다.
- **YOLO-World는 문구에 매우 민감하다.** "fire extinguisher"는 0점인데 "extinguisher"는 0.45점이었다. 박스마다 가장 높은 문구 하나만 남아서 비슷한 문구끼리 표를 뺏는다.
  사람, 병, 상자 같은 **미끼 문구**를 같이 넣으면 헷갈리는 물체를 흡수해서 F1이 0.615에서 0.701로 올랐다.
- 타일링은 처음 구현에서 정확도를 떨어뜨렸다(잘린 물체 조각, 중복 박스). 타일은 작은 물체를 **추가**만 하도록 바꾸고 나서야 효과가 났다.

## 재현

```bash
mkdir -p datasets && curl -L -o datasets/coco128.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip && unzip -q datasets/coco128.zip -d datasets
.venv/bin/python benchmark.py
.venv/bin/python eval_hazard.py
.venv/bin/python eval_hazard.py --pipeline --preset max
.venv/bin/python benchmark.py --configs team max_light --corrupt clean dark very_dark glare overexposed
.venv/bin/python eval_hazard.py --pipeline --preset max --corrupt clean dark very_dark glare overexposed
```
