# 로봇(라즈베리파이)에 탐지기 넣는 법

팀 서버(`GUIServer_custom.py`, `GUIServer_mobile.py`)는 **라즈베리파이에서** 카메라 프레임에 YOLO 박스를 그린 뒤
PC와 폰으로 보냅니다. 그래서 탐지기도 파이에 넣어야 합니다.

노트북용 `balanced`/`max`는 파이에서 너무 무거워서, 파이 전용 **`pi` 프리셋**을 씁니다.

| | 팀 현재 | `pi` 프리셋 |
|---|---|---|
| 일반 물체 모델 | yolov8n, 입력 320 | yolo26n, 입력 416 |
| 위험물 탐지 | 없음 | 소화기, 안전모, 교통 콘, 안전조끼 |
| 어두운 화면 보정 | 없음 | 있음 |
| 모델 크기 | 6MB | 5MB + 26MB |

## 전체 순서

1. **노트북**: 파이에 넣을 파일 묶음(zip) 만들기
2. **노트북 → 파이**: zip 복사
3. **파이**: 압축 풀고 확인 스크립트 실행
4. **파이**: 팀 서버에 연결 (서버 코드 몇 줄 수정)
5. 서버 실행해서 확인. 문제 생기면 한 줄로 원래대로 되돌리기

---

## 1. 노트북에서 파일 묶음 만들기

```bash
cd ~/IdeaProjects/HazardScout-AI/yolo_lab
.venv/bin/python prepare_robot.py --ncnn
```

`yolo_lab/robot_package.zip`(약 40MB)이 생깁니다. 안에 들어 있는 것:

| 파일 | 설명 |
|---|---|
| `detect_pro.py` | 탐지기 본체 |
| `pi_check.py` | 파이에서 동작과 속도를 확인하는 스크립트 |
| `models/yolo26n.pt` | 일반 물체 모델 (5MB) |
| `models/hazard_world_s.pt` | 위험물 모델, 문구를 미리 넣어둬서 **CLIP 없이 동작** (26MB) |
| `models/yolo26n_ncnn_model/` | 파이에서 더 빠를 수 있는 변환본 (속도 비교용) |
| `test_images/` | 확인용 사진 3장 |
| `requirements-pi.txt`, `ROBOT_DEPLOY.md` | 필요한 패키지, 이 문서 |

## 2. 파이로 복사

파이 IP 주소와 사용자 이름은 팀 환경에 맞게 바꾸세요. 아래 예시는 사용자 `pi`, IP `192.168.0.10`입니다.

```bash
scp ~/IdeaProjects/HazardScout-AI/yolo_lab/robot_package.zip pi@192.168.0.10:~/
```

윈도우라면 WinSCP나 FileZilla로 zip을 파이 홈 폴더(`/home/pi/`)에 올려도 됩니다.

## 3. 파이에서 확인

PuTTY(또는 `ssh pi@192.168.0.10`)로 파이에 접속해서:

```bash
cd ~
unzip -o robot_package.zip
cd robot_package
```

**팀 서버를 돌릴 때 쓰는 가상환경을 켠 상태에서** 실행하세요. (10월 7일에 YOLO, Picamera2용으로 만든 그 환경)

```bash
python3 pi_check.py
```

확인하는 것: 버전, 모델 파일, 사진 3장 탐지, **입력 크기별 속도**.
박스가 그려진 사진은 `robot_package/results/pi_check/`에 저장됩니다.

카메라까지 확인하려면 **팀 서버를 끈 상태에서**(카메라는 한 프로그램만 쓸 수 있음):

```bash
python3 pi_check.py --camera
```

### 경고가 나오면
- **`ultralytics가 8.4보다 낮아서`**: yolo26n은 ultralytics 8.4 이상이 필요합니다. 두 가지 방법이 있습니다.
  - 업데이트: `pip install -U "ultralytics>=8.4"`. 10월 7일에 맞춰둔 PyTorch, NumPy 버전이 꼬일 수 있으니 **먼저 백업**하세요(`pip freeze > ~/pip_backup.txt`).
  - 업데이트 없이: 4단계 코드에서 `models=["yolo11n.pt"]`를 추가합니다. 처음 한 번은 인터넷이 필요합니다.
- **CLIP을 다운로드하려는 메시지가 나옴**: `hazard_world_s.pt`가 잘못된 파일이라는 뜻입니다. 노트북에서 `prepare_robot.py`를 다시 실행해서 zip을 다시 복사하세요.

## 4. 팀 서버에 연결

**이 파일들은 팀원(김유민, 김민준) 코드입니다. 수정 전에 담당자와 꼭 상의하세요.**
아래처럼 고치면 `USE_PRO_DETECTOR = False` 한 줄로 언제든 원래대로 돌아갑니다.

`GUIServer_custom.py`를 예로 듭니다. `GUIServer_mobile.py`도 똑같이 고치면 됩니다.

### (1) import 부분 (58번째 줄 근처)

기존:
```python
from YOLO_detector import YOLODetector
```

변경:
```python
from YOLO_detector import YOLODetector

# ---- detect_pro (위험물 탐지) 연결 ----
import sys
sys.path.insert(0, "/home/pi/robot_package")   # robot_package 압축 푼 경로
USE_PRO_DETECTOR = True                         # False로 바꾸면 기존 탐지기로 돌아감
try:
    from detect_pro import ProYOLODetector
except Exception as e:
    ProYOLODetector = None
    print("[YOLO] detect_pro 불러오기 실패, 기존 탐지기 사용:", repr(e))
```

### (2) 탐지기 만드는 부분 (`main()` 안, 752번째 줄 근처)

기존:
```python
    try:
        yolo_detector = YOLODetector(
            model_path=YOLO_MODEL_PATH,
            confidence=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            detect_every_n_frames=YOLO_DETECT_EVERY_N_FRAMES,
        )
```

변경:
```python
    try:
        if USE_PRO_DETECTOR and ProYOLODetector is not None:
            yolo_detector = ProYOLODetector(
                preset="pi",
                detect_every_n_frames=YOLO_DETECT_EVERY_N_FRAMES,
            )
        else:
            yolo_detector = YOLODetector(
                model_path=YOLO_MODEL_PATH,
                confidence=YOLO_CONFIDENCE,
                imgsz=YOLO_IMAGE_SIZE,
                detect_every_n_frames=YOLO_DETECT_EVERY_N_FRAMES,
            )
```

`except` 부분과 카메라 루프의 `yolo_detector.detect_and_draw(frame_bgr)`는 **그대로 둡니다.**
`ProYOLODetector`는 기존 `YOLODetector`와 사용법이 같습니다(`detect_and_draw`, `last_detections`).

### 속도가 느리면 (3단계 `pi_check.py` 결과를 보고 고르기)

```python
ProYOLODetector(preset="pi", imgsz=320, detect_every_n_frames=...)        # 일반 물체 입력을 320으로
ProYOLODetector(preset="pi", world="none", detect_every_n_frames=...)     # 위험물 탐지 끔 (일반 물체만)
ProYOLODetector(preset="pi", models=["/home/pi/robot_package/models/yolo26n_ncnn_model"],
                detect_every_n_frames=...)                                # NCNN 변환본 사용
```

`YOLO_DETECT_EVERY_N_FRAMES`를 3에서 5로 늘리면 영상은 부드러워지고, 박스 갱신이 조금 늦어집니다.

## 5. 실행과 확인

평소처럼 서버를 실행합니다. 콘솔에 이런 줄이 나오면 연결된 것입니다.

```
[ProDetector] loading yolo26n.pt
[ProDetector] loading hazard_world_s.pt with 21 hazard prompts
```

PC GUI나 폰 화면에서 **위험물은 빨간 박스, 일반 물체는 초록 박스**로 보입니다.

### 원래대로 되돌리기
`USE_PRO_DETECTOR = False`로 바꾸고 서버를 다시 실행하면 됩니다.
`detect_pro`를 불러오다 실패해도 서버는 기존 탐지기로 계속 돌아갑니다.

## 성능 (노트북에서 측정)

**파이에서의 속도는 아직 재지 않았습니다.** 3단계 `pi_check.py` 결과를 여기에 채워 주세요.

일반 물체 (COCO128 128장, mAP50-95):

| 구성 | 정상 | 어두움 | 매우 어두움 | 난반사 | 흔들림 |
|---|---|---|---|---|---|
| 팀 현재 (yolov8n @320) | 0.346 | 0.257 | 0.084 | 0.206 | 0.183 |
| `pi` 입력 320 | 0.341 | 0.315 | 0.240 | 0.223 | 0.151 |
| **`pi` 기본 (입력 416)** | **0.387** | **0.342** | **0.261** | **0.262** | **0.156** |

위험물 (사진 28장, 찾음/전체, 괄호는 오탐):

| 클래스 | 정상 | 어두움 | 매우 어두움 | 난반사 | 과노출 |
|---|---|---|---|---|---|
| 소화기 | 6/6 (1) | 5/6 (1) | 3/6 (1) | 3/6 (1) | 5/6 (1) |
| 안전모 | 4/5 | **1/5** | **1/5** | 2/5 | 3/5 |
| 교통 콘 | 4/4 (1) | 3/4 | 2/4 | 4/4 | 4/4 |
| 안전조끼 | 2/2 | 2/2 | 0/2 | 2/2 | 2/2 |

- 밝은 곳에서는 노트북용 큰 모델과 비슷하게 찾습니다.
- **어두운 곳의 안전모가 가장 약합니다.** 시연 때는 조명을 켜두는 게 좋습니다.
- **가스통은 작은 모델에서 쓸 수 없어서 뺐습니다.** 구급상자는 검증이 안 됐습니다. 둘 다 라벨링 데이터로 학습이 필요합니다.
- 흔들림에는 팀 현재 설정보다 약간 약합니다. 로봇이 멈춘 상태에서 판단하는 게 안전합니다.
