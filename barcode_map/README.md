# 바코드로 막힌 길 인식하기

현장에 붙인 바코드를 로봇이 읽으면 그 칸을 **막힌 길**로 지도에 기록하고, 바코드마다 정해둔 방향(왼쪽/오른쪽/유턴/정지)으로 피합니다.

```
바코드 1100 인식
  → markers.json 에서 찾기: (1,1) 막힘, 로봇은 북쪽을 보고 읽음, 오른쪽으로
  → 지도에 (1,1) 막힘 표시
  → 로봇 위치를 (1,2), 북쪽 방향으로 보정
  → 정지 → 살짝 후진 → 우회전
```

## 파일

| 파일 | 설명 |
|---|---|
| `markers.json` | 바코드 번호별 막힌 칸, 읽을 때 로봇 방향, 회피 동작 |
| `barcode_map.py` | 바코드 인식, 격자 지도, 위치 보정, 미니맵, 경로 찾기 |
| `barcode_avoid.py` | 회피 동작 실행 (별도 스레드, 수동 조작 시 중단) |
| `barcode_nav.py` | 팀 서버에 붙이는 연결 모듈 (위 두 개를 묶음) |
| `make_barcodes.py` | 인쇄용 라벨 PNG 생성 |
| `test_barcode_map.py` | 로봇 없이 노트북에서 돌려보는 시뮬레이션 |

## 좌표 정하는 법

- 장소를 **30cm 칸**으로 나눕니다. 왼쪽 위가 (0,0)이고, x는 오른쪽으로, y는 아래쪽으로 커집니다.
- 방향은 지도 기준으로 N(위), E(오른쪽), S(아래), W(왼쪽)입니다.

`markers.json`에는 바코드마다 아래 값을 적습니다.

```json
"1100": {"cell": [1, 1], "robot_heading": "N", "action": "right", "note": "복도 끝 막힘"}
```

| 항목 | 뜻 |
|---|---|
| 번호 (`"1100"`) | 바코드에 들어가는 값. **숫자 4자리**를 씁니다. 바코드가 짧아져서 더 멀리서 읽힙니다 |
| `cell` | 막힌 칸 좌표 |
| `robot_heading` | 로봇이 이 바코드를 읽을 때 바라보는 방향. 로봇은 `cell` 바로 앞 칸에 있다고 판단합니다 |
| `action` | `left` / `right` / `back`(유턴) / `stop` |

서버를 켜면 좌표가 지도 밖이거나 회피 방향이 지도 밖을 향하는 경우 `[지도 경고]`가 출력됩니다.

## 위치 추정 방식

로봇에 바퀴 회전 센서가 없어서 **전진한 시간 × 속도**로 몇 칸 갔는지 추정합니다. 오래 갈수록 오차가 쌓이지만, 바코드를 읽는 순간 위치가 바코드 바로 앞 칸으로 다시 맞춰집니다.

## 1. 노트북에서 확인

```bash
cd ~/IdeaProjects/HazardScout-AI/barcode_map
pip3 install zxing-cpp "python-barcode[images]" opencv-python numpy
python3 make_barcodes.py        # labels/ 에 라벨 생성
python3 test_barcode_map.py     # 시뮬레이션, 마지막 줄에 "통과"가 나오면 정상
```

결과 화면은 `results/sim_frame.png`에 저장됩니다. 오른쪽 아래 미니맵에서 빨간 칸은 막힌 곳, 파란 점은 로봇, 초록 칸은 지나온 길입니다.

## 2. 라벨 인쇄와 부착

- `labels/*.png`를 **실제 크기(100%)**로 인쇄합니다. 가로 20cm이고, A4 가로 방향으로 들어갑니다.
- 로봇 카메라 높이에 맞춰 **바닥에서 20~40cm 높이**에 붙입니다. 로봇이 다가오는 쪽을 정면으로 보게 하세요.
- 시뮬레이션 기준으로 약 0.7m 거리부터 인식됩니다. 실제 카메라에서는 더 가까이 가야 할 수 있습니다.
- 무광 용지를 쓰세요. 조명이 반사되면 인식이 안 됩니다.

## 3. 라즈베리파이에 복사

```bash
scp -r ~/IdeaProjects/HazardScout-AI/barcode_map pi@192.168.0.10:~/
```

파이에서 **팀 서버를 돌리는 가상환경을 켠 상태로** 설치합니다.

```bash
pip install zxing-cpp
```

## 4. 팀 서버에 연결

**서버 파일은 팀원(김유민, 김민준) 코드입니다. 수정 전에 담당자와 꼭 상의하세요.**
`USE_BARCODE_MAP = False` 한 줄로 언제든 원래대로 돌아갑니다.

`GUIServer_custom.py` 기준입니다. `GUIServer_mobile.py`도 똑같이 고치면 됩니다. 변수 이름(`current_motion`, `robot_ctrl`, `safe_stop`)이 두 파일에서 같습니다.

### (1) import 부분 (YOLO import 아래)

```python
# ---- 바코드 지도 연결 ----
import sys
sys.path.insert(0, "/home/pi/barcode_map")      # barcode_map 폴더 경로
USE_BARCODE_MAP = True                          # False로 바꾸면 바코드 기능 끔
try:
    from barcode_nav import BarcodeNavigator
except Exception as e:
    BarcodeNavigator = None
    print("[지도] barcode_nav 불러오기 실패, 바코드 기능 없이 실행:", repr(e))

barcode_nav = None
```

### (2) `main()` 안, YOLO 탐지기 만드는 부분 아래

```python
    global barcode_nav
    if USE_BARCODE_MAP and BarcodeNavigator is not None:
        try:
            barcode_nav = BarcodeNavigator(
                robot_ctrl,
                safe_stop,
                forward_m_per_sec=0.25,   # 실측해서 바꿀 것
            )
        except Exception as e:
            barcode_nav = None
            print("[지도] 초기화 실패:", repr(e))
```

`main()` 맨 위의 `global oled_eyes, yolo_detector` 줄에 `barcode_nav`를 추가해도 됩니다.

### (3) 카메라 루프, YOLO 그리는 줄 바로 아래

`GUIServer_custom.py`는 `camera_stream()`, `GUIServer_mobile.py`는 `camera_worker()` 안에 있습니다.

```python
            if yolo_detector is not None:
                frame_bgr = yolo_detector.detect_and_draw(frame_bgr)

            if barcode_nav is not None:
                frame_bgr = barcode_nav.on_frame(frame_bgr, current_motion)
```

서버에 `AsyncYOLO`가 있는 버전(10월 15일 파이 버전)도 마찬가지로, **박스를 그린 프레임을 JPEG로 압축하기 직전**에 이 두 줄을 넣으면 됩니다.

### (4) 수동 조작 시 회피 중단 (선택)

회피 동작 중에 사용자가 방향키를 누르면 회피를 멈추고 사용자 조작을 따르게 합니다. `GUIServer_custom.py`의 `handle_client()`에서 `robot_ctrl(command)`를 부르기 직전에 넣습니다.

```python
                if barcode_nav is not None:
                    barcode_nav.cancel()
                robot_ctrl(command)
```

## 5. 실측해서 바꿀 값

| 값 | 위치 | 재는 법 |
|---|---|---|
| `forward_m_per_sec` | (2)의 `BarcodeNavigator(...)` | 서버 전진 속도로 5초 달린 거리(m) ÷ 5 |
| `TURN_90_SEC` | `barcode_avoid.py` | `left`/`right` 명령으로 정확히 90° 도는 데 걸리는 시간 |
| `TURN_180_SEC` | `barcode_avoid.py` | `backleft` 명령으로 180° 도는 데 걸리는 시간 |
| `BACK_SEC` | `barcode_avoid.py` | 회피 전 후진 시간. 벽에 너무 붙으면 늘립니다 |
| `min_width_px` | (2)의 `BarcodeNavigator(...)` | 바코드가 이 폭(px)보다 작게 보이면 무시합니다. 키우면 더 가까이 가서 멈춥니다 |

PiCar-Pro는 앞바퀴 조향식이라 제자리 회전이 안 됩니다. 90°를 도는 동안 앞으로 나아가니 회전할 공간을 1~2칸 비워두세요.

## 6. 대시보드, AI 연동

- 지도 상태는 바코드를 읽을 때마다 `barcode_map/map_state.json`에 저장됩니다. 서버 코드에서는 `barcode_nav.map_dict()`로 바로 가져올 수 있습니다.
  ```json
  {"blocked": [[1, 1]], "robot_cell": [1, 2], "heading": "E", "visited": [...], "last_event": {...}}
  ```
- `barcode_nav.map.find_path((시작x, 시작y), (목표x, 목표y))`는 막힌 칸을 피한 최단 경로를 칸 목록으로 돌려줍니다. "창고로 가" 같은 음성 명령을 경로로 바꿀 때 씁니다.
- `BarcodeNavigator(..., on_event=함수)`로 바코드를 읽을 때마다 대시보드 로그나 음성 안내를 보낼 수 있습니다.
- 자율주행 모드에서는 `auto_resume=True`로 두면 회피 후 자동으로 다시 전진합니다.
