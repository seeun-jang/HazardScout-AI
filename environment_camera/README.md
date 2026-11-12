# Dashboard UI v3 (2026-11-12)

새로운 관제 UI, 미연결 카메라 안내, 실제 현장 사진을 활용한 분리된 데모 모드를 추가했습니다.
**기존 설정과 저장 기록을 유지하면서 업데이트/시연하려면 [DASHBOARD_V3.md](DASHBOARD_V3.md)를 먼저 확인하세요.**

---

# HazardScout AI — 환경카메라 관제 대시보드 (노트북 전용)

**구조:** 고정형 상단/천장 카메라 → 노트북 Python + OpenCV + YOLO → 노트북 HTTP 서버 → 웹 대시보드. 기존 라즈베리파이용 코드는 수정하지 않습니다.

**빨간 UNKNOWN** = 기준 화면과 달라진 영역 중 YOLO가 알려진 객체로 설명하지 못한 *후보*. 충분히 지속되면 빨간 박스를 표시하고 해당 프레임과 A/B/C구역 정보를 SQLite에 저장합니다. **위험물로 판정한 결과가 아니며, 관리자 현장 확인이 필요합니다.** 이동하는 로봇, 그림자, 조명 변화 등이 오탐을 유발할 수 있습니다.

## 1. 폴더 배치

`HazardScout-AI`의 `YOLO-Seeun` 브랜치 최상단에 `environment_camera/` 폴더만 추가하세요. 이 폴더는 라즈베리파이에 설치하지 **않고**, 환경카메라가 연결된 **노트북에서** 실행합니다.

```text
HazardScout-AI/
  yolo_lab/                    # 기존 파일 그대로
  ...                          # 기존 파일 그대로
  environment_camera/          # 새 폴더
    server.py                  # 노트북 웹서버/API/카메라 캡처
    vision.py                  # YOLO + UNKNOWN 후보 감지
    storage.py                 # SQLite 순찰/관리자 확인 이력
    config.py
    requirements.txt
    diagnose_camera.py
    run_windows.bat
    .env.example
    .gitignore
    static/
      index.html
      style.css
      app.js
    data/                      # 실행 시 생성, Git 제외
      monitor.db
      reference.jpg
      captures/
```

## 2. Windows VS Code 실행

**Python 3.11** 설치 권장. VS Code에서 `environment_camera` 폴더를 터미널 작업 위치로 열고 다음 실행:

```powershell
cd environment_camera
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`를 열어서 **실제 환경카메라 장치 번호**(`CAMERA_SOURCE=0`, `1`, `2` 중 맞는 값)와 안전한 **ADMIN_PIN**을 설정합니다. USB 카메라가 아닌 IP 카메라라면 `CAMERA_SOURCE`에 RTSP URL을 넣을 수 있지만, 신뢰할 수 있는 로컬망에서만 사용하세요. 실제 비밀번호/토큰은 Git에 올리면 안 됩니다.

```powershell
python server.py
```

설치 후에는 폴더의 `run_windows.bat`을 더블클릭해도 실행할 수 있습니다. 카메라 번호가 불확실하면 `python diagnose_camera.py`로 0~4번 장치를 확인하세요.

브라우저에서 **http://127.0.0.1:8000** 접속. 처음에는 모델이 자동 다운로드될 수 있으므로 인터넷이 필요할 수 있습니다. 오프라인 환경이라면 `yolov8n.pt`를 미리 폴더에 복사하세요.

### 다른 기기에서 대시보드 접속하기 (같은 Wi-Fi)

`.env`에 `HOST=0.0.0.0` 설정 후 서버를 재실행합니다. 노트북 PowerShell에서 `ipconfig` 실행 → Wi-Fi `IPv4 주소` 확인 → 다른 기기에서 `http://노트북_IP:8000` 접속. Windows Defender 방화벽에서 사설 네트워크로 포트 8000을 허용해야 할 수 있습니다. **인터넷 공개·포트포워딩·공용 Wi-Fi 사용 금지:** 현재 버전은 LAN 시연용이며 대시보드 자체 로그인/HTTPS가 없습니다. 관리자 확인 처리만 PIN으로 제한합니다.

## 3. 시연 순서

1. **로봇이 없고 조명이 안정적인 빈 공간**을 카메라로 비춘 뒤 **[기준 화면 저장]**. `data/reference.jpg`에 보관됨.
2. **작업 시작 날짜/시간** 입력 → 대시보드에 작업 시작까지 카운트다운 표시.
3. **[순찰 시작]**. 웹 화면에 LIVE 영상과 YOLO의 녹색 박스, `UNKNOWN` 빨간 박스 표시.
4. 새로운 물체가 화면에 오래 유지되고 YOLO가 설명하지 못하면 빨간 `UNKNOWN`이 나타나며, 관리자 확인 대기 경고 및 스냅샷 자동 저장.
5. 순찰 1회가 끝나면 **특이사항** 입력. `A구역에서 확인되지 않은 물체 발견함`처럼 입력한 뒤 **[순찰 완료 · 보고서 저장]** → 수동 `UNKNOWN` 확인 요청도 자동 등록됨.
6. 관리자가 실제 현장을 확인한 뒤 **[확인 처리]** → PIN 입력 → **확인 완료** 또는 **오탐 처리**. 미처리 `UNKNOWN`이 남아 있으면 새 순찰을 시작하지 못함.

> 초기 버전은 로봇 주행 완료 신호와 자동 연동되지 않습니다. 순찰 완료는 대시보드에서 직접 누릅니다. 이후 Raspberry Pi에서 노트북 API로 서명된 메시지를 보내는 별도 통합을 할 수 있습니다.

## 4. 모델/현장 맞춤 조정

- 기본 모델: Ultralytics `yolov8n.pt` — COCO의 일반 물체 인식. 안전모·소화기 같은 맞춤 클래스는 기본 모델로 보장할 수 없습니다.
- 옵션: 상단 시점으로 촬영하고 라벨링한 **자체 YOLO 모델**(로봇, 보호구, 위험 표식 등)을 `CUSTOM_YOLO_MODEL=...pt`에 등록하면 기본 YOLO와 함께 사용합니다. 로봇을 `robot`으로 학습하면 로봇 자체가 UNKNOWN으로 표시될 가능성을 줄일 수 있습니다.
- UNKNOWN은 별도 YOLO `unknown` 클래스가 아닙니다. OpenCV 기준 장면 차이에서 후보를 제안하고, 알려진 YOLO 박스와 겹치는 부분을 제외합니다. 모든 미인식 물체를 발견하는 기술은 아닙니다.
- 초기 구역 경계는 화면을 가로 3등분해 **왼쪽 A, 가운데 B, 오른쪽 C**입니다. 실제 테스트 공간 사진·카메라 위치를 받은 뒤 다각형 ROI로 교체해야 정확합니다.
- 먼 거리의 작은 물체가 잘 안 보이면 `YOLO_IMGSZ=960`으로 테스트하세요. 노트북이 느리면 640으로 낮출 수 있습니다.
- 오탐이 많으면 `.env`에서 `DIFF_THRESHOLD`(조명 차이), `UNKNOWN_MIN_AREA`(최소 크기), `UNKNOWN_STABLE_FRAMES`(지속 프레임)를 조절한 뒤 **서버 재시작**.
- 고정 카메라가 흔들리면 기준 화면 차이 방식의 정확도가 떨어집니다. 실제 적용 전 최소 수십 회 현장 검증이 필요합니다.
- 로봇이 이동하면서 화면을 가리므로 현재 방식은 로봇을 UNKNOWN으로 오인할 수 있습니다. 로봇 모델 학습/추적/무시 마스크를 실제 사진으로 조정하는 것을 권장합니다.

## 5. 기록 데이터

모든 알림 및 순찰 기록은 `data/monitor.db`에 SQLite로 기록하고, 탐지 당시 이미지는 `data/captures/`에 저장합니다. `.env`, DB, 현장 카메라 이미지, 모델 `.pt`는 `.gitignore` 처리되어 GitHub에 올라가지 않습니다.

## 6. 문제 해결

- 화면이 검정색/오프라인: `.env`의 `CAMERA_SOURCE=0`과 `1`을 차례로 시험하고 Windows의 카메라 권한·카메라를 이미 사용하는 앱을 확인.
- YOLO 오류: 서버 콘솔과 카메라 하단 상태 표시 확인. 모델 파일 경로 및 `pip install -r requirements.txt` 확인.
- `ADMIN_PIN` 오류: `.env`에 설정되어 있는지 확인 후 서버 재시작.
- PowerShell에서 `Activate.ps1` 실행 차단: `cmd`에서 `.venv\Scripts\activate.bat`을 사용하거나 `.venv\Scripts\python.exe server.py` 직접 실행.
- 관리자 확인 후에도 작업이 시작 안 될 때: 대시보드의 미처리 `UNKNOWN`이 0건인지 확인.

## 7. 검증 범위

`tests/`에는 가상 화면에서 UNKNOWN 지속시간, 기존 탐지 박스 제외, DB 순찰 및 관리자 확인을 확인하는 테스트가 있습니다. **실제 카메라 장치 / 현장 로봇 / 학습 모델 / 전체 통합 운용은 현장 테스트 전까지 검증 완료로 간주하면 안 됩니다.**

## 8. 현장 사진 + 4륜 로봇팔을 반영한 운영 설정 (2026-11-12)

- 사진으로 확인한 코스 소품: **안전콘 / 소화기 / 안전모 / 안전조끼 / 장갑 / 귀마개 / 보안경**. `yolo_lab/models/safety7.pt`는 이 7종을 인식하는 기존 모델이며, `environment_camera`와 같은 저장소에 있으면 **자동 로드**합니다. 별도 경로 지정도 가능: `.env`의 `CUSTOM_YOLO_MODEL=../yolo_lab/models/safety7.pt`. 별도 모델을 복사할 필요는 없습니다.
- `YOLO_MODEL=yolov8n.pt`는 필요하면 COCO baseline을 추가합니다. 노트북에서는 `.env`에 `YOLO_MODEL=none`을 넣어 safety7만 추론할 수 있습니다. 이 경우 `person`, `stop sign` 탐지는 사라집니다. 기본 COCO 출력은 `COCO_LABELS=person,stop sign`으로 제한합니다.
- 사진 속 4륜 로봇팔은 검은 차체, 4바퀴, 회전 관절 팔, 2지 그리퍼로 구성됩니다. **기존 safety7와 기본 COCO에는 로봇 클래스가 없습니다.** 따라서 로봇 자체를 항상 정상 장비로 자동 제외한다고 볼 수 없습니다. 현장에 환경카메라를 고정한 상태에서 실제 주행 영상을 확보하고, 다양한 로봇 팔 자세를 bbox 라벨링해 `robot` 클래스를 별도로 학습해야 안정적으로 제외할 수 있습니다. 그런 모델을 학습·검증한 뒤 `.env`의 `ROBOT_YOLO_MODEL=...pt`에 지정합니다.
- 현재 코드는 이동하는 물체가 약간 겹치기만 해도 지속 객체로 분류되는 문제를 줄이도록 `UNKNOWN_TRACK_IOU=0.70`을 적용합니다. 그래도 **천천히 이동하거나 정차한 로봇, 조명·그림자 변화는 UNKNOWN으로 오인될 수 있습니다.** 영상 검증 전까지 보호기능으로 간주하지 마세요.
- 배경 차이 탐지를 코스 영역으로 제한하려면, **실제로 설치된 고정 카메라의 한 프레임**을 바탕으로 `.env`의 `MONITOR_ROI`에 `x,y;x,y;x,y` 형태의 0~1 범위 다각형을 넣으세요. 현재 보내준 현장 사진들은 서로 시점이 달라 좌표를 확정할 수 없습니다. A/B/C도 아직 임시 화면 3등분입니다.
- 기준 화면 저장: **고정 코스 소품은 모두 설치하되, 이동 로봇과 미확인 신규 물체는 없는 상태**에서 저장하는 것을 권장합니다. 첫 촬영 이후 카메라 위치를 바꾸면 기준 화면을 새로 저장하세요.

### 카메라 연결 전 현장 사진 테스트

카메라 없이도 저장된 사진을 YOLO 모델에 넣고 녹색 박스를 확인할 수 있습니다. `environment_camera/` 폴더에서:

```powershell
python analyze_photos.py "C:\Photos\scene1.jpg" "C:\Photos\scene2.jpg"
```

출력: `data/photo_tests/YYYYMMDD_HHMMSS/`에 탐지 이미지와 `results.json`이 생성됩니다. 사진에 `UNKNOWN`이 표시되지 않는 것은 정상입니다. **서로 다른 각도의 사진은 기준 화면 차이 감지에 사용할 수 없기 때문**입니다. 실제 UNKNOWN 기능은 고정 카메라의 **같은 시점에서 찍은 전/후 영상**으로만 시험하세요.

> 기존 `code/`와 `yolo_lab/`은 수정하지 않으며, 이 노트북용 서버를 라즈베리파이에 설치하지 마세요.
