# HAZARDSCOUT / MISSION CONTROL v3

노트북 서버 전용 환경카메라 대시보드. **라즈베리파이의 `code/` 및 `yolo_lab/`은 수정하지 않습니다.**

## 1. 기존 v2를 v3으로 업데이트

1. 서버가 실행 중이면 VS Code 터미널에서 `Ctrl + C`로 종료합니다.
2. v3 ZIP의 `environment_camera/` **안에 있는 파일과 폴더**만 기존 프로젝트의 `environment_camera/`에 붙여넣어 덮어씁니다.
3. **기존 `.env` 및 `data/` 기록은 삭제하거나 덮어쓰지 마세요.** `data/demo_scene.jpg`만 새로 넣습니다. 다운로드한 ZIP에 `.env`와 실사용 DB는 포함하지 않았습니다.
4. 기존 `.venv`를 그대로 사용합니다. 새로운 패키지는 추가하지 않았습니다.

```powershell
cd C:\Users\AISW-510-102\Desktop\HazardScout-AI\environment_camera
python server.py
```

브라우저: **http://127.0.0.1:8000**. 서버 터미널은 열어둡니다.

## 2. 아직 USB 카메라가 없는 경우: 데모 모드

가장 간편한 방법: **`run_demo_windows.bat` 더블클릭**. 로컬 전용 데모를 자동으로 켜고 데모 PIN `DEMO-ONLY-2468`을 설정합니다. 이 PIN은 **데모 실행에만 임시 적용**되며 실제 운용에 사용하면 안 됩니다.

또는 아래처럼 `.env`에서 직접 설정할 수 있습니다.


`environment_camera/.env`에 다음 줄을 추가하거나 기존 값을 수정합니다.

```ini
HOST=127.0.0.1
DEMO_MODE=1
ADMIN_PIN=본인이_정한_긴_비공개_PIN
```

`.env` 파일이 없으면 `.env.example`을 복사해 `.env`로 이름을 변경합니다. 서버 재시작 후 데모 배너와 샘플 현장 사진이 표시됩니다. `data/demo_scene.jpg`는 사용자가 제공한 현장 사진의 축소본이며 **실제 영상이 아닙니다**.

**데모 시연 흐름:**

1. `작업 시작 예약`에 시간을 지정하고 저장합니다.
2. `순찰 시작`을 누릅니다.
3. `UNKNOWN 시뮬레이션` 버튼으로 **가상** 빨간 UNKNOWN 박스와 관리자 확인 알림을 생성합니다.
4. `순찰 종료 보고` → `순찰 완료 보고서`에 내용 입력 → `순찰 완료 및 보고서 저장`.
5. `관리자 확인 센터`에서 PIN을 입력하고 `확인 완료` 또는 `오탐 처리`합니다.
6. 대기 0건이 되면 다음 순찰을 시작할 수 있습니다.

**중요:** 데모 모드에서는 YOLO를 실행하지 않습니다. `data/demo_monitor.db`에 데모 기록을 **분리** 저장합니다. 실제 순찰 DB는 `data/monitor.db`입니다. 데모 모드 사용 중에는 로봇/환경카메라를 직접 제어하지 않습니다. 데모 모드는 반드시 노트북 로컬 주소에서만 실행할 수 있습니다.

## 3. USB 환경카메라 연결 후 실사용 모드

`.env`에서:

```ini
DEMO_MODE=0
CAMERA_SOURCE=1
HOST=127.0.0.1
# YOLO_MODEL=yolov8n.pt      # COCO 일반 객체
# CUSTOM_YOLO_MODEL=         # 공백이면 ../yolo_lab/models/safety7.pt 자동 사용
```

서버를 재시작합니다. 연결 실패하면 `python diagnose_camera.py`로 카메라 번호를 확인합니다. **로봇이나 새로운 물체가 없는 기준 현장**을 비춘 후 `기준 화면 저장`을 누르고 순찰을 시작합니다. 화면의 녹색 객체 박스는 모델이 인식한 클래스를, 빨간 UNKNOWN은 기준 프레임 대비 지속되는 **미분류 장면 변화 후보**를 뜻합니다. `UNKNOWN`은 YOLO가 모든 모르는 물체를 알아낸 결과가 아니며 위험 판정도 아닙니다. 상단 시점에서 로봇이 UNKNOWN으로 오탐될 수 있어 추가 촬영·라벨링·모델 검증이 필요합니다.

## 4. 화면 구성

| 화면 영역 | 기능 |
| --- | --- |
| 미션 상태 | 순찰 대기/진행/확인 필요 상태, 작업 시작까지 카운트다운 |
| KPI 4개 | 카메라 상태, 순찰 회차, 현재 YOLO 탐지 수, 관리자 확인 대기 |
| LIVE CAMERA | JPEG 영상, 모델 및 기준 화면 상태, 순찰 시작·종료 |
| SAFETY REVIEW | 빨간 UNKNOWN 알림, 증거 사진, 관리자 PIN 확인/오탐 처리 |
| SCHEDULE | 작업 시작 시간 및 작업 순서 |
| PATROL REPORT | A/B/C구역 선택, 수동 메모, 관리자 확인 요청 |
| MISSION HISTORY | 순찰 회차별 시작/종료, 특이사항 및 확인 상태 |

노트북 브라우저의 모바일 화면 폭에도 맞게 자동 배치됩니다. 외부 CSS/JavaScript CDN이 필요 없습니다.

## 5. 보안 및 업로드

- **공용 인터넷에 공개하지 마세요.** 이 서버는 운영용 인증·HTTPS가 없는 로컬 시연 서버입니다. 관리자 확인 요청 처리만 PIN으로 제한합니다.
- `.env`, `data/`, `*.pt` 및 개인 사진/캡처는 `.gitignore`로 추적하지 않습니다. `data/demo_scene.jpg`를 GitHub에 올리지 않도록 주의합니다.
- ZIP 적용 시 `data/monitor.db`, `data/reference.jpg`, `data/captures/`를 유지해야 지난 기록이 보존됩니다.
- 현장 카메라가 연결되지 않았더라도 서버와 대시보드는 실행되며, 실제 순찰 시작은 비활성화됩니다. 화면·보고 기능을 시험하려면 위 데모 모드로 전환합니다.

## 6. 테스트

```powershell
python -m unittest discover -s tests -v
```

v3 기본 테스트 10건과 데모 API 흐름(작업시간→시작→UNKNOWN 시뮬레이션→보고→관리자 확인)을 점검했습니다. 실제 카메라·로봇·YOLO의 현장 정확도는 아직 검증 완료가 아닙니다.
