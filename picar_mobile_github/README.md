# PiCar-Pro 모바일 로봇 제어 서버

> **HazardScout-AI | `mj` 브랜치 | Raspberry Pi 4 기반 모바일 제어 모듈**
>
> 모바일 서버 실행 코드와 YOLO 가중치를 관리합니다. Sherpa 실행 파일과 한국어 모델은 Pi에 별도로 설치합니다.

## 1. 프로젝트 개요

이 디렉터리는 **Adeept PiCar-Pro와 Raspberry Pi 4**를 이용해 제작한 모바일 로봇 제어 프로그램입니다. 사용자는 같은 네트워크의 스마트폰 또는 PC 브라우저에서 로봇에 접속하여 주행, 카메라 목 관절, 집게와 조명을 조작할 수 있습니다. 카메라 영상에는 학습된 YOLO 모델의 객체 탐지 결과를 표시하며, OLED에는 눈동자·깜빡임 애니메이션이 출력됩니다.

특히 카메라·객체 탐지·OLED가 동시에 실행될 때 발생하던 서보 지연을 줄이기 위해 **I2C 통신 속도, PWM 객체 재사용, OLED 전송 우선순위**를 조정했습니다.

이 폴더는 **모바일 서버 실행본**을 다룹니다. PC GUI/VNC 연결용 `GUIServer_custom.py`, 개발 중 만든 `stage*.py` 실험 코드, 임시 백업 파일은 포함하지 않습니다.

## 2. 구현 기능 및 검증 상태

| 구분 | 구현 내용 | 실제 확인 상태 |
|---|---|---|
| 모바일 웹 제어 | 브라우저 기반 주행 및 관절 조작 | 정상 동작 |
| 로봇 주행 | 전진·후진·좌회전·우회전 및 관련 주행 명령 | 기존 기능 유지 |
| 실시간 카메라 | 640 × 480 영상, 설정 목표 12 FPS | 화면 출력 정상 |
| AI 객체 탐지 | Safety7 7종 YOLO 모델 적용 | 2026-11-09 교통콘 인식 확인 |
| 목 A 좌우 | 방향 보정, 양쪽 이동 범위 확장 | 부드러운 동작 확인 |
| 목 A 상하 | 부드러운 단계별 이동 | 부드러운 동작 확인 |
| 목 B 상하 | 아래쪽 이동 범위 추가 확장 | 동작 확인, 미세한 떨림 잔존 |
| 집게 | 잡기·놓기 및 정지 | 부드러운 동작 확인 |
| OLED | 눈동자 방향·깜빡임·미세 애니메이션 | 정상 출력 |
| LED 조명 | 기존 로봇 조명 제어 | 기존 로직 유지 |
| 음성·텍스트 명령 | Sherpa 한국어 인식과 Claude 분석을 기존 제어기에 연결 | 사용자 명령 실행 확인, 마이크·장시간 시험 별도 |

> **검증 범위:** YOLO가 소화기 사진을 인식하는 상황에서 카메라·OLED·목 관절·집게의 동시 동작을 확인했습니다. 모든 환경, 모든 인식 대상, 장시간 운용 또는 음성 기능까지 검증되었다는 의미는 아닙니다.

## 3. 디렉터리 구성

```text
picar_mobile_github/
├── README.md
├── GUIServer_mobile_final.py   # 최종 실행 파일 / 최적화 설정 적용
├── GUIServer_mobile.py         # 모바일 웹 UI, HTTP API, 로봇 제어 본체
├── SmoothServo.py              # 목 관절·집게 부드러운 이동 제어
├── RPIservo.py                 # 서보 제어 및 PCA9685 연결
├── OLED_eyes.py                # OLED 눈동자·깜빡임 애니메이션
├── YOLO_detector.py            # YOLO 모델 로드, 추론 및 탐지 결과 표시
├── Move.py                     # 바퀴 모터 구동
├── Switch.py                   # LED 조명 제어
├── PiCommandCore.py           # Sherpa 음성 인식 및 Claude 명령 분석
├── PiCommandService.py        # 명령 API와 기존 제어기 연결
├── YOLO_overlay.py            # YOLOv8n 추가 영상 표시
├── hazard_map/                # 좌표 매핑과 바코드 자동 회피
└── models/
    ├── safety7.pt             # 안전 장비 7종 모델
    └── yolov8n.pt             # 일반 물체 탐지 모델
```

### 주요 파일 관계

- **`GUIServer_mobile_final.py`**: 실행 진입점입니다. 기존 모바일 서버를 불러와 목 PWM 캐시, OLED 갱신 우선순위, 목 이동 범위와 YOLO 탐지 로그 기능을 적용합니다.
- **`GUIServer_mobile.py`**: 웹 UI, 조작 명령 수신, 카메라 스트리밍, 주행·목·집게·조명 제어를 수행합니다.
- **`SmoothServo.py` + `RPIservo.py`**: 서보 이동 단계와 실제 PWM 출력을 담당합니다.
- **`OLED_eyes.py`**: OLED 눈동자 화면을 생성하며, 최종 실행 파일에서 전송 주기를 최적화합니다.
- **`YOLO_detector.py` + `models/safety7.pt`**: 카메라 프레임의 실제 객체 탐지를 담당합니다.
- **`PiCommandCore.py` + `PiCommandService.py`**는 음성·텍스트 명령을 분석하고 기존 주행·목·집게 제어기에 전달합니다. 수동 조작은 진행 중인 AI 명령을 취소합니다.
- **`YOLO_overlay.py` + `models/yolov8n.pt`**는 일반 물체를 16프레임 간격으로 탐지해 파란 박스로 표시합니다. 지도에는 Safety7 결과를 전달합니다.

웹 화면은 `GUIServer_mobile.py`에 포함되어 있어, 별도의 `index.html`, CSS, JavaScript 파일이 필요하지 않습니다.

## 4. 하드웨어 및 실행 환경

| 항목 | 사용 환경 / 설정 |
|---|---|
| 메인 보드 | Raspberry Pi 4 |
| 로봇 키트 | Adeept PiCar-Pro |
| 카메라 | 실제 테스트에 연결한 USB 카메라, Picamera2/libcamera 기반 |
| 서보 드라이버 | PCA9685, I2C 주소 `0x5F`, PWM `50 Hz` |
| OLED | SSD1306, I2C 주소 `0x3C` |
| I2C 버스 | `/dev/i2c-1` |
| I2C 통신 속도 | `400000` baud (400 kHz) |
| 웹 서버 | HTTP, 포트 `5000` |
| AI 모델 | Safety7 7종 + 기본 YOLOv8n |
| YOLO 설정 | 입력 크기 `224`, Safety7 8프레임·YOLOv8n 16프레임 간격 |
| 카메라 설정 | `640 × 480`, 목표 `12 FPS` |
| 주행 속도 설정 | 전후진 `60`, 회전 `50` |
| Python 실행 환경 | 로봇에서 준비한 `~/picar-yolo/bin/python` |
| 음성 실행 환경 | `~/picar-yolo/bin/python` + Sherpa 공식 ALSA 실행 파일 |

Python 패키지는 기존 Raspberry Pi 실행 환경을 기준으로 합니다. 주요 의존성에는 `ultralytics`, `torch`, `opencv-python`, `picamera2`, `Pillow`, `adafruit-blinka`, `adafruit-circuitpython-pca9685`, `adafruit-circuitpython-motor`, `luma.oled`, `gpiozero` 등이 있습니다. 음성 기능은 공식 Sherpa ALSA 실행 파일과 한국어 ONNX 모델을 사용합니다. **이 저장소에 전체 의존성 잠금 파일이 포함된 것은 아니므로**, 다른 Raspberry Pi에서 재현할 때는 운영체제·하드웨어에 맞게 설치 및 권한을 설정해야 합니다.

## 5. 라즈베리파이 실행 방법

### 5.1 파일 배치

최종 서버 코드는 기존 Adeept 서버 경로를 기준으로 사용했습니다.

```text
/home/pi/Adeept_PiCar-Pro/Server/
```

`picar_mobile_github/` 안의 `.py` 파일과 `models/` 폴더를 이 경로에 배치합니다. **기존 로봇에 이미 정상 실행 중인 파일이 있다면 백업한 뒤 비교하여 반영하세요.** 별도 Git 저장소나 컴퓨터에서 다운로드한 폴더만으로 다른 환경에서 즉시 실행되는 구성은 아닙니다.

> **경로 주의:** 현재 `GUIServer_mobile.py`의 모델 위치는 `/home/pi/Adeept_PiCar-Pro/Server/models/safety7.pt`로 설정되어 있습니다. 로봇 경로를 변경했다면 코드 내 `YOLO_MODEL_PATH`도 맞춰야 합니다.

### 5.2 I2C 설정 확인

Raspberry Pi에서 아래 설정을 사용하여 통신 속도를 개선했습니다.

```ini
# /boot/firmware/config.txt
dtparam=i2c_arm=on,i2c_arm_baudrate=400000
```

이미 설정되어 있다면 다시 수정할 필요가 없습니다. 운영체제에 따라 설정 파일 위치가 다를 수 있으며, 변경 전에는 원본 백업과 연결 장치의 400 kHz 지원 여부를 확인해야 합니다. 부팅 설정 변경은 재부팅 후 적용됩니다.

### 5.3 파일 및 문법 확인

```bash
cd ~/Adeept_PiCar-Pro/Server
ls -lh GUIServer_mobile_final.py models/safety7.pt
~/picar-yolo/bin/python -m py_compile GUIServer_mobile_final.py
```

### 5.4 모바일 서버 실행

음성·텍스트 명령은 Pi의 ~/.config/hazardscout/claude.key에 저장한 키를 서버 시작 시 읽습니다. 파일 권한은 600으로 설정합니다. 최초 저장 명령은 다음과 같습니다.

```bash
(
  set -eu
  umask 077
  install -d -m 700 "$HOME/.config/hazardscout"
  read -rsp 'Claude API 키: ' hazard_key
  printf '\n'
  test -n "$hazard_key"
  printf '%s\n' "$hazard_key" > "$HOME/.config/hazardscout/claude.key"
  chmod 600 "$HOME/.config/hazardscout/claude.key"
)
```

음성 인식은 Pi에 별도로 설치한 자원을 사용합니다.

- 실행 파일 `~/sherpa-stt/sherpa-onnx-v1.12.1-linux-aarch64-shared-cpu/bin/sherpa-onnx-alsa`
- 한국어 모델 `~/Adeept_PiCar-Pro/Server/models/sherpa-onnx-streaming-zipformer-korean-2024-06-16/`
- 모델 폴더의 `tokens.txt`, `encoder-epoch-99-avg-1.int8.onnx`, `decoder-epoch-99-avg-1.onnx`, `joiner-epoch-99-avg-1.int8.onnx`
- ReSpeaker Lite 입력 장치와 Claude API 접속

`PICAR_AI_MOTION=1`은 실제 AI 동작을 켭니다. 이 옵션을 생략하면 AI 명령은 미리보기로 표시됩니다.


```bash
cd ~/Adeept_PiCar-Pro/Server
ANTHROPIC_API_KEY="$(<"$HOME/.config/hazardscout/claude.key")" \
  sudo --preserve-env=ANTHROPIC_API_KEY env PICAR_AI_MOTION=1 \
  ~/picar-yolo/bin/python GUIServer_mobile_final.py
```

로봇과 같은 네트워크에 연결한 스마트폰 또는 PC에서 다음 주소로 접속합니다.

```text
http://<라즈베리파이-IP>:5000
```

IP 주소는 라즈베리파이에서 `hostname -I` 명령으로 확인할 수 있습니다. PuTTY에서 실행한 서버는 해당 터미널에서 `Ctrl + C`로 종료합니다. **같은 하드웨어를 제어하는 다른 서버를 동시에 실행하지 마세요.**

### 5.5 모바일 지도 모드

Pi의 `~/hazard_map_80c7951`에 이 패키지의 `hazard_map/` 실행 파일을 배치합니다. 현장 보정값은 Pi의 `robot_config.json`과 `site_map.json`에 유지하세요. 바코드 인식에는 `~/picar-yolo` 환경의 `zxing-cpp`가 필요합니다.

```bash
cd ~/Adeept_PiCar-Pro/Server
ANTHROPIC_API_KEY="$(<"$HOME/.config/hazardscout/claude.key")" \
  sudo --preserve-env=ANTHROPIC_API_KEY env PICAR_AI_MOTION=1 \
  ~/picar-yolo/bin/python ~/hazard_map_80c7951/run_mobile_map.py
```

지도는 `http://<Pi-IP>:5001/map`에서 엽니다. 배포 기본값에서 바코드 자동 회피는 켜져 있습니다. Pi에서 이미 설정을 수정했다면 실제 실행값은 Pi의 `robot_config.json`을 따릅니다.

[지도 코드 출처](https://github.com/seeun-jang/HazardScout-AI/tree/80c7951c4c3cd9d40a95cff71a4725dad41eacc9/hazard_map)

## 6. 모바일 제어 기능

| 조작 영역 | 동작 |
|---|---|
| 주행 조이스틱 | 로봇 전진·후진·좌우 이동 |
| 목 A 좌우 | 카메라 목의 좌우 회전 |
| 목 A 상하 | 카메라 목의 상하 이동 |
| 목 B 상하 | 두 번째 목 관절의 상하 이동 |
| 집게 | 잡기·놓기·정지 |
| 영상 | 카메라 스트리밍 및 객체 인식 결과 확인 |
| 조명 | 로봇 LED 제어 |
| 음성·텍스트 입력 | 로봇 마이크 6초 수집, 텍스트 전송, AI 중지 |

기본 웹 화면과 제어 요청은 `GUIServer_mobile.py`에서 처리합니다. `GUIServer_mobile_final.py`는 기존 UI를 수정하지 않고 제어 성능과 일부 이동 범위를 조정합니다.

## 7. 서보 최적화 및 각도 설정

### 7.1 PWM 객체 재사용

목 서보 채널 `1`, `2`, `3`은 PCA9685 PWM 제어 객체를 한 번 생성해 재사용합니다. 매 명령마다 제어 객체를 다시 준비하면서 발생할 수 있는 I2C 지연을 줄이는 방식입니다. 집게(`4`)와 조향(`0`)은 기존 출력 함수를 유지합니다.

| PCA9685 채널 | 기능 |
|---|---|
| `0` | 앞바퀴 조향 |
| `1` | 목 A 좌우 |
| `2` | 목 B 상하 |
| `3` | 목 A 상하 |
| `4` | 집게 |

### 7.2 OLED 갱신 우선순위

- 화면 변화가 없으면 중복된 OLED 전송을 생략합니다.
- 목 또는 집게를 조작하는 동안 OLED 전송을 잠시 보류합니다.
- 조작이 끝난 뒤 약 `0.2초` 동안 안정화 시간을 둔 다음 OLED 화면을 갱신합니다.
- 눈동자 방향·깜빡임은 유지하며, 미세 애니메이션은 최대 약 `2 FPS` 갱신 예산을 사용합니다.
- OLED 상태 통계는 실행 로그에서 확인할 수 있습니다.

이는 OLED와 서보가 동일한 I2C 버스를 사용하는 환경에서 동작 우선순위를 조정하기 위한 처리입니다.

### 7.3 최종 목 이동 범위

아래 수치는 초기 기준 자세에서의 **상대 이동각**이며, 물리적인 절대 서보 각도가 아닙니다.

| 관절 | 최종 범위 | 비고 |
|---|---|---|
| A 목 좌우 (`a_pan`) | `-90° ~ +90°` | 수동 조작, 방향 반전 보정 |
| A 목 상하 (`a_tilt`) | `-80° ~ +80°` | 수동 조작 |
| B 목 상하 (`b_tilt`) | `-90° ~ +90°` | 수동 조작 |

서보의 이동 단계·속도는 기존 `SmoothServo.py` 설정을 유지합니다. 각도 범위를 더 크게 변경할 경우 관절 걸림, 케이블 장력, 서보 발열 여부를 먼저 점검해야 합니다.

## 8. AI 객체 탐지

- 사용 모델: `models/safety7.pt`
- 학습 클래스 수: **7개**
- 추론 크기: `224`
- 탐지 주기 설정: `8`프레임
- 객체 탐지 시 브라우저 영상에 탐지 결과 표시
- 2026-11-09 Safety7 실기 확인: **교통콘(`traffic_cone`) 탐지 성공**
- `models/yolov8n.pt`는 16프레임 간격으로 일반 물체를 추가 탐지해 파란색 박스와 `YOLOv8n` 접두어로 표시합니다. Safety7은 초록색 박스를 사용합니다.
- 두 모델의 지도 전달 분리와 스레드 종료를 모의 검증했습니다. Pi의 동시 추론 속도는 실기 확인이 필요합니다.

최종 실행본은 탐지 결과가 있을 때 로그를 제한적으로 출력합니다. 같은 객체를 연속으로 인식하더라도 로그를 매 프레임 출력하지 않고, 최대 약 3초 간격으로 출력하므로 **로그에 결과가 없다고 해서 화면 탐지가 실패한 것은 아닙니다.**

```text
[YOLO] 모델 로드 완료
[YOLO 인식] 1개: <학습 모델에 등록된 클래스 이름>
```

> 이전 3종 모델의 실기 기록에는 소화기 사진 탐지가 포함됩니다. Safety7에서 확인한 사례는 교통콘입니다.

## 9. 실제 테스트 결과

| 테스트 | 관찰 결과 |
|---|---|
| I2C OLED 단독 전송, 기존 설정 | 약 `94.63 ms` 평균 |
| I2C OLED 단독 전송, 400 kHz | 약 `24.80 ms` 평균 |
| YOLO 모델 로딩 및 추론 | 정상 실행 |
| 소화기 사진 탐지 | 정상 인식 |
| YOLO·카메라·OLED 동시 실행 | 정상 작동 확인 |
| A 목 좌우 | 부드러운 조작 확인 |
| A 목 상하 | 부드러운 조작 확인 |
| B 목 상하 | 동작 정상, 미세한 떨림 존재 |
| 집게 | 부드러운 조작 확인 |
| 최종 각도 확장 | 사용자 실기 테스트 완료 |

이 값들은 특정 라즈베리파이와 실제 로봇 환경에서 측정·관찰한 결과이며, 모든 장치에서 같은 성능을 보장하는 벤치마크는 아닙니다.

## 10. 주의사항 및 남은 확인 항목

1. **목 B 상하의 미세 진동:** PWM 지연은 크게 줄었으나 약한 떨림이 남아 있습니다. 기계적 유격, 관절 부하, 전원 상태 등 추가 점검 여지가 있으며 원인을 하드웨어로 확정하지 않았습니다.
2. **음성 인식:** 명령 실행은 사용자 확인을 받았습니다. 마이크 입력과 장시간 연속 동작은 실제 환경에서 확인합니다. AI 주행 속도는 전후진 40·회전 30이며, AI 목 명령은 중앙 기준 ±20도 안에서 처리합니다.
3. **하드웨어 의존성:** 이 프로젝트는 Raspberry Pi GPIO, 카메라, I2C 및 실제 서보 하드웨어에 의존합니다. 일반 PC에서 실행하면 관련 모듈 또는 장치 오류가 발생할 수 있습니다.
4. **AI 탐지 한계:** 모델이 인식하지 못하는 물체·상황에 대한 안전한 자동 판단을 보장하지 않습니다. 현재 문서의 테스트는 수동 제어와 객체 탐지 중심입니다.
5. **전원·기구부 안전:** 서보가 끝부분에서 떨리거나 걸리면 즉시 입력을 중단합니다. 주행 테스트는 장애물 없는 공간에서 실시합니다.

## 11. 문제 해결

| 증상 | 확인할 사항 |
|---|---|
| `ModuleNotFoundError` | `~/picar-yolo/bin/python`으로 실행했는지, 필요한 라이브러리가 설치됐는지 확인 |
| YOLO 모델 파일 오류 | `models/safety7.pt`가 실제 경로에 있는지, 코드의 `YOLO_MODEL_PATH`가 맞는지 확인 |
| 모바일 페이지 접속 불가 | 라즈베리파이 IP, 같은 네트워크 사용 여부, 포트 `5000`, 서버 실행 로그 확인 |
| 카메라 화면 없음 | USB 카메라 연결, 장치 인식, 다른 카메라 프로세스 실행 여부 확인 |
| OLED 표시 오류 | I2C 활성화, 주소 `0x3C`, 연결 상태 확인 |
| 목·집게 동작 오류 | PCA9685 주소 `0x5F`, 연결 상태, 기구부 간섭, 전원 상태 확인 |
| 음성 인식 실패 | `arecord -l`, Sherpa 실행 파일·한국어 모델 경로, API 키 파일 확인 |
| `ConnectionResetError` | 브라우저 새로고침·연결 종료 시 발생할 수 있음. 반복적 접속 실패가 있는지 별도 확인 |

## 12. 브랜치 관리 원칙

이 폴더는 **`mj` 브랜치의 모바일 서버 실행 코드**를 위한 공간입니다.

- 실제 사용한 소스 코드와 학습 모델을 보관합니다.
- 실험용 `stage*.py`, 임시 백업, 로그, 캐시 파일은 올리지 않습니다.
- PC GUI 서버는 `picar_PC_github/`에서 관리합니다.
- `GUIServer_mobile_final.py`를 공식 실행 진입점으로 사용합니다.
- 팀 프로젝트 전체 소개와 역할 분담은 저장소 최상위 `README.md`에서 관리하며, 이 README는 모바일 모듈의 기술적 설명에 집중합니다.

---

**관련 프로젝트:** [HazardScout-AI — `mj` 브랜치](https://github.com/seeun-jang/HazardScout-AI/tree/mj)  
**실행 모듈:** `picar_mobile_github/GUIServer_mobile_final.py`
