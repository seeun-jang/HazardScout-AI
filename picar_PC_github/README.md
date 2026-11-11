# HazardScout-AI — PC 전용(TigerVNC) 코드

PC는 `GUIServer_custom.py`와 TigerVNC의 Tkinter GUI로 제어합니다. 음성·텍스트 명령과 Safety7·YOLOv8n 영상 탐지를 포함합니다.

## 구성

- `Server/GUIServer_custom.py` — PC 전용 확장 서버, TCP(10223) 및 ZeroMQ(5555) 기존 구조 유지
- `move/robot_gui.py` — Tkinter PC 제어 화면
- `move/fullscreen_gui.py` — 전체화면 카메라·조작 화면
- `Server/SmoothServo.py` — 목 A/B 및 집게 소프트 제어
- `Server/RPIservo.py` — 서보 드라이버
- `Server/Move.py` — 모터 제어
- `Server/Switch.py` — 라이트 제어
- `Server/YOLO_detector.py` — YOLO 추론 및 프레임 표시
- `Server/YOLO_overlay.py` — 기본 YOLOv8n 추가 영상 표시
- `Server/PiCommandCore.py`, `Server/PiCommandService.py` — 음성·텍스트 명령 분석과 제어
- `move/PiCommandPanel.py` — 음성·텍스트 명령 패널

## 주요 설정

- 목 A 좌우: ±90°, 목 A 상하: ±80°, 목 B 상하: ±90° (상대각)
- 주행: 직진/후진 60, 회전 50
- 집게·모터·카메라·OLED·YOLO를 PC 화면에서 조작
- 음성·텍스트 패널과 AI 중지, 수동 조작 시 AI 명령 취소
- AI 주행 속도는 전후진 40·회전 30, AI 목 명령은 중앙 기준 ±20도

## 실행 모델과 의존 파일

Safety7은 8프레임 간격으로 탐지하고 지도에 결과를 전달합니다. `Server/models/yolov8n.pt`는 16프레임 간격으로 일반 물체를 추가 표시합니다. Safety7은 초록색, YOLOv8n은 파란색 박스를 사용합니다. 2026-11-09 Safety7 교통콘 탐지를 사용자 실기에서 확인했습니다.

기본 서버 `Server/GUIServer_custom_base.py`, OLED 모듈 `Server/OLED_eyes.py`, 모델 `Server/models/safety7.pt`가 이 폴더에 포함됩니다. Python 라이브러리와 장치 설정은 기존 Pi 환경을 사용합니다.

[Safety7 모델 출처](https://github.com/seeun-jang/HazardScout-AI/blob/bf14e2af669d481f97044b5236aed7ab9a61c8ac/yolo_lab/models/safety7.pt)

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
  ~/picar-yolo/bin/python GUIServer_custom.py
```

TigerVNC 터미널에서 GUI를 엽니다.

```bash
cd ~/move
~/picar-yolo/bin/python robot_gui.py
```

## 지도 모드

Pi의 `~/hazard_map_80c7951`에 이 패키지의 `hazard_map/` 실행 파일을 배치합니다. 현장 보정값은 Pi의 `robot_config.json`과 `site_map.json`에 유지하세요. 바코드 인식에는 `~/picar-yolo` 환경의 `zxing-cpp`가 필요합니다.

```bash
cd ~/Adeept_PiCar-Pro/Server
ANTHROPIC_API_KEY="$(<"$HOME/.config/hazardscout/claude.key")" \
  sudo --preserve-env=ANTHROPIC_API_KEY env PICAR_AI_MOTION=1 \
  ~/picar-yolo/bin/python ~/hazard_map_80c7951/run_pc_map.py
```

지도는 `http://<Pi-IP>:5001/map`에서 엽니다. 배포 기본값에서 바코드 자동 회피는 켜져 있습니다. Pi에서 이미 설정을 수정했다면 실제 실행값은 Pi의 `robot_config.json`을 따릅니다.

[지도 코드 출처](https://github.com/seeun-jang/HazardScout-AI/tree/80c7951c4c3cd9d40a95cff71a4725dad41eacc9/hazard_map)

## 검증

Python 문법, 명령 API·취소, 두 모델의 지도 전달 분리와 스레드 종료를 확인했습니다. PC 기존 주행·목·집게 동작과 새 명령 실행은 사용자 확인을 기반으로 기록합니다. 두 모델의 Pi 동시 추론 속도는 실제 환경에서 확인합니다. 공용 기본 서버는 기존 mj 소스를 유지하며, Pi의 구형 음성 참조가 남은 기본 서버와 세부 차이가 있습니다.

API 키는 Pi의 개인 설정 파일에 저장합니다. Sherpa 실행 파일과 한국어 모델은 Pi에 별도로 설치합니다.
