# HazardScout-AI — PC 전용(TigerVNC) 코드

PC 서버와 모바일 서버의 기능적 목표는 같지만, PC는 별도의 `GUIServer_custom.py`와 Tkinter GUI를 이용합니다. 이 패키지는 **음성 기능을 제외한 버전**입니다.

## 구성

- `Server/GUIServer_custom.py` — PC 전용 확장 서버, TCP(10223) 및 ZeroMQ(5555) 기존 구조 유지
- `move/robot_gui.py` — Tkinter PC 제어 화면
- `move/fullscreen_gui.py` — 전체화면 카메라·조작 화면
- `Server/SmoothServo.py` — 목 A/B 및 집게 소프트 제어
- `Server/RPIservo.py` — 서보 드라이버
- `Server/Move.py` — 모터 제어
- `Server/Switch.py` — 라이트 제어
- `Server/YOLO_detector.py` — YOLO 추론 및 프레임 표시

## 주요 설정

- 목 A 좌우: ±90°, 목 A 상하: ±80°, 목 B 상하: ±90° (상대각)
- 주행: 직진/후진 60, 회전 50
- 집게·모터·카메라·OLED·YOLO를 PC 화면에서 조작
- 음성 관련 파일/버튼/연동: 제외

## 실행 모델과 의존 파일

`Server/GUIServer_custom.py`는 `Server/models/safety7.pt`를 사용합니다. 2026-11-09 Pi에서 7개 클래스 모델을 로드하고 교통콘(`traffic_cone`) 탐지를 확인했습니다.

기본 서버 `Server/GUIServer_custom_base.py`, OLED 모듈 `Server/OLED_eyes.py`, 모델 `Server/models/safety7.pt`가 이 폴더에 포함됩니다. Python 라이브러리와 장치 설정은 기존 Pi 환경을 사용합니다.

[Safety7 모델 출처](https://github.com/seeun-jang/HazardScout-AI/blob/bf14e2af669d481f97044b5236aed7ab9a61c8ac/yolo_lab/models/safety7.pt)

## 검증

이 ZIP 내 Python 파일은 문법 검사를 통과했습니다. 실제 주행/목/집게 등 PC 기존 기능은 사용자 실기 동작 확인을 기반으로 기록합니다. GitHub 공통 소스와 기존 실기 파일 간 세부 차이 및 이 ZIP으로 새 환경에 배포한 결과는 별도 확인이 필요합니다.

내부 서버 접속 방법·주소·비밀번호·개별 테스트 기록·모바일 실행 파일·음성 파일은 포함하지 않습니다.
