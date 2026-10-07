# PiCar-Pro Robot Control System

Raspberry Pi 4와 Adeept PiCar-Pro를 이용해 로봇을 제어하는 프로젝트입니다.

이 프로젝트는 **PC용 GUI 제어 방식**과 **모바일 웹 GUI 제어 방식**을 분리해 두었습니다. PC에서는 Tkinter 기반 GUI를 사용하고, 모바일에서는 별도 앱 없이 스마트폰 브라우저로 접속해 로봇을 제어합니다.

> **중요**  
> `GUIServer_custom.py`와 `GUIServer_mobile.py`는 동시에 실행하지 않습니다.  
> 두 서버 모두 동일한 모터, Servo, GPIO, 카메라 하드웨어에 직접 접근하므로 한 번에 하나의 서버만 실행해야 합니다.

---

## 1. 프로젝트 구성

프로젝트에서 직접 관리하는 핵심 파일은 다음 3개입니다.

| 파일 | 역할 | 사용 환경 |
| --- | --- | --- |
| `GUIServer_custom.py` | PC GUI와 실제 로봇 하드웨어 사이를 연결하는 서버 | PC GUI 사용 시 |
| `robot_gui.py` | 컴퓨터에서 사용하는 Tkinter 기반 제어 GUI | PC GUI 사용 시 |
| `GUIServer_mobile.py` | 모바일 웹 화면 제공 + 실제 로봇 직접 제어 | 스마트폰 사용 시 |

단, 위 3개 파일만으로 하드웨어를 직접 구동할 수 있는 것은 아닙니다. Adeept 프로젝트의 하드웨어 제어 모듈인 `Move.py`, `RPIservo.py`, `Switch.py`도 함께 필요합니다.

---

## 2. 전체 구조

### PC 제어 방식

```text
┌────────────────────────────┐
│       robot_gui.py         │
│                            │
│ Tkinter GUI                │
│ - 이동 버튼                │
│ - 후레시 ON/OFF            │
│ - 집게 제어                │
│ - 카메라 표시              │
│ - 카메라 전체화면          │
└─────────────┬──────────────┘
              │
              │ TCP 10223
              │ ZMQ 5555
              ▼
┌────────────────────────────┐
│   GUIServer_custom.py      │
│                            │
│ - 명령 수신                │
│ - 모터 제어                │
│ - 조향 Servo 제어          │
│ - 후레시 제어              │
│ - 집게 Servo 제어          │
│ - 카메라 영상 전송         │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│  Move / RPIservo / Switch  │
│        실제 하드웨어        │
└────────────────────────────┘
```

### 모바일 제어 방식

```text
┌────────────────────────────┐
│ 스마트폰 Chrome / Safari   │
│                            │
│ http://RaspberryPi-IP:5000 │
└─────────────┬──────────────┘
              │ HTTP
              ▼
┌────────────────────────────┐
│   GUIServer_mobile.py      │
│                            │
│ - 모바일 Web GUI 제공      │
│ - 이동 제어                │
│ - 후레시 제어              │
│ - 집게 제어                │
│ - 카메라 MJPEG 스트리밍    │
│ - 실제 하드웨어 직접 제어  │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│  Move / RPIservo / Switch  │
│        실제 하드웨어        │
└────────────────────────────┘
```

---

## 3. 파일별 역할

### `GUIServer_custom.py`

PC용 `robot_gui.py`와 실제 로봇을 연결하는 **PC 전용 제어 서버**입니다.

주요 역할:

- TCP `10223` 포트에서 PC GUI의 명령 수신
- 전진 / 후진 제어
- 전진 좌회전 / 우회전 제어
- 후진 좌회전 / 우회전 제어
- 앞바퀴 조향 Servo 제어
- 후레시 ON/OFF 및 방향별 자동 후레시 제어
- 집게 및 로봇 팔 Servo 명령 처리
- Picamera2 카메라 영상 획득
- ZMQ `5555` 포트를 이용해 `robot_gui.py`에 카메라 영상 전송
- 연결 종료 시 모터를 안전하게 정지

PC GUI에서 보내는 대표 명령은 다음과 같습니다.

| 명령 | 기능 |
| --- | --- |
| `forward` | 전진 |
| `backward` | 후진 |
| `left` | 전진 좌회전 |
| `right` | 전진 우회전 |
| `backleft` | 후진 좌회전 |
| `backright` | 후진 우회전 |
| `DS` | 주행 모터 정지 |
| `TS` | 조향 중앙 복귀 + 정지 |
| `wsB 숫자` | 전진/후진 속도 변경 |
| `light_on` | 양쪽 후레시 계속 ON |
| `light_off` | 자동 방향등 모드로 복귀 |
| `grab` | 집게 잡기 |
| `loose` | 집게 놓기 |
| `stop` | 집게 Servo 정지 |

### `robot_gui.py`

컴퓨터에서 사용하는 **Tkinter 기반 제어 화면**입니다.

이 파일은 실제 GPIO나 모터를 직접 제어하지 않습니다. 대신 `GUIServer_custom.py`에 명령을 보내고 서버가 실제 하드웨어를 움직입니다.

주요 기능:

- 전진 / 후진
- 좌회전 / 우회전
- 후진 좌회전 / 후진 우회전
- 버튼을 누르고 있는 동안만 이동
- 버튼을 놓으면 즉시 정지
- 방향키를 이용한 키보드 주행
- `Z` 키로 집게 잡기
- `X` 키로 집게 놓기
- 전진/후진 부드러운 속도 증가
- 현재 방향 및 현재 속도 표시
- 서버 연결 상태 표시
- 로봇 카메라 실시간 표시
- 후레시 ON/OFF 제어
- 카메라 전체화면 보기
- 전체화면 상태에서도 이동 / 후레시 / 집게 조작 가능

현재 설정에서는 다음 주소를 사용합니다.

```python
SERVER_IP = "127.0.0.1"
SERVER_PORT = 10223
VIDEO_PORT = 5555
```

따라서 현재 구조는 `robot_gui.py`와 `GUIServer_custom.py`가 **같은 Raspberry Pi에서 실행되는 구성**입니다. PC에서는 VNC 등의 원격 화면을 통해 Raspberry Pi의 GUI를 보는 방식으로 사용할 수 있습니다.

### `GUIServer_mobile.py`

스마트폰에서 사용하는 **독립형 모바일 웹 제어 서버**입니다.

`robot_gui.py`를 사용하지 않으며, `GUIServer_mobile.py` 하나가 다음 두 가지 역할을 동시에 담당합니다.

1. 스마트폰에 Web GUI 제공
2. 실제 로봇 하드웨어 직접 제어

주요 기능:

- HTTP `5000` 포트에서 모바일 GUI 제공
- 스마트폰 Chrome / Safari에서 접속 가능
- 별도의 VNC 앱 필요 없음
- 전진 / 후진
- 좌회전 / 우회전
- 후진 좌회전 / 후진 우회전
- 후레시 ON/OFF
- 집게 잡기 / 놓기
- Picamera2 카메라 촬영
- MJPEG 방식 실시간 영상 스트리밍
- 브라우저 화면에서 벗어날 때 안전 정지 처리

접속 형식:

```text
http://<Raspberry-Pi-IP>:5000
```

예:

```text
http://192.168.25.112:5000
```

Raspberry Pi의 IP 주소가 변경된 경우 다음 명령으로 다시 확인합니다.

```bash
hostname -I
```

---

## 4. 후레시 동작 방식

PC 서버와 모바일 서버 모두 같은 후레시 동작 규칙을 사용합니다.

### 후레시 ON

`ON`을 누르면 이동 방향과 관계없이 **양쪽 후레시가 계속 켜진 상태를 유지**합니다.

| 로봇 상태 | 왼쪽 | 오른쪽 |
| --- | :---: | :---: |
| 정지 | ON | ON |
| 전진 | ON | ON |
| 후진 | ON | ON |
| 좌회전 | ON | ON |
| 우회전 | ON | ON |
| 후진 좌회전 | ON | ON |
| 후진 우회전 | ON | ON |

### 후레시 OFF

이 프로젝트에서 `OFF`는 후레시 기능을 완전히 사용하지 않는다는 의미가 아니라 **자동 방향등 모드로 복귀한다는 의미**입니다.

| 로봇 상태 | 왼쪽 | 오른쪽 |
| --- | :---: | :---: |
| 정지 | OFF | OFF |
| 전진 | ON | ON |
| 후진 | ON | ON |
| 좌회전 | ON | OFF |
| 후진 좌회전 | ON | OFF |
| 우회전 | OFF | ON |
| 후진 우회전 | OFF | ON |

> 실제 좌/우 라이트 방향이 로봇 배선 기준과 반대로 보이면 `switch.switch(1, ...)`, `switch.switch(2, ...)` 부분을 서로 교체해 조정할 수 있습니다.

---

## 5. 주행 방식

지원하는 이동 방향은 총 6가지입니다.

```text
        전진
         ▲

좌회전  ◀   ▶  우회전

후진좌  ↙   ↘  후진우
         ▼
        후진
```

### PC 키보드

| 키 | 기능 |
| --- | --- |
| `↑` | 전진 |
| `↓` | 후진 |
| `←` | 좌회전 |
| `→` | 우회전 |
| `↓ + ←` | 후진 좌회전 |
| `↓ + →` | 후진 우회전 |
| `Z` | 집게 잡기 |
| `X` | 집게 놓기 |
| `F11` | 카메라 전체화면 |
| `ESC` | 전체화면에서는 전체화면 종료 / 메인 화면에서는 프로그램 종료 |

---

## 6. 카메라 구조

### PC 카메라

`GUIServer_custom.py`가 Picamera2로 카메라 영상을 획득한 뒤 JPEG로 압축합니다.

압축된 영상은 Base64 형태로 변환한 후 ZMQ `5555` 포트를 통해 `robot_gui.py`로 전달합니다.

```text
Picamera2
    ↓
OpenCV Frame
    ↓
JPEG Encode
    ↓
Base64
    ↓
ZMQ :5555
    ↓
robot_gui.py
```

PC GUI에서는 기본 화면에서 카메라를 확인할 수 있고, 전체화면 버튼을 누르면 카메라를 크게 보면서 오른쪽 제어 패널로 로봇을 계속 조작할 수 있습니다.

### 모바일 카메라

`GUIServer_mobile.py`가 Picamera2 영상을 JPEG로 변환하고 브라우저에 MJPEG 스트림 형태로 전달합니다.

```text
Picamera2
    ↓
OpenCV Frame
    ↓
JPEG
    ↓
HTTP MJPEG
    ↓
스마트폰 브라우저
```

---

## 7. Servo 구성

현재 프로젝트에서 사용하는 Servo 역할은 다음과 같습니다.

| Servo 번호 | 역할 |
| --- | --- |
| Servo 0 | 앞바퀴 조향 |
| Servo 1 | 카메라 목 |
| Servo 2 | 로봇 팔 |
| Servo 3 | 손목 기능이 있는 서버 버전에서 사용 |
| Servo 4 | 집게 |

### 앞바퀴 중앙값

조향 중앙은 `RPIservo.py`의 `init_pwm0` 값을 기준으로 합니다.

현재 로봇에서 확인한 중앙값은 다음과 같습니다.

```python
init_pwm0 = 60
```

따라서:

```python
steering.moveAngle(0, 0)
```

은 Servo를 절대각 0도로 보내는 명령이 아니라 `init_pwm0` 기준 중앙 위치로 보내는 명령입니다.

### 카메라 목

현재 카메라 목은 자동 추적하지 않습니다.

카메라 선 길이와 안정성을 위해 실행 중 좌우로 계속 움직이지 않도록 구성되어 있으며, 센서를 추가한 이후 추적 기능을 별도로 구현할 예정입니다.

---

## 8. 필요한 하드웨어 제어 모듈

핵심 프로젝트 파일은 3개지만 실제 로봇 동작을 위해 다음 Adeept 관련 파일이 필요합니다.

```text
Move.py
RPIservo.py
Switch.py
```

각 역할은 다음과 같습니다.

| 파일 | 역할 |
| --- | --- |
| `Move.py` | DC 모터 전진 / 후진 / 정지 제어 |
| `RPIservo.py` | 조향, 로봇 팔, 집게 등 Servo 제어 |
| `Switch.py` | 후레시 GPIO ON/OFF 제어 |

따라서 다른 작업자가 Repository를 새로 Clone하여 실제 로봇에서 실행하려면 위 하드웨어 모듈도 접근할 수 있어야 합니다.

---

## 9. 권장 폴더 구조

현재 Raspberry Pi에서는 다음 구조를 기준으로 사용합니다.

```text
/home/pi/
│
├── Adeept-PiCar-Pro/
│   └── Server/
│       ├── GUIServer_custom.py
│       ├── GUIServer_mobile.py
│       ├── Move.py
│       ├── RPIservo.py
│       ├── Switch.py
│       └── ...
│
└── move/
    └── robot_gui.py
```

GitHub Repository 하나로 정리할 경우 다음처럼 구성해도 이해하기 쉽습니다.

```text
PiCar-Pro-Control/
│
├── README.md
│
├── pc/
│   └── robot_gui.py
│
├── server/
│   ├── GUIServer_custom.py
│   └── GUIServer_mobile.py
│
└── adeept/
    ├── Move.py
    ├── RPIservo.py
    └── Switch.py
```

단, 폴더 구조를 변경하면 Python import 경로도 맞게 수정해야 하므로 현재 실제 Raspberry Pi에서 동작하는 구조를 그대로 업로드하는 것이 가장 안전합니다.

---

## 10. Python 의존성

코드에서 사용하는 주요 Python 모듈은 다음과 같습니다.

```text
opencv-python / cv2
pyzmq / zmq
numpy
Pillow
Picamera2
tkinter
```

또한 Raspberry Pi GPIO / PCA9685 / Adeept 모터 제어 환경이 정상적으로 구성되어 있어야 합니다.

---

## 11. PC GUI 실행 방법

PC GUI를 사용할 때는 **PC용 서버를 먼저 실행**합니다.

### 1단계 - PC용 서버 실행

```bash
cd ~/Adeept-PiCar-Pro/Server
sudo python3 GUIServer_custom.py
```

정상 실행되면 TCP `10223` 포트에서 GUI 연결을 기다립니다.

### 2단계 - PC GUI 실행

다른 터미널에서:

```bash
cd ~/move
DISPLAY=:0 python3 robot_gui.py
```

현재 `robot_gui.py`의 서버 주소가 `127.0.0.1`이므로 같은 Raspberry Pi의 `GUIServer_custom.py`에 연결됩니다.

---

## 12. 모바일 GUI 실행 방법

먼저 PC용 `GUIServer_custom.py`가 실행되고 있다면 `Ctrl + C`로 종료합니다.

그 다음:

```bash
cd ~/Adeept-PiCar-Pro/Server
sudo python3 GUIServer_mobile.py
```

IP 확인:

```bash
hostname -I
```

스마트폰을 Raspberry Pi와 같은 Wi-Fi에 연결한 뒤 브라우저에서 다음 형식으로 접속합니다.

```text
http://RaspberryPi-IP:5000
```

예:

```text
http://192.168.25.112:5000
```

---

## 13. PC 서버와 모바일 서버를 동시에 실행하면 안 되는 이유

다음 두 파일은 둘 다 실제 하드웨어를 직접 사용합니다.

```text
GUIServer_custom.py
GUIServer_mobile.py
```

둘을 동시에 실행하면 다음 문제가 발생할 수 있습니다.

- 동일 모터에 서로 다른 이동 명령 전달
- 동일 Servo에 동시에 명령 전달
- 후레시 상태 충돌
- Pi Camera 동시 사용 충돌
- 로봇이 예상하지 않은 방향으로 움직일 가능성

따라서 반드시 다음처럼 사용합니다.

```text
PC 사용
→ GUIServer_custom.py + robot_gui.py

모바일 사용
→ GUIServer_mobile.py + 스마트폰 브라우저
```

---

## 14. 주요 설정값

### `GUIServer_custom.py`

```python
PORT = 10223
VIDEO_PORT = 5555
speed_set = 25
TURN_SPEED = 30
TURN_ANGLE = 30
```

### `robot_gui.py`

```python
SERVER_IP = "127.0.0.1"
SERVER_PORT = 10223
VIDEO_PORT = 5555

START_SPEED = 20
MAX_SPEED = 50
SPEED_STEP = 5
TURN_SPEED = 30
```

### `GUIServer_mobile.py`

```python
WEB_PORT = 5000
DRIVE_SPEED = 25
TURN_SPEED = 30
TURN_ANGLE = 30
CAMERA_FIXED_ANGLE = 90
```

> 모바일 전진/후진 속도를 높이려면 `DRIVE_SPEED` 값만 변경하면 됩니다. 좌우회전 속도는 `TURN_SPEED`와 별도로 관리됩니다.

---

## 15. 안전 관련 주의사항

실제 로봇을 테스트할 때는 처음에는 바퀴가 바닥에 닿지 않도록 들어 올린 상태에서 테스트하는 것을 권장합니다.

특히 다음 항목을 수정한 후에는 저속으로 먼저 확인합니다.

- 모터 방향
- 조향 방향
- 조향 중앙값
- Servo 각도
- 후레시 GPIO
- 전진/후진 속도

프로그램을 비정상 종료했을 때 모터가 계속 동작하지 않도록 서버 코드에는 정지 처리가 포함되어 있지만, 테스트 중에는 항상 로봇 전원을 바로 차단할 수 있는 상태에서 작업하는 것이 좋습니다.

---

## 16. 코드 수정 시 주의사항

### 카메라 Servo

공식 Adeept 코드의 일부 Servo 초기화 함수는 여러 Servo 채널을 한 번에 초기화할 수 있습니다.

이 프로젝트에서는 카메라 목이 의도하지 않게 움직이는 문제를 줄이기 위해 카메라 Servo를 불필요하게 초기화하거나 반복 제어하지 않도록 구성했습니다.

### 조향 Servo

조향 중앙값 `60`은 실제 로봇에서 측정하여 맞춘 값입니다.

`RPIservo.py`를 새 버전으로 덮어쓰면 이 값이 기본값으로 돌아갈 수 있으므로 반드시 확인해야 합니다.

### 후레시

후레시 ON/OFF 동작의 실제 판단은 GUI가 아니라 서버에서 처리합니다.

즉 `robot_gui.py`의 ON/OFF 버튼은 서버에 다음 명령을 보냅니다.

```text
light_on
light_off
```

실제 GPIO 상태는 `GUIServer_custom.py`가 결정합니다.

---

## 17. Git에 올리지 않아도 되는 파일

Python 실행 후 자동 생성되는 캐시 파일은 Git에 올리지 않아도 됩니다.

`.gitignore` 예시:

```gitignore
__pycache__/
*.pyc
*.pyo
*.log
.DS_Store
.vscode/
```

`.vscode/` 설정을 팀원들과 공유할 필요가 있다면 해당 줄은 제거하면 됩니다.

---

## 18. 작업자용 빠른 확인

프로젝트를 처음 받은 작업자는 먼저 다음 관계를 이해하면 됩니다.

```text
[PC 방식]
robot_gui.py
     ↓
GUIServer_custom.py
     ↓
Move.py / RPIservo.py / Switch.py
     ↓
실제 로봇

[모바일 방식]
스마트폰 브라우저
     ↓
GUIServer_mobile.py
     ↓
Move.py / RPIservo.py / Switch.py
     ↓
실제 로봇
```

핵심적으로:

- `robot_gui.py` = PC 화면
- `GUIServer_custom.py` = PC 화면과 실제 로봇 사이의 서버
- `GUIServer_mobile.py` = 모바일 화면 + 모바일용 로봇 서버
- `Move.py` = 모터
- `RPIservo.py` = Servo
- `Switch.py` = 후레시

이 구조만 이해하면 각 파일의 역할을 빠르게 파악할 수 있습니다.

---

## 19. 현재 개발 방향

현재 구현된 주요 기능:

- PC GUI 제어
- 모바일 Web GUI 제어
- 전진 / 후진
- 전진 좌우회전
- 후진 좌우회전
- 집게 제어
- 후레시 상시 ON
- 방향별 자동 후레시
- 로봇 카메라 실시간 영상
- PC 카메라 전체화면 제어
- 카메라 목 고정

추후 확장 예정 기능:

- 센서를 이용한 카메라 목 자동 추적
- 환경 카메라 실제 연결
- 센서 기반 장애물 / 작업구역 판단
- 모바일 GUI 기능 개선

---

## 20. 요약

```text
GUIServer_custom.py
→ PC 제어용 로봇 서버

robot_gui.py
→ PC에서 보는 GUI

GUIServer_mobile.py
→ 핸드폰 Web GUI + 모바일용 로봇 서버
```

세 파일이 프로젝트의 핵심 제어 코드이며, 실제 하드웨어 제어를 위해 Adeept의 `Move.py`, `RPIservo.py`, `Switch.py`가 함께 필요합니다.
