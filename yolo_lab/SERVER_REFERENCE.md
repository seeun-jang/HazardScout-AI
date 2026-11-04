# 기존 서버·GUI 읽기 전용 참고 — 2026-11-05

검토 범위는 저장소 code/의 README, GUIServer_custom.py, GUIServer_mobile.py,
YOLO_detector.py, robot_gui.py, fullscreen_gui.py, OLED_eyes.py입니다.
파일을 수정하거나 서버/GUI/하드웨어 모듈을 실행하지 않았습니다.
검토 전후 해시는 backups/20261104_before_missing_classes/team_code_sha256.json으로 확인합니다.

## 영상과 탐지 데이터 흐름

```text
PC: Picamera2 -> 서버의 YOLODetector.detect_and_draw(BGR)
    -> JPEG -> Base64 -> ZeroMQ PAIR -> robot_gui.py -> fullscreen_gui.py
Mobile: Picamera2 -> 같은 detector -> JPEG -> 최신 프레임 공유/Condition
        -> MJPEG HTTP stream -> 브라우저
```

- PC 제어 TCP 10223과 영상 ZeroMQ 5555는 별개입니다. 모델 추론을 PC GUI로 옮기지 않습니다.
- PC 서버는 GUI의 영상 수신 포트로 connect하고 robot_gui는 bind합니다.
- GUI에 들어오는 것은 상자가 이미 그려진 JPEG입니다. 클래스별 JSON 탐지 이벤트를 전달하는 구조가 아닙니다.
- 따라서 7종 가중치로 교체해도 GUI의 영상 표시 방식은 유지되지만, 별도 클래스 목록/모드 선택 UI는 자동 생성되지 않습니다.
- fullscreen_gui는 robot_gui의 get_frame과 주행/집게/라이트 콜백을 재사용합니다. 별도 YOLO 추론을 하지 않습니다.
- OLED는 이동 방향/자체 스레드로 동작합니다. 현재 YOLO 클래스와 연동되어 장비별 행동을 하는 코드는 없습니다.

## 모델 연결 계약

- YOLO_detector.py:21의 YOLODetector는 model_path/confidence/imgsz/detect_every_n_frames를 받습니다.
- __init__은 Ultralytics YOLO 한 개만 로딩합니다. 기본값은 .pt/0.35/320/3입니다.
- _run_inference는 BGR numpy frame을 받고 각 객체를 `(x1,y1,x2,y2,class_name,confidence)`로 구성합니다.
- class_name은 model.names[class_id]에서 가져오므로 custom 7종의 영어 이름을 그대로 표시할 수 있습니다.
- detect_and_draw는 입력과 같은 형태의 영상에 상자를 그려 반환합니다. 독립 시험의 CSV/summary와 서버 인터페이스를 혼합하지 않습니다.
- 숫자 COCO ID를 GUI로 전송하는 부분은 이 영상 흐름에서 발견하지 못했습니다. 향후 별도 동작 연동을 추가한다면 ID 의미를 다시 정의해야 합니다.
- GUIServer_custom.py:114와 GUIServer_mobile.py:140의 YOLO_MODEL_PATH는 상대 문자열 yolov8n.pt입니다.
  실행 폴더에 따라 해석되므로 담당자가 검증된 가중치의 절대 경로로 지정하는 방법만 README에 기록합니다.
- 두 서버 모두 모델 로딩 실패 시 detector=None으로 두고 카메라는 계속 실행합니다.
  영상이 보인다는 사실만으로 custom 모델 로딩 성공을 판단하면 안 됩니다. 시작 로그의 모델/클래스 수와 실제 상자를 확인해야 합니다.

## yolo_lab 개발에 반영할 차이

1. 독립 테스트/패키지는 모터·Servo·OLED·네트워크 제어 모듈을 import하지 않습니다.
   이 모듈은 import/시작 과정에서 하드웨어 초기화가 발생할 수 있으므로 Windows에서 서버 파일을 실행해 시험하지 않습니다.
2. 서버 둘 모두 RGB888을 capture한 뒤 RGB2BGR 변환을 수행합니다
   (custom.py:435,469 / mobile.py:935,992). Picamera2 공식 구현은 RGB888 배열을 BGR로 다룹니다.
   따라서 추가 변환이 R/B를 뒤바꿀 가능성이 있습니다. 독립 시험은 변환 없이 BGR을 사용합니다.
   이 차이는 실제 Pi의 빨간 소화기/노란 장비로 담당자가 확인해야 하며 여기서 서버 코드는 변경하지 않았습니다.
   근거: https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py
3. 기존 detector의 추론 프레임은 `첫 프레임 또는 frame_count % N == 0`이므로 N=3에서 1,3,6,9입니다.
   독립 시험은 일정 간격 1,4,7,10입니다. 같은 3 설정이라도 최초 구간 횟수가 다를 수 있습니다.
   Pi 속도를 비교할 때 summary의 실제 inference 수를 함께 확인합니다.
4. 기존 detector는 건너뛴 프레임에 이전 상자를 다시 그리며 추론 예외 때도 이전 값이 남을 수 있습니다.
   독립 시험은 건너뛴 프레임에 오래된 상자를 그리지 않습니다. 모델 경로 변경이 서버의 이 정책을 바꾸지는 않습니다.
5. 모델 추론이 서버 카메라 루프 안에 있으므로 지연이 영상 전송률에 영향을 줍니다.
   Windows CPU 수치를 GUI 전체 지연으로 해석하지 않고 Pi에서 별도 측정합니다.
6. PC/모바일 서버와 독립 카메라 시험을 동시에 실행하지 않습니다. 카메라 장치는 하나의 프로세스가 소유해야 합니다.

위 내용은 정적 코드 검토이며 실제 Pi 접속/모터 작동/GUI 통신/하드웨어 성능 검증은 수행하지 않았습니다.
