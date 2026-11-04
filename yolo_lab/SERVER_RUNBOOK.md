# 서버 담당자용 YOLO 실행 명령서 — 2026-11-05

이 문서는 사용자 제공 실기 배포 기록과 yolo_lab의 현재 결과를 정리한 안내다.
SSH/VNC 주소·Wi-Fi·계정 인증 정보는 원래 팀 내부 자료에서 확인한다. 해당 내부 TXT 원문은 커밋하지 않는다.
아래 명령은 서버 담당자가 Raspberry Pi의 PuTTY에서 실행할 명령이며 이번 작업에서 원격 실행하지 않았다.

## 먼저 구분할 모델과 서버

| 항목 | 현재 상태 |
|---|---|
| Pi 기존 models/hazard_3class_v1.pt | 제공 기록의 기존3종 모델. 새7종 모델이 아님 |
| yolo_lab models/yolov8n.pt | COCO80 일반 객체 모델. CPU 추론 확인 |
| yolo_lab models/safety7.pt | 아직 없음. 학교 CUDA 학습·평가 후 생성 |
| 모바일 실행 파일 | 실기 배포 기록 기준 GUIServer_mobile_final.py |
| PC 서버 실행 파일 | 실기 배포 기록 기준 GUIServer_custom.py, 기본 기능은 GUIServer_custom_base.py |

현재 YOLO-Seeun code/에는 GUIServer_mobile_final.py와 GUIServer_custom_base.py가 없다.
따라서 아래 통합 서버 명령은 **이미 동작 중인 Pi 배포 폴더** 기준이며, 이 브랜치를 받아
실기 배포 파일에 덮어쓰거나 다른 브랜치로 전환하라는 안내가 아니다.
기존3종 모델, COCO80 모델, 새7종 모델의 클래스 순서를 혼용하지 않는다.

## A. 기존 모바일 서버를 그대로 실행

기존 PC/모바일 서버가 실행 중이면 그 PuTTY 창에서 Ctrl+C를 한 번 누르고 정상 종료를 확인한다.

```bash
cd ~/Adeept_PiCar-Pro/Server
ls -lh GUIServer_mobile_final.py GUIServer_mobile.py YOLO_detector.py
ls -lh models/hazard_3class_v1.pt
sudo ~/picar-yolo/bin/python GUIServer_mobile_final.py
```

브라우저는 기존 Pi 주소의5000포트로 접속한다: `http://<현재_PI_IP>:5000`.
시작 로그에서 YOLO 모델 로드 완료와 카메라 시작을 확인한다.
이 명령만으로 새7종 모델로 바뀌지 않는다. 기존 서버 설정의 모델을 그대로 로드한다.

## B. 기존 PC 서버 + TigerVNC 실행

### PuTTY1 — PC 서버

```bash
cd ~/Adeept_PiCar-Pro/Server
ls -lh GUIServer_custom.py GUIServer_custom_base.py YOLO_detector.py
ls -lh models/hazard_3class_v1.pt
sudo ~/picar-yolo/bin/python GUIServer_custom.py
```

### PuTTY2 — Pi의 Tkinter GUI

```bash
cd ~/move
DISPLAY=:0 ~/picar-yolo/bin/python robot_gui.py
```

기존 VNC 세션이 :1이면 위 DISPLAY만 :1로 바꾼다. 실제 디스플레이 번호를 새로 가정하지 않는다.
Windows TigerVNC Viewer는 팀에서 기존에 사용하던 연결로 접속한다. 새 VNC 서버/세션을 만들 필요는 없다.
PC GUI는 Pi 안에서 실행되어127.0.0.1의 제어 서버에 연결한다.
제어 TCP10223, 영상 ZeroMQ5555이며 모바일 웹5000과 별개다.
모바일 서버와 PC 서버는 같은 하드웨어를 사용하므로 함께 실행하지 않는다.

## C. 기존3종 모델만 카메라 없이 확인

서버를 종료한 상태에서 기존 모델의 실제 클래스 수/이름을 확인한다.
설정/캐시는 yolo_lab 안에 둔다. 모델 파일이 없으면 실행하지 말고 먼저 전송 위치를 확인한다.

```bash
cd ~/Adeept_PiCar-Pro/Server
mkdir -p yolo_lab/.config/Ultralytics
YOLO_CONFIG_DIR="$PWD/yolo_lab/.config" ~/picar-yolo/bin/python -B -c "from pathlib import Path; from ultralytics import YOLO; p=Path('models/hazard_3class_v1.pt'); assert p.is_file(), p; m=YOLO(str(p)); print('classes=',len(m.names)); print(m.names)"
```

이 명령은 기존3종 모델용이다. 새 yolo_lab의 --mode custom 검사는7종을 요구하므로 기존3종 가중치에 적용하지 않는다.
영상이 보여도 모델 로딩이 실패했을 수 있다. 클래스 수·이름·시작 로그를 별도로 확인한다.

## D. 새 baseline 패키지로 YOLO만 독립 시험

이 시험에는 기존 서버/GUI/모터/OLED가 필요하지 않다. 카메라를 쓰는 기존 서버는 먼저 정상 종료한다.

### Windows에서 전달 폴더 만들기

YOLO 학습/실험용 Python 환경에서 실행한다.

```powershell
Set-Location (Join-Path $env:USERPROFILE 'Documents\HazardScout-AI\yolo_lab')
python prepare_models.py
python package_pi.py --mode baseline
```

출력된 deploy/baseline_<실제시간> 폴더만 WinSCP로 Pi의 다음 **새 폴더**에 전송한다:
`~/Adeept_PiCar-Pro/Server/yolo_lab/runtime_baseline_20261105`.
기존 같은 이름의 폴더가 있다면 다른 새 이름을 사용하고 아래 cd 경로도 함께 바꾼다.
Git 커밋에는 가중치·데이터·deploy가 없으므로 커밋만 받아서는 모델이 전달되지 않는다.

### PuTTY에서 무결성·모델 로딩·카메라 확인

```bash
cd ~/Adeept_PiCar-Pro/Server/yolo_lab/runtime_baseline_20261105
~/picar-yolo/bin/python -B - <<'PY'
from pathlib import Path
import hashlib, json
for name, expected in json.loads(Path('SHA256.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('SHA256 OK')
PY
~/picar-yolo/bin/python -B -c "from lab_utils import load_model; import torch,ultralytics; m=load_model('baseline'); print(torch.__version__,ultralytics.__version__); print(len(m.names),m.names)"
~/picar-yolo/bin/python -B -c "from picamera2 import Picamera2; print('Picamera2 OK')"
~/picar-yolo/bin/python -B test_pi_camera.py --mode baseline --imgsz 320 --conf 0.35 --every 3 --max-frames 120 --no-show
```

baseline은 사람·의자·병·가방 등의 COCO80이다. 안전 장비7종이 이 모델에 추가되어 있지는 않다.
PuTTY는 --no-show를 사용한다. 장치 접근 권한 오류가 나면 담당자가 기존 실행 환경/카메라 점유 상태를 확인한다.
결과는 이 runtime 폴더의 results/<실행시간>/summary.json, detections.csv, first_frame.jpg에 저장된다.
워밍업 제외 추론 평균/p50/p95, 실제 추론 수, 처리율, Linux peak RSS·온도를 확인한다.
속도 비교는 같은 조건에서 **--every 1**과 **--every 3**으로 각각 실행한다.

## E. 7종 모델 학습 후 전달·확인

**지금 models/safety7.pt는 없다.** 학교 RTX3080에서 먼저 학습·평가한다.

```powershell
# Windows yolo_lab, CUDA PyTorch 환경
python prepare_dataset.py --check
python train_yolo.py --epochs 50 --batch 8 --device 0
python test_yolo.py --mode custom --evaluate val --imgsz 320
# 설정/모델을 확정한 뒤 독립 test 평가
python test_yolo.py --mode custom --evaluate test --imgsz 320
python package_pi.py --mode custom
```

학습 성공 파일: models/safety7.pt, 원본 best.pt: runs/safety7_<시간>/weights/best.pt.
출력된 deploy/custom_<시간> 폴더만 Pi의 새 yolo_lab/runtime_custom_<배포일> 폴더로 전송한다.

```bash
# 실제 전송한 폴더 이름으로 변경
cd ~/Adeept_PiCar-Pro/Server/yolo_lab/runtime_custom_배포일
~/picar-yolo/bin/python -B -c "from lab_utils import load_model; m=load_model('custom'); print(len(m.names),m.names)"
~/picar-yolo/bin/python -B test_pi_camera.py --mode custom --imgsz 320 --conf 0.35 --every 3 --max-frames 120 --no-show
```

기대 ID는0콘/1소화기/2안전모/3조끼/4장갑/5귀덮개/6보안경이다.
custom 검사는 정확한7종 이름·순서가 다르면 중지한다.
이 모델은 COCO80을 유지하지 않는다. 일반 객체 모드로 돌아가려면 baseline 모델을 사용한다.

## F. 통합 서버가 새 모델을 사용하도록 연결

이 단계는 **서버 담당자가** 실제 배포 파일의 설정을 확인하고 적용한다. 이번 작업은 서버 파일을 변경하지 않았다.
첨부 기록의 모바일 final/PC base 구조는 현 브랜치와 다르므로 기존 코드의 줄 번호를 그대로 적용하지 않는다.

```bash
cd ~/Adeept_PiCar-Pro/Server
# 존재하는 배포 파일에서 실제 모델 설정 위치만 읽기 확인
for f in GUIServer_mobile_final.py GUIServer_mobile.py GUIServer_custom.py GUIServer_custom_base.py YOLO_detector.py; do
  if [ -f "$f" ]; then
    grep -nE 'YOLO_MODEL_PATH|hazard_3class|safety7|YOLODetector' "$f"
  fi
done
```

독립 시험을 통과한 뒤 실제로 적용되는 YOLO_MODEL_PATH를 새 모델의 절대 경로로 지정한다.
예시 값: `/home/사용자/Adeept_PiCar-Pro/Server/yolo_lab/runtime_custom_배포일/models/safety7.pt`.
실제 경로는 다음처럼 확인할 수 있다:

```bash
realpath ~/Adeept_PiCar-Pro/Server/yolo_lab/runtime_custom_배포일/models/safety7.pt
```

모델 경로를 환경변수로 export하는 것만으로 기존 서버가 이를 읽는다고 가정하지 않는다.
기존3종 모델은 그대로 보존하고 모델 파일을 덮어쓰거나 이름만 safety7.pt로 바꾸지 않는다.
담당자가 경로를 적용한 뒤 모바일은A, PC는B의 기존 실행 명령으로 재시작한다.
카메라 독립 시험과 통합 서버는 동시에 실행하지 않는다.

GUI는 상자가 그려진 영상을 받으므로 표시 형식은 유지되지만 custom에서는7종만 나타난다.
별도 클래스 목록·모드 선택 UI는 자동 추가되지 않는다.
현재 서버의 색상 변환·이전 박스 재사용은 SERVER_REFERENCE.md의 확인 사항을 실제 장비로 점검한다.

## 종료·문제 확인

- GUI/로봇을 정지시키고 GUI를 종료한 다음 서버 PuTTY에서 Ctrl+C를 한 번 눌러 종료 로그/프롬프트를 확인한다.
- VNC Viewer만 닫아도 서버/GUI가 남아 있을 수 있다. 서버 재시작에는 Pi 전원 종료가 필요하지 않다.
- ModuleNotFoundError면 ~/picar-yolo/bin/python 사용 여부, Picamera2 접근과 실제 배포 파일 위치를 확인한다.
- 모델 로딩 실패면 파일 존재, 절대 경로, 실제 클래스 이름을 확인한다. 모델이 없을 때 자동 대체하지 않는다.
- 카메라 충돌이면 기존 PC/모바일/독립 시험 중 다른 프로세스가 장치를 사용 중인지 확인한다.
- 7종 정확도·실제 Pi 지연/메모리·GUI 전환은 아직 실기 검증 전이다.
