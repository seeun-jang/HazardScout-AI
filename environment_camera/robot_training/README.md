# HazardScout: 로봇팔 + 이동로봇 YOLO 1차 학습

## 목표와 현재 한계

- **두 개의 분리된 클래스:** `0 mobile_robot` (검정 바퀴차체), `1 robot_arm` (다관절팔·그리퍼).
- 기존 7종 `yolo_lab/models/safety7.pt`는 **절대 수정하지 않습니다.** 환경카메라에서는 기존 safety7 + 새 robot 모델을 각각 읽습니다.
- **현재 ZIP에는 실제 현장 로봇 2장(4개 bbox 라벨)과 로봇이 없는 현장 4장(음성 예시), 2개의 공개 데이터셋 링크·병합 코드가 들어 있습니다.** 공개 데이터셋 원본은 Roboflow 로그인/다운로드 절차가 필요해 ZIP에 포함하지 않았습니다.
- 기존 사진 2장만으로 학습할 경우 `--allow-starter-only`가 필요하며, 이 경우 **코드 작동 확인용**에 불과합니다. 실전 정확도 평가/발표 결과로 사용하지 마세요.
- 정지된 소품과 움직이는 로봇은 구별해야 합니다. 새 모델을 학습하고도 카메라 시점·거리·팔 각도에 따른 미탐을 확인해야 합니다.

## 1. 공개 데이터셋 다운로드: 최소 1~2개

Roboflow 회원 계정으로 접속 후 각 페이지에서 **Download Dataset → YOLOv8 → Download ZIP** 선택. 별도 파일을 GitHub에 올리지 마세요.

1. [Robot Arm — Moscow State University, 576장, Public Domain](https://universe.roboflow.com/moscow-state-university-gprff/robot-arm-tkb7h/dataset/1) — **추천 1순위**
2. [Robot Arm — Moscow State University, 89장, Public Domain](https://universe.roboflow.com/moscow-state-university-gprff/robot-arm-2njnl) — **보완용**
3. [Robotic Arm — gripper/Servo, teamproject, CC BY 4.0](https://universe.roboflow.com/teamproject-3f4si/robotic-arm-akk9a) — 이번 2종 모델에 자동 통합하면 클래스 의미가 달라지므로 **참고만**.

예시 배치 (압축 해제 폴더에는 `data.yaml`, `train/images`, `train/labels`가 포함되어야 함):

```text
HazardScout-AI/
  environment_camera/
    robot_training/
      starter/images + starter/labels   # 실제 사진 6장 + 라벨
      external/
        robot_arm_576/                   # Roboflow ZIP을 해제한 데이터
          data.yaml
          train/images ...
        robot_arm_89/
          data.yaml
          train/images ...
      prepare_dataset.py
      train_robot.py
      test_robot.py
      models/                           # 학습 결과 robot_v1.pt (Git 제외)
```

> 외부 데이터의 저작권·출처·라이선스 표시를 유지하세요. Public Domain 표시는 해당 프로젝트 페이지의 표시이며, 원본과 이용조건은 다운로드할 때 다시 검토하세요. 여러 데이터셋에 사진이 중복될 수 있으므로 분리 전 중복검사 및 현장 이미지 독립검증이 필요합니다.

## 2. Windows VS Code 터미널에서 데이터 합치기

`environment_camera`에서 실행하고 이미 활성화된 프로젝트 `.venv`를 그대로 사용합니다.

```powershell
python -m pip install pyyaml
python robot_training/prepare_dataset.py --external robot_training/external/robot_arm_576 --external robot_training/external/robot_arm_89
```

**공개 데이터 다운로드 전 데이터 생성 흐름만 시험하려면:**

```powershell
python robot_training/prepare_dataset.py
```

그러면 `robot_training/dataset/combined/{images,labels}/{train,val}`, `data.yaml`, `manifest.csv`가 생성됩니다. 라벨은 정규화된 YOLOv8 박스 2종으로 통일됩니다.

## 3. 학습 (CPU를 기본 설정)

```powershell
python -m pip install -r requirements.txt
python robot_training/train_robot.py --epochs 40 --imgsz 640 --device cpu
```

CUDA PyTorch가 설치되어 있다면 `--device 0`. GPU가 없어도 CPU로 작동하지만 오래 걸릴 수 있습니다. 공개 데이터가 없다면 학습 스크립트는 기본적으로 실행을 중지합니다. **실행 파이프라인 점검만** 하려면 `--allow-starter-only --epochs 3`를 추가하세요(실전 탐지용이 아님).

학습이 끝나면 **로봇 전용** 파일만 생성합니다.

```text
robot_training/models/robot_v1.pt
```

기존 safety7 모델 파일을 덮어쓰지 않습니다.

## 4. 사진으로 로봇 인식 시험

```powershell
python robot_training/test_robot.py robot_training/starter/images/own_top.jpg robot_training/starter/images/own_side.jpg
```

출력: `robot_training/results/*_robot.jpg`. 위 두 사진은 학습에도 사용되므로, 반드시 **처음 보는 실제 환경카메라 사진/영상**으로 최종 성능을 검증해야 합니다.

## 5. 기존 환경카메라 대시보드에 연결

`environment_camera/.env`를 열고 아래 행을 추가하거나 수정:

```env
ROBOT_YOLO_MODEL=robot_training/models/robot_v1.pt
```

기존 설정은 유지합니다. 서버 종료 `Ctrl+C` 후 다시 `python server.py`.

탐지 성공 시: 기존 안전용품 녹색 박스 + **로봇 전용 파란 박스**. 로봇 주변 배경 변화가 `UNKNOWN`으로 오인되는 문제를 줄이는 용도지만 모델 미탐이 있으면 빨간 UNKNOWN이 여전히 발생할 수 있습니다.

## 6. 데이터 품질 주의

- `starter/preview_own_*.jpg`에서 수동 라벨을 확인하세요. 로봇 팔의 선/그리퍼 일부가 박스 바깥으로 뻗거나 카메라 프레임 밖으로 나간 경우는 **임시 라벨**이며 재촬영·재라벨링을 권장합니다.
- 576장 데이터셋 버전은 자체 검증 세트가 없는 것으로 표시되어, 코드에서 일부 데이터를 자동 분리합니다. 공용 데이터의 동일 촬영 장면·증강 이미지가 양쪽으로 갈라지면 성능 지표가 과대평가될 수 있습니다. **최종 평가는 현장에서 새롭게 찍은 별도 영상으로 하세요.**
- 현장 카메라가 아직 연결되지 않았으므로 실시간 인식 성능이나 `UNKNOWN` 오탐 개선은 **검증 전**입니다.
- 원본 공개 데이터 / 사용자 사진 / 생성된 학습 데이터 / 가중치 파일을 공개 GitHub에 푸시하지 않도록 `environment_camera/.gitignore`를 보강했습니다.
