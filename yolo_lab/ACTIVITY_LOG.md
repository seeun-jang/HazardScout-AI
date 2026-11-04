# YOLO 담당 활동기록 — 2026-11-04~2026-11-05

## 활동 주제와 담당 범위

Raspberry Pi4(8GB) 자율 탐사 로봇 HazardScout-AI에서 담당 영역인 yolo_lab을
YOLOv8n 기반의 작은 학습·추론 프로젝트로 재구성했다.
COCO 일반 객체 인식과 실제 구매한 안전 장비7종 인식을 별도 모델로 운영할 수 있도록 구성했다.
작업 브랜치는 YOLO-Seeun이며 팀원의 서버·GUI·모터·Servo·OLED·음성 코드는 읽기 전용으로 참고했다.
사용자 요청에 따라 변경 내용을 로컬 커밋하고 Push/PR/Merge/Rebase는 수행하지 않는다.

## 1. 기존 프로젝트 분석과 복구 가능한 정리

### 발견한 문제

- YOLO-World/YOLOE 준비·실행 옵션과 YOLOv8n 실험 코드가 섞여 있었다.
- 소화기·안전모·콘3종 병합 코드와 대량 데이터, 과거 학습 결과가 활성 폴더에 남아 있었다.
- 기본 가중치가 중복 저장되어 있었으며 world_hazard.pt는 약328MB였다.
- 데이터/가중치/학습 결과를 그대로 Pi에 전달하면 불필요한 저장 공간과 관리 비용이 발생했다.
- 기존 데이터 설정이7종의 고정 ID를 설명하지 않아 모델/라벨의 클래스 의미를 혼동할 수 있었다.

### 수행한 작업

- 시작 전 git branch --show-current로 YOLO-Seeun을 확인했다.
- 기존 파일13,065개(668,020,832bytes, 약637MiB)의 경로·크기·SHA-256을 inventory.json으로 기록했다.
- 기존 dataset/models/runs/results, 3종 병합 코드, 과거 안내문과 중복/World 가중치를 yolo_lab/backups에 보존했다.
- 원본 전체의 SHA-256을 정리 후 다시 확인했다. 복구 불가능한 대량 삭제는 하지 않았다.
- 활성 기본 모델은 models/yolov8n.pt 하나로 모았다.
- yolo_lab/.gitignore에서 백업·원본 사진/라벨·가중치·학습 결과·전송 패키지·캐시·가상환경을 제외했다.
- 저장소 최상위 .gitignore/README/code/runs/weights는 변경하지 않았다.

## 2. 7종 데이터셋 구성

### 클래스 정의

| ID | 클래스 | 구매한 장비 |
|---:|---|---|
| 0 | traffic_cone | 주황색 미니 교통콘 |
| 1 | fire_extinguisher | 빨간 소화기 모형, 노란 호스 |
| 2 | safety_helmet | 노란 CAT 안전모 |
| 3 | safety_vest | 노란 망사 형광 조끼 |
| 4 | safety_gloves | 회색 줄무늬·노란 작업장갑 |
| 5 | earmuffs | 노랑/검정 귀덮개와 밝은 밴드 |
| 6 | safety_goggles | 노란 테두리·검정 스트랩 보안경 |

### 수집·검토 절차

1. 기존 Roboflow 자료의 출처·라이선스·라벨을 확인하고 작은 시험용 부분집합을 골랐다.
2. 처음 선정한 street-work 후보에 의료용 일회용 장갑과 의료용/일반 안경이 많아 구매 장비에 맞는 양성 데이터로 계산하지 않았다.
3. 부족했던 장갑·귀덮개·보안경을 보충하기 위해 SH17 원저자 자료와 공개 후보를 추가 검토했다.
4. 공개 미러의 COCO 숫자 클래스 순서가 원본 YOLO ID와 다른 것을 발견했다.
   숫자만 보고 라벨을 재매핑하지 않고 원저자 sh17.yaml 및 공식 Kaggle의 후보419개 원본 라벨을 직접 받아 좌표·클래스를 대조했다.
5. 최종 SH17 사진105장은 원저자가 제공한 Pexels URL에서 직접 확보했다.
   전체 구도를 자르지 않고640×640으로 변환하고 검토 이미지와 레이아웃 해시를 비교했다.
6. Bounding Box 확대와 전체 장면을 모두 확인했다. 일반 안경·통화/음악용 헤드셋·의료용 장갑·다른 품목은 제외했다.
7. 함께 보이는 다른 대상 장비의 누락 라벨은 기존 원저자 bbox 좌표로 보완했다.
   모델 예측 pseudo-label이나 합성 이미지로 수량을 채우지 않았다.
8. 실제 제공 사진8장은 원본을 보존하고 EXIF 방향을 정규화한 별도 복사본에 보이는 장비의 bbox를 수동 지정했다.
   Overlay를 확인해 콘 끝·소화기 호스·장갑 끝·귀덮개 밴드가 잘리지 않도록 보정했다.
9. SHA-256, 64-bit dHash 거리<=5, 동일 Roboflow 원본 이름으로 중복을 검사했다.
10. 같은 연구실/작업장/제품 촬영 세션은 하나의 group/split으로 묶었다.
    실제 구매 사진8장은 같은 배경의 촬영 세션이므로 모두 train에 두었다.
11. 안전모 등의 수량이 커진 부분은 기존 샘플 일부를 내부 백업으로 옮겨 전체 학습 데이터를 작게 유지했다.

### 최종 수량

서로 다른 이미지213장: train145 / val40 / test28.
한 이미지에 여러 장비가 있어 아래 양성 이미지 수의 합은213보다 크다.

| ID | 클래스 | train | val | test | 합계 |
|---:|---|---:|---:|---:|---:|
| 0 | traffic_cone | 29 | 6 | 6 | 41 |
| 1 | fire_extinguisher | 30 | 6 | 6 | 42 |
| 2 | safety_helmet | 37 | 8 | 4 | 49 |
| 3 | safety_vest | 20 | 6 | 4 | 30 |
| 4 | safety_gloves | 37 | 6 | 7 | 50 |
| 5 | earmuffs | 32 | 6 | 4 | 42 |
| 6 | safety_goggles | 28 | 13 | 5 | 46 |

모든 클래스 총30장 이상, 모든 split에 양성 이미지가 있는 상태로 검사에 통과했다.
출처별 서로 다른 사진은 cone40 + fire40 + RF100 20 + SH17 105 + 실물8 =213장이다.
원본 이미지와 라벨은 로컬에서 보존하고 Git에서는 제외했다. Git에는 설정·manifest·출처·검증 보고서를 남겼다.

### 출처·이용 조건

- cone: project-k84yn / Roboflow Universe, CC BY4.0.
- fire extinguisher: CV projects / Roboflow Universe, CC BY4.0.
- construction safety: Roboflow100, CC BY4.0.
- SH17: Hafiz Mughees Ahmad·Afshin Rahimi, 원저자 CC BY-NC-SA4.0 및 Pexels 조건을 유지한 학교 비상업 학습용.
- 실물 사진: 사용자 제공 로컬 학습용. 외부 재배포 권한을 가정하지 않았다.

구체 URL·파일별 해시·변환·라벨 대조 기록은 dataset/SOURCES.md, sources.json, raw/curated/sh17/PROVENANCE.json에 기록했다.

## 3. 모델 운영 방식 결정

| 검토 방식 | 판단 |
|---|---|
| COCO80+장비7 단일 모델 | 공동 데이터셋과 재라벨링, 두 클래스군의 정확도 검증 부담이 큼 |
| 기본/커스텀 중 하나 선택 | Pi CPU의 메모리·추론 부담을 줄일 수 있어 우선 구현 |
| 두 모델 동시 추론 | Pi 실측 및 학습된7종 모델이 없어 이번 단계에서는 미구현 |

7종만 fine-tuning한 모델이 COCO80을 유지한다고 가정하지 않았다.
현재 baseline은 COCO80 yolov8n.pt, custom은 별도7종 safety7.pt를 사용하는 구조다.
모드 전환은 실행 종료 후 다른 모드로 재실행한다.
YOLO-World/YOLOE는 활성 준비·실행 경로에서 제거했다.
ONNX/NCNN/양자화는 호환성과 정확도 비교를 거치기 전이므로 기본 적용하지 않았다.

## 4. 파일별 구현 내용

| 파일 | 변경/추가 내용 |
|---|---|
| prepare_models.py | 기본 YOLOv8n만 준비하고 COCO80 모델인지 확인 |
| prepare_dataset.py | 검토 manifest를 데이터셋으로 빌드; 라벨 좌표/ID·누락·중복·세션 누출·7종 수량 검사 |
| train_yolo.py | YOLOv8n 초기 가중치로 CUDA 학습; Windows workers0, imgsz640, seed42, batch8/epochs50 기본; 출력 모델 덮어쓰기 방지 |
| lab_utils.py | 클래스 순서·경로·모델 검증·공통 인자·내부 캐시 경로 관리 |
| inference.py | CPU 추론 공통 루프, 3프레임마다 탐지, CSV/첫 이미지/시간 통계, 자원 반환 |
| test_yolo.py | 사진/영상/USB 카메라 시험과 custom val/test 평가; 확장자 없는 JPEG/EXIF 처리 |
| test_pi_camera.py | Picamera2 독립 시험; RGB888 배열을 BGR로 전달, 카메라 stop/close |
| package_pi.py | 모델1개+추론 코드/설정만 복사한 최소 Pi 패키지와 SHA256 목록 생성 |
| tests/test_workflow.py | 부족 데이터·오류 라벨·중복·세션 누출·출력 범위·모델 누락·카메라 반환·현재7종 데이터 검사 |
| requirements-laptop.txt | Pi 목표 Ultralytics8.4.174 고정; CUDA PyTorch 설치는 공식 선택기로 별도 안내 |
| README.md / REPORT.md | 실행 구조, 결과, 검증 범위, 남은 작업과 Git 범위 정리 |
| SERVER_REFERENCE.md | 현 브랜치 팀 코드의 읽기 전용 연동 분석 |
| SERVER_RUNBOOK.md | 제공된 실기 배포 기록을 반영한 서버 담당자용 실행 명령과7종 전환 절차 |

추론 초기값은 imgsz320/conf0.35/every3이다. CPU 스레드는2로 제한했다.
첫 워밍업은 속도 통계에서 제외하고 평균/p50/p95 추론 시간, 실제 프레임/추론 수,
Linux peak RSS와 종료 온도를 기록한다. 건너뛴 프레임에는 오래된 상자를 다시 그리지 않는다.

## 5. 기존 서버·GUI 읽기 전용 분석

- YOLO_detector는 모델 하나를 로드하고 model.names에서 클래스 이름을 얻어 영상에 그린다.
- PC는 상자가 그려진 JPEG를 Base64/ZeroMQ로 전송하고 robot_gui/fullscreen_gui는 영상을 표시한다.
- 모바일은 같은 방식으로 처리한 영상을 MJPEG로 전송한다.
- 따라서 모델 교체로 GUI에 클래스 목록·모드 선택 UI가 자동 추가되지 않는다.
- 모델 로딩 실패 시 카메라만 실행될 수 있어 영상 표시만으로 YOLO 성공을 판단하면 안 된다.
- 서버의 RGB888 이후 추가 색상 변환과 이전 박스 재사용은 독립 시험과 차이가 있어 실제 Pi에서 담당자 확인이 필요하다.
- 팀 코드7개 파일의 SHA-256이 검토 전후 동일함을 확인했다. 서버/GUI/하드웨어 코드는 실행하거나 수정하지 않았다.
- 제공된 실기 문서의 최종 모바일 파일은 GUIServer_mobile_final.py이고 PC는 GUIServer_custom_base.py를 사용하는 배포 구조다.
  이 두 파일은 현재 YOLO-Seeun code/에 없으므로 실기 배포 기록과 현 브랜치 분석을 구분해 문서화했다.

## 6. 실제로 검증한 내용

- 기존13,065개 원본 파일 백업의 SHA-256 일치.
- yolo_lab 외부 추적 파일과 팀 code/7개 파일의 해시 불변.
- 최종213장7종 데이터의 라벨·중복·세션·수량 검사 통과.
- 기능 테스트8개와 Python 문법 검사 통과.
- baseline COCO80 가중치 로딩과 실제 사진 CPU 추론.
-12프레임 시험에서4회 추론 및 스트림 반환.
- CSV/결과 이미지/summary 생성과 최소 Pi 패키지의 독립 CPU 실행.
- 기존 Python3.12.15/torch2.14.1+cpu/Ultralytics8.4.175 환경 유지.
- 공식 Ultralytics8.4.174를 내부 .config에 분리해 실제 사진 추론 호환성 확인.
- 학습 전 데이터 검사는 통과하고 현재 PC의 CUDA 부재로 학습은 명확한 오류와 함께 중지하는 것을 확인.

## 7. 아직 하지 않은 작업과 한계

- 학교 RTX3080 CUDA 학습과 models/safety7.pt 생성은 아직 수행하지 않았다.
- 새7종 모델의 precision/recall/mAP와 실물 인식 정확도는 측정하지 않았다.
- 실제 Pi4/Python3.11.2 카메라·지연·RSS·발열·GUI 통신은 검증하지 않았다.
- 두 모델 동시 추론과 ONNX/양자화 최적화는 검증하지 않았다.
- 실제 구매 장비 학습 사진은 클래스당1장(소화기2장)뿐이다. 실물의 별도 촬영 val/test가 필요하다.
- 현재 Pi에서 사용 중인 hazard_3class_v1.pt는 제공된 서버 기록의 기존3종 모델이며 새7종 모델이 아니다.
  기존 모델을 이름만 safety7.pt로 바꿔 사용하지 않았다.

## 제출용 서술 예시

HazardScout-AI의 YOLO 담당 영역을 Raspberry Pi4 CPU 환경에 맞게 경량화하고 재구성했다.
먼저 YOLO-Seeun 브랜치와 변경 범위를 확인하고 기존13,065개 파일을 해시와 함께 내부 백업했다.
YOLO-World 및 과거3종 병합 실험을 활성 경로에서 정리하고 YOLOv8n 기반의 준비·학습·평가·카메라 시험 흐름을 구성했다.
구매한 안전 장비7종에 대해 공개 데이터와 실물8장을 검토해213장의 소규모 Bounding Box 데이터셋을 만들었다.
원저자 라벨419개를 대조해 숫자 클래스 재정렬 문제를 확인했고 일반 안경·음악 헤드셋·의료용 장갑과 중복 이미지를 제외했다.
모든 클래스가30~50장 범위에서 train/val/test에 포함되도록 구성했으며 실물 사진은 수동 bbox를 검토했다.
COCO 일반 객체와 커스텀7종을 별도 모델로 선택 실행하도록 설계하고 기본값을320/0.35/3프레임으로 설정했다.
기능 테스트8개, 기본 모델 실추론과 최소 Pi 전달 패키지 실행을 검증했다.
팀원의 서버·GUI·하드웨어 코드는 변경하지 않고 연동 구조와 실제 서버 실행 명령을 문서화했다.
현재 PC의 CUDA 부재로 커스텀 학습과 Pi 실측은 후속 작업으로 남겨 두었다.
