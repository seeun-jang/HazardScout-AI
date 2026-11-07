# HazardScout-AI 작업 결과 — 2026-11-05 (Asia/Seoul)

> **2026-11-07 추가:** 노트북에서 safety7 213쌍을 SHA-256으로 검증(213/213)하고, 노트북 CPU로 7종 YOLOv8n을 실제 학습했습니다.
> `models/safety7.pt`(100 epoch, SHA-256 `91e388a7…`)를 Git에 포함합니다. 데이터셋 이미지·라벨은 라이선스 때문에 계속 Git에서 제외합니다.
> 결과·한계·테스트 절차: [SAFETY7_MODEL.md](SAFETY7_MODEL.md). 복구 검증 도구: `recover_safety7.py`.
> 아래 2026-11-05 기록 중 "safety7.pt 없음/학습 미수행" 항목은 이 업데이트로 대체됩니다.

저장소: `C:\Users\dnflw\Documents\HazardScout-AI`. 작업 시작/최종 브랜치는 **YOLO-Seeun**입니다.
사용자 요청에 따라 YOLO-Seeun에 로컬 커밋합니다. Push/PR/Merge/Rebase/브랜치 전환은 수행하지 않습니다.
최종 Git 변경은 yolo_lab 내부뿐이며 code/ 서버·GUI·모터·OLED는 읽기 전용으로 참고했습니다.

## 1. 유지·변경·백업

- COCO YOLOv8n은 models/yolov8n.pt 한 개를 활성 유지하고 실제80클래스 로딩/CPU 추론을 확인했습니다.
- prepare_models.py에서 YOLO-World/YOLOE를 제거했습니다. train/test/Pi camera는 간단한 baseline/custom 선택 구조로 정리했습니다.
- 데이터 준비·검증, 공통 추론 루프, 최소 Pi 패키지, 테스트와 README/출처/서버 참고 문서를 추가했습니다.
- 기존 데이터·모델·중복 가중치·과거 runs/results·3종 병합 코드·이전 안내/YAML은 backups/20261104_before_rebuild에 보존했습니다.
- 기존 원본13,065개, 총668,020,832bytes(약637MiB)의 SHA-256을 최종에 모두 재확인했습니다.
- 3종이0장이었던 초기 출력, 중간 검토 출력과 제외 사진은 backups/20261104_before_missing_classes 및 backups/20261105_dataset_candidate_review에 보존했습니다.
- 클래스 비중 조정에서 기존 Hard-Hat15장과 RF10011장, 잘못된 품목/과다한 단독 장갑 후보는 삭제하지 않고 내부 백업으로 옮겼습니다.
- 최상위 .gitignore/README/code/runs/weights는 수정하지 않았습니다. yolo_lab/.gitignore에서만 백업·이미지·라벨·가중치·캐시를 제외했습니다.
- 영구 대량 삭제는 수행하지 않았습니다. 백업을 유지하므로 전체 로컬 디스크 사용량은 그만큼 줄지 않습니다.

## 2. 최종 구조

```text
yolo_lab/
  README.md / REPORT.md / SERVER_REFERENCE.md
  prepare_models.py / prepare_dataset.py / train_yolo.py
  test_yolo.py / test_pi_camera.py / package_pi.py
  lab_utils.py / inference.py / requirements-laptop.txt
  tests/test_workflow.py
  dataset/
    SOURCES.md / sources.json / manifest.csv / manifest.example.csv
    reference_manifest.json
    raw/curated/            (선정 공개 이미지·원본/재매핑 라벨·출처, Git 제외)
    raw/local_reference/    (실물8장 + 수동 bbox, Git 제외)
    references/             (제공 원본 참고 사진8장, 변경 없음, Git 제외)
    safety7/
      data.yaml / build_report.json
      images/{train,val,test}/ / labels/{train,val,test}/ (Git 제외)
  models/yolov8n.pt          (준비·추론 검증, Git 제외)
  models/safety7.pt          (아직 없음; 학교 학습 후 생성)
  deploy/baseline_20261104_232436_672336/ (최소 Pi 전달 폴더)
  results/ / runs/ / .config/ / backups/ (Git 제외)
```

## 3. 7종 데이터 수량과 출처

중복을 제외한 서로 다른 이미지 **213장**입니다(train145/val40/test28).
각 숫자는 해당 장비를 하나 이상 포함한 이미지 수이며 여러 장비가 함께 등장합니다.

| ID | 클래스 | train/val/test | 합계 | 출처별 양성 이미지 |
|---:|---|---|---:|---|
| 0 | traffic_cone | 29/6/6 | 41 | project-k84yn40 + 실물1 |
| 1 | fire_extinguisher | 30/6/6 | 42 | CV projects40 + 실물2 |
| 2 | safety_helmet | 37/8/4 | 49 | RF100 19 + SH17 29 + 실물1 |
| 3 | safety_vest | 20/6/4 | 30 | RF100 20 + SH17 9 + 실물1 |
| 4 | safety_gloves | 37/6/7 | 50 | SH17 49 + 실물1 |
| 5 | earmuffs | 32/6/4 | 42 | SH17 41 + 실물1 |
| 6 | safety_goggles | 28/13/5 | 46 | SH17 45 + 실물1 |

서로 다른 채택 사진은 cone40 + fire40 + RF100 20 + SH17 105 + 실물8 =213입니다.
0장 클래스가 없고 각 클래스 총30장 이상·모든 split 양성 검사를 통과했습니다.

출처: [cone](https://universe.roboflow.com/project-k84yn/traffic-cone-ry08d/dataset/3),
[fire](https://universe.roboflow.com/cv-projects-ab4ye/fire-extinguisher-8kk5k/dataset/1),
[RF100](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1),
[SH17 원저자](https://github.com/ahmadmughees/SH17dataset).
앞의3개는 CC BY4.0, SH17은 원저자 CC BY-NC-SA4.0 및 Pexels 조건을 유지한 학교 비상업 학습용입니다.
사용자 사진의 외부 재배포 권한은 가정하지 않았습니다.

SH17 검토 후보419개의 원저자 YOLO 라벨을 공식 Kaggle에서 받아 미러 COCO 숫자 재정렬/좌표를 검증했습니다.
최종105장 이미지는 원저자가 제공한 Pexels URL에서 직접 받았고 crop 없이640×640으로 변환했습니다.
원본 라벨 좌표를 그대로 재매핑하고 의료용 장갑/음악 헤드셋/일반 안경/잘못된 품목은 제외했습니다.
전체 장면을 재검토해 함께 보이는 다른 실제 대상 장비의 누락 라벨도 원본 bbox로 보완했습니다.
실물8장만 별도 수동 시각 bbox를 만들고 overlay로 장비 끝·호스·귀덮개 밴드가 포함되는지 확인했습니다.
모델 예측 pseudo-label/합성 이미지/임의 bbox를 사용해 수량을 늘리지 않았습니다.

파일별 출처/라이선스/변경과 실물 촬영 방법은 dataset/SOURCES.md 및 sources.json,
SH17 개별 원본 URL·SHA·좌표는 raw/curated/sh17/PROVENANCE.json에 기록했습니다.

## 4. 모델 학습·추론 방식

- baseline: YOLOv8n의 COCO80. custom: 별도로 YOLOv8n을7종 fine-tuning한 safety7.pt.
- 한 번에 한 모델만 로드하며 모드 전환은 종료 후 재실행합니다.
- 7종만 학습한 모델이 COCO80을 유지한다고 가정하지 않습니다.
- 단일87클래스 공동 학습은 공동 데이터/평가 비용 때문에 구현하지 않았습니다.
- 두 모델 동시 추론은 Pi 실측 전이므로 미구현·미측정입니다.
- ONNX/NCNN/양자화는 호환성·정확도 비교 전이라 기본 적용하지 않았습니다.
- safety7.pt는 아직 없습니다. 현재 PC에서 데이터 검사는 통과하고 CUDA 부재 때문에 학습을 중지합니다.

## 5. 실행 명령

Windows에서 yolo_lab로 이동하고 학교 CUDA PyTorch 환경을 준비한 후:

```powershell
python prepare_models.py
python prepare_dataset.py --check
python train_yolo.py --epochs 50 --batch 8 --device 0
# 학습/모델 선택 후:
python test_yolo.py --mode custom --evaluate val --imgsz 320
python test_yolo.py --mode custom --evaluate test --imgsz 320
python test_yolo.py --mode custom --source C:/test/robot_scene.jpg --no-show
python package_pi.py --mode baseline
# 커스텀 학습 완료 시에만:
python package_pi.py --mode custom
python -B -m unittest discover -s tests -v
```

학습 가중치는 runs/safety7_<시간>/weights/best.pt 및 models/safety7.pt입니다.
기존 출력/모델은 덮어쓰지 않으며 내부 백업 후 새 출력으로 작업합니다.
현재 baseline 패키지는8개 파일/6,565,452bytes(약6.3MB)로, 모델 한 개와 추론 코드·설정만 포함합니다.
원격 접속/전송은 수행하지 않았습니다. PuTTY에서:

```bash
source ~/picar-yolo/bin/activate
cd ~/HazardScout-AI/yolo_lab/runtime_baseline
python -c "from lab_utils import load_model; print(load_model('baseline').names)"
python test_pi_camera.py --mode baseline --no-show --max-frames 120
```

## 6. 검증 완료·미검증

완료:
- 최종213장/7종의 라벨·이미지·중복·세션 split 검사, 모든 클래스 총30장 이상/모든 split 양성 통과.
- 기능 테스트8개 및 문법 검사 통과. 부족/잘못된 라벨, 세션 누출, 중복 test 유입, 누락 모델을 차단합니다.
- 원본13,065개 백업 SHA-256과 외부 추적 파일/현재 code 폴더 전체 SHA 불변 확인.
- baseline COCO80 로딩·실제 사진 추론·CSV/이미지/summary 생성,12프레임에서4회 추론과 자원 반환 확인.
- Pi 전달 폴더만으로 baseline 실제 사진 CPU 추론 확인.
- 기존 환경은 Python3.12.15/torch2.14.1+cpu/Ultralytics8.4.175이며 변경하지 않았습니다.
- 공식 Ultralytics8.4.174를 .config 안에 별도로 분리해 실제 사진 baseline CPU 추론도 확인했습니다.
- 최종 데이터 검사는 통과하며 train_yolo.py는 현 PC의 CUDA 부재를 명확히 알리고 exit2로 중지합니다.

미검증:
학교 RTX3080 CUDA 학습·학습된7종 모델 정확도·Pi의 Python3.11.2/실제 카메라·지연·RSS·발열·장시간 구동,
GUI 통신/모델 교체 실연동, 두 모델 동시 추론, ONNX/양자화입니다.
현재 CPU 시험 속도는 Windows 사진 표본이며 Pi FPS나 실제 로봇 성능으로 해석할 수 없습니다.

## 7. Pi·기존 서버 참고와 주의

기본값은 CPU/imgsz320/conf0.35/every3, 카메라640×480, CPU 스레드2입니다.
워밍업1회를 제외한 추론 평균/p50/p95, 실제 프레임/탐지 횟수, Linux peak RSS·온도를 기록합니다.
실물 학습 예시는 클래스당1~2장뿐이며 실물의 독립 val/test는 없습니다.
특히 공개 test의 귀덮개4/안전모4/조끼4는 적으므로 실제 장비 촬영을 더 확보해야 합니다.

code/의 README, 서버2개, YOLO_detector.py, robot_gui.py, fullscreen_gui.py, OLED_eyes.py를
**읽기 전용으로 검토했고 실행/수정하지 않았습니다.**
결과는 SERVER_REFERENCE.md에 정리했습니다:
- Pi에서 상자를 그린 JPEG를 PC(ZeroMQ)/모바일(MJPEG)로 전송하며 GUI 자체에서 YOLO를 실행하지 않습니다.
- 현재 detector는 한 모델만 로드합니다. GUI에 별도 클래스 목록/모드 선택이 자동 추가되지 않습니다.
- 담당자가 서버의 YOLO_MODEL_PATH를 검증한 가중치 절대 경로로 지정하고 재시작하는 방법만 안내했습니다.
- 서버는 모델 로딩 실패 시 카메라만 실행하므로 영상 표시만으로 모델 성공을 판단하면 안 됩니다.
- 서버의 RGB888 이후 추가 RGB2BGR 변환은 독립 시험의 BGR 처리와 다를 수 있습니다. 실제 빨강/노랑 장비로 담당자 확인이 필요합니다.
- 기존 서버는 추론하지 않은 프레임에 이전 박스를 재사용합니다. 이 정책을 변경하지 않았습니다.
- 실제 카메라 시험 전 기존 서버를 정상 종료하고 SSH에서는 --no-show를 사용하세요.

초기 검증에서 라이브러리가 최상위에 자동 생성했던 단일 Ultralytics 설정 폴더는 내부 .config로 회수했습니다.
설정 부모 경로를 먼저 만들어 재발을 막았고 최종 외부 폴더/파일 변경은 없습니다.

## 8. 최종 Git 변경 목록

브랜치 YOLO-Seeun. 재구성 중 HEAD를 유지했고 사용자 승인 후 로컬 커밋합니다. 커밋 직전 git diff --check 통과.
git diff HEAD -- . ':(exclude)yolo_lab/**' 출력 없음.
git status --short -uall의 모든 경로가 yolo_lab/로 시작하는지 확인했습니다.

```text
 M yolo_lab/.gitignore
 D yolo_lab/README_START_HERE.md
 D yolo_lab/dataset/data.yaml
 D yolo_lab/merge_hazard_datasets.py
 M yolo_lab/prepare_models.py
 M yolo_lab/requirements-laptop.txt
 M yolo_lab/test_pi_camera.py
 M yolo_lab/test_yolo.py
 M yolo_lab/train_yolo.py
?? yolo_lab/ACTIVITY_LOG.md
?? yolo_lab/README.md
?? yolo_lab/REPORT.md
?? yolo_lab/SERVER_REFERENCE.md
?? yolo_lab/SERVER_RUNBOOK.md
?? yolo_lab/dataset/SOURCES.md
?? yolo_lab/dataset/manifest.csv
?? yolo_lab/dataset/manifest.example.csv
?? yolo_lab/dataset/reference_manifest.json
?? yolo_lab/dataset/safety7/build_report.json
?? yolo_lab/dataset/safety7/data.yaml
?? yolo_lab/dataset/sources.json
?? yolo_lab/inference.py
?? yolo_lab/lab_utils.py
?? yolo_lab/package_pi.py
?? yolo_lab/prepare_dataset.py
?? yolo_lab/tests/test_workflow.py
```

D 표시는 활성 위치에서 내부 백업으로 옮긴 파일입니다. 복구 불가능한 대량 삭제가 아닙니다.
이미지/라벨/가중치/백업/캐시는 ignore 상태입니다. 사용자 승인으로 yolo_lab 변경만 스테이징·로컬 커밋하며 Push하지 않습니다.

활동기록용 상세 문서는 ACTIVITY_LOG.md, 실기 배포의 final/base 실행 파일을 반영한 명령서는 SERVER_RUNBOOK.md입니다. 첨부 내부 TXT 원문과 접속 IP/Wi-Fi/인증정보는 커밋하지 않았습니다.
