# 7종 데이터 출처·라벨 검토 — 2026-11-05

현재 데이터셋은 중복을 제외한 **213장**, train145 / val40 / test28입니다.
실제 구매 장비 사진 8장은 직접 시각 검토한 bbox로 train에 포함했습니다.
0장 클래스는 없으며 모든 클래스가 train/val/test에 각각 존재합니다.

| ID | 클래스 | train/val/test | 총 양성 이미지 | 현재 출처 |
|---:|---|---|---:|---|
| 0 | traffic_cone | 29/6/6 | 41 | project-k84yn 40 + 실제 사진1 |
| 1 | fire_extinguisher | 30/6/6 | 42 | CV projects 40 + 실제 사진2 |
| 2 | safety_helmet | 37/8/4 | 49 | RF100 19 + SH17 29 + 실제 사진1 |
| 3 | safety_vest | 20/6/4 | 30 | RF100 20 + SH17 9 + 실제 사진1 |
| 4 | safety_gloves | 37/6/7 | 50 | SH17 49 + 실제 사진1 |
| 5 | earmuffs | 32/6/4 | 42 | SH17 41 + 실제 사진1 |
| 6 | safety_goggles | 28/13/5 | 46 | SH17 45 + 실제 사진1 |

한 사진에 여러 장비가 있으므로 양성 이미지 수의 합은 213보다 큽니다.

## 채택 출처와 사용 조건

| 출처 | 서로 다른 이미지 수 | 라이선스·변경 |
|---|---:|---|
| [traffic cone / project-k84yn](https://universe.roboflow.com/project-k84yn/traffic-cone-ry08d/dataset/3) | 40 | CC BY 4.0; 기존 3종 병합 bbox ID 2→0. 과거 polygon→bbox 변환 이력 보존 |
| [fire extinguisher / CV projects](https://universe.roboflow.com/cv-projects-ab4ye/fire-extinguisher-8kk5k/dataset/1) | 40 | CC BY 4.0; 기존 bbox ID 0→1 |
| [construction safety / Roboflow 100](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/1) | 20 | CC BY 4.0; 원본 helmet0→2, vest4→3; person/no-* 제외. 원출처 Roboflow100/Anonymous |
| [SH17 / Hafiz Mughees Ahmad, Afshin Rahimi](https://github.com/ahmadmughees/SH17dataset) | 105 | 원저자 CC BY-NC-SA 4.0 및 원본 Pexels 조건. **학교 비상업 연구/학습용**으로 구성; 원본 좌표 검증·실제 장비 유형만 선별 |
| 사용자 실제 장비 사진 | 8 | 사용자 제공 로컬 학습용. 외부 재배포 허락을 가정하지 않음. EXIF 방향 정규화한 별도 복사본에 보이는 객체 bbox를 직접 지정 |

- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- 원본·학습 이미지·라벨은 Git에서 제외합니다. 원본 사진을 임의로 공개 저장소에 재배포하지 않습니다.
- SH17 저자 문서는 Pexels 원본 이용 조건도 명시합니다. [원저자 라이선스와 조건](https://github.com/ahmadmughees/SH17dataset#license)을 함께 보존했습니다.
- 원본 소화기와 콘은 산업용/제품 사진이 포함되어 실제 노란 호스 모형·미니콘과 외형 차이가 있습니다.
- safety_goggles는 보호용 고글·보호안경만 사용했습니다. 일반 시력교정 안경·선글라스는 제외했습니다. 구매한 두꺼운 노란 프레임과 다른 투명 프레임도 보조 예시로 포함됩니다.
- earmuffs는 산업용 귀덮개만 사용합니다. 통화용/음악용 헤드셋, 이어버드/귀마개를 이 클래스로 넣지 않았습니다.
- safety_gloves는 작업·보호 장갑입니다. 의료용 일회용 장갑, 복싱 장갑 등은 제외했습니다. 회색 직물·노란/주황색/검정 보호장갑 등 외형 차이가 남습니다.

## 원저자 라벨과 이미지 검증

1. Roboflow/공개 미러의 Bounding Box 메타데이터로 작은 후보 집합만 검토했습니다.
   SH17 전체 8,099장 이미지를 내려받지 않았습니다.
2. [SH17 미러](https://huggingface.co/datasets/fathansanum/SH-17-Dataset)의 revision
   `c979be777eaa1c6b5338410788b65926a033e3a0`은 검토용이었습니다.
   미러 README의 CC BY 4.0 표기로 원저자의 NC-SA 조건을 완화하지 않았습니다.
3. 미러 COCO 숫자 클래스가 원본과 다르게 정렬된 것을 발견했습니다.
   원저자 [sh17.yaml](https://raw.githubusercontent.com/ahmadmughees/SH17dataset/master/sh17.yaml)과
   [공식 Kaggle](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)의
   **후보 419개 원본 YOLO 라벨**을 직접 받아 좌표·클래스를 모두 대조했습니다.
   숫자 문자열 순서로 추정한 매핑만으로 최종 라벨을 만들지 않았습니다.
4. 최종 bbox는 원저자 YOLO의 정규화 좌표를 그대로 사용합니다.
   원본 ID2→earmuffs5, ID9→safety_gloves4, ID8→safety_goggles6은 실제 품목이 맞는 경우만 적용합니다.
   helmet10→2, safety-vest16→3도 함께 보이는 대상에 적용했습니다.
5. 원저자가 제공한 Pexels URL에서 최종 105장 모두 직접 받았습니다.
   crop 없이 전체 구도를 640×640으로 stretch했고 기존 검토 이미지와 레이아웃 dHash 거리를 확인했습니다(105/105 통과).
   최종 이미지·원본 라벨 SHA-256과 개별 원본 URL은 `raw/curated/sh17/PROVENANCE.json`, `ORIGINAL_IMAGE_URLS.json`에 있습니다.
6. 객체 crop뿐 아니라 전체 장면도 확인해 일반 안경·잘못된 품목을 추가로 제외했습니다.
   같은 이미지에 함께 보이는 다른 실제 대상 장비의 누락 라벨은 **기존 원본 bbox 좌표로만** 보완했습니다.
   bbox가 없는 곳에 임의 박스를 만들거나 모델 예측으로 pseudo-label을 생성하지 않았습니다.
7. 실물 8장은 `raw/local_reference/ANNOTATIONS.json`에 좌표·원본·수동 검토 방법을 기록했습니다.
   bbox overlay를 보고 소화기 호스·장갑 끝·귀덮개 밴드 등이 잘리지 않도록 보완했습니다.
   참고 원본 `references/`와 Downloads 원본은 변경하지 않았습니다.

원본 RF100은 [LibreYOLO 미러](https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb)의
revision `342e545489a6b5f76d6c8225f1ef2629c5a4770a`에서 필요한 파일만 받았습니다.
원본 cone 버전3 이력은 이전 SOURCES를 보존했으며 현재 공개 페이지에는 최신 버전 정보만 확인되었습니다.
기존 Hard-Hat/tunyong 데이터는 전체를 백업했습니다. 현재 활성 manifest에는 사용하지 않습니다.

## 중복·촬영 세션과 한계

- SHA-256, 64-bit dHash 거리 <=5, 동일 Roboflow 원본 이름으로 중복을 검사합니다.
- 전체 장면에서 확인한 같은 연구실/작업장/제품 촬영 세션은 하나의 group/split으로 묶었습니다.
- 실제 구매 사진은 같은 배경·촬영 세션이므로 8장 모두 train에 두었습니다.
- `manifest.csv`의 group이 다른 split에 걸치면 빌더가 중지합니다.
- 결과와 제외 이력은 `safety7/build_report.json`, `results/review_missing/`에 있습니다.
- 3종이 0장이었던 초기 출력, 중간 검토 출력, 후보 전체와 제외 이미지는 `backups/`에 보존했습니다.
- 해시와 시각 검토로 모든 유사 촬영 장면을 완벽하게 찾았다고 보장할 수 없습니다.
- 모든 split에 양성은 있지만 test의 귀덮개4/안전모4/조끼4는 매우 작습니다.
  공개 split의 mAP가 실제 모형 또는 로봇 성능을 증명하지는 않습니다.

## 실제 장비 추가 촬영

현재 실제 장비는 클래스당1장(소화기2장)만 학습에 포함되어 있습니다.
모형/노란 프레임/회색·노란 직물 같은 실물 차이를 배우고 평가하려면 각 장비를 추가 촬영하세요.

- 초기 추가 촬영 예: 각40장, train28/val6/test6. 촬영 세션·배경을 분리하세요.
- 로봇 카메라 높이에서 정면/측면/비스듬한 방향, 실제 사용할 거리 0.3/0.6/1.0m, 조명2~3종을 섞습니다.
- 손에서 내려놓기/착용/바닥·책상·의자 배경, 일부 가림, 장비 없는 배경과 노란 유사 물체도 확보합니다.
- 장갑: 한 짝/두 짝, 손등/손바닥. 귀덮개: 펼침/접힘. 고글: 렌즈 반사/스트랩.
- 소화기는 노란 호스 모형의 후면과 원거리, 콘은 미니콘의 실제 관측 크기를 포함합니다.
- 연속 영상의 인접 프레임이나 증강본을 서로 다른 split에 넣지 않습니다.
- 사진 속 모든 대상 장비에 실제 bbox를 검토해 붙이고 손·배경을 장비로 라벨링하지 않습니다.
- 신규 사진/라벨은 `raw/local/`에 두고 `manifest.example.csv`를 참고해 manifest에 추가합니다.
