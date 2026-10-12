"""로봇(라즈베리파이)에 넣을 파일 묶음을 노트북에서 만드는 스크립트 (prepare_robot.py)

노트북에서 한 번 실행하면 yolo_lab/robot_package/ 폴더와 robot_package.zip이 생깁니다.
이 zip 하나만 라즈베리파이로 복사하면 됩니다. 자세한 순서는 ROBOT_DEPLOY.md 참고.

만드는 것:
  models/yolo26n.pt           일반 물체(COCO 80종) 탐지 모델, 5MB
  models/hazard_world_s.pt    위험물 탐지 모델(문구를 미리 넣어둠), 26MB
                              -> 파이에서 CLIP(338MB)을 받을 필요가 없음
  models/yolo26n_ncnn_model/  (--ncnn 옵션) 파이에서 더 빠를 수 있는 NCNN 변환본
  detect_pro.py, pi_check.py, ROBOT_DEPLOY.md, requirements-pi.txt, 확인용 사진 3장

실행 (yolo_lab/ 폴더에서, 노트북에서):
  python prepare_robot.py
  python prepare_robot.py --ncnn      # NCNN 변환본도 같이 (처음엔 변환 도구 설치로 시간 걸림)

hazard_world_s.pt를 만들 때만 CLIP이 필요합니다 (노트북에는 이미 있음).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO, YOLOWorld

from detect_pro import HAZARD_PROFILES, MODELS, PRESETS, ROOT, resolve_weights

PKG = ROOT / "robot_package"
SAMPLE_IMAGES = ["extinguisher_hallway2.jpg", "hardhat_amsterdam.jpg", "cones_car.jpg"]


def build_hazard_model(out_path):
    """작은 YOLO-World 모델에 파이용 위험물 문구를 넣고, CLIP을 뺀 채로 저장."""
    prompts = list(HAZARD_PROFILES["s"]["prompts"])
    m = YOLOWorld(resolve_weights("yolov8s-worldv2.pt"))
    m.set_classes(prompts)
    # set_classes() 하면서 붙은 CLIP 모델(338MB)을 지워야 파일이 26MB로 작아짐
    if hasattr(m.model, "clip_model"):
        del m.model.clip_model
    m.save(str(out_path))
    # 제대로 저장됐는지 확인: 문구가 그대로 들어 있어야 detect_pro가 CLIP 없이 씀
    names = list(YOLO(str(out_path)).names.values())
    assert names == prompts, "저장된 문구가 HAZARD_PROMPTS_SMALL과 다릅니다"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncnn", action="store_true", help="yolo26n NCNN 변환본도 포함 (파이 속도 비교용)")
    args = ap.parse_args()

    pi = PRESETS["pi"]
    if PKG.exists():
        shutil.rmtree(PKG)
    (PKG / "models").mkdir(parents=True)
    (PKG / "test_images").mkdir()

    # 1) 일반 물체 모델 (없으면 자동 다운로드)
    det_model = pi["models"][0]
    YOLO(resolve_weights(det_model))
    shutil.copy2(MODELS / det_model, PKG / "models" / det_model)
    print(f"[1/4] {det_model} 준비 완료")

    # 2) 위험물 모델 (문구를 넣어서 새로 만듦)
    build_hazard_model(MODELS / pi["world"])
    shutil.copy2(MODELS / pi["world"], PKG / "models" / pi["world"])
    print(f"[2/4] {pi['world']} 생성 완료 (CLIP 없이 동작)")

    # 3) (선택) NCNN 변환. 파이 CPU에서 PyTorch보다 빠른 경우가 많음
    if args.ncnn:
        out = YOLO(str(MODELS / det_model)).export(format="ncnn", imgsz=pi["imgsz"])
        shutil.copytree(out, PKG / "models" / Path(out).name)
        print(f"[3/4] NCNN 변환 완료: {Path(out).name}")
    else:
        print("[3/4] NCNN 변환 건너뜀 (--ncnn 으로 포함 가능)")

    # 4) 코드, 문서, 확인용 사진
    for f in ["detect_pro.py", "pi_check.py", "ROBOT_DEPLOY.md"]:
        shutil.copy2(ROOT / f, PKG / f)
    for name in SAMPLE_IMAGES:
        shutil.copy2(ROOT / "test_images" / "hazard" / name, PKG / "test_images" / name)
    (PKG / "requirements-pi.txt").write_text(
        "# 라즈베리파이에 필요한 패키지 (팀 서버 환경에 이미 대부분 설치되어 있음)\n"
        "# yolo26n 모델은 ultralytics 8.4 이상이 필요함\n"
        "ultralytics>=8.4.0\n"
        "opencv-python\n"
        "numpy\n", encoding="utf-8")
    print("[4/4] 코드와 문서 복사 완료")

    zip_path = shutil.make_archive(str(ROOT / "robot_package"), "zip", PKG.parent, PKG.name)
    size = Path(zip_path).stat().st_size / 1e6
    print(f"\n완료: {zip_path} ({size:.0f}MB)")
    print("이 zip을 라즈베리파이로 복사하세요. 순서는 ROBOT_DEPLOY.md 참고.")


if __name__ == "__main__":
    main()
