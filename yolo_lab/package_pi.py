"""학습 데이터 없이 필요한 추론 파일만 deploy/<이름>/에 복사합니다."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from lab_utils import ROOT, model_path, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "custom"], default="baseline")
    args = parser.parse_args()
    model = load_model(args.mode)
    folder = ROOT / "deploy" / (args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    for name in ["lab_utils.py", "inference.py", "test_yolo.py", "test_pi_camera.py"]:
        shutil.copy2(ROOT / name, folder / name)
    (folder / "README.md").write_text(
        f"# HazardScout Pi inference — {args.mode}\n\n"
        "학습 데이터/코드는 포함하지 않습니다. 기존 카메라 사용 서버를 먼저 종료하세요.\n"
        "기존 ~/picar-yolo 환경을 사용합니다. SSH에서는 --no-show를 지정하세요.\n\n"
        "```bash\nsource ~/picar-yolo/bin/activate\n"
        f"python -c \"from lab_utils import load_model; print(load_model('{args.mode}').names)\"\n"
        f"python test_pi_camera.py --mode {args.mode} --max-frames 120 --no-show\n"
        "```\n\n"
        "기본값: CPU, imgsz=320, conf=0.35, 3프레임당 1회. 결과는 results/에 저장합니다.\n"
        "Picamera2 RGB888 배열은 BGR이므로 추가 색상 변환을 하지 않습니다.\n"
        "Ultralytics 목표 버전 8.4.174. Python 3.11/Pi4 실제 검증은 별도로 수행해야 합니다.\n"
        "test_yolo.py --source 파일 --no-show로 이미지/영상 시험도 가능합니다.\n"
        "--evaluate는 학습용 전체 프로젝트에서만 사용하세요.\n",
        encoding="utf-8")
    (folder / "models").mkdir()
    shutil.copy2(model_path(args.mode), folder / "models" / model_path(args.mode).name)
    (folder / "MODEL_INFO.json").write_text(json.dumps({"mode": args.mode, "classes": model.names,
        "ultralytics_target": "8.4.174", "imgsz": 320, "conf": 0.35, "every": 3,
        "note": "No dataset or training code included. --evaluate is available only in full Windows project."},indent=2),encoding="utf-8")
    manifest = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in folder.rglob("*") if p.is_file()}
    (folder / "SHA256.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Pi 전송 폴더: {folder}")


if __name__ == "__main__":
    main()
