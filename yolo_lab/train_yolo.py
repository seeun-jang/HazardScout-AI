"""Windows CUDA에서만 커스텀 7종 YOLOv8n을 학습합니다."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from lab_utils import ROOT, load_model
from prepare_dataset import check_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="NVIDIA CUDA 장치 번호")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs와 batch는 양수여야 합니다.")
    try:
        check_dataset(ROOT / "dataset/safety7")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"학습 중지: {exc}\n")
    output = ROOT / "models/safety7.pt"
    if output.exists():
        parser.error("models/safety7.pt가 이미 있습니다. 기존 모델을 먼저 backups/로 보존하세요.")
    import torch
    import yaml
    import ultralytics
    if not torch.cuda.is_available() or args.device == "cpu":
        parser.error("CUDA가 필요합니다. 학교 RTX 3080 환경에서 PyTorch CUDA 설치를 확인하세요.")
    from ultralytics import settings
    settings.update({key: False for key in ("sync", "tensorboard", "wandb", "mlflow", "comet", "clearml", "neptune", "hub", "dvc") if key in settings})
    cfg = yaml.safe_load((ROOT / "dataset/safety7/data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(ROOT / "dataset/safety7")
    run_name = "safety7_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runtime = ROOT / "results" / run_name
    runtime.mkdir(parents=True)
    data_file = runtime / "data_runtime.yaml"
    data_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    model = load_model("baseline")
    model.train(data=str(data_file), epochs=args.epochs, batch=args.batch, imgsz=640,
                device=args.device, project=str(ROOT / "runs"), name=run_name,
                workers=0, seed=42, deterministic=True, cache=False, amp=False)
    # amp=False는 AMP 점검에서 다른 모델을 자동 다운로드하는 부작용도 방지합니다.
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    load_model("custom", str(best))
    shutil.copy2(best, output)
    (ROOT / "models/training_info.json").write_text(json.dumps({
        "python_torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "run": run_name, "dataset": "dataset/safety7", "note": "COCO 80 classes are not retained"}, indent=2), encoding="utf-8")
    print(f"학습 완료: {output}. test_yolo.py --evaluate test로 별도 평가하세요.")


if __name__ == "__main__":
    main()
