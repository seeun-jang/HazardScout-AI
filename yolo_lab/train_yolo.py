"""커스텀 7종 YOLOv8n 학습. 기본은 Windows CUDA, --device cpu --allow-cpu로 CPU 학습 허용."""
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
    parser.add_argument("--device", default="0", help="NVIDIA CUDA 장치 번호 또는 cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="학습 이미지 크기(32의 배수)")
    parser.add_argument("--output", default="models/safety7.pt", help="저장할 가중치(yolo_lab/models 내부 .pt, 기존 파일 덮어쓰기 금지)")
    parser.add_argument("--allow-cpu", action="store_true", help="CUDA가 없을 때 CPU 학습을 명시적으로 허용 (느림)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs와 batch는 양수여야 합니다.")
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error("imgsz는 32 이상의 32 배수여야 합니다.")
    try:
        check_dataset(ROOT / "dataset/safety7")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"학습 중지: {exc}\n")
    output = (ROOT / args.output).resolve()
    if output.parent != (ROOT / "models").resolve() or output.suffix != ".pt":
        parser.error("--output은 yolo_lab/models 안의 .pt 파일이어야 합니다.")
    if output.exists():
        parser.error(f"{output.name}이 이미 있습니다. 기존 모델을 먼저 보존하거나 다른 --output을 지정하세요.")
    import torch
    import yaml
    import ultralytics
    if args.device == "cpu":
        if not args.allow_cpu:
            parser.error("CPU 학습은 --allow-cpu를 함께 지정해야 합니다.")
    elif not torch.cuda.is_available():
        parser.error("CUDA가 필요합니다. 학교 RTX 3080 환경에서 PyTorch CUDA 설치를 확인하거나 --device cpu --allow-cpu를 사용하세요.")
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
    model.train(data=str(data_file), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, project=str(ROOT / "runs"), name=run_name,
                workers=0, seed=42, deterministic=True, cache=False, amp=False)
    # amp=False는 AMP 점검에서 다른 모델을 자동 다운로드하는 부작용도 방지합니다.
    best = Path(model.trainer.save_dir) / "weights/best.pt"
    load_model("custom", str(best))
    shutil.copy2(best, output)
    info = "training_info.json" if output.name == "safety7.pt" else f"{output.stem}_training_info.json"
    (ROOT / "models" / info).write_text(json.dumps({
        "python_torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "run": run_name, "weights": output.name, "dataset": "dataset/safety7", "device": args.device, "epochs": args.epochs,
        "batch": args.batch, "imgsz": args.imgsz, "note": "COCO 80 classes are not retained"}, indent=2), encoding="utf-8")
    print(f"학습 완료: {output}. test_yolo.py --mode custom --model models/{output.name} --evaluate test로 별도 평가하세요.")


if __name__ == "__main__":
    main()
