"""Train a standalone YOLO object detector from a LABELED dataset.
Requires dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml.
Usage: python train_yolo.py --epochs 50 --name hazard_v1
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/data.yaml", help="YAML path relative to yolo_lab/")
    p.add_argument("--model", default="yolov8n.pt", help="Pretrained weights, filename/path")
    p.add_argument("--name", default="hazard_v1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu or 0 for NVIDIA GPU (omit for auto)")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        p.error(f"Missing data.yaml: {data_path}")
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root_dataset = Path(cfg.get("path", data_path.parent))
    if not root_dataset.is_absolute():
        root_dataset = (data_path.parent / root_dataset).resolve()
    for part in ("train", "val"):
        rel = cfg.get(part)
        if not rel or not (root_dataset / rel).exists():
            p.error(f"Missing dataset {part} directory: {root_dataset / str(rel)}")
    if not cfg.get("names"):
        p.error("data.yaml must include class names")
    for part in ("train", "val"):
        paths = list((root_dataset / cfg[part]).rglob("*.jpg")) + list((root_dataset / cfg[part]).rglob("*.png")) + list((root_dataset / cfg[part]).rglob("*.jpeg"))
        if not paths:
            p.error(f"No labeled images in {part}: {root_dataset / cfg[part]}")
    cfg["path"] = str(root_dataset)
    runtime_yaml = ROOT / "results" / "data_runtime.yaml"
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    weights = Path(args.model)
    if not weights.is_absolute() and (ROOT / weights).exists():
        weights = ROOT / weights
    output = ROOT / "models" / f"{args.name}.pt"
    if output.exists():
        p.error(f"Model already exists: {output}; use another --name to keep earlier models")
    model = YOLO(str(weights))
    options = dict(data=str(runtime_yaml), epochs=args.epochs, batch=args.batch,
                   imgsz=args.imgsz, project=str(ROOT / "runs"), name=args.name,
                   plots=True, workers=0)
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)
    run_dir = Path(model.trainer.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Training ended, but best.pt not found: {best_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output)
    print(f"Training completed. Saved: {output}")
    print(f"Full experiment results: {run_dir}")
    print("Note: If you trained only hazard classes, the model is NOT guaranteed to retain COCO 80-class detection.")


if __name__ == "__main__":
    main()
