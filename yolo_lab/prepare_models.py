"""Download baseline model and make a zero-shot YOLO-World model with saved prompts.
Run from anywhere: python yolo_lab/prepare_models.py
Does not access robot control or team server code.
"""
from pathlib import Path
import argparse
import shutil
from ultralytics import YOLO, YOLOWorld

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
PROMPTS = [
    "person", "chair", "backpack", "bottle", "laptop",
    "fire extinguisher", "safety helmet", "traffic cone",
    "cardboard box", "first aid kit", "smoke detector",
]


def main():
    parser = argparse.ArgumentParser(description="Download models for tomorrow's demo")
    parser.add_argument("--with-yoloe", action="store_true", help="also download YOLOE prompt-free (experimental, heavier)")
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)

    print("[1/2] Standard YOLOv8n model...")
    baseline = MODELS / "yolov8n.pt"
    if not baseline.exists():
        model = YOLO("yolov8n.pt")
        src = Path(model.ckpt_path).resolve()
        if src != baseline.resolve():
            shutil.copy2(src, baseline)
    print(f"OK: {baseline}")

    print("[2/2] YOLO-World v2 (zero-shot) vocabulary setup...")
    world_path = MODELS / "world_hazard.pt"
    if not world_path.exists():
        model = YOLOWorld("yolov8s-worldv2.pt")
        model.set_classes(PROMPTS)
        model.save(str(world_path))
    print(f"OK: {world_path}")
    print("Classes:", ", ".join(PROMPTS))
    print("Note: world_hazard.pt is a pre-trained zero-shot model with saved prompts, NOT a newly fine-tuned model.")

    if args.with_yoloe:
        print("[Optional] YOLOE prompt-free model download...")
        path = MODELS / "yoloe-26n-seg-pf.pt"
        if not path.exists():
            model = YOLO("yoloe-26n-seg-pf.pt")
            src = Path(model.ckpt_path).resolve()
            if src != path.resolve():
                shutil.copy2(src, path)
        print(f"OK: {path}")


if __name__ == "__main__":
    main()
