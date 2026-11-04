"""YOLOv8n COCO 기본 모델만 준비합니다. 다른 모델은 다운로드하지 않습니다."""
import os
from lab_utils import ROOT, load_model


def main():
    folder = ROOT / "models"
    folder.mkdir(exist_ok=True)
    if not (folder / "yolov8n.pt").is_file():
        # 명시적으로 실행한 이 준비 단계에서만 공식 가중치 다운로드 허용.
        os.environ["YOLO_OFFLINE"] = "false"
        from ultralytics import YOLO
        previous = os.getcwd()
        try:
            os.chdir(folder)
            YOLO("yolov8n.pt")
        finally:
            os.chdir(previous)
    model = load_model("baseline")
    print(f"준비 완료: {folder / 'yolov8n.pt'} ({len(model.names)} COCO classes)")


if __name__ == "__main__":
    main()
