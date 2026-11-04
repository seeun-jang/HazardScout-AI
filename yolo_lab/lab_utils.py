"""공통 경로, 클래스, 모델 로딩. 팀 서버와 독립적으로 동작합니다."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Ultralytics/Matplotlib 캐시도 이 폴더 안에 둡니다.
for key, folder in {"YOLO_CONFIG_DIR": ".config/ultralytics", "MPLCONFIGDIR": ".config/matplotlib",
                    "TORCH_HOME": ".config/torch", "XDG_CACHE_HOME": ".config/cache"}.items():
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(ROOT / folder)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
NAMES = ["traffic_cone", "fire_extinguisher", "safety_helmet", "safety_vest",
         "safety_gloves", "earmuffs", "safety_goggles"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def local_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def output_path(value):
    path = local_path(value)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("출력은 yolo_lab 내부의 하위 경로여야 합니다.")
    return path


def model_path(mode, override=None):
    path = local_path(override or f"models/{'yolov8n' if mode == 'baseline' else 'safety7'}.pt")
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"모델 없음: {path}. baseline은 prepare_models.py, custom은 학습이 필요합니다.")
    return path


def load_model(mode, override=None):
    from ultralytics import YOLO
    model = YOLO(str(model_path(mode, override)))
    if model.task != "detect":
        raise ValueError("Bounding Box detection 모델만 사용할 수 있습니다.")
    names = [model.names[i] for i in range(len(model.names))]
    if mode == "custom" and names != NAMES:
        raise ValueError(f"custom 클래스 순서 불일치: {names}; 기대값: {NAMES}")
    if mode == "baseline" and (len(names) != 80 or names[0] != "person"):
        raise ValueError("baseline에는 COCO 80클래스 yolov8n.pt를 사용하세요.")
    return model


def inference_args(parser):
    parser.add_argument("--mode", choices=["baseline", "custom"], default="baseline")
    parser.add_argument("--model", help="신뢰할 수 있는 로컬 .pt 파일; 상대 경로는 yolo_lab 기준")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=120, help="0=무제한")
    parser.add_argument("--no-show", action="store_true", help="SSH/PuTTY에서는 필수")


def validate_inference_args(parser, args):
    if not 0 < args.conf <= 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("conf는 0~1, imgsz는 32 이상의 32 배수여야 합니다.")
    if args.every < 1 or args.max_frames < 0:
        parser.error("every >= 1, max-frames >= 0을 사용하세요.")
