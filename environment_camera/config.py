"""Local laptop surveillance server configuration (separate from Raspberry Pi)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_env_file():
    path = BASE_DIR / '.env'
    if not path.exists():
        return
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()
DATA_DIR = BASE_DIR / 'data'
CAPTURE_DIR = DATA_DIR / 'captures'
DB_PATH = DATA_DIR / 'monitor.db'
REFERENCE_PATH = DATA_DIR / 'reference.jpg'
DATA_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', '8000'))
CAMERA_SOURCE_RAW = os.getenv('CAMERA_SOURCE', '0')
CAMERA_SOURCE = int(CAMERA_SOURCE_RAW) if CAMERA_SOURCE_RAW.isdecimal() else CAMERA_SOURCE_RAW
ADMIN_PIN = os.getenv('ADMIN_PIN', '')
# Explicit opt-in: independent demo records, never presented as real YOLO inference.
DEMO_MODE = os.getenv('DEMO_MODE', '0').strip() == '1'
DEMO_DB_PATH = DATA_DIR / 'demo_monitor.db'
DEMO_SCENE_PATH = DATA_DIR / 'demo_scene.jpg'
# Both optional. The already-trained safety7 model lives in the same repo.
YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolov8n.pt').strip()
CUSTOM_YOLO_MODEL = os.getenv('CUSTOM_YOLO_MODEL', '').strip()
if not CUSTOM_YOLO_MODEL and (BASE_DIR.parent / 'yolo_lab' / 'models' / 'safety7.pt').is_file():
    CUSTOM_YOLO_MODEL = str(BASE_DIR.parent / 'yolo_lab' / 'models' / 'safety7.pt')
# Future overhead-trained robot detector (not part of safety7; never fake its existence).
ROBOT_YOLO_MODEL = os.getenv('ROBOT_YOLO_MODEL', '').strip()
# The class filter reduces false positives from the general-purpose COCO model.
COCO_LABELS = {s.strip() for s in os.getenv('COCO_LABELS', 'person,stop sign').split(',') if s.strip()}
# Optional normalized area-of-interest polygon: "x,y;x,y;x,y" (0..1 coordinates).
MONITOR_ROI = os.getenv('MONITOR_ROI', '').strip()
ENABLE_YOLO = os.getenv('ENABLE_YOLO', '1') == '1'
DETECTION_CONF = float(os.getenv('DETECTION_CONF', '0.35'))
DIFF_THRESHOLD = int(os.getenv('DIFF_THRESHOLD', '44'))
UNKNOWN_MIN_AREA = int(os.getenv('UNKNOWN_MIN_AREA', '950'))
UNKNOWN_STABLE_FRAMES = int(os.getenv('UNKNOWN_STABLE_FRAMES', '9'))
UNKNOWN_COOLDOWN_SEC = int(os.getenv('UNKNOWN_COOLDOWN_SEC', '100'))
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '960'))
YOLO_IMGSZ = int(os.getenv('YOLO_IMGSZ', '960'))
YOLO_EVERY_N_FRAMES = max(1, int(os.getenv('YOLO_EVERY_N_FRAMES', '3')))

# Better protection against slow-moving rover false alerts: boxes must remain
# in approximately the same position instead of merely overlapping slightly.
UNKNOWN_TRACK_IOU = float(os.getenv('UNKNOWN_TRACK_IOU', '0.70'))
