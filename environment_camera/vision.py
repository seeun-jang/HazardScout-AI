"""Fixed overhead camera: known YOLO classes + scene-change UNKNOWN proposals.

The safety7 model recognizes only safety gear. A moving arm-equipped rover is NOT
covered by that model. Robot rejection is reliable only with a dedicated model
trained and validated from the *actual overhead camera viewpoint*.
UNKNOWN means an unexplained, persistent *change*, never a hazard classification.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from config import (
    BASE_DIR, REFERENCE_PATH, YOLO_MODEL, CUSTOM_YOLO_MODEL, ROBOT_YOLO_MODEL,
    COCO_LABELS, MONITOR_ROI, ENABLE_YOLO, DETECTION_CONF, DIFF_THRESHOLD,
    UNKNOWN_MIN_AREA, UNKNOWN_STABLE_FRAMES, UNKNOWN_TRACK_IOU,
    UNKNOWN_COOLDOWN_SEC, FRAME_WIDTH, YOLO_IMGSZ, YOLO_EVERY_N_FRAMES,
)

SAFETY_CLASSES = frozenset({
    'traffic_cone', 'fire_extinguisher', 'safety_helmet', 'safety_vest',
    'safety_gloves', 'earmuffs', 'safety_goggles',
})
ROBOT_CLASSES = frozenset({'robot', 'robot_arm', 'rover', 'mobile_robot'})


def resize(frame):
    height, width = frame.shape[:2]
    if width == FRAME_WIDTH:
        return frame.copy()
    return cv2.resize(frame, (FRAME_WIDTH, max(1, round(height * FRAME_WIDTH / width))))


def overlap_ratio(a, b):
    """Intersection / area of candidate a (not IoU)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return max(0, x2 - x1) * max(0, y2 - y1) / area


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1, aa + bb - intersection)


def zone_for_box(box, width):
    """Temporary horizontal thirds until one *fixed camera image* defines real zones."""
    cx = (box[0] + box[2]) / 2
    return 'A구역' if cx < width / 3 else 'B구역' if cx < width * 2 / 3 else 'C구역'


def parse_roi(value):
    """Return normalized polygon; invalid configuration must be visible."""
    if not value.strip():
        return None
    points = []
    for pair in value.split(';'):
        values = pair.strip().split(',')
        if len(values) != 2:
            raise ValueError('MONITOR_ROI 형식: x,y;x,y;x,y (각 좌표 0~1)')
        x, y = (float(v.strip()) for v in values)
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError('MONITOR_ROI는 0~1 범위로 설정하세요.')
        points.append([x, y])
    if len(points) < 3:
        raise ValueError('MONITOR_ROI에는 최소 3개의 좌표가 필요합니다.')
    return points


def roi_points(points, shape):
    h, w = shape[:2]
    return np.array([[round(x * (w - 1)), round(y * (h - 1))] for x, y in points], dtype=np.int32)


def resolve_model(path):
    """Prefer repo-relative paths. The standard 'yolov8n.pt' is a valid model name."""
    if not path or path.lower() in ('none', 'off', 'disabled'):
        return None
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if (BASE_DIR / p).is_file():
        return str((BASE_DIR / p).resolve())
    # Ultralytics can download pretrained model names by their official basename.
    if '/' not in path and '\\' not in path:
        return path
    raise FileNotFoundError(f'모델 파일을 찾을 수 없음: {path} (environment_camera 폴더 기준)')


class OverheadDetector:
    def __init__(self):
        self.reference = None
        self.models = []
        self.model_error = ''
        self.frame_no = 0
        self.known_cache = []
        self.tracks = []
        self.last_emitted = []
        self.roi = parse_roi(MONITOR_ROI)
        if REFERENCE_PATH.exists():
            self.reference = cv2.imread(str(REFERENCE_PATH))
        if ENABLE_YOLO:
            try:
                from ultralytics import YOLO
                for role, path in [('coco', YOLO_MODEL), ('safety', CUSTOM_YOLO_MODEL),
                                   ('robot', ROBOT_YOLO_MODEL)]:
                    resolved = resolve_model(path)
                    if resolved:
                        # A supplied custom path must exist; do not silently use wrong weights.
                        if role != 'coco' and not Path(resolved).is_file():
                            raise FileNotFoundError(f'{role} 모델 파일이 없습니다: {resolved}')
                        self.models.append({'role': role, 'model': YOLO(resolved), 'path': resolved})
            except Exception as error:
                self.models = []
                self.model_error = f'YOLO 로딩 실패: {error}'
        else:
            self.model_error = 'YOLO 비활성화: ENABLE_YOLO=0'

    def set_reference(self, frame):
        frame = resize(frame)
        if not cv2.imwrite(str(REFERENCE_PATH), frame):
            raise RuntimeError('기준 화면을 저장하지 못했습니다.')
        self.reference = frame.copy()
        self.tracks.clear()
        self.last_emitted.clear()

    def known_objects(self, frame):
        if not self.models:
            return []
        self.frame_no += 1
        if (self.frame_no - 1) % YOLO_EVERY_N_FRAMES != 0:
            return self.known_cache
        known = []
        for entry in self.models:
            try:
                model, role = entry['model'], entry['role']
                res = model.predict(frame, conf=DETECTION_CONF,
                                    imgsz=YOLO_IMGSZ, verbose=False)[0]
                for obj in res.boxes:
                    label = str(res.names[int(obj.cls[0])])
                    # Only allow intentional classes in each model. The actual
                    # rover is not a standard COCO 'car' from an overhead view.
                    if role == 'coco' and label not in COCO_LABELS:
                        continue
                    if role == 'safety' and label not in SAFETY_CLASSES:
                        continue
                    if role == 'robot' and label not in ROBOT_CLASSES:
                        continue
                    known.append({'box': [int(v) for v in obj.xyxy[0].tolist()],
                                  'label': label, 'conf': round(float(obj.conf[0]), 2),
                                  'role': role})
            except Exception as error:
                self.model_error = f'YOLO 추론 실패({entry["role"]}): {error}'
        self.known_cache = known
        return known

    def scene_candidates(self, frame, known):
        if self.reference is None:
            return []
        base = self.reference
        if base.shape != frame.shape:
            base = cv2.resize(base, (frame.shape[1], frame.shape[0]))
        a = cv2.GaussianBlur(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        b = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
        diff = cv2.absdiff(a, b)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        if self.roi is not None:
            area = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(area, [roi_points(self.roi, frame.shape)], 255)
            mask = cv2.bitwise_and(mask, area)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # Exclude YOLO boxes *before* contour grouping. Otherwise one large
        # changed blob can contain both a recognized helmet and another object.
        for item in known:
            x1, y1, x2, y2 = item['box']
            pad = max(7, round(max(x2 - x1, y2 - y1) * .10))
            cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)),
                          (min(frame.shape[1] - 1, x2 + pad),
                           min(frame.shape[0] - 1, y2 + pad)), 0, thickness=-1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        proposals = []
        max_area = frame.shape[0] * frame.shape[1] * 0.40
        for contour in contours:
            area = cv2.contourArea(contour)
            if not UNKNOWN_MIN_AREA <= area <= max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidate = [x, y, x+w, y+h]
            if any(overlap_ratio(candidate, k['box']) > .35 for k in known):
                continue
            proposals.append(candidate)
        return proposals

    def _track(self, candidates):
        used = set()
        current = []
        for box in candidates:
            match = None
            best = UNKNOWN_TRACK_IOU
            for i, track in enumerate(self.tracks):
                if i in used:
                    continue
                score = iou(box, track['box'])
                if score > best:
                    best, match = score, i
            if match is None:
                current.append({'box': box, 'hits': 1, 'emitted': False})
            else:
                track = self.tracks[match]
                used.add(match)
                current.append({'box': box, 'hits': track['hits'] + 1, 'emitted': track['emitted']})
        self.tracks = current
        return [t for t in current if t['hits'] >= UNKNOWN_STABLE_FRAMES]

    def process(self, raw_frame, patrol_id=None, on_alert=None):
        frame = resize(raw_frame)
        known = self.known_objects(frame)
        candidates = self.scene_candidates(frame, known) if patrol_id else []
        stable = self._track(candidates)
        alerts = []
        now = time.monotonic()
        self.last_emitted = [(box, t) for box, t in self.last_emitted if now-t < UNKNOWN_COOLDOWN_SEC]
        for track in stable:
            box = track['box']
            zone = zone_for_box(box, frame.shape[1])
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(frame, (x1, max(0, y1-27)),
                          (min(frame.shape[1]-1, x1+240), y1), (0, 0, 200), -1)
            cv2.putText(frame, f'UNKNOWN {zone[0]} - ADMIN CHECK',
                        (x1+4, max(16, y1-7)), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (255, 255, 255), 2)
            if track['emitted'] or any(overlap_ratio(box, old) > .30 for old, _ in self.last_emitted):
                continue
            track['emitted'] = True
            self.last_emitted.append((box.copy(), now))
            alerts.append({'box': box.copy(), 'zone': zone})
        for item in known:
            x1, y1, x2, y2 = item['box']
            color = (255, 165, 50) if item.get('role') == 'robot' else (118, 210, 112)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{item['label']} {item['conf']:.2f}",
                        (x1, max(16, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .53, color, 2)
        if self.roi is not None:
            cv2.polylines(frame, [roi_points(self.roi, frame.shape)], True, (255, 200, 65), 2)
        cv2.putText(frame, 'PATROL LIVE' if patrol_id else 'PATROL IDLE',
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (30, 190, 250) if patrol_id else (240, 240, 240), 2)
        if self.reference is None:
            cv2.putText(frame, 'CAPTURE REFERENCE BEFORE PATROL', (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 190, 255), 2)
        if self.model_error:
            cv2.putText(frame, 'YOLO UNAVAILABLE - CHECK SERVER STATUS',
                        (16, frame.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (30, 110, 250), 2)
        if on_alert:
            for alert in alerts:
                on_alert(alert, frame.copy())
        return frame, {'known': len(known), 'unknown_visible': len(stable),
                       'reference_ready': self.reference is not None}
