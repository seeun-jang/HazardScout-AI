"""Draw a second detector's results in the video stream."""

from pathlib import Path

import cv2

from YOLO_detector import YOLODetector


YOLOV8N_INTERVAL = 16
YOLOV8N_FRAME_OFFSET = 4


def with_yolov8n(primary_worker, plain_worker, primary_model, confidence, imgsz):
    """Keep the primary worker's inference and map hooks intact."""
    model_path = Path(primary_model).with_name("yolov8n.pt")

    class AsyncYOLOWithOverlay(primary_worker):
        def __init__(self, detector, *args, **kwargs):
            super().__init__(detector, *args, **kwargs)
            self.overlay = None
            self.video_frames = 0

        def start(self):
            super().start()
            if not model_path.is_file():
                print(f"[YOLOv8n] 모델 파일 없음: {model_path}")
                return
            try:
                detector = YOLODetector(
                    model_path=str(model_path), confidence=confidence, imgsz=imgsz
                )
                # The unwrapped worker never forwards these results to the map.
                overlay = plain_worker(detector)
                overlay.start()
                self.overlay = overlay
                print(f"[YOLOv8n] 영상 표시 시작 · {YOLOV8N_INTERVAL}프레임 간격")
            except Exception as exc:
                print("[YOLOv8n] 초기화 실패:", repr(exc))

        def draw(self, frame):
            if self.overlay is None:
                return super().draw(frame)

            self.video_frames += 1
            if self.video_frames % YOLOV8N_INTERVAL == YOLOV8N_FRAME_OFFSET:
                self.overlay.submit(frame)
            frame = super().draw(frame)

            with self.overlay.result_lock:
                detections = list(self.overlay.latest_detections)
            color = (255, 160, 0)
            try:
                for x1, y1, x2, y2, name, score in detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    y_label = min(frame.shape[0] - 4, y2 + 16)
                    cv2.putText(
                        frame, f"YOLOv8n {name} {score:.2f}", (x1, y_label),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
                    )
            except cv2.error as exc:
                print("[YOLOv8n] 표시 오류:", repr(exc))
            return frame

        def stop(self):
            try:
                super().stop()
            finally:
                if self.overlay is not None:
                    self.overlay.stop()
                    self.overlay = None

    return AsyncYOLOWithOverlay
