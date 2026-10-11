"""Independent YOLO test for laptop webcam / image / video (no GUI/server edits).
Examples:
  python test_yolo.py --mode baseline --source 0
  python test_yolo.py --mode world --source 0
  python test_yolo.py --mode world --source C:/Videos/test.mp4 --save
  python test_yolo.py --mode custom --source 0 --model models/hazard_v1.pt
  python test_yolo.py --mode yoloe --source 0
Press s to save a screenshot, q to quit. Detections CSV is always written.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "baseline": ROOT / "models/yolov8n.pt",
    "world": ROOT / "models/world_hazard.pt",
    "custom": ROOT / "models/hazard_v1.pt",
    "yoloe": ROOT / "models/yoloe-26n-seg-pf.pt",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=DEFAULT_MODELS, default="baseline")
    p.add_argument("--model", help="Optional custom .pt path; relative to yolo_lab/")
    p.add_argument("--source", default="0", help="0 for webcam, otherwise image/video path")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--every", type=int, default=1, help="Infer every N frames, reusing the last result for display")
    p.add_argument("--max-frames", type=int, default=0, help="0 = no frame limit")
    p.add_argument("--no-show", action="store_true", help="No OpenCV window (headless)")
    p.add_argument("--save", action="store_true", help="Save annotated video/image and screenshots")
    args = p.parse_args()

    if not 0.0 < args.conf <= 1.0 or args.every < 1 or args.imgsz < 32:
        p.error("Use 0 < --conf <= 1; --every >= 1; --imgsz >= 32")
    model_path = Path(args.model) if args.model else DEFAULT_MODELS[args.mode]
    if not model_path.is_absolute():
        model_path = ROOT / model_path if args.model else model_path
    if not model_path.exists():
        p.error(f"Model not found: {model_path}. Run 'python prepare_models.py' first (or train).")

    model = YOLO(str(model_path))
    run_dir = ROOT / "results" / f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "detections.csv"
    headers = ["frame", "class", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y", "inference_ms"]

    src_path = Path(args.source)
    is_image = src_path.is_file() and src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_camera = args.source.isdigit()
    source = int(args.source) if is_camera else args.source
    writer = None
    cap = None

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            if is_image:
                frame = cv2.imread(str(src_path))
                if frame is None:
                    raise RuntimeError(f"Cannot open image: {src_path}")
                annotated, count, ms = predict_one(model, frame, args, csv_writer, 1)
                cv2.imwrite(str(run_dir / "image_result.jpg"), annotated)
                print(f"image -> {count} detections, {ms:.1f} ms. Saved {run_dir / 'image_result.jpg'}")
                if not args.no_show:
                    cv2.imshow("HazardScout YOLO (press any key)", annotated)
                    cv2.waitKey(0)
                return

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera/video: {source}")
            count_frame = 0
            last_annotated = None
            last_ms = 0.0
            last_count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                count_frame += 1
                if count_frame == 1 or (count_frame - 1) % args.every == 0:
                    last_annotated, last_count, last_ms = predict_one(model, frame, args, csv_writer, count_frame)
                    annotated = last_annotated.copy()
                else:
                    # Cached boxes are visual only; they may lag behind moving objects.
                    annotated = frame.copy()
                    cv2.putText(annotated, "INFERENCE SKIPPED (cached last result not drawn)", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255), 1)
                overlay = f"{args.mode} | frame {count_frame} | last {last_count} objects | {last_ms:.0f} ms"
                cv2.putText(annotated, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)
                if args.save:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        video_fps = cap.get(cv2.CAP_PROP_FPS)
                        fps = video_fps if 1 <= video_fps <= 120 else 15.0
                        writer = cv2.VideoWriter(str(run_dir / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("Cannot create output.mp4; check codec/path")
                    writer.write(annotated)
                if not args.no_show:
                    cv2.imshow("HazardScout YOLO (q=quit, s=screenshot)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("s"):
                        path = run_dir / f"screenshot_{count_frame:05d}.jpg"
                        cv2.imwrite(str(path), annotated)
                        print(f"Saved: {path}")
                    if key == ord("q"):
                        break
                if args.max_frames > 0 and count_frame >= args.max_frames:
                    break
            print(f"Processed {count_frame} frames. Results: {run_dir}")
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
    print(f"Detections CSV: {csv_path}")


def predict_one(model, frame, args, writer, frame_index):
    t0 = time.perf_counter()
    r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            idx = int(box.cls.item())
            name = r.names.get(idx, str(idx)) if isinstance(r.names, dict) else r.names[idx]
            writer.writerow([frame_index, name, round(float(box.conf.item()), 5),
                             x1, y1, x2, y2, (x1 + x2) // 2, (y1 + y2) // 2, round(ms, 2)])
    return r.plot(), (len(r.boxes) if r.boxes is not None else 0), ms


if __name__ == "__main__":
    main()
