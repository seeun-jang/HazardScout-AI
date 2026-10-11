"""Independent Raspberry Pi Picamera2 detection test (NO server/motor/GPIO imports).
Run on Raspberry Pi only after teammates stop their camera-using server.
  python3 test_pi_camera.py --mode baseline --max-frames 120
  python3 test_pi_camera.py --mode world --max-frames 120
  python3 test_pi_camera.py --mode custom --max-frames 120
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

ROOT = Path(__file__).resolve().parent
MODELS = {"baseline": "yolov8n.pt", "world": "world_hazard.pt", "custom": "hazard_v1.pt"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODELS, default="baseline")
    p.add_argument("--model", help="Explicit .pt file relative to yolo_lab/")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--no-show", action="store_true", help="For Raspberry Pi without desktop")
    args = p.parse_args()
    if args.every < 1:
        p.error("--every must be >= 1")
    model_path = ROOT / args.model if args.model else ROOT / "models" / MODELS[args.mode]
    if not model_path.exists():
        p.error(f"Missing model: {model_path}. Copy it from the laptop first.")

    model = YOLO(str(model_path))
    folder = ROOT / "results" / f"pi_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True, exist_ok=True)
    camera = None
    total = 0
    n_inferences = 0
    total_ms = 0.0
    last_annotated = None
    try:
        camera = Picamera2()
        cfg = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        camera.configure(cfg)
        camera.start()
        time.sleep(1)
        with (folder / "results.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["frame", "object_count", "inference_ms"])
            while args.max_frames <= 0 or total < args.max_frames:
                rgb = camera.capture_array()
                if rgb is None:
                    continue
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                total += 1
                if total == 1 or (total - 1) % args.every == 0:
                    start = time.perf_counter()
                    result = model.predict(bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                    ms = (time.perf_counter() - start) * 1000
                    n_inferences += 1
                    total_ms += ms
                    n = len(result.boxes) if result.boxes is not None else 0
                    out.writerow([total, n, round(ms, 2)])
                    last_annotated = result.plot()
                    if total == 1 or n > 0:
                        cv2.imwrite(str(folder / f"frame_{total:04d}.jpg"), last_annotated)
                    print(f"frame={total} objects={n} inference={ms:.0f}ms")
                if not args.no_show and last_annotated is not None:
                    cv2.imshow("HazardScout Pi YOLO (q=quit)", last_annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        avg = total_ms / n_inferences if n_inferences else 0
        print(f"Done: {total} camera frames, {n_inferences} inferences, average {avg:.1f} ms/inference")
        print(f"Saved evidence: {folder}")
    finally:
        if camera is not None:
            try:
                camera.stop()
                camera.close()
            except Exception as exc:
                print("Camera cleanup warning:", exc)
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
