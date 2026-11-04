"""PC와 Pi에서 같은 추론 루프를 사용합니다. CSV/속도 결과는 results/에 저장."""
import csv
import json
import platform
import statistics
import time
from datetime import datetime

from lab_utils import ROOT, load_model, model_path


def run_frames(frames, args):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    # Pi에서 CPU 스레드 경쟁을 제한합니다. PC 추론도 재현 가능한 CPU 기준입니다.
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = load_model(args.mode, args.model)
    folder = ROOT / "results" / (args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True)
    timings = []; count = 0; last_ms = 0.0; start_loop = None
    started = time.perf_counter(); stopped = "end_of_source"
    try:
        with (folder / "detections.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2", "inference_ms"])
            for frame in frames:
                if count == 0:
                    # 동형 입력으로 워밍업 1회; 통계에서 제외합니다.
                    model.predict(np.zeros_like(frame), imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)
                    start_loop = time.perf_counter()
                count += 1
                annotated = frame.copy()
                detected = (count - 1) % args.every == 0
                if detected:
                    before = time.perf_counter()
                    result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
                    last_ms = (time.perf_counter() - before) * 1000
                    timings.append(last_ms)
                    for box in result.boxes:
                        writer.writerow([count, result.names[int(box.cls.item())], round(float(box.conf.item()), 4),
                                         *[round(v, 1) for v in box.xyxy[0].tolist()], round(last_ms, 2)])
                    if len(result.boxes) == 0:
                        writer.writerow([count, "", "", "", "", "", "", round(last_ms, 2)])
                    if not args.no_show or count == 1:
                        annotated = result.plot()
                    if count == 1:
                        path = folder / "first_frame.jpg"
                        if not cv2.imwrite(str(path), annotated):
                            raise RuntimeError(f"이미지 저장 실패: {path}")
                # 건너뛴 프레임에는 오래된 박스를 재사용하지 않습니다.
                if not args.no_show:
                    text = f"{args.mode} | {'infer' if detected else 'skipped'} | last {last_ms:.0f} ms"
                    cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.imshow("HazardScout YOLO - q:quit", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped = "q"; break
                if args.max_frames and count >= args.max_frames:
                    stopped = "max_frames"; break
    except KeyboardInterrupt:
        stopped = "keyboard_interrupt"
    except Exception:
        stopped = "error"
        raise
    finally:
        frames.close()
        if not args.no_show:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - (start_loop or started)
        ordered = sorted(timings)
        summary = {"mode": args.mode, "model": str(model_path(args.mode,args.model)), "platform": platform.platform(),
                   "python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                   "device": "cpu", "torch_threads": 2, "imgsz": args.imgsz, "conf": args.conf, "every": args.every,
                   "frames": count, "inferences": len(timings), "elapsed_s": round(elapsed,3),
                   "processed_frames_per_s": round(count/elapsed,2) if elapsed else None,
                   "detections_per_s": round(len(timings)/elapsed,2) if elapsed else None,
                   "mean_inference_ms": round(statistics.mean(timings),2) if timings else None,
                   "p50_inference_ms": round(statistics.median(timings),2) if timings else None,
                   "p95_inference_ms": round(ordered[max(0, int(len(ordered)*0.95+0.999)-1)],2) if ordered else None,
                   "warmup_excluded": True, "stop_reason": stopped}
        if platform.system() == "Linux":
            import resource
            summary["peak_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2)
            temp = __import__('pathlib').Path('/sys/class/thermal/thermal_zone0/temp')
            if temp.exists():
                summary["final_temperature_c"] = int(temp.read_text())/1000
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"결과: {folder}")
    if count == 0:
        raise RuntimeError("읽은 프레임이 없습니다. 카메라/영상 경로를 확인하세요.")
