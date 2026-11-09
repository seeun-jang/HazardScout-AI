#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pi_check_map.py

라즈베리파이에서 좌표 매핑을 쓰기 전에 확인/보정하는 스크립트
(카메라를 쓰는 옵션은 팀 서버를 끈 상태에서 실행. 카메라는 한 프로그램만 쓸 수 있음)

  ~/picar-yolo/bin/python pi_check_map.py
      버전, 설정 파일 오류, 라벨 인식 확인

  ~/picar-yolo/bin/python pi_check_map.py --camera
      실제 카메라로 바코드를 읽어서 거리/방향 출력

  ~/picar-yolo/bin/python pi_check_map.py --calibrate 0.50
      카메라 렌즈에서 바코드 라벨까지 정확히 0.50m 에 두고 실행
      -> focal_px 를 계산해서 알려줌 (robot_config.json 에 넣기)

  ~/picar-yolo/bin/python pi_check_map.py --camera --yolo
      Safety7 YOLO 탐지 결과를 거리로 바꿔서 출력
      (줄자로 잰 거리와 비교해서 robot_config.json 의 height_m 을 맞춤)
"""

import argparse
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from map_barcodes import BarcodeWatcher  # noqa: E402
from map_config import Config  # noqa: E402
from map_hazards import HazardTracker  # noqa: E402

DEFAULT_MODEL = "/home/pi/Adeept_PiCar-Pro/Server/models/safety7.pt"


def check_versions():
    print("=" * 50)
    print(" 좌표 매핑 확인")
    print("=" * 50)
    print("Python :", platform.python_version(), "|", sys.executable)
    print("OpenCV :", cv2.__version__)
    print("NumPy  :", np.__version__)
    try:
        import zxingcpp  # noqa: F401
        print("zxing  : 설치됨")
    except ImportError:
        print("zxing  : 없음 -> ~/picar-yolo/bin/pip install zxing-cpp")


def check_config():
    print("\n[설정 파일]")
    cfg = Config()
    print(f"격자 {cfg.grid_w}x{cfg.grid_h}, 칸 {cfg.cell_m}m "
          f"(실제 {cfg.grid_w * cfg.cell_m:.1f}m x {cfg.grid_h * cfg.cell_m:.1f}m), "
          f"바코드 {len(cfg.barcodes)}개")
    print("[지도 경고]가 없으면 정상" if not cfg.warnings else f"경고 {len(cfg.warnings)}개를 고치세요")
    return cfg


def check_labels(cfg, watcher):
    labels = os.path.join(HERE, "labels")
    if not os.path.isdir(labels):
        print("\n[라벨] labels/ 없음, 건너뜀 (노트북에서 make_barcodes.py 로 만든 뒤 같이 복사)")
        return
    print("\n[라벨] 0.5m 거리 크기로 줄여서 인식 + 거리 계산")
    focal = float(cfg.camera["focal_px"])
    for code in cfg.barcodes:
        path = os.path.join(labels, f"{code}.png")
        if not os.path.exists(path):
            print(f"  {code}: 라벨 파일 없음")
            continue
        img = cv2.imread(path)
        # 0.5m 에서 라벨 전체(20cm)가 보이는 폭
        w = int(focal * 0.20 / 0.5)
        h = int(img.shape[0] * w / img.shape[1])
        frame = np.full((480, 640, 3), 110, np.uint8)
        frame[100:100 + h, 320 - w // 2:320 - w // 2 + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        t = time.time()
        found = watcher.reader.read(frame)
        ms = (time.time() - t) * 1000
        hit = [box for text, box in found if text == code]
        if hit:
            d, b = watcher.distance_and_bearing(hit[0], 640, 0.0)
            print(f"  {code}: OK  계산 거리 {d:.2f}m (정답 0.50m)  {ms:.0f}ms")
        else:
            print(f"  {code}: 인식 실패")


def open_camera(cfg):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": (int(cfg.camera["width"]), int(cfg.camera["height"]))}
    ))
    cam.start()
    time.sleep(1.0)
    return cam


def run_camera(cfg, watcher, seconds, calibrate, use_yolo, model_path):
    try:
        cam = open_camera(cfg)
    except Exception as e:
        print("\n[카메라] 열 수 없음:", repr(e))
        print("  팀 서버가 켜져 있으면 끄고 다시 실행하세요")
        return

    yolo = None
    tracker = HazardTracker(cfg)
    if use_yolo:
        try:
            from ultralytics import YOLO
            yolo = YOLO(model_path)
            print(f"[YOLO] {model_path} 로드: {list(yolo.names.values())}")
        except Exception as e:
            print("[YOLO] 로드 실패:", repr(e))

    print(f"\n[카메라] {seconds}초 동안 확인합니다 (Ctrl+C 로 중단)")
    widths = []
    frames = 0
    last_print = 0.0
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    saved = False
    end = time.time() + seconds
    pose = {"x": 0.0, "y": 0.0, "heading": 0.0, "motion": "stop"}

    try:
        while time.time() < end:
            frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
            frames += 1
            found = watcher.reader.read(frame)
            lines = []

            for text, box in found:
                if text not in cfg.barcodes:
                    lines.append(f"  모르는 바코드 '{text}'")
                    continue
                width = box[2] - box[0]
                widths.append(width)
                d, b = watcher.distance_and_bearing(box, frame.shape[1], 0.0)
                bc = cfg.barcode
                state = "회피 실행 거리" if d <= float(bc["trigger_distance_m"]) else "지도에만 표시"
                lines.append(f"  바코드 {text}: 폭 {width}px, 거리 {d:.2f}m, 방향 {b:+.0f}도 -> {state}")

            if yolo is not None and frames % 5 == 0:
                r = yolo.predict(frame, imgsz=224, conf=0.35, verbose=False)[0]
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    name = r.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    p = tracker.project((x1, y1, x2, y2, name, conf), frame.shape, pose, 0.0)
                    if p is None:
                        lines.append(f"  YOLO {name} {conf:.2f}: 거리 계산 안 함 (신뢰도 낮음/화면 끝에 걸림/범위 밖)")
                    else:
                        lines.append(f"  YOLO {name} {conf:.2f}: 높이 {y2 - y1}px -> 거리 {p[2]:.2f}m")

            if lines and time.time() - last_print > 0.7:
                last_print = time.time()
                print("\n".join(lines))
                if not saved:
                    cv2.imwrite(os.path.join(out_dir, "camera_check.jpg"), frame)
                    saved = True
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cam.close()

    print(f"\n프레임 {frames}장")
    if saved:
        print("화면 저장:", os.path.join(out_dir, "camera_check.jpg"))

    if calibrate:
        if len(widths) < 5:
            print("\n[보정] 바코드를 충분히 못 읽었습니다. 거리/조명을 확인하세요")
            return
        width = statistics.median(widths)
        focal = width * calibrate / float(cfg.barcode["bar_width_m"])
        print("\n[보정 결과]")
        print(f"  바코드 막대 폭 중앙값 {width:.0f}px @ {calibrate:.2f}m")
        print(f"  focal_px = {focal:.0f}   (지금 설정값 {cfg.camera['focal_px']})")
        print("  robot_config.json 의 camera.focal_px 를 이 값으로 바꾸세요")


def main():
    parser = argparse.ArgumentParser(description="좌표 매핑 확인/보정")
    parser.add_argument("--camera", action="store_true", help="실제 카메라로 바코드 확인")
    parser.add_argument("--calibrate", type=float, metavar="거리m", help="라벨을 이 거리에 두고 focal_px 측정")
    parser.add_argument("--yolo", action="store_true", help="Safety7 탐지 거리도 확인")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()

    check_versions()
    cfg = check_config()
    watcher = BarcodeWatcher(cfg)
    print("인식 엔진:", watcher.reader.backend)
    check_labels(cfg, watcher)

    if args.camera or args.calibrate or args.yolo:
        run_camera(cfg, watcher, args.seconds, args.calibrate, args.yolo, args.model)


if __name__ == "__main__":
    main()
