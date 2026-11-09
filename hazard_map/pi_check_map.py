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

  ~/picar-yolo/bin/python pi_check_map.py --model
      서버가 쓰는 YOLO 모델이 Safety7 7종(순서까지)인지 확인. 카메라 안 씀
      (YOLO-Seeun 브랜치 yolo_lab/lab_utils.py 의 검사와 같은 기준)

  ~/picar-yolo/bin/python pi_check_map.py --camera --yolo
      위 모델 확인 후, Safety7 탐지 결과를 거리로 바꿔서 출력
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

# YOLO-Seeun 브랜치 yolo_lab/lab_utils.py 의 NAMES 와 같은 순서 (클래스 번호 0~6)
SAFETY7_NAMES = ["traffic_cone", "fire_extinguisher", "safety_helmet", "safety_vest",
                 "safety_gloves", "earmuffs", "safety_goggles"]
SAFETY7_SHA256 = "91e388a76c09f24141ea4a522466ef81bc92a071e7898dfdee0789746928d8b9"


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


def check_model_names(cfg, names):
    """
    모델 클래스 목록 검사. 문제 목록을 돌려줌 (비어 있으면 정상)
    names: 모델의 클래스 이름 목록 (번호 순서대로)
    """
    problems = []
    if names != SAFETY7_NAMES:
        if sorted(names) == sorted(SAFETY7_NAMES):
            problems.append(f"클래스 이름은 같지만 순서가 다름: {names}")
        else:
            problems.append(f"Safety7 7종 모델이 아님 ({len(names)}종): {names[:8]}"
                            + (" ..." if len(names) > 8 else ""))

    configured = set(cfg.hazards["classes"])
    missing = [n for n in names if n not in configured]
    unused = [n for n in configured if n not in names]
    if missing:
        problems.append(f"robot_config.json 에 없어서 지도에 안 올라가는 클래스: {missing}")
    if unused:
        problems.append(f"robot_config.json 에만 있고 모델은 못 찾는 클래스: {unused}")
    return problems


def load_yolo(cfg, model_path):
    """모델을 불러와 Safety7 인지 확인. 실패하면 None"""
    print(f"\n[YOLO 모델] {model_path}")
    if not os.path.isfile(model_path):
        print("  파일 없음. --model-path 로 서버의 models/safety7.pt 경로를 지정하세요")
        return None

    import hashlib
    with open(model_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    if sha == SAFETY7_SHA256:
        print("  SHA-256 일치: YOLO-Seeun 11월 7일 safety7.pt (mj 서버와 같은 파일)")
    else:
        print(f"  SHA-256 {sha[:16]}... : 11월 7일 safety7.pt 와 다른 파일 (새로 학습했다면 정상)")

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print("  로드 실패:", repr(e))
        return None

    if getattr(model, "task", "detect") != "detect":
        print(f"  박스 탐지(detect) 모델이 아님: {model.task}")
        return None

    names = [model.names[i] for i in range(len(model.names))]
    problems = check_model_names(cfg, names)
    if problems:
        for p in problems:
            print("  [주의]", p)
    else:
        print("  클래스 7종, 순서, robot_config.json 설정 모두 일치")
    return model


def open_camera(cfg):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": (int(cfg.camera["width"]), int(cfg.camera["height"]))}
    ))
    cam.start()
    time.sleep(1.0)
    return cam


def run_camera(cfg, watcher, seconds, calibrate, yolo):
    try:
        cam = open_camera(cfg)
    except Exception as e:
        print("\n[카메라] 열 수 없음:", repr(e))
        print("  팀 서버가 켜져 있으면 끄고 다시 실행하세요")
        return

    tracker = HazardTracker(cfg)

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
    parser.add_argument("--yolo", action="store_true", help="Safety7 탐지 거리도 확인 (카메라 사용)")
    parser.add_argument("--model", action="store_true", help="YOLO 모델이 Safety7 7종인지만 확인 (카메라 안 씀)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()

    check_versions()
    cfg = check_config()
    watcher = BarcodeWatcher(cfg)
    print("인식 엔진:", watcher.reader.backend)
    check_labels(cfg, watcher)

    yolo = None
    if args.model or args.yolo:
        yolo = load_yolo(cfg, args.model_path)

    if args.camera or args.calibrate or args.yolo:
        run_camera(cfg, watcher, args.seconds, args.calibrate, yolo)


if __name__ == "__main__":
    main()
