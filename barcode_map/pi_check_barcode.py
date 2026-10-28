#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pi_check_barcode.py

라즈베리파이에서 바코드 기능이 동작하는지 확인하는 스크립트
- 버전, zxing-cpp 설치, markers.json 오류 확인
- labels/ 가 있으면 라벨 사진으로 인식 확인
- --camera: 실제 카메라로 바코드를 읽어서 거리별 인식 여부 확인
  (카메라는 한 프로그램만 쓸 수 있으므로 팀 서버를 끈 상태에서 실행)

실행 (PuTTY):
  cd ~/barcode_map
  ~/picar-yolo/bin/python pi_check_barcode.py
  ~/picar-yolo/bin/python pi_check_barcode.py --camera
"""

import argparse
import os
import platform
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from barcode_map import BarcodeMap  # noqa: E402


def check_versions():
    print("==========================================")
    print(" 바코드 기능 확인")
    print("==========================================")
    print("Python :", platform.python_version(), "|", sys.executable)
    print("OpenCV :", cv2.__version__)
    print("NumPy  :", np.__version__)

    try:
        import zxingcpp
        print("zxing  :", getattr(zxingcpp, "__version__", "설치됨"))
    except ImportError:
        print("zxing  : 없음 -> ~/picar-yolo/bin/pip install zxing-cpp")


def check_markers():
    print("\n[markers.json]")
    # 확인용이라 지도 상태는 임시 폴더에 저장
    state = os.path.join(tempfile.gettempdir(), "barcode_check_state.json")
    m = BarcodeMap(state_path=state)
    print(f"격자 {m.w}x{m.h}, 칸 {m.cell_m}m, 바코드 {len(m.markers)}개")
    print("위에 [지도 경고]가 없으면 정상")
    return m


def check_labels(m):
    labels_dir = os.path.join(HERE, "labels")
    if not os.path.isdir(labels_dir):
        print("\n[라벨] labels/ 없음, 건너뜀 (노트북에서 make_barcodes.py 로 만든 뒤 같이 복사)")
        return

    print("\n[라벨] 카메라 0.5m 거리 크기(약 210px)로 줄여서 인식")
    ok = 0
    for code in m.markers:
        path = os.path.join(labels_dir, f"{code}.png")
        if not os.path.exists(path):
            print(f"  {code}: 라벨 파일 없음")
            continue

        img = cv2.imread(path)
        w = 210
        h = int(img.shape[0] * w / img.shape[1])
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        frame = np.full((480, 640, 3), 110, np.uint8)
        frame[100:100 + h, 200:200 + w] = small

        t = time.time()
        found = [text for text, _ in m.reader.read(frame)]
        ms = (time.time() - t) * 1000
        hit = code in found
        ok += hit
        print(f"  {code}: {'OK' if hit else '실패'}  ({ms:.0f}ms)")

    print(f"  {ok}/{len(m.markers)} 인식")


def check_camera(m, seconds):
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("\n[카메라] picamera2 를 불러올 수 없음")
        return

    print(f"\n[카메라] {seconds}초 동안 바코드 읽기. 바코드를 들고 가까이/멀리 움직여 보세요.")
    print(f"  폭이 {m.min_width_px}px 이상이면 서버에서 회피 동작이 실행됩니다.")

    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    ))
    camera.start()
    time.sleep(1.0)

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    frames = 0
    read_ms = 0.0
    last_print = 0.0
    saved = False
    end = time.time() + seconds

    try:
        while time.time() < end:
            frame = cv2.cvtColor(camera.capture_array(), cv2.COLOR_RGB2BGR)

            t = time.time()
            found = m.reader.read(frame)
            read_ms += (time.time() - t) * 1000
            frames += 1

            if found and time.time() - last_print > 0.5:
                last_print = time.time()
                for text, (x1, y1, x2, y2) in found:
                    width = x2 - x1
                    known = text in m.markers
                    near = width >= m.min_width_px
                    state = "회피 실행" if (known and near) else ("멀어서 무시" if known else "모르는 바코드")
                    print(f"  {text}: 폭 {width}px -> {state}")

                if not saved:
                    m.last_boxes = found
                    cv2.imwrite(os.path.join(out_dir, "camera_check.jpg"), m.draw(frame))
                    saved = True
    finally:
        camera.stop()
        camera.close()

    if frames:
        print(f"  프레임 {frames}장, 바코드 읽기 평균 {read_ms / frames:.0f}ms")
    if saved:
        print("  인식된 화면:", os.path.join(out_dir, "camera_check.jpg"))
    else:
        print("  바코드를 한 번도 못 읽었습니다. 조명, 거리, 라벨 인쇄 크기를 확인하세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="실제 카메라로 확인 (팀 서버 끄고 실행)")
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()

    check_versions()
    m = check_markers()
    check_labels(m)
    if args.camera:
        check_camera(m, args.seconds)


if __name__ == "__main__":
    main()
