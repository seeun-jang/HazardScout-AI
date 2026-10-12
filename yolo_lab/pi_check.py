"""라즈베리파이에서 탐지가 잘 돌아가는지, 얼마나 빠른지 확인하는 스크립트 (pi_check.py)

팀 서버에 붙이기 전에 이걸 먼저 돌려서 문제가 없는지 확인하세요.
모터, 서보, GPIO는 전혀 건드리지 않습니다.

확인하는 것:
  1) 파이썬, ultralytics, torch 버전 (yolo26n은 ultralytics 8.4 이상 필요)
  2) 모델 파일이 있는지
  3) 확인용 사진 3장 탐지 -> results/pi_check/ 에 박스 그린 사진 저장
  4) 속도: 입력 크기와 위험물 탐지 on/off 조합별로 프레임당 시간 측정
  5) (--camera) 파이 카메라로 직접 찍어서 탐지

실행 (robot_package/ 폴더에서, 라즈베리파이에서):
  python3 pi_check.py
  python3 pi_check.py --camera        # 카메라까지 확인 (팀 서버가 꺼져 있어야 함)

주의: 카메라는 한 프로그램만 쓸 수 있습니다. --camera 전에 팀 서버(GUIServer)를 끄세요.
"""
import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def check_versions():
    import torch
    import ultralytics
    print(f"Python      : {platform.python_version()}  ({platform.machine()})")
    print(f"ultralytics : {ultralytics.__version__}")
    print(f"torch       : {torch.__version__}")
    major, minor = (int(x) for x in ultralytics.__version__.split(".")[:2])
    if (major, minor) < (8, 4):
        print("\n[경고] ultralytics가 8.4보다 낮아서 yolo26n을 못 읽을 수 있습니다.")
        print("  해결 1) pip install -U 'ultralytics>=8.4'  (팀 환경이 깨질 수 있으니 가상환경 백업 후)")
        print("  해결 2) ProYOLODetector(preset='pi', models=['yolo11n.pt']) 로 yolo11n 사용")


def check_files():
    from detect_pro import MODELS, PRESETS
    ok = True
    for name in PRESETS["pi"]["models"] + [PRESETS["pi"]["world"]]:
        exists = (MODELS / name).exists()
        ok &= exists
        print(f"  {'OK ' if exists else '없음'} models/{name}")
    if not ok:
        print("모델 파일이 없습니다. 노트북에서 prepare_robot.py로 만든 zip을 다시 복사하세요.")
    return ok


def timed(det, frame, n=10):
    """프레임 하나를 n번 탐지해서 중앙값(ms)을 돌려줌. 첫 2번은 예열이라 뺌."""
    for _ in range(2):
        det.detect(frame)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        det.detect(frame)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", action="store_true", help="파이 카메라로 찍어서도 확인")
    ap.add_argument("--repeat", type=int, default=10, help="속도 측정 반복 횟수")
    args = ap.parse_args()

    print("=== 1. 버전 ===")
    check_versions()
    print("\n=== 2. 모델 파일 ===")
    if not check_files():
        return

    from detect_pro import ProDetector, HAZARD_CLASSES
    out_dir = ROOT / "results" / "pi_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== 3. 확인용 사진 탐지 (pi 프리셋) ===")
    det = ProDetector(preset="pi", device="cpu")
    frames = []
    for f in sorted((ROOT / "test_images").glob("*.jpg")):
        img = cv2.imread(str(f))
        img = cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))  # 파이 카메라 크기(640)에 맞춤
        frames.append(img)
        dets = det.detect(img)
        names = [d.name for d in dets]
        hazards = [n for n in names if n in HAZARD_CLASSES]
        cv2.imwrite(str(out_dir / f"{f.stem}_pi.jpg"), det.draw(img, dets))
        print(f"  {f.name}: 물체 {len(dets)}개, 위험물 {hazards}")
    print(f"  박스 그린 사진: {out_dir}")

    print("\n=== 4. 속도 (프레임당, 640x480 기준) ===")
    frame = cv2.resize(frames[0], (640, 480))
    combos = [("pi 프리셋 그대로 (일반 416 + 위험물 320)", dict()),
              ("일반 320 + 위험물 320 (더 빠르게)", dict(imgsz=320)),
              ("일반 416만, 위험물 끔", dict(world="none")),
              ("일반 320만, 위험물 끔 (팀 현재와 비슷)", dict(imgsz=320, world="none"))]
    ncnn = ROOT / "models" / "yolo26n_ncnn_model"
    if ncnn.exists():
        combos.append(("NCNN 일반 416 + 위험물 320", dict(models=[str(ncnn)])))
    for label, kw in combos:
        d = ProDetector(preset="pi", device="cpu", verbose=False, **kw)
        ms = timed(d, frame, args.repeat)
        every3 = 3 * 1000 / ms if ms else 0
        print(f"  {label:38s} {ms:7.0f} ms  (매 프레임 {1000 / ms:4.1f} FPS, 3프레임마다 탐지 시 영상 최대 {every3:4.1f} FPS)")
    print("  -> 팀 서버는 3프레임마다 탐지합니다 (YOLO_DETECT_EVERY_N_FRAMES = 3).")

    if args.camera:
        print("\n=== 5. 파이 카메라 ===")
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
        cam.start()
        time.sleep(1.0)  # 자동 노출이 안정될 때까지 대기
        for i in range(3):
            # 팀 서버(GUIServer_custom.py)와 똑같이 변환해야 서버에서 보는 결과와 같아짐
            frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
            dets = det.detect(frame)
            cv2.imwrite(str(out_dir / f"camera_{i}.jpg"), det.draw(frame, dets))
            print(f"  camera_{i}.jpg: {[d.name for d in dets]}")
            time.sleep(0.5)
        cam.stop()
        print(f"  카메라 사진: {out_dir}")

    print("\n확인 완료. 문제없으면 ROBOT_DEPLOY.md의 '팀 서버에 연결' 단계로 넘어가세요.")


if __name__ == "__main__":
    main()
