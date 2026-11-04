"""Picamera2 카메라 시험. 기존 서버를 종료한 후 Pi에서만 실행하세요."""
import argparse
import time
from lab_utils import inference_args, validate_inference_args


def pi_frames():
    from picamera2 import Picamera2
    camera = Picamera2()
    try:
        camera.configure(camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}, buffer_count=4))
        camera.start()
        time.sleep(1)
        while True:
            # Picamera2 RGB888은 메모리상 [B,G,R]. cvtColor 없이 OpenCV/YOLO에 전달합니다.
            yield camera.capture_array("main")
    finally:
        # configure/start/추론 오류나 Ctrl+C에도 장치 반환
        try:
            camera.stop()
        finally:
            camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inference_args(parser)
    args = parser.parse_args()
    validate_inference_args(parser, args)
    from inference import run_frames
    run_frames(pi_frames(), args)


if __name__ == "__main__":
    main()
