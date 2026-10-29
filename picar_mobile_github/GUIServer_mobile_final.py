#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PiCar-Pro mobile server: validated stage-14 settings, without stage test files.

Run on the Pi:
    sudo ~/picar-yolo/bin/python GUIServer_mobile_final.py

Original GUIServer_mobile.py / OLED_eyes.py / RPIservo.py / SmoothServo.py
are not modified. Assumes Pi I2C speed was already set to 400kHz.
"""

import threading
import time

import busio
from board import SCL, SDA
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from PIL import Image, ImageDraw

import RPIservo
import SmoothServo
import GUIServer_mobile as server
from OLED_eyes import OLEDEyes


# Same hardware settings verified in stage8.
NECK_CHANNELS = (1, 2, 3)
PWM_ADDRESS = 0x5F
PWM_FREQUENCY = 50

# Relative offsets from the original startup positions (degrees).
# Increase movement conservatively: stop if a servo strains or binds.
# B positive is the UP direction in the tested mobile controls.
NECK_TRAVEL_LIMITS = {
    "a_pan": (-25, 25),     # A left/right: symmetric +/-25 from starting pose
    "b_tilt": (-20, 12),   # B down: -16 -> -20; B up remains +12
}

# Same OLED scheduling verified in stage12.
OLED_POLL_SECONDS = 0.08
OLED_ANIMATION_SECONDS = 0.5
OLED_SERVO_IDLE_SECONDS = 0.20
OLED_REPORT_SECONDS = 10.0


class FinalOLEDEyes(OLEDEyes):
    """Send changed OLED frames, deferring writes during neck/grip movement."""

    @staticmethod
    def servo_busy():
        now = time.monotonic()
        neck = getattr(server, "neck_controller", None)
        if neck is not None:
            with neck.lock:
                for axis in neck.axes:
                    if neck.direction[axis] != 0 and neck.deadline[axis] > now:
                        return True

        grip = getattr(server, "grip_controller", None)
        if grip is not None:
            with grip.lock:
                if grip.direction != 0 and grip.deadline > now:
                    return True
        return False

    def run(self):
        if not self.connected:
            print("[최종 OLED] 연결 실패")
            return

        print("[최종 OLED] 변경 화면 전송 + 서보 우선 갱신 시작")
        last_image = None
        last_direction = None
        last_blinking = None
        last_send = 0.0
        last_busy = time.monotonic()
        last_report = time.monotonic()
        sent = skipped = deferred = 0

        while self.running:
            now = time.monotonic()
            if now - last_report >= OLED_REPORT_SECONDS:
                print(
                    "[OLED 통계] "
                    f"전송={sent}회, 중복 생략={skipped}회, "
                    f"서보 우선 보류={deferred}회",
                    flush=True,
                )
                sent = skipped = deferred = 0
                last_report = now

            if self.servo_busy():
                last_busy = now
                deferred += 1
                time.sleep(OLED_POLL_SECONDS)
                continue

            if now - last_busy < OLED_SERVO_IDLE_SECONDS:
                deferred += 1
                time.sleep(OLED_POLL_SECONDS)
                continue

            with self.lock:
                direction = self.direction

            frame = Image.new(self.device.mode, self.device.size)
            draw = ImageDraw.Draw(frame)
            self._draw_face(draw, direction)
            blinking = self.blinking
            frame_bytes = frame.tobytes()
            changed = frame_bytes != last_image
            important = direction != last_direction or blinking != last_blinking
            animation_due = now - last_send >= OLED_ANIMATION_SECONDS

            # Check again before an I2C transfer (as in stage12).
            if self.servo_busy():
                last_busy = time.monotonic()
                deferred += 1
            elif changed and (important or animation_due):
                try:
                    self.device.display(frame)
                    last_image = frame_bytes
                    last_direction = direction
                    last_blinking = blinking
                    last_send = time.monotonic()
                    sent += 1
                except Exception as exc:
                    print("[최종 OLED] 전송 오류:", repr(exc))
            else:
                skipped += 1

            self.frame_count += 1
            time.sleep(OLED_POLL_SECONDS)

        print("[최종 OLED] Thread 종료")


class ExtendedSmoothNeck(SmoothServo.SmoothNeck):
    """Keep the original stepping, heartbeat and stop logic; change only limits."""

    def _run(self):
        while not self.stop_event.is_set():
            loop_start = time.monotonic()
            moves = []

            with self.lock:
                for axis in self.axes:
                    if self.direction[axis] and loop_start > self.deadline[axis]:
                        self.direction[axis] = 0

                    sign = self.direction[axis] * SmoothServo.NECK_SIGNS[axis]
                    if not sign:
                        continue

                    position = self.position[axis]
                    original_limit = SmoothServo.NECK_MAX_OFFSET[axis]
                    lower_limit, upper_limit = NECK_TRAVEL_LIMITS.get(
                        axis, (-original_limit, original_limit)
                    )
                    next_pos = max(
                        lower_limit,
                        min(upper_limit, position + sign * SmoothServo.NECK_STEP),
                    )
                    if next_pos == position:
                        continue
                    moves.append((axis, next_pos))

            for axis, next_pos in moves:
                if self.stop_event.is_set():
                    break
                try:
                    self.servo.moveAngle(SmoothServo.NECK_CHANNELS[axis], next_pos)
                    with self.lock:
                        self.position[axis] = next_pos
                    self.last_error = None
                except Exception as exc:
                    self.last_error = repr(exc)
                    print(f"[목 제어 오류] {axis}: {exc!r}")
                    self.release_all()
                    self.stop_event.wait(0.5)
                    break

            elapsed = time.monotonic() - loop_start
            self.stop_event.wait(max(0.001, SmoothServo.NECK_TICK - elapsed))


class FinalYOLO(server.YOLODetector):
    """Original YOLO inference, with throttled positive-detection logging."""

    def __init__(self, *args, **kwargs):
        self._last_detection_log = 0.0
        super().__init__(*args, **kwargs)

    def _run_inference(self, frame):
        detections = super()._run_inference(frame)
        now = time.monotonic()
        if detections and now - self._last_detection_log >= 3.0:
            try:
                names = [
                    str(d[4]) for d in detections
                    if isinstance(d, (tuple, list)) and len(d) > 4
                ]
                print(
                    f"[YOLO 인식] {len(detections)}개: "
                    + (", ".join(names[:4]) if names else "객체 감지"),
                    flush=True,
                )
            except Exception as exc:
                # Logging must never break a successful inference.
                print("[YOLO 로그 경고]", repr(exc))
            self._last_detection_log = now
        return detections


def main():
    print("=" * 52)
    print("PiCar-Pro 모바일 최종 통합 서버")
    print("I2C: 부팅 설정 400kHz 사용 (프로그램이 설정 변경하지 않음)")
    print("카메라 / YOLO 실제 추론 / OLED / 음성: 기존 기능 유지")
    print("목 1·2·3: PCA9685 PWM 재사용")
    print("OLED: 변경 화면만 전송, 목/집게 동작 중 갱신 보류")
    print("집게 / 조향: 기존 서보 함수 유지")
    print("기존 서버·OLED·서보 파일: 수정 없음")
    print("=" * 52)

    original_set_angle = RPIservo.ServoCtrl.set_angle
    original_oled_class = server.OLEDEyes
    original_yolo_class = server.YOLODetector
    original_a_pan_sign = SmoothServo.NECK_SIGNS["a_pan"]
    original_neck_class = server.SmoothNeck

    # Same cached driver settings as stage8. Keep objects alive for the server.
    i2c = busio.I2C(SCL, SDA)
    pwm = PCA9685(i2c, address=PWM_ADDRESS)
    pwm.frequency = PWM_FREQUENCY
    neck_servos = {
        ch: servo.Servo(
            pwm.channels[ch],
            min_pulse=500,
            max_pulse=2400,
            actuation_range=180,
        )
        for ch in NECK_CHANNELS
    }
    neck_lock = threading.Lock()

    def cached_set_angle(self, channel, angle):
        if channel in NECK_CHANNELS:
            with neck_lock:
                neck_servos[channel].angle = angle
            return
        return original_set_angle(self, channel, angle)

    try:
        RPIservo.ServoCtrl.set_angle = cached_set_angle
        server.OLEDEyes = FinalOLEDEyes
        server.YOLODetector = FinalYOLO
        # Invert only the A neck left/right direction; all other axes unchanged.
        SmoothServo.NECK_SIGNS["a_pan"] = -1
        server.SmoothNeck = ExtendedSmoothNeck
        print("[각도 확장] A 좌우: -25~+25도, B 상하: -20~+12도")
        print("[최종] A 목 좌우 방향 반전 적용")
        print("[최종] 목 PWM 캐시 준비 완료")
        print("[최종] 원본 YOLO 사용 + 실제 인식 시 로그 출력")
        server.start_voice()
        server.main()
    except KeyboardInterrupt:
        print("\n[최종] 종료 요청")
    finally:
        # Restore process-local patches. Hardware outputs are not recentered.
        RPIservo.ServoCtrl.set_angle = original_set_angle
        server.OLEDEyes = original_oled_class
        server.YOLODetector = original_yolo_class
        SmoothServo.NECK_SIGNS["a_pan"] = original_a_pan_sign
        server.SmoothNeck = original_neck_class
        print("[최종] 실행 중 변경한 Python 함수 복원")


if __name__ == "__main__":
    main()

