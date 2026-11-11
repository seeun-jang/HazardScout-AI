#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PiCar-Pro PC(TigerVNC) server: GUIServer_custom.py entrypoint.

Use on the Raspberry Pi (not in parallel with any mobile server):
    sudo ~/picar-yolo/bin/python GUIServer_custom.py

The original PC server logic is preserved in GUIServer_custom_base.py.
The OLED, servo driver, and SmoothServo files are not modified.
Neck ranges were requested and physically checked by the operator:
  A pan +/-90, A tilt +/-80, B tilt +/-90 (relative to servo 90-degree neutral).
Actual mechanical limits and sustained combined-load behavior remain operator-dependent.
"""
import os
import threading
import time

import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from PIL import Image, ImageDraw

import RPIservo
import SmoothServo
import GUIServer_custom_base as pc
from OLED_eyes import OLEDEyes
from YOLO_overlay import with_yolov8n
from PiCommandService import ExistingControls, start_command_service

# Capture the camera worker before the map launcher wraps it.
PlainAsyncYOLO = pc.AsyncYOLO

# PCA9685 / neck servo settings from the mobile Final implementation.
PCA_ADDRESS = 0x5F
PWM_FREQUENCY = 50
NECK_CHANNELS = (1, 2, 3)
NECK_LIMITS = {
    'a_pan': (-90, 90),
    'a_tilt': (-80, 80),
    'b_tilt': (-90, 90),
}
OLED_POLL = 0.08
OLED_ANIMATION_INTERVAL = 0.5
OLED_AFTER_SERVO_IDLE = 0.20
DRIVE_TIMEOUT = 1.2  # PC command keepalive is sent at 0.25-second intervals.

neck_controller = None
grip_controller = None
last_drive_command = 0.0
last_drive_motion = False
drive_lock = threading.Lock()
watchdog_stop = threading.Event()


class FinalPCOLED(OLEDEyes):
    """Defer I2C OLED writes during servo motion; skip unchanged displays."""

    @staticmethod
    def servo_busy():
        now = time.monotonic()
        nc = neck_controller
        if nc is not None:
            with nc.lock:
                for axis in nc.axes:
                    if nc.direction[axis] and nc.deadline[axis] > now:
                        return True
        gc = grip_controller
        if gc is not None:
            with gc.lock:
                if gc.direction and gc.deadline > now:
                    return True
        return False

    def run(self):
        if not self.connected:
            return
        previous_image = None
        previous_direction = None
        previous_blink = None
        sent_at = 0.0
        idle_since = time.monotonic()
        while self.running:
            now = time.monotonic()
            if self.servo_busy():
                idle_since = now
                time.sleep(OLED_POLL)
                continue
            if now - idle_since < OLED_AFTER_SERVO_IDLE:
                time.sleep(OLED_POLL)
                continue
            with self.lock:
                direction = self.direction
            frame = Image.new(self.device.mode, self.device.size)
            draw = ImageDraw.Draw(frame)
            self._draw_face(draw, direction)
            blink = self.blinking
            frame_bytes = frame.tobytes()
            changed = frame_bytes != previous_image
            important = direction != previous_direction or blink != previous_blink
            if self.servo_busy():
                idle_since = time.monotonic()
            elif changed and (important or now - sent_at >= OLED_ANIMATION_INTERVAL):
                try:
                    self.device.display(frame)
                    previous_image = frame_bytes
                    previous_direction = direction
                    previous_blink = blink
                    sent_at = time.monotonic()
                except Exception as exc:
                    print('[PC OLED 갱신 오류]', repr(exc))
            self.frame_count += 1
            time.sleep(OLED_POLL)


class ExtendedPCNeck(SmoothServo.SmoothNeck):
    """Keep original mobile one-degree stepping, heartbeat and halt behavior."""

    def _run(self):
        while not self.stop_event.is_set():
            tick = time.monotonic()
            changes = []
            with self.lock:
                for axis in self.axes:
                    if self.direction[axis] and tick > self.deadline[axis]:
                        self.direction[axis] = 0
                    sign = self.direction[axis] * SmoothServo.NECK_SIGNS[axis]
                    if not sign:
                        continue
                    before = self.position[axis]
                    low, high = NECK_LIMITS[axis]
                    after = max(low, min(high, before + sign * SmoothServo.NECK_STEP))
                    if after != before:
                        changes.append((axis, after))
            for axis, angle in changes:
                if self.stop_event.is_set():
                    break
                try:
                    self.servo.moveAngle(SmoothServo.NECK_CHANNELS[axis], angle)
                    with self.lock:
                        self.position[axis] = angle
                    self.last_error = None
                except Exception as exc:
                    self.last_error = repr(exc)
                    print('[PC 목 제어 오류]', axis, repr(exc))
                    self.release_all()
                    self.stop_event.wait(0.5)
                    break
            duration = time.monotonic() - tick
            self.stop_event.wait(max(0.001, SmoothServo.NECK_TICK - duration))


class FinalPCYOLO(pc.YOLODetector):
    def __init__(self, *args, **kwargs):
        self._last_report = 0.0
        super().__init__(*args, **kwargs)

    def _run_inference(self, frame):
        result = super()._run_inference(frame)
        now = time.monotonic()
        if result and now - self._last_report >= 3.0:
            try:
                print('[PC YOLO 인식]', ', '.join(str(d[4]) for d in result[:4]), flush=True)
            except Exception:
                pass
            self._last_report = now
        return result


def _pc_neck_command(message, seq=None):
    if neck_controller is None:
        raise RuntimeError('PC 목 제어기 초기화되지 않음')
    neck_controller.command(message, seq)


def _pc_grip_start(direction):
    if grip_controller is None:
        raise RuntimeError('PC 집게 제어기 초기화되지 않음')
    grip_controller.hold(direction)


def _pc_grip_stop():
    if grip_controller is not None:
        grip_controller.release()


def _drive_watchdog():
    global last_drive_motion
    while not watchdog_stop.wait(0.10):
        expired = False
        with drive_lock:
            if last_drive_motion and time.monotonic() - last_drive_command > DRIVE_TIMEOUT:
                last_drive_motion = False
                expired = True
        if expired:
            try:
                pc.safe_stop()
                print('[PC 안전] 주행 명령 타임아웃 → 정지', flush=True)
            except Exception as exc:
                print('[PC 안전 정지 오류]', repr(exc))


def run():
    global neck_controller, grip_controller, last_drive_motion, last_drive_command
    # Store originals before process-local overrides; restore on exit.
    original_set_angle = RPIservo.ServoCtrl.set_angle
    original_oled = pc.OLEDEyes
    original_yolo = pc.YOLODetector
    original_async_yolo = pc.AsyncYOLO
    original_robot_ctrl = pc.robot_ctrl
    original_handle_client = pc.handle_client
    original_gripper_start = pc.gripper_start
    original_gripper_stop = pc.gripper_stop
    original_sign = SmoothServo.NECK_SIGNS['a_pan']
    original_destroy = pc.move.destroy
    command_service = None

    # Avoid repeated I2C/PCA9685 construction for neck channels 1, 2, 3.
    bus = busio.I2C(SCL, SDA)
    pwm = PCA9685(bus, address=PCA_ADDRESS)
    pwm.frequency = PWM_FREQUENCY
    neck_pwm = {
        ch: servo.Servo(pwm.channels[ch], min_pulse=500,
                        max_pulse=2400, actuation_range=180)
        for ch in NECK_CHANNELS
    }
    pwm_lock = threading.Lock()

    def cached_set_angle(self, channel, angle):
        if channel in NECK_CHANNELS:
            with pwm_lock:
                neck_pwm[channel].angle = angle
            return
        return original_set_angle(self, channel, angle)

    drive_commands = {'forward', 'backward', 'left', 'right', 'backleft', 'backright'}
    neck_legacy_map = {
        'lookleft': 'neck:a_pan:-1',
        'lookright': 'neck:a_pan:1',
        'LRstop': 'neck:a_pan:0',
        'handup': 'neck:a_tilt:1',
        'handdown': 'neck:a_tilt:-1',
        'HAstop': 'neck:a_tilt:0',
        'armup': 'neck:b_tilt:1',
        'armdown': 'neck:b_tilt:-1',
        'armstop': 'neck:b_tilt:0',
    }

    def pc_robot_ctrl(command):
        global last_drive_motion, last_drive_command
        if command in neck_legacy_map:
            command = neck_legacy_map[command]
        if command.startswith('neck:'):
            _pc_neck_command(command)
            return
        if command == 'grab':
            _pc_grip_start(-1)
            return
        if command == 'loose':
            _pc_grip_start(1)
            return
        if command in ('stop', 'grip_stop'):
            _pc_grip_stop()
            return
        if command == 'drive_ping':
            with drive_lock:
                if last_drive_motion:
                    last_drive_command = time.monotonic()
            return
        if command in drive_commands:
            with drive_lock:
                last_drive_motion = True
                last_drive_command = time.monotonic()
            return original_robot_ctrl(command)
        if command in ('DS', 'TS', 'stop_move'):
            with drive_lock:
                last_drive_motion = False
            return original_robot_ctrl('DS')
        # All remaining commands use the original PC implementation.
        return original_robot_ctrl(command)

    def route_command(command):
        if command == 'drive_ping' or command_service is None:
            return pc_robot_ctrl(command)
        return command_service.manual(pc_robot_ctrl, command)

    def refresh_ai_watchdog():
        global last_drive_motion, last_drive_command
        with drive_lock:
            last_drive_motion = True
            last_drive_command = time.monotonic()

    def destroy_with_ai_cleanup():
        if command_service is not None:
            command_service.close()
        original_destroy()

    def guarded_handle_client(client_socket, client_address):
        try:
            return original_handle_client(client_socket, client_address)
        finally:
            if command_service is not None:
                command_service.cancel()
            if neck_controller is not None:
                neck_controller.release_all()
            if grip_controller is not None:
                grip_controller.release()

    try:
        RPIservo.ServoCtrl.set_angle = cached_set_angle
        pc.OLEDEyes = FinalPCOLED
        pc.YOLODetector = FinalPCYOLO
        pc.robot_ctrl = route_command
        pc.move.destroy = destroy_with_ai_cleanup
        pc.handle_client = guarded_handle_client
        pc.gripper_start = _pc_grip_start
        pc.gripper_stop = _pc_grip_stop
        SmoothServo.NECK_SIGNS['a_pan'] = -1  # mobile Final direction correction

        pc.speed_set = 60
        pc.TURN_SPEED = 50
        pc.CAMERA_FPS = 12
        pc.JPEG_QUALITY = 60
        pc.YOLO_MODEL_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models', 'safety7.pt',
        )
        pc.YOLO_IMAGE_SIZE = 224
        pc.YOLO_DETECT_EVERY_N_FRAMES = 8

        neck_controller = ExtendedPCNeck()
        neck_controller.start()
        grip_controller = SmoothServo.SmoothGripper(pc.gripper_servo, speed=pc.GRIPPER_SPEED)
        watchdog_stop.clear()
        watcher = threading.Thread(target=_drive_watchdog, daemon=True,
                                   name='PCDriveWatchdog')
        watcher.start()

        print('=' * 62)
        print('PC(TigerVNC) 통합 서버 / TCP 10223 / ZeroMQ 5555')
        print('목 A 좌우 ±90°, 목 A 상하 ±80°, 목 B 상하 ±90°')
        print('모바일 Final의 PWM 캐시·OLED 보호·YOLO 모델·집게 보정')
        print('PC GUI는 ~/move/robot_gui.py로 실행하세요.')
        print('모바일 서버와 동시에 실행하지 마세요.')
        print('=' * 62)
        controls = ExistingControls(
            vars(pc),
            lambda: (neck_controller, grip_controller),
            lambda: pc_robot_ctrl('DS'),
            refresh_ai_watchdog,
        )
        try:
            command_service = start_command_service(controls, local_http=True)
        except OSError as exc:
            print('[AI 명령] PC 패널 포트 초기화 실패:', exc, flush=True)
        pc.AsyncYOLO = with_yolov8n(
            original_async_yolo, PlainAsyncYOLO, pc.YOLO_MODEL_PATH,
            pc.YOLO_CONFIDENCE, pc.YOLO_IMAGE_SIZE,
        )
        pc.main()
    except KeyboardInterrupt:
        print('\n[PC Final] 종료 요청')
    finally:
        if command_service is not None:
            command_service.close()
        pc.move.destroy = original_destroy
        watchdog_stop.set()
        with drive_lock:
            last_drive_motion = False
        if neck_controller is not None:
            neck_controller.close()
        if grip_controller is not None:
            grip_controller.close()
        RPIservo.ServoCtrl.set_angle = original_set_angle
        pc.OLEDEyes = original_oled
        pc.YOLODetector = original_yolo
        pc.AsyncYOLO = original_async_yolo
        pc.robot_ctrl = original_robot_ctrl
        pc.handle_client = original_handle_client
        pc.gripper_start = original_gripper_start
        pc.gripper_stop = original_gripper_stop
        SmoothServo.NECK_SIGNS['a_pan'] = original_sign
        try:
            pwm.deinit()
        except Exception:
            pass
        print('[PC Final] 기존 제어 함수 복원 / 종료')


if __name__ == '__main__':
    run()
