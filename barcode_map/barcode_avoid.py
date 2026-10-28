#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barcode_avoid.py

바코드 이벤트("길 없음 -> 왼쪽/오른쪽/돌아가기/정지")를 실제 주행 동작으로 바꾸는 모듈
- GUIServer 의 robot_ctrl / safe_stop 을 그대로 받아서 사용
- 카메라 루프를 막지 않도록 별도 스레드에서 실행
- 사용자가 수동 명령을 보내면 cancel() 로 즉시 중단

PiCar-Pro 는 조향식이라 제자리 회전이 안 되므로
  정지 -> 살짝 후진 -> 조향 상태로 전진(또는 후진) 하며 90도 / 180도 회전
시간값(TURN_90_SEC 등)은 실제 바닥에서 재보고 맞춰야 한다.
"""

import threading
import time


class BarcodeAvoider:
    STOP_SEC = 0.4
    BACK_SEC = 0.6
    TURN_90_SEC = 1.6
    TURN_180_SEC = 3.0

    def __init__(self, robot_ctrl, safe_stop, barcode_map=None,
                 auto_resume=False, on_event=None):
        """
        auto_resume : 회피 후 자동으로 다시 전진할지 (자율주행 모드에서 True)
        on_event    : 이벤트를 GUI/대시보드로 보내고 싶을 때 쓰는 콜백 (event dict 받음)
        """
        self.robot_ctrl = robot_ctrl
        self.safe_stop = safe_stop
        self.barcode_map = barcode_map
        self.auto_resume = auto_resume
        self.on_event = on_event

        self.cancel_event = threading.Event()
        self.thread = None

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def handle(self, event):
        """BarcodeMap.process() 가 돌려준 이벤트를 받아서 동작 시작"""
        if event is None or self.busy:
            return False

        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as e:
                print("[회피] on_event 오류:", repr(e))

        self.cancel_event.clear()
        self.thread = threading.Thread(target=self._run, args=(event,), daemon=True)
        self.thread.start()
        return True

    def cancel(self):
        """수동 조작이 들어오면 호출"""
        if self.busy:
            self.cancel_event.set()
            print("[회피] 수동 명령으로 중단")

    def _wait(self, sec):
        """sec 만큼 기다림. 중간에 취소되면 False"""
        return not self.cancel_event.wait(sec)

    def _drive(self, command, sec):
        # 정지는 robot_ctrl 명령 대신 safe_stop 으로 한다.
        # 정지 명령 이름이 서버마다 다름 (custom: "TS", mobile: "stop_move")
        self.robot_ctrl(command)
        ok = self._wait(sec)
        self.safe_stop()
        return ok

    def _run(self, event):
        action = event["action"]
        print(f"[회피] {event['code']} -> {action} 시작")

        try:
            self.safe_stop()
            if not self._wait(self.STOP_SEC):
                return

            if action == "stop":
                if self.barcode_map is not None:
                    self.barcode_map.apply_turn("stop")
                return

            if not self._drive("backward", self.BACK_SEC):
                return

            if action == "left":
                ok = self._drive("left", self.TURN_90_SEC)
            elif action == "right":
                ok = self._drive("right", self.TURN_90_SEC)
            elif action == "back":
                ok = self._drive("backleft", self.TURN_180_SEC)
            else:
                ok = True

            if not ok:
                return

            if self.barcode_map is not None:
                self.barcode_map.apply_turn(action)

            if self.auto_resume:
                self.robot_ctrl("forward")
            else:
                self.safe_stop()

            print(f"[회피] {action} 완료")

        except Exception as e:
            print("[회피] 오류:", repr(e))
            self.safe_stop()
