#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_avoider.py

바코드 회피 동작 (정지 -> 살짝 후진 -> 왼쪽/오른쪽/유턴/정지)

- 별도 스레드에서 실행해서 카메라 영상이 멈추지 않음
- 주행 명령을 refresh_sec 마다 다시 보냄
  -> PC 서버의 '1.2초 동안 명령 없으면 정지' 안전장치에 걸리지 않음
- 사용자가 방향키/조이스틱을 누르면 cancel() 로 즉시 중단 (사용자 조작이 우선)
- 회전 시간은 robot_config.json 의 yaw_deg_per_sec 로 계산
  (90도 = 90 / yaw_deg_per_sec 초)

PiCar-Pro는 앞바퀴 조향식이라 제자리 회전이 안 되고, 도는 동안 앞으로 나아간다.
"""

import threading
import time


class Avoider:
    def __init__(self, cfg, robot_ctrl, stop, on_done=None):
        """
        robot_ctrl : 서버의 주행 명령 함수 (forward/backward/left/right/backleft/backright)
        stop       : 정지 함수
        on_done    : 동작이 끝나면 action 을 받아 호출 (방향 정리 등)
        """
        self.cfg = cfg
        self.robot_ctrl = robot_ctrl
        self.stop = stop
        self.on_done = on_done
        self.cancel_event = threading.Event()
        self.thread = None
        self.current = None

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def is_own_thread(self):
        return self.thread is not None and threading.current_thread() is self.thread

    def start(self, action, code=""):
        if self.busy:
            return False
        self.cancel_event.clear()
        self.current = action
        self.thread = threading.Thread(target=self._run, args=(action, code),
                                       daemon=True, name="HazardMapAvoid")
        self.thread.start()
        return True

    def cancel(self):
        if self.busy and not self.is_own_thread():
            self.cancel_event.set()
            print("[회피] 사용자 조작으로 중단", flush=True)

    def _hold(self, command, sec):
        """command 를 sec 초 동안 유지. 취소되면 False"""
        # 기다린 횟수가 아니라 실제 경과 시간으로 끝냄
        # (파이가 바쁘면 wait 가 길어져서 횟수로 세면 더 많이 돌게 됨)
        refresh = max(0.02, float(self.cfg.avoid["refresh_sec"]))
        end = time.monotonic() + sec
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return True
            self.robot_ctrl(command)
            if self.cancel_event.wait(min(refresh, remaining)):
                return False

    def _run(self, action, code):
        av = self.cfg.avoid
        yaw = max(1.0, float(self.cfg.motion["yaw_deg_per_sec"]))
        print(f"[회피] 바코드 {code} -> {action} 시작", flush=True)
        finished = False

        try:
            self.stop()
            if self.cancel_event.wait(float(av["stop_sec"])):
                return

            if action == "stop":
                finished = True
                return

            if not self._hold("backward", float(av["back_sec"])):
                return
            self.stop()

            if action == "left":
                ok = self._hold("left", 90.0 / yaw)
            elif action == "right":
                ok = self._hold("right", 90.0 / yaw)
            elif action == "back":
                ok = self._hold("backleft", 180.0 / yaw)
            else:
                ok = True

            if not ok:
                return

            if av.get("auto_resume"):
                self.robot_ctrl("forward")
            else:
                self.stop()

            finished = True
            print(f"[회피] {action} 완료", flush=True)

        except Exception as e:
            print("[회피] 오류:", repr(e), flush=True)
            try:
                self.stop()
            except Exception:
                pass
        finally:
            self.current = None
            if self.on_done is not None:
                try:
                    self.on_done(action, finished)
                except Exception as e:
                    print("[회피] 마무리 오류:", repr(e), flush=True)
