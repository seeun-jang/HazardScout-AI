#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_pose.py

로봇 위치/방향 추적 (데드레커닝 + 바코드 보정)

PiCar-Pro는 바퀴 회전 센서가 없어서, 서버의 현재 주행 상태(current_motion)와
흐른 시간으로 위치를 계산한다.

  forward   : 바라보는 방향으로 forward_m_per_sec 만큼 이동
  backward  : 반대 방향으로 이동
  left/right: 조향 상태로 전진. 이동하면서 yaw_deg_per_sec 만큼 회전
  backleft  : 조향 상태로 후진. 뒤로 가면서 차체는 오른쪽(시계 방향)으로 돈다
  backright : 뒤로 가면서 차체는 왼쪽으로 돈다

오차는 시간이 갈수록 쌓이지만, 바코드를 읽을 때마다 set() 으로 다시 맞춘다.
"""

import threading

from map_config import HEADING_DEG, heading_vec, norm_deg

# motion -> (이동 방향 부호, 회전 부호, 회전 중인지)
MOTION_MODEL = {
    "forward": (1, 0, False),
    "backward": (-1, 0, False),
    "left": (1, -1, True),
    "right": (1, 1, True),
    "backleft": (-1, 1, True),
    "backright": (-1, -1, True),
}


class Pose:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        x, y = self.cfg.cell_center(self.cfg.start_cell)
        with self.lock:
            self.x = x
            self.y = y
            self.heading = HEADING_DEG[self.cfg.start_heading]
            self.motion = "stop"
            self.corrected_at = None   # 마지막 바코드 보정 시각
            self.odometry_m = 0.0      # 마지막 보정 후 추정으로만 움직인 거리

    def update(self, motion, dt):
        """카메라 루프에서 매 프레임 호출"""
        if dt <= 0 or dt > 1.0:
            # 첫 프레임이거나 루프가 멈췄다 돌아온 경우. 큰 점프를 막는다
            dt = 0.0

        m = self.cfg.motion
        with self.lock:
            self.motion = motion
            model = MOTION_MODEL.get(motion)
            if model is None or dt == 0.0:
                return

            sign, turn, turning = model
            speed = m["turn_m_per_sec"] if turning else m["forward_m_per_sec"]

            # 회전과 이동을 반씩 나눠 적용 (호를 따라가는 근사)
            self.heading = norm_deg(self.heading + turn * m["yaw_deg_per_sec"] * dt / 2)
            dx, dy = heading_vec(self.heading)
            self.x += sign * speed * dt * dx
            self.y += sign * speed * dt * dy
            self.heading = norm_deg(self.heading + turn * m["yaw_deg_per_sec"] * dt / 2)
            self.odometry_m += speed * dt

    def set(self, x, y, heading, now):
        with self.lock:
            self.x, self.y = x, y
            self.heading = norm_deg(heading)
            self.corrected_at = now
            self.odometry_m = 0.0

    def snap_heading(self, max_diff=30.0):
        """회피 동작 직후 방향을 가까운 90도 단위로 맞춤 (회피는 원래 직각 회전)"""
        with self.lock:
            target = round(self.heading / 90.0) * 90.0
            if abs((self.heading - target + 180) % 360 - 180) <= max_diff:
                self.heading = norm_deg(target)

    def snapshot(self):
        with self.lock:
            return {
                "x": self.x,
                "y": self.y,
                "heading": self.heading,
                "motion": self.motion,
            }

    def cell(self):
        with self.lock:
            return self.cfg.cell_of(self.x, self.y)
