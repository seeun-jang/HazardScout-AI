#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_config.py

site_map.json(현장)과 robot_config.json(로봇 보정값)을 읽고 검사한다.

좌표 규칙 (모든 모듈 공통)
- 지도 좌표는 미터 단위 (x, y). x는 오른쪽(E), y는 아래쪽(S)으로 증가
- 칸 좌표는 (cx, cy) = (x // 칸 크기, y // 칸 크기)
- 방향(heading)은 나침반 각도. N=0, E=90, S=180, W=270, 시계 방향으로 증가
  -> 오른쪽으로 돌면 각도가 커지고, 왼쪽으로 돌면 작아진다
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEADING_DEG = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
ACTIONS = ("left", "right", "back", "stop", "none")


def heading_vec(deg):
    """방향 각도 -> 지도 위 단위 벡터 (dx, dy)"""
    rad = math.radians(deg)
    return math.sin(rad), -math.cos(rad)


def norm_deg(deg):
    return deg % 360.0


def nearest_heading_name(deg):
    deg = norm_deg(deg)
    return min(HEADING_DEG, key=lambda k: abs((deg - HEADING_DEG[k] + 180) % 360 - 180))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Config:
    def __init__(self, site_path=None, robot_path=None):
        self.site_path = site_path or os.path.join(HERE, "site_map.json")
        self.robot_path = robot_path or os.path.join(HERE, "robot_config.json")

        site = _load(self.site_path)
        robot = _load(self.robot_path)

        self.cell_m = float(site["cell_size_m"])
        self.grid_w = int(site["grid_w"])
        self.grid_h = int(site["grid_h"])
        self.start_cell = tuple(site.get("start_cell", [0, 0]))
        self.start_heading = site.get("start_heading", "N")
        self.barcodes = site.get("barcodes", {})

        self.motion = robot["motion"]
        self.camera = robot["camera"]
        self.barcode = dict(robot["barcode"])
        self.avoid = robot["avoid"]
        self.hazards = robot["hazards"]

        # 현장마다 다른 회피 시작 거리 (좁은 트랙 등). 있으면 robot_config.json 값 대신 사용
        if site.get("trigger_distance_m") is not None:
            self.barcode["trigger_distance_m"] = float(site["trigger_distance_m"])

        self.warnings = []
        self._validate()
        for w in self.warnings:
            print("[지도 경고]", w)

    # -----------------------------------------------------
    # 좌표 변환
    # -----------------------------------------------------

    def in_bounds(self, cell):
        return 0 <= cell[0] < self.grid_w and 0 <= cell[1] < self.grid_h

    def cell_of(self, x, y):
        return int(math.floor(x / self.cell_m)), int(math.floor(y / self.cell_m))

    def cell_center(self, cell):
        return (cell[0] + 0.5) * self.cell_m, (cell[1] + 0.5) * self.cell_m

    def barcode_face(self, code):
        """바코드 라벨이 붙은 지점 (막힌 칸에서 로봇 쪽을 향한 면의 가운데)"""
        m = self.barcodes[code]
        cx, cy = self.cell_center(m["cell"])
        dx, dy = heading_vec(HEADING_DEG[m["robot_heading"]])
        return cx - dx * self.cell_m / 2, cy - dy * self.cell_m / 2

    # -----------------------------------------------------
    # 설정 실수 검사
    # -----------------------------------------------------

    def _validate(self):
        w = self.warnings

        if not self.in_bounds(self.start_cell):
            w.append(f"출발 칸 {self.start_cell} 이 지도 밖")
        if self.start_heading not in HEADING_DEG:
            w.append(f"start_heading '{self.start_heading}' 은 N/E/S/W 중 하나여야 함")
            self.start_heading = "N"

        for code, m in self.barcodes.items():
            if not (code.isdigit() and len(code) % 2 == 0):
                w.append(f"바코드 {code}: 짝수 자리 숫자(예: 1100)를 써야 멀리서 읽힘")

            cell = tuple(m.get("cell", (-1, -1)))
            heading = m.get("robot_heading")
            action = m.get("action")

            if heading not in HEADING_DEG or action not in ACTIONS:
                w.append(f"바코드 {code}: robot_heading(N/E/S/W) 또는 action({'/'.join(ACTIONS)}) 값 확인")
                continue
            if not self.in_bounds(cell):
                w.append(f"바코드 {code}: 막힌 칸 {cell} 이 지도 밖")
                continue

            dx, dy = heading_vec(HEADING_DEG[heading])
            stand = (cell[0] - round(dx), cell[1] - round(dy))
            if not self.in_bounds(stand):
                w.append(f"바코드 {code}: 로봇이 다가올 칸 {stand} 이 지도 밖")
                continue

            if action in ("left", "right", "back"):
                turn = {"left": -90, "right": 90, "back": 180}[action]
                nx, ny = heading_vec(HEADING_DEG[heading] + turn)
                nxt = (stand[0] + round(nx), stand[1] + round(ny))
                if not self.in_bounds(nxt):
                    w.append(f"바코드 {code}: '{action}' 하면 지도 밖 {nxt} 으로 감")

        for name in ("focal_px",):
            if float(self.camera.get(name, 0)) <= 0:
                w.append(f"camera.{name} 값이 0 이하")
