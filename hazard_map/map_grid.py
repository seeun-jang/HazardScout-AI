#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_grid.py

격자 지도: 바코드로 막힌 칸, 지나온 칸, 경로 찾기, 지도 그림
"""

import heapq
import threading

import cv2
import numpy as np

from map_config import heading_vec

# 지도 색 (BGR)
COLOR_BG = (40, 40, 40)
COLOR_LINE = (75, 75, 75)
COLOR_VISITED = (70, 105, 70)
COLOR_BLOCKED = (40, 40, 210)
COLOR_HAZARD_BLOCK = (0, 140, 255)
COLOR_HAZARD = (0, 215, 255)
COLOR_ROBOT = (255, 200, 0)
COLOR_PATH = (120, 255, 120)

# 영상에는 한글을 못 쓰므로 영문 약자
CLASS_SHORT = {
    "traffic_cone": "C",
    "fire_extinguisher": "F",
    "safety_helmet": "H",
    "safety_vest": "V",
    "safety_gloves": "G",
    "earmuffs": "E",
    "safety_goggles": "O",
}


class GridMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.blocked = {}      # (cx, cy) -> {"code", "note"}
            self.visited = set()

    def block(self, cell, code, note=""):
        cell = tuple(cell)
        with self.lock:
            if cell in self.blocked:
                return False
            self.blocked[cell] = {"code": code, "note": note}
            return True

    def visit(self, cell):
        if self.cfg.in_bounds(cell):
            with self.lock:
                self.visited.add(tuple(cell))

    def blocked_cells(self, hazards=()):
        with self.lock:
            cells = set(self.blocked)
        for h in hazards:
            if h.get("blocks_path"):
                cells.add(tuple(h["cell"]))
        return cells

    def find_path(self, start, goal, hazards=()):
        """막힌 칸(바코드 + 통행을 막는 위험물)을 피한 최단 경로. 없으면 None"""
        start, goal = tuple(start), tuple(goal)
        if not (self.cfg.in_bounds(start) and self.cfg.in_bounds(goal)):
            return None

        blocked = self.blocked_cells(hazards)
        blocked.discard(start)
        if goal in blocked:
            return None

        def h(c):
            return abs(c[0] - goal[0]) + abs(c[1] - goal[1])

        prev = {start: None}
        cost = {start: 0}
        heap = [(h(start), start)]

        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(list(cur))
                    cur = prev[cur]
                return path[::-1]

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + dx, cur[1] + dy)
                if not self.cfg.in_bounds(nxt) or nxt in blocked:
                    continue
                c = cost[cur] + 1
                if c < cost.get(nxt, 1e9):
                    cost[nxt] = c
                    prev[nxt] = cur
                    heapq.heappush(heap, (c + h(nxt), nxt))
        return None

    def to_dict(self):
        with self.lock:
            return {
                "blocked": [
                    {"cell": list(c), "code": v["code"], "note": v["note"]}
                    for c, v in sorted(self.blocked.items())
                ],
                "visited": [list(c) for c in sorted(self.visited)],
            }

    # -----------------------------------------------------
    # 그림
    # -----------------------------------------------------

    def render(self, pose, hazards, cell_px=16, path=None):
        """지도 그림 (BGR). 영상 미니맵과 map.png 에 같이 씀"""
        cfg = self.cfg
        w, h = cfg.grid_w * cell_px, cfg.grid_h * cell_px
        img = np.full((h, w, 3), COLOR_BG, np.uint8)

        with self.lock:
            visited = set(self.visited)
            blocked = set(self.blocked)

        for (cx, cy) in visited:
            cv2.rectangle(img, (cx * cell_px, cy * cell_px),
                          ((cx + 1) * cell_px - 1, (cy + 1) * cell_px - 1), COLOR_VISITED, -1)

        for (cx, cy) in blocked:
            cv2.rectangle(img, (cx * cell_px, cy * cell_px),
                          ((cx + 1) * cell_px - 1, (cy + 1) * cell_px - 1), COLOR_BLOCKED, -1)

        for x in range(cfg.grid_w + 1):
            cv2.line(img, (x * cell_px, 0), (x * cell_px, h), COLOR_LINE, 1)
        for y in range(cfg.grid_h + 1):
            cv2.line(img, (0, y * cell_px), (w, y * cell_px), COLOR_LINE, 1)

        if path:
            pts = [(int((c[0] + 0.5) * cell_px), int((c[1] + 0.5) * cell_px)) for c in path]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, COLOR_PATH, max(1, cell_px // 8))

        scale = cell_px / cfg.cell_m
        for hz in hazards:
            px, py = int(hz["x"] * scale), int(hz["y"] * scale)
            color = COLOR_HAZARD_BLOCK if hz.get("blocks_path") else COLOR_HAZARD
            r = max(3, cell_px // 3)
            cv2.circle(img, (px, py), r, color, -1)
            if cell_px >= 14:
                cv2.putText(img, CLASS_SHORT.get(hz["class"], "?"), (px - r + 1, py + r - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, cell_px / 40.0, (0, 0, 0), 1, cv2.LINE_AA)

        rx, ry = int(pose["x"] * scale), int(pose["y"] * scale)
        dx, dy = heading_vec(pose["heading"])
        r = max(3, cell_px // 2 - 1)
        cv2.circle(img, (rx, ry), r, COLOR_ROBOT, -1)
        cv2.line(img, (rx, ry), (int(rx + dx * r * 2), int(ry + dy * r * 2)), (255, 255, 255), 2)

        return img


def overlay(frame, img, margin=8, alpha=0.8, max_w=230, max_h=170):
    """frame 오른쪽 아래에 지도 그림을 반투명하게 붙임"""
    ih, iw = img.shape[:2]
    s = min(max_w / iw, max_h / ih, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
        ih, iw = img.shape[:2]

    fh, fw = frame.shape[:2]
    x, y = fw - iw - margin, fh - ih - margin
    if x < 0 or y < 0:
        return frame
    roi = frame[y:y + ih, x:x + iw]
    frame[y:y + ih, x:x + iw] = cv2.addWeighted(img, alpha, roi, 1 - alpha, 0)
    return frame

