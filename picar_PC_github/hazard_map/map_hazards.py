#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_hazards.py

YOLO(Safety7) 탐지 결과를 지도 좌표로 바꾸고, 여러 번 본 것만 지도에 올린다.

1) 거리: 실물 높이를 알고 있으므로
       거리(m) = focal_px × 실물 높이(m) / 박스 높이(px)
   (박스가 화면 위/아래 끝에 닿아 잘린 경우는 높이를 믿을 수 없어서 버림)
2) 방향: 로봇 방향 + 목(카메라) 좌우 각도 + 화면 속 좌우 각도
3) 위치: 로봇 위치에서 그 방향으로 거리만큼 간 지점
4) 합치기: 같은 종류가 merge_radius_m 안에서 다시 보이면 같은 물체로 보고
   위치를 평균. confirm_hits 번 이상 보이면 '확정'으로 지도에 표시
   확정 안 된 후보는 forget_sec 동안 안 보이면 지움

로봇이 회전하는 중에는 방향 추정이 부정확해서 탐지를 지도에 넣지 않는다.
"""

import math
import threading

from map_config import heading_vec

EDGE_PX = 3


class HazardTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.items = []
        self.next_id = 1

    def reset(self):
        with self.lock:
            self.items = []
            self.next_id = 1

    def project(self, det, frame_shape, pose, pan_deg):
        """탐지 1개 -> (x, y, 거리) 또는 None"""
        x1, y1, x2, y2, name, conf = det[:6]
        hz = self.cfg.hazards
        spec = hz["classes"].get(str(name))
        if spec is None or float(conf) < float(hz["min_confidence"]):
            return None

        frame_h, frame_w = frame_shape[:2]
        if y1 <= EDGE_PX or y2 >= frame_h - EDGE_PX:
            return None

        box_h = max(1.0, float(y2 - y1))
        cam = self.cfg.camera
        focal = float(cam["focal_px"]) * frame_w / float(cam["width"])

        distance = focal * float(spec["height_m"]) / box_h
        if distance < 0.15 or distance > float(hz["max_range_m"]):
            return None

        cx = (x1 + x2) / 2.0
        bearing = math.degrees(math.atan2(cx - frame_w / 2.0, focal))
        bearing += float(cam.get("pan_sign", 1)) * pan_deg

        dx, dy = heading_vec(pose["heading"] + bearing)
        return pose["x"] + dx * distance, pose["y"] + dy * distance, distance

    def observe(self, detections, frame_shape, pose, pan_deg, now):
        """YOLO 추론 1회 결과를 반영. 새로 확정된 물체 목록을 돌려줌"""
        if pose.get("motion") not in ("stop", "forward", "backward"):
            return []

        hz = self.cfg.hazards
        radius = float(hz["merge_radius_m"])
        confirmed_now = []

        with self.lock:
            for det in detections:
                try:
                    p = self.project(det, frame_shape, pose, pan_deg)
                except Exception:
                    p = None
                if p is None:
                    continue

                x, y, distance = p
                name, conf = str(det[4]), float(det[5])

                best = None
                best_d = radius
                for item in self.items:
                    if item["class"] != name:
                        continue
                    d = math.hypot(item["x"] - x, item["y"] - y)
                    if d <= best_d:
                        best, best_d = item, d

                if best is None:
                    best = {
                        "id": self.next_id,
                        "class": name,
                        "x": x,
                        "y": y,
                        "hits": 0,
                        "conf": 0.0,
                        "confirmed": False,
                        "first_seen": now,
                    }
                    self.next_id += 1
                    self.items.append(best)

                # 가까이서 본 결과일수록 거리 오차가 작아서 가중치를 크게 준다
                w = 1.0 / max(0.3, distance)
                total = best.get("_w", 0.0) + w
                best["x"] = (best["x"] * best.get("_w", 0.0) + x * w) / total
                best["y"] = (best["y"] * best.get("_w", 0.0) + y * w) / total
                best["_w"] = total
                best["hits"] += 1
                best["conf"] = max(best["conf"], conf)
                best["last_seen"] = now
                best["last_distance"] = distance

                if not best["confirmed"] and best["hits"] >= int(hz["confirm_hits"]):
                    best["confirmed"] = True
                    confirmed_now.append(dict(best))

            forget = float(hz["forget_sec"])
            self.items = [
                it for it in self.items
                if it["confirmed"] or now - it.get("last_seen", now) <= forget
            ]

        return confirmed_now

    def confirmed(self):
        out = []
        with self.lock:
            for it in self.items:
                if not it["confirmed"]:
                    continue
                spec = self.cfg.hazards["classes"].get(it["class"], {})
                out.append({
                    "id": it["id"],
                    "class": it["class"],
                    "name_ko": spec.get("name_ko", it["class"]),
                    "blocks_path": bool(spec.get("blocks_path", False)),
                    "x": round(it["x"], 3),
                    "y": round(it["y"], 3),
                    "cell": list(self.cfg.cell_of(it["x"], it["y"])),
                    "hits": it["hits"],
                    "conf": round(it["conf"], 2),
                    "last_seen": it.get("last_seen"),
                })
        return out
