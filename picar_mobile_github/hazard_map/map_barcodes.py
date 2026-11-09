#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_barcodes.py

카메라 프레임에서 바코드를 읽고, 거리/방향을 계산한다.

거리 계산 (핀홀 카메라 모델)
    거리(m) = focal_px × 바코드 막대 실제 폭(m) / 화면 속 막대 폭(px)
방향 계산
    카메라 기준 각도 = atan((바코드 중심 x - 화면 중심 x) / focal_px)   (+ 는 오른쪽)
    로봇 기준 각도   = 카메라 기준 각도 + 목(카메라) 좌우 각도

인식 엔진 (있는 것을 위에서부터 사용)
    1) zxing-cpp  : ~/picar-yolo/bin/pip install zxing-cpp  (권장)
    2) pyzbar     : sudo apt install libzbar0 + pip install pyzbar
    3) OpenCV QR  : 추가 설치 없음, QR 코드만
"""

import math

import cv2

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None


class BarcodeReader:
    def __init__(self):
        if zxingcpp is not None:
            self.backend = "zxing-cpp"
        elif pyzbar is not None:
            self.backend = "pyzbar"
        else:
            self.backend = "opencv-qr"
            self.qr = cv2.QRCodeDetector()

    def read(self, frame_bgr):
        """[(text, (x1, y1, x2, y2)), ...]"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out = []

        if self.backend == "zxing-cpp":
            for r in zxingcpp.read_barcodes(gray):
                p = r.position
                xs = (p.top_left.x, p.top_right.x, p.bottom_right.x, p.bottom_left.x)
                ys = (p.top_left.y, p.top_right.y, p.bottom_right.y, p.bottom_left.y)
                out.append((r.text.strip(), (min(xs), min(ys), max(xs), max(ys))))

        elif self.backend == "pyzbar":
            for r in pyzbar.decode(gray):
                x, y, w, h = r.rect
                out.append((r.data.decode(errors="ignore").strip(), (x, y, x + w, y + h)))

        else:
            ok, texts, points, _ = self.qr.detectAndDecodeMulti(gray)
            if ok:
                for text, pts in zip(texts, points):
                    if text:
                        xs, ys = pts[:, 0], pts[:, 1]
                        out.append((text.strip(), (int(xs.min()), int(ys.min()),
                                                   int(xs.max()), int(ys.max()))))
        return out


class BarcodeWatcher:
    """
    프레임마다 바코드를 읽어서 관측 결과를 돌려준다.
    - min_hits 프레임 연속으로 보여야 확정 (오인식 방지)
    - trigger_distance_m 안으로 들어오면 회피 동작 대상 (cooldown 동안 한 번만)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reader = BarcodeReader()
        self.hits = {}
        self.last_trigger = {}
        self.last_boxes = []

    def distance_and_bearing(self, box, frame_w, pan_deg):
        cam = self.cfg.camera
        x1, _, x2, _ = box
        width = max(1, x2 - x1)
        focal = float(cam["focal_px"]) * frame_w / float(cam["width"])
        distance = focal * float(self.cfg.barcode["bar_width_m"]) / width
        cx = (x1 + x2) / 2.0
        bearing = math.degrees(math.atan2(cx - frame_w / 2.0, focal))
        bearing += float(cam.get("pan_sign", 1)) * pan_deg
        return distance, bearing

    def process(self, frame_bgr, pan_deg, now):
        """
        관측 목록을 돌려준다. 확정된 바코드만 포함.
          {"code", "distance", "bearing", "pan_ok", "trigger"}
        몇 프레임마다 부를지는 map_core 가 정한다 (barcode.detect_every_n_frames)
        """
        bc = self.cfg.barcode

        try:
            found = self.reader.read(frame_bgr)
        except Exception as e:
            print("[바코드] 인식 오류:", repr(e))
            return []

        self.last_boxes = found
        frame_w = frame_bgr.shape[1]
        seen = set()
        observations = []

        for code, box in found:
            if code not in self.cfg.barcodes:
                continue

            distance, bearing = self.distance_and_bearing(box, frame_w, pan_deg)
            if distance > float(bc["max_read_distance_m"]):
                continue

            seen.add(code)
            self.hits[code] = self.hits.get(code, 0) + 1
            if self.hits[code] < int(bc["min_hits"]):
                continue

            pan_ok = abs(pan_deg) <= float(bc["max_pan_deg"])
            trigger = (
                pan_ok
                and distance <= float(bc["trigger_distance_m"])
                and now - self.last_trigger.get(code, -1e9) >= float(bc["cooldown_sec"])
            )
            if trigger:
                self.last_trigger[code] = now

            observations.append({
                "code": code,
                "distance": distance,
                "bearing": bearing,
                "pan_ok": pan_ok,
                "trigger": trigger,
            })

        for code in list(self.hits):
            if code not in seen:
                self.hits[code] = 0

        return observations
