#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barcode_map.py

바코드 기반 위험 구역 좌표 매핑 모듈
- 현장에 붙인 바코드(Code128 / QR 등)를 카메라 프레임에서 인식
- markers.json 에서 바코드 ID -> (막힌 칸 좌표, 회피 방향) 을 찾음
- 격자 지도에 막힌 칸을 표시하고, 로봇 위치/방향을 바코드 기준으로 보정
- 바코드를 읽지 않는 동안에는 주행 시간으로 위치를 대략 추정 (데드레커닝)
- 카메라 화면 오른쪽 아래에 미니맵을 그려줌

좌표계:
  x 는 오른쪽(E)으로, y 는 아래쪽(S)으로 증가. (0,0) 은 지도 왼쪽 위 칸.

바코드 인식 라이브러리 (위에서부터 있는 것을 사용):
  1) zxing-cpp  : ~/picar-yolo/bin/pip install zxing-cpp   (권장, Code128/QR 모두 지원)
  2) pyzbar     : sudo apt install libzbar0 && pip3 install pyzbar
  3) OpenCV QR  : 추가 설치 없음, QR 코드만 지원
"""

import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

try:
    from pyzbar import pyzbar
except Exception:
    pyzbar = None


HERE = os.path.dirname(os.path.abspath(__file__))

FREE = 0
BLOCKED = 1

HEADINGS = ["N", "E", "S", "W"]

HEADING_VEC = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}

# action 별로 몇 번 오른쪽으로 90도 도는지
ACTION_TURNS = {
    "left": -1,
    "right": 1,
    "back": 2,
    "stop": 0,
}

ACTION_TEXT = {
    "left": "LEFT",
    "right": "RIGHT",
    "back": "U-TURN",
    "stop": "STOP",
}


def turn_heading(heading, turns):
    return HEADINGS[(HEADINGS.index(heading) + turns) % 4]


# =========================================================
# 바코드 인식
# =========================================================

class BarcodeReader:
    def __init__(self):
        if zxingcpp is not None:
            self.backend = "zxing-cpp"
        elif pyzbar is not None:
            self.backend = "pyzbar"
        else:
            self.backend = "opencv-qr"
            self.qr = cv2.QRCodeDetector()

        print("[바코드] 인식 엔진:", self.backend)

    def read(self, frame_bgr):
        """[(text, (x1, y1, x2, y2)), ...] 반환"""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        results = []

        if self.backend == "zxing-cpp":
            for r in zxingcpp.read_barcodes(gray):
                p = r.position
                xs = [p.top_left.x, p.top_right.x, p.bottom_right.x, p.bottom_left.x]
                ys = [p.top_left.y, p.top_right.y, p.bottom_right.y, p.bottom_left.y]
                results.append((r.text, (min(xs), min(ys), max(xs), max(ys))))

        elif self.backend == "pyzbar":
            for r in pyzbar.decode(gray):
                x, y, w, h = r.rect
                results.append((r.data.decode(errors="ignore"), (x, y, x + w, y + h)))

        else:
            ok, texts, points, _ = self.qr.detectAndDecodeMulti(gray)
            if ok:
                for text, pts in zip(texts, points):
                    if text:
                        xs, ys = pts[:, 0], pts[:, 1]
                        results.append((text, (int(xs.min()), int(ys.min()),
                                               int(xs.max()), int(ys.max()))))

        return results


# =========================================================
# 격자 지도
# =========================================================

class BarcodeMap:
    def __init__(
        self,
        config_path=None,
        state_path=None,
        min_hits=2,
        cooldown_sec=5.0,
        min_width_px=60,
        forward_m_per_sec=0.25,
        detect_every_n_frames=2,
    ):
        """
        min_hits            : 연속 몇 프레임 인식돼야 확정할지 (오인식 방지)
        cooldown_sec        : 같은 바코드를 다시 처리하기까지 기다리는 시간
        min_width_px        : 이 폭보다 작게 보이면 아직 멀다고 보고 무시
        forward_m_per_sec   : 전진 속도 (실측해서 바꿀 것). 데드레커닝용

        서버는 ~/Adeept_PiCar-Pro/Server 에서 실행되므로
        경로를 안 주면 이 파일이 있는 폴더의 markers.json / map_state.json 을 쓴다.
        """
        if config_path is None:
            config_path = os.path.join(HERE, "markers.json")
        if state_path is None:
            state_path = os.path.join(HERE, "map_state.json")

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        self.cell_m = cfg["cell_size_m"]
        self.w = cfg["grid_w"]
        self.h = cfg["grid_h"]
        self.markers = cfg["markers"]
        self.start_cell = tuple(cfg.get("start_cell", [0, 0]))
        self.start_heading = cfg.get("start_heading", "N")

        self.state_path = state_path
        self.min_hits = min_hits
        self.cooldown_sec = cooldown_sec
        self.min_width_px = min_width_px
        self.forward_m_per_sec = forward_m_per_sec
        self.detect_every_n_frames = max(1, int(detect_every_n_frames))

        self.reader = BarcodeReader()
        self.lock = threading.Lock()

        self.frame_count = 0
        self.hits = {}
        self.last_handled = {}
        self.last_boxes = []
        self.last_event = None
        self.last_event_time = 0.0
        self.travel_m = 0.0

        self.reset()
        self._validate()

    # ---------------------------------------------------------
    # 상태
    # ---------------------------------------------------------

    def reset(self):
        with self.lock:
            self.grid = np.zeros((self.h, self.w), np.uint8)
            self.robot_cell = self.start_cell
            self.heading = self.start_heading
            self.visited = {self.start_cell}
            self.travel_m = 0.0

    def _in_bounds(self, cell):
        return 0 <= cell[0] < self.w and 0 <= cell[1] < self.h

    def _validate(self):
        """markers.json 실수 미리 잡기"""
        for code, m in self.markers.items():
            cell = tuple(m["cell"])
            heading = m["robot_heading"]
            action = m["action"]

            if not self._in_bounds(cell):
                print(f"[지도 경고] {code}: 칸 {cell} 이 지도 밖")
                continue
            if heading not in HEADING_VEC or action not in ACTION_TURNS:
                print(f"[지도 경고] {code}: robot_heading/action 값 확인")
                continue

            dx, dy = HEADING_VEC[heading]
            stand = (cell[0] - dx, cell[1] - dy)
            if not self._in_bounds(stand):
                print(f"[지도 경고] {code}: 로봇이 설 칸 {stand} 이 지도 밖")
                continue

            if action != "stop":
                new_heading = turn_heading(heading, ACTION_TURNS[action])
                nx, ny = HEADING_VEC[new_heading]
                nxt = (stand[0] + nx, stand[1] + ny)
                if not self._in_bounds(nxt):
                    print(f"[지도 경고] {code}: '{action}' 하면 지도 밖 {nxt} 으로 감")

    # ---------------------------------------------------------
    # 데드레커닝
    # ---------------------------------------------------------

    def update_motion(self, motion, dt):
        """
        카메라 루프에서 매 프레임 호출.
        motion 은 GUIServer 의 current_motion ("forward", "backward", ...)
        조향 회전 중에는 위치를 바꾸지 않는다 (회전은 바코드 동작에서 처리).
        """
        if motion not in ("forward", "backward"):
            return

        sign = 1 if motion == "forward" else -1

        with self.lock:
            self.travel_m += self.forward_m_per_sec * dt * sign

            while abs(self.travel_m) >= self.cell_m - 1e-6:
                step = 1 if self.travel_m > 0 else -1
                self.travel_m -= step * self.cell_m

                dx, dy = HEADING_VEC[self.heading]
                nxt = (self.robot_cell[0] + dx * step, self.robot_cell[1] + dy * step)

                # 지도 밖이나 막힌 칸으로는 추정 위치를 옮기지 않음
                if self._in_bounds(nxt) and self.grid[nxt[1], nxt[0]] == FREE:
                    self.robot_cell = nxt
                    self.visited.add(nxt)

    def apply_turn(self, action):
        """회피 동작이 끝난 뒤 호출. 로봇 방향만 바꾼다."""
        with self.lock:
            self.heading = turn_heading(self.heading, ACTION_TURNS.get(action, 0))
            self.travel_m = 0.0

    # ---------------------------------------------------------
    # 바코드 처리
    # ---------------------------------------------------------

    def process(self, frame_bgr):
        """
        카메라 프레임에서 바코드를 찾는다.
        새로 확정된 바코드가 있으면 이벤트 dict 를 반환, 없으면 None.

        이벤트 예:
          {"code": "1100", "blocked_cell": (1, 1), "robot_cell": (1, 2),
           "heading": "N", "action": "right", "note": "복도 끝 막힘"}
        """
        self.frame_count += 1
        if self.frame_count % self.detect_every_n_frames != 0:
            return None

        try:
            found = self.reader.read(frame_bgr)
        except Exception as e:
            print("[바코드] 인식 오류:", repr(e))
            return None

        self.last_boxes = found
        seen = set()
        event = None
        now = time.time()

        for text, (x1, y1, x2, y2) in found:
            code = text.strip()
            m = self.markers.get(code)
            if m is None:
                continue

            # 아직 멀리 있는 바코드는 무시 -> 가까이 왔을 때 처리
            if (x2 - x1) < self.min_width_px:
                continue

            seen.add(code)
            self.hits[code] = self.hits.get(code, 0) + 1

            if self.hits[code] < self.min_hits:
                continue
            if now - self.last_handled.get(code, 0) < self.cooldown_sec:
                continue

            self.last_handled[code] = now
            event = self._apply_marker(code, m)

        # 이번 프레임에 안 보인 바코드는 연속 카운트 초기화
        for code in list(self.hits):
            if code not in seen:
                self.hits[code] = 0

        return event

    def _apply_marker(self, code, m):
        cell = tuple(m["cell"])
        heading = m["robot_heading"]
        dx, dy = HEADING_VEC[heading]
        stand = (cell[0] - dx, cell[1] - dy)

        with self.lock:
            if self._in_bounds(cell):
                self.grid[cell[1], cell[0]] = BLOCKED

            # 바코드를 읽었다 = 막힌 칸 바로 앞에서 그 방향을 보고 있다 -> 위치 보정
            if self._in_bounds(stand):
                self.robot_cell = stand
                self.visited.add(stand)
            self.heading = heading
            self.travel_m = 0.0

        event = {
            "code": code,
            "blocked_cell": cell,
            "robot_cell": stand,
            "heading": heading,
            "action": m["action"],
            "note": m.get("note", ""),
            "time": time.time(),
        }

        self.last_event = event
        self.last_event_time = event["time"]

        print(f"[지도] {code} 인식: 칸 {cell} 막힘 / 로봇 {stand} {heading} / 동작 {m['action']}")
        self.save()
        return event

    # ---------------------------------------------------------
    # 경로 탐색 (AI 판단 / 자율주행용)
    # ---------------------------------------------------------

    def find_path(self, start, goal):
        """막힌 칸을 피해 start -> goal 최단 경로 [(x, y), ...]. 없으면 None"""
        start, goal = tuple(start), tuple(goal)
        with self.lock:
            grid = self.grid.copy()

        prev = {start: None}
        q = deque([start])

        while q:
            cur = q.popleft()
            if cur == goal:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]

            for dx, dy in HEADING_VEC.values():
                nxt = (cur[0] + dx, cur[1] + dy)
                if self._in_bounds(nxt) and grid[nxt[1], nxt[0]] == FREE and nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)

        return None

    # ---------------------------------------------------------
    # 저장 / 대시보드 전송용
    # ---------------------------------------------------------

    def to_dict(self):
        with self.lock:
            return {
                "cell_size_m": self.cell_m,
                "grid_w": self.w,
                "grid_h": self.h,
                "blocked": [[int(x), int(y)] for y, x in zip(*np.nonzero(self.grid))],
                "robot_cell": list(self.robot_cell),
                "heading": self.heading,
                "visited": [list(c) for c in sorted(self.visited)],
                "last_event": self.last_event,
            }

    def save(self):
        if not self.state_path:
            return
        try:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_path)
        except Exception as e:
            print("[지도] 저장 실패:", repr(e))

    # ---------------------------------------------------------
    # 화면 표시
    # ---------------------------------------------------------

    def draw(self, frame_bgr, cell_px=14, margin=10):
        """바코드 박스 + 안내 문구 + 오른쪽 아래 미니맵"""
        for text, (x1, y1, x2, y2) in self.last_boxes:
            known = text.strip() in self.markers
            color = (0, 0, 255) if known else (160, 160, 160)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_bgr, text, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        if self.last_event and time.time() - self.last_event_time < 3.0:
            msg = f"NO PATH -> {ACTION_TEXT[self.last_event['action']]}"
            cv2.putText(frame_bgr, msg, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

        with self.lock:
            grid = self.grid.copy()
            robot = self.robot_cell
            heading = self.heading
            visited = set(self.visited)

        fh, fw = frame_bgr.shape[:2]
        mw, mh = self.w * cell_px, self.h * cell_px
        ox, oy = fw - mw - margin, fh - mh - margin
        if ox < 0 or oy < 0:
            return frame_bgr

        roi = frame_bgr[oy:oy + mh, ox:ox + mw]
        overlay = np.full_like(roi, 40)

        for y in range(self.h):
            for x in range(self.w):
                p1 = (x * cell_px, y * cell_px)
                p2 = (p1[0] + cell_px - 1, p1[1] + cell_px - 1)
                if grid[y, x] == BLOCKED:
                    cv2.rectangle(overlay, p1, p2, (0, 0, 220), -1)
                elif (x, y) in visited:
                    cv2.rectangle(overlay, p1, p2, (90, 110, 90), -1)
                cv2.rectangle(overlay, p1, p2, (80, 80, 80), 1)

        cx = robot[0] * cell_px + cell_px // 2
        cy = robot[1] * cell_px + cell_px // 2
        dx, dy = HEADING_VEC[heading]
        cv2.circle(overlay, (cx, cy), cell_px // 2 - 1, (255, 200, 0), -1)
        cv2.line(overlay, (cx, cy), (cx + dx * cell_px, cy + dy * cell_px), (255, 255, 255), 2)

        frame_bgr[oy:oy + mh, ox:ox + mw] = cv2.addWeighted(overlay, 0.8, roi, 0.2, 0)
        return frame_bgr
