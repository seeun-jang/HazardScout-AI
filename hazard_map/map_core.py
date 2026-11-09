#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_core.py

좌표 매핑 전체를 묶는 모듈
- on_frame()     : 카메라 스레드에서 매 프레임. 위치 갱신 + 미니맵
                   바코드 읽기는 별도 스레드 (파이에서 한 번에 수십~수백 ms 라 영상이 멈추지 않게)
- on_inference() : YOLO 스레드에서 추론 1회마다. 위험물 위치를 지도에 반영
- to_dict()      : 웹/대시보드용 지도 데이터
- 지도 상태는 map_state.json, 지도 그림은 map.png 로 저장 (1초에 한 번까지)
"""

import json
import os
import threading
import time
from collections import deque

import cv2

from map_avoider import Avoider
from map_barcodes import BarcodeWatcher
from map_config import HERE, HEADING_DEG, Config, heading_vec, nearest_heading_name
from map_grid import GridMap, overlay
from map_hazards import HazardTracker
from map_pose import Pose

ACTION_TEXT = {"left": "LEFT", "right": "RIGHT", "back": "U-TURN", "stop": "STOP"}
CHECKPOINT = "none"     # 막힌 곳이 아니라 위치 보정만 하는 기준점 바코드
SAVE_INTERVAL = 1.0


class HazardMapper:
    def __init__(self, cfg=None, state_dir=None, pan_provider=None, async_barcodes=True):
        """async_barcodes=False 는 테스트용 (on_frame 안에서 바로 바코드 처리)"""
        self.cfg = cfg or Config()
        self.state_dir = state_dir or HERE
        self.pan_provider = pan_provider

        self.pose = Pose(self.cfg)
        self.grid = GridMap(self.cfg)
        self.hazards = HazardTracker(self.cfg)
        self.barcodes = BarcodeWatcher(self.cfg)
        self.avoider = None

        self.events = deque(maxlen=30)
        self.banner = None
        self.banner_until = 0.0
        self.last_frame_t = None
        self.last_saved = 0.0
        self.dirty = True
        self.save_lock = threading.Lock()
        self.goal = None

        self.frame_count = 0
        self.async_barcodes = async_barcodes
        self.bc_frame = None
        self.bc_lock = threading.Lock()
        self.bc_event = threading.Event()
        self.bc_stop = threading.Event()
        self.bc_busy = False
        self.bc_thread = None

        self.grid.visit(self.pose.cell())
        print(f"[지도] 바코드 인식 엔진: {self.barcodes.reader.backend}", flush=True)
        print(f"[지도] 격자 {self.cfg.grid_w}x{self.cfg.grid_h}, 칸 {self.cfg.cell_m}m, "
              f"바코드 {len(self.cfg.barcodes)}개, 위험물 {len(self.cfg.hazards['classes'])}종", flush=True)

    # -----------------------------------------------------
    # 서버 연결
    # -----------------------------------------------------

    def attach_controls(self, robot_ctrl, stop):
        self.avoider = Avoider(self.cfg, robot_ctrl, stop, on_done=self._avoid_done)

    def pan(self):
        if self.pan_provider is None:
            return 0.0
        try:
            return float(self.pan_provider() or 0.0)
        except Exception:
            return 0.0

    def _log(self, kind, text, **extra):
        item = {"time": time.time(), "type": kind, "text": text}
        item.update(extra)
        self.events.appendleft(item)
        self.dirty = True
        print(f"[지도] {text}", flush=True)

    # -----------------------------------------------------
    # 카메라 스레드
    # -----------------------------------------------------

    def on_frame(self, frame_bgr, motion):
        now = time.monotonic()
        dt = 0.0 if self.last_frame_t is None else now - self.last_frame_t
        self.last_frame_t = now

        before = self.pose.cell()
        self.pose.update(motion, dt)
        after = self.pose.cell()
        if after != before:
            self.grid.visit(after)
            self.dirty = True

        self.frame_count += 1
        if self.frame_count % max(1, int(self.cfg.barcode["detect_every_n_frames"])) == 0:
            if self.async_barcodes:
                self._submit_barcode_frame(frame_bgr)
            else:
                self._read_barcodes(frame_bgr)

        self._draw(frame_bgr, now)
        self._maybe_save(now)
        return frame_bgr

    def _read_barcodes(self, frame_bgr):
        now = time.monotonic()
        for obs in self.barcodes.process(frame_bgr, self.pan(), now):
            self._handle_barcode(obs, now)

    def _submit_barcode_frame(self, frame_bgr):
        if self.bc_thread is None:
            self.bc_thread = threading.Thread(target=self._barcode_worker, daemon=True,
                                              name="HazardMapBarcode")
            self.bc_thread.start()
        if self.bc_busy:
            return          # 이전 프레임을 아직 읽는 중이면 이번 프레임은 건너뜀
        with self.bc_lock:
            self.bc_frame = frame_bgr.copy()     # 미니맵을 그리기 전의 깨끗한 프레임
        self.bc_event.set()

    def _barcode_worker(self):
        while not self.bc_stop.is_set():
            if not self.bc_event.wait(0.2):
                continue
            self.bc_event.clear()
            with self.bc_lock:
                frame, self.bc_frame = self.bc_frame, None
            if frame is None:
                continue
            self.bc_busy = True
            try:
                self._read_barcodes(frame)
            except Exception as e:
                print("[지도] 바코드 처리 오류:", repr(e), flush=True)
            finally:
                self.bc_busy = False

    def close(self):
        self.bc_stop.set()
        self.bc_event.set()
        if self.bc_thread is not None:
            self.bc_thread.join(timeout=1.0)

    def _handle_barcode(self, obs, now):
        code = obs["code"]
        m = self.cfg.barcodes[code]
        cell = tuple(m["cell"])
        checkpoint = m["action"] == CHECKPOINT

        if checkpoint:
            pass
        elif self.grid.block(cell, code, m.get("note", "")):
            self._log("barcode", f"바코드 {code}: 칸 {cell} 막힘 ({m.get('note', '')})",
                      code=code, cell=list(cell))

        if obs["pan_ok"]:
            self._correct_pose(code, obs, now)

        if obs["trigger"] and not checkpoint:
            action = m["action"]
            self.banner = f"NO PATH -> {ACTION_TEXT.get(action, action)}"
            self.banner_until = now + 3.0
            self._log("avoid", f"바코드 {code} {obs['distance']:.2f}m: {action}",
                      code=code, action=action)
            if self.cfg.barcode.get("auto_avoid", True) and self.avoider is not None:
                self.avoider.start(action, code)

    def _correct_pose(self, code, obs, now):
        """바코드 위치를 기준으로 로봇 위치/방향을 다시 맞춤"""
        m = self.cfg.barcodes[code]
        face_x, face_y = self.cfg.barcode_face(code)
        barcode_heading = HEADING_DEG[m["robot_heading"]]

        snap = self.pose.snapshot()
        # 1D 바코드는 라벨 앞면을 대략 마주 봐야 읽힘 -> 읽혔다면 로봇은 바코드 방향을 보고 있음.
        # 시간으로 어림잡은 방향은 수동 회전 몇 번이면 크게 틀어지므로 바코드 쪽을 믿는다.
        heading = barcode_heading

        dx, dy = heading_vec(heading + obs["bearing"])
        x = face_x - dx * obs["distance"]
        y = face_y - dy * obs["distance"]

        if not self.cfg.in_bounds(self.cfg.cell_of(x, y)):
            return

        moved = ((x - snap["x"]) ** 2 + (y - snap["y"]) ** 2) ** 0.5
        self.pose.set(x, y, heading, now)
        self.grid.visit(self.cfg.cell_of(x, y))
        self.dirty = True
        if moved > self.cfg.cell_m / 2:
            self._log("pose", f"바코드 {code} 기준 위치 보정 ({moved:.2f}m 이동)", code=code)

    def _avoid_done(self, action, finished):
        if finished and action != "stop":
            self.pose.snap_heading()
        self.dirty = True

    # -----------------------------------------------------
    # YOLO 스레드
    # -----------------------------------------------------

    def on_inference(self, detections, frame_shape, pose_snapshot, pan):
        if not detections:
            return
        new = self.hazards.observe(detections, frame_shape, pose_snapshot, pan, time.monotonic())
        for item in new:
            spec = self.cfg.hazards["classes"].get(item["class"], {})
            cell = self.cfg.cell_of(item["x"], item["y"])
            self._log("hazard", f"{spec.get('name_ko', item['class'])} 발견: 칸 {cell}",
                      cls=item["class"], cell=list(cell))

    # -----------------------------------------------------
    # 표시 / 저장
    # -----------------------------------------------------

    def render(self, cell_px=16):
        hazards = self.hazards.confirmed()
        path = None
        if self.goal is not None:
            path = self.grid.find_path(self.pose.cell(), self.goal, hazards)
        return self.grid.render(self.pose.snapshot(), hazards, cell_px=cell_px, path=path)

    def _draw(self, frame, now):
        for text, (x1, y1, x2, y2) in self.barcodes.last_boxes:
            known = text in self.cfg.barcodes
            color = (0, 0, 255) if known else (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)

        if self.banner and now < self.banner_until:
            cv2.putText(frame, self.banner, (12, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

        snap = self.pose.snapshot()
        cell = self.cfg.cell_of(snap["x"], snap["y"])
        status = (f"MAP cell({cell[0]},{cell[1]}) "
                  f"{nearest_heading_name(snap['heading'])} {snap['heading']:.0f}deg")
        if self.avoider is not None and self.avoider.busy:
            status += " AVOIDING"
        fh = frame.shape[0]
        cv2.putText(frame, status, (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)

        overlay(frame, self.render(cell_px=14))

    def to_dict(self):
        snap = self.pose.snapshot()
        cfg = self.cfg
        hazards = self.hazards.confirmed()
        data = {
            "time": time.time(),
            "cell_size_m": cfg.cell_m,
            "grid_w": cfg.grid_w,
            "grid_h": cfg.grid_h,
            "robot": {
                "x": round(snap["x"], 3),
                "y": round(snap["y"], 3),
                "heading": round(snap["heading"], 1),
                "heading_name": nearest_heading_name(snap["heading"]),
                "cell": list(cfg.cell_of(snap["x"], snap["y"])),
                "motion": snap["motion"],
                "avoiding": bool(self.avoider is not None and self.avoider.busy),
                # +는 오른쪽. 카메라를 오른쪽으로 돌렸는데 -가 나오면 camera.pan_sign 을 바꿀 것
                "camera_pan": round(float(cfg.camera.get("pan_sign", 1)) * self.pan(), 1),
            },
            "barcodes": [
                {"code": code, "cell": m["cell"], "robot_heading": m["robot_heading"],
                 "action": m["action"], "note": m.get("note", "")}
                for code, m in cfg.barcodes.items()
            ],
            "hazards": hazards,
            "events": list(self.events),
            "goal": list(self.goal) if self.goal else None,
        }
        data.update(self.grid.to_dict())
        if self.goal is not None:
            data["path"] = self.grid.find_path(data["robot"]["cell"], self.goal, hazards)
        return data

    def _maybe_save(self, now):
        """카메라 스레드에서 호출. 저장은 수십 ms 걸려서 별도 스레드로 넘김"""
        if not self.dirty or now - self.last_saved < SAVE_INTERVAL:
            return
        if self.save_lock.locked():
            return
        self.last_saved = now
        threading.Thread(target=self._save_now, daemon=True, name="HazardMapSave").start()

    def _save_now(self):
        if not self.save_lock.acquire(blocking=False):
            return
        try:
            self.dirty = False
            state = os.path.join(self.state_dir, "map_state.json")
            tmp = state + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, state)
            cv2.imwrite(os.path.join(self.state_dir, "map.png"), self.render(cell_px=40))
        except Exception as e:
            print("[지도] 저장 실패:", repr(e), flush=True)
        finally:
            self.save_lock.release()

    def save(self):
        """바로 저장 (종료할 때). 저장 중이면 끝날 때까지 기다림"""
        with self.save_lock:
            pass
        self._save_now()

    # -----------------------------------------------------
    # 조작
    # -----------------------------------------------------

    def reset(self):
        if self.avoider is not None:
            self.avoider.cancel_event.set()
        self.pose.reset()
        self.grid.reset()
        self.hazards.reset()
        self.events.clear()
        self.goal = None
        self.grid.visit(self.pose.cell())
        self._log("reset", "지도 초기화 (출발 위치로)")

    def set_goal(self, cell):
        if cell is None:
            self.goal = None
        else:
            cell = (int(cell[0]), int(cell[1]))
            self.goal = cell if self.cfg.in_bounds(cell) else None
        self.dirty = True
        return self.goal
