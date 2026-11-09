#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_map.py

로봇 없이 노트북에서 좌표 매핑 전체를 확인하는 시뮬레이션
- 위치 추적(데드레커닝), 바코드 거리 계산과 위치 보정, 회피 동작 순서
- YOLO 위험물 좌표 변환, 경로 찾기, 지도 저장, 지도 웹 서버

실행:
  pip3 install zxing-cpp "python-barcode[images]" opencv-python numpy
  python3 make_barcodes.py
  python3 test_map.py
"""

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import map_web  # noqa: E402
from map_config import HEADING_DEG, Config, heading_vec  # noqa: E402
from map_core import HazardMapper  # noqa: E402

failed = []


def check(name, ok, detail=""):
    print(("  [OK]   " if ok else "  [실패] ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failed.append(name)


def barcode_frame(cfg, code, distance, offset_px=0):
    """라벨을 distance(m) 거리에서 본 것처럼 640x480 화면에 붙임"""
    label = cv2.imread(os.path.join(HERE, "labels", f"{code}.png"))
    w = int(float(cfg.camera["focal_px"]) * 0.20 / distance)
    h = int(label.shape[0] * w / label.shape[1])
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_AREA)
    label = cv2.GaussianBlur(label, (3, 3), 0)
    frame = np.random.default_rng(0).integers(60, 120, (480, 640, 3), dtype=np.uint8)
    x = 320 - w // 2 + offset_px
    y = 120
    x0, x1 = max(0, x), min(640, x + w)
    frame[y:y + h, x0:x1] = label[:480 - y, x0 - x:x1 - x]
    return frame


def cone_detection(cfg, robot, target, height_m=0.30):
    """target(m) 위치에 있는 교통콘을 robot 자세에서 본 YOLO 박스"""
    dx, dy = target[0] - robot["x"], target[1] - robot["y"]
    distance = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dx, -dy)) - robot["heading"]
    f = float(cfg.camera["focal_px"])
    cx = 320 + f * math.tan(math.radians(angle))
    box_h = f * height_m / distance
    return (int(cx - box_h / 3), int(300 - box_h / 2), int(cx + box_h / 3), int(300 + box_h / 2),
            "traffic_cone", 0.8)


def main():
    if not os.path.isdir(os.path.join(HERE, "labels")):
        print("labels/ 가 없습니다. 먼저 python3 make_barcodes.py 를 실행하세요.")
        return 1

    tmp = tempfile.mkdtemp(prefix="hazard_map_test_")
    try:
        return run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(tmp):
    cfg = Config()
    check("설정 파일 경고 없음", not cfg.warnings, "; ".join(cfg.warnings))

    # 테스트에서는 회피 동작 시간을 짧게
    cfg.avoid.update({"stop_sec": 0.01, "back_sec": 0.05, "refresh_sec": 0.02})
    cfg.motion["yaw_deg_per_sec"] = 180.0

    pan = {"deg": 0.0}
    m = HazardMapper(cfg, state_dir=tmp, pan_provider=lambda: pan["deg"], async_barcodes=False)

    commands = []
    lock = threading.Lock()
    state = {"motion": "stop"}

    def robot_ctrl(c):
        with lock:
            commands.append(c)
            state["motion"] = c

    def stop():
        with lock:
            commands.append("STOP")
            state["motion"] = "stop"

    def simulate_motion(until):
        """서버 카메라 루프처럼 현재 주행 상태로 위치를 계속 갱신"""
        last = time.monotonic()
        while not until():
            time.sleep(0.005)
            now = time.monotonic()
            m.pose.update(state["motion"], now - last)
            last = now

    m.attach_controls(robot_ctrl, stop)

    print("\n1) 위치 추적")
    start = m.pose.snapshot()
    check("출발 칸 (1,6), 북쪽", m.pose.cell() == (1, 6) and start["heading"] == 0.0)
    for _ in range(30):
        m.pose.update("forward", 0.1)          # 0.3m/s x 3초 = 0.9m = 3칸
    check("3초 전진하면 3칸 위 (1,3)", m.pose.cell() == (1, 3), str(m.pose.cell()))
    yaw = cfg.motion["yaw_deg_per_sec"]
    m.pose.update("right", 90.0 / yaw)
    check("오른쪽 90도 회전 후 동쪽", abs(m.pose.snapshot()["heading"] - 90) < 1)
    m.pose.update("backleft", 90.0 / yaw)
    check("후진 왼쪽 조향이면 차체는 시계 방향으로 돔", abs(m.pose.snapshot()["heading"] - 180) < 1)
    m.pose.reset()

    print("\n2) 바코드 거리 계산 / 위치 보정")
    # 추정으로는 0.75m 왔다고 생각하지만, 실제로는 0.9m 온 상황 (추정 오차 15cm)
    for _ in range(25):
        m.pose.update("forward", 0.1)
    face = cfg.barcode_face("1100")
    true_robot = (face[0], face[1] + 0.45)        # 바코드 0.45m 앞
    frame = barcode_frame(cfg, "1100", 0.45)
    obs = []
    for _ in range(int(cfg.barcode["detect_every_n_frames"]) * 2):
        obs += m.barcodes.process(frame, 0.0, time.monotonic())
    check("바코드 1100 인식", bool(obs))
    d = obs[0]["distance"] if obs else 0
    check("거리 계산 0.45m ±10%", abs(d - 0.45) / 0.45 < 0.10, f"{d:.3f}m")

    snap = m.pose.snapshot()
    err_before = math.hypot(snap["x"] - true_robot[0], snap["y"] - true_robot[1])

    m.barcodes.hits.clear()
    m.barcodes.last_trigger.clear()
    commands.clear()
    cfg.avoid["stop_sec"] = 0.3          # 위치를 확인할 시간
    for _ in range(int(cfg.barcode["detect_every_n_frames"]) * 2):
        m.on_frame(frame.copy(), "forward")
    snap = m.pose.snapshot()
    err = math.hypot(snap["x"] - true_robot[0], snap["y"] - true_robot[1])
    check("막힌 칸 (1,1) 표시", (1, 1) in m.grid.blocked)
    check("로봇 위치가 실제 위치 ±5cm 로 보정", err < 0.05,
          f"보정 전 {err_before * 100:.1f}cm -> 보정 후 {err * 100:.1f}cm")

    if m.avoider.thread:
        sim = threading.Thread(target=simulate_motion, args=(lambda: not m.avoider.busy,))
        sim.start()
        m.avoider.thread.join(5)
        sim.join(2)
    cfg.avoid["stop_sec"] = 0.01

    print("\n3) 회피 동작")
    seq = [c for i, c in enumerate(commands) if i == 0 or c != commands[i - 1]]
    check("정지 -> 후진 -> 정지 -> 우회전 -> 정지 순서", seq == ["STOP", "backward", "STOP", "right", "STOP"], str(seq))
    check("우회전 명령을 계속 다시 보냄 (PC 서버 1.2초 안전정지 대비)", commands.count("right") >= 2)
    check("회피 후 방향 동쪽(90도)으로 정리", m.pose.snapshot()["heading"] == 90.0, str(m.pose.snapshot()["heading"]))

    m.on_frame(frame.copy(), "stop")
    m.on_frame(frame.copy(), "stop")
    m.on_frame(frame.copy(), "stop")
    check("같은 바코드는 cooldown 동안 다시 회피 안 함", not m.avoider.busy)

    # 사용자 조작으로 중단
    cfg.avoid["back_sec"] = 2.0
    m.avoider.start("left", "test")
    time.sleep(0.05)
    m.avoider.cancel()
    m.avoider.thread.join(2)
    check("사용자 조작이 들어오면 회피 즉시 중단", not m.avoider.busy)
    cfg.avoid["back_sec"] = 0.05

    print("\n4) 카메라를 옆으로 돌린 상태")
    m.reset()
    m.barcodes.hits.clear()
    m.barcodes.last_trigger.clear()
    pan["deg"] = 40.0
    for _ in range(int(cfg.barcode["detect_every_n_frames"]) * 2):
        m.on_frame(barcode_frame(cfg, "1101", 0.45), "stop")
    check("목을 40도 돌리면 막힌 칸만 표시하고 회피는 안 함",
          (8, 2) in m.grid.blocked and not m.avoider.busy)
    pan["deg"] = 0.0

    print("\n4-1) 기준점 바코드 (action: none)")
    m.reset()
    m.barcodes.hits.clear()
    m.barcodes.last_trigger.clear()
    cfg.barcodes["1103"]["action"] = "none"
    face = cfg.barcode_face("1103")                    # (4,4) 칸 동쪽 면, 로봇은 서쪽을 봄
    expect = (face[0] + 0.5, face[1])
    for _ in range(int(cfg.barcode["detect_every_n_frames"]) * 2):
        m.on_frame(barcode_frame(cfg, "1103", 0.5), "stop")
    snap = m.pose.snapshot()
    err = math.hypot(snap["x"] - expect[0], snap["y"] - expect[1])
    check("기준점 바코드: 막힌 칸 표시 안 함, 회피 안 함", (4, 4) not in m.grid.blocked and not m.avoider.busy)
    check("기준점 바코드: 위치/방향 보정", err < 0.05 and snap["heading"] == 270.0,
          f"오차 {err * 100:.1f}cm, 방향 {snap['heading']:.0f}")
    cfg.barcodes["1103"]["action"] = "left"

    print("\n5) YOLO 위험물 좌표")
    m.reset()
    robot = m.pose.snapshot()                       # (1,6) 가운데, 북쪽
    cone = (robot["x"] + 0.3, robot["y"] - 1.2)    # 1.2m 앞, 0.3m 오른쪽
    det = cone_detection(cfg, robot, cone)
    for _ in range(int(cfg.hazards["confirm_hits"])):
        m.on_inference([det], (480, 640, 3), robot, 0.0)
    hz = m.hazards.confirmed()
    check("교통콘 확정", len(hz) == 1)
    if hz:
        e = math.hypot(hz[0]["x"] - cone[0], hz[0]["y"] - cone[1])
        check("교통콘 위치 ±10cm", e < 0.10, f"오차 {e * 100:.1f}cm, 칸 {hz[0]['cell']}")

    turning = dict(robot, motion="left")
    m.on_inference([cone_detection(cfg, robot, (0.4, 0.4))] * 5, (480, 640, 3), turning, 0.0)
    check("회전 중 탐지는 지도에 안 넣음", len(m.hazards.confirmed()) == 1)

    # 카메라를 오른쪽으로 30도 돌린 채 본 물체
    side = (robot["x"] + 1.0 * math.sin(math.radians(30)), robot["y"] - 1.0 * math.cos(math.radians(30)))
    look = dict(robot, heading=robot["heading"] + 30)
    det2 = list(cone_detection(cfg, look, side))
    det2[4] = "fire_extinguisher"
    det2[1] = int(300 - float(cfg.camera["focal_px"]) * 0.40 / 1.0 / 2)
    det2[3] = int(300 + float(cfg.camera["focal_px"]) * 0.40 / 1.0 / 2)
    for _ in range(3):
        m.on_inference([tuple(det2)], (480, 640, 3), robot, 30.0 / float(cfg.camera["pan_sign"]))
    ext = [h for h in m.hazards.confirmed() if h["class"] == "fire_extinguisher"]
    ok = bool(ext) and math.hypot(ext[0]["x"] - side[0], ext[0]["y"] - side[1]) < 0.10
    check("목을 돌린 상태로 본 소화기 위치 ±10cm", ok)

    print("\n6) 경로 찾기")
    start_cell = m.pose.cell()
    goal = (1, 0)
    cone_cell = tuple(hz[0]["cell"]) if hz else None
    path = m.grid.find_path(start_cell, goal, m.hazards.confirmed())
    check("경로 있음", path is not None)
    check("교통콘 칸을 피해 감", path is not None and cone_cell is not None and list(cone_cell) not in path,
          f"콘 {cone_cell}")

    print("\n7) 저장 / 웹 서버")
    m.save()
    with open(os.path.join(tmp, "map_state.json"), encoding="utf-8") as f:
        saved = json.load(f)
    check("map_state.json 저장", saved["robot"]["cell"] == list(start_cell) and len(saved["hazards"]) == 2)
    check("map.png 저장", os.path.exists(os.path.join(tmp, "map.png")))

    web = map_web.start(m, port=0, host="127.0.0.1")
    port = web.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        data = json.load(urllib.request.urlopen(base + "/api/map", timeout=3))
        check("GET /api/map", data["grid_w"] == cfg.grid_w and "hazards" in data)
        html = urllib.request.urlopen(base + "/map", timeout=3).read().decode()
        check("GET /map 지도 페이지", "HazardScout" in html)

        req = urllib.request.Request(base + "/api/map/goal", data=b'{"cell":[1,0]}', method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=3).read()
        data = json.load(urllib.request.urlopen(base + "/api/map", timeout=3))
        check("목표 지정 후 경로 표시", data["goal"] == [1, 0] and data.get("path"))

        bad = urllib.request.Request(base + "/api/map/reset", data=b"{}", method="POST",
                                     headers={"Origin": "http://evil.example"})
        try:
            urllib.request.urlopen(bad, timeout=3)
            check("다른 사이트에서 온 초기화 요청 거부", False)
        except urllib.error.HTTPError as e:
            check("다른 사이트에서 온 초기화 요청 거부", e.code == 403)
    finally:
        web.shutdown()

    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    frame = barcode_frame(cfg, "1101", 0.5)
    m._draw(frame, time.monotonic())
    cv2.imwrite(os.path.join(out, "sim_frame.png"), frame)
    cv2.imwrite(os.path.join(out, "sim_map.png"), m.render(cell_px=40))
    print("\n화면 저장:", os.path.join(out, "sim_frame.png"), os.path.join(out, "sim_map.png"))

    print("\n통과" if not failed else f"\n실패 {len(failed)}개: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
