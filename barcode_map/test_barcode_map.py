#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_barcode_map.py

로봇 없이 노트북에서 바코드 지도/회피 동작을 확인하는 시뮬레이션
- labels/ 의 라벨을 가짜 카메라 화면(640x480)에 붙여서 인식시킴
- 모터 대신 명령 기록만 함
- 결과 화면은 results/sim_frame.png 로 저장

실행:
  pip3 install zxing-cpp "python-barcode[images]" opencv-python numpy
  python3 make_barcodes.py
  python3 test_barcode_map.py
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from barcode_avoid import BarcodeAvoider  # noqa: E402
from barcode_nav import BarcodeNavigator  # noqa: E402

# 라벨 20cm, 카메라 수평 화각 약 62도, 640px 기준: 거리 d(m) 에서 라벨 폭 ≈ 106.5 / d (px)
PX_PER_M = 106.5


def fake_frame(label_path=None, distance_m=0.5):
    rng = np.random.default_rng(0)
    frame = rng.integers(60, 120, (480, 640, 3), dtype=np.uint8)
    if label_path is None:
        return frame

    label = cv2.imread(label_path)
    w = int(PX_PER_M / distance_m)
    h = int(label.shape[0] * w / label.shape[1])
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_AREA)

    # 살짝 비스듬하게 + 흐리게 (실제 카메라 흉내)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 8], [w, 0], [w, h], [0, h - 8]])
    label = cv2.warpPerspective(label, cv2.getPerspectiveTransform(src, dst), (w, h),
                                borderValue=(255, 255, 255))
    label = cv2.GaussianBlur(label, (3, 3), 0)

    x, y = (640 - w) // 2, 120
    frame[y:y + h, x:x + w] = label[:480 - y, :640 - x]
    return frame


def main():
    labels_dir = os.path.join(HERE, "labels")
    if not os.path.isdir(labels_dir):
        print("labels/ 가 없습니다. 먼저 python3 make_barcodes.py 를 실행하세요.")
        return 1

    def label(code):
        return os.path.join(labels_dir, f"{code}.png")

    commands = []

    def robot_ctrl(command):
        commands.append(command)

    def safe_stop():
        commands.append("safe_stop")

    # 테스트에서는 회피 동작 시간을 짧게
    BarcodeAvoider.STOP_SEC = BarcodeAvoider.BACK_SEC = 0.01
    BarcodeAvoider.TURN_90_SEC = BarcodeAvoider.TURN_180_SEC = 0.02

    nav = BarcodeNavigator(
        robot_ctrl,
        safe_stop,
        state_path=os.path.join(HERE, "results", "map_state.json"),
        detect_every_n_frames=1,
    )
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    m = nav.map
    failed = []

    def check(name, ok):
        print(("  [OK]  " if ok else "  [실패] ") + name)
        if not ok:
            failed.append(name)

    def show(code, distance_m, frames=3):
        for _ in range(frames):
            event = m.process(fake_frame(label(code), distance_m))
            if event is not None:
                nav.avoider.handle(event)
                nav.avoider.thread.join()
                return event
        return None

    print("1) 출발", m.robot_cell, m.heading)
    m.update_motion("forward", 0.9 / m.forward_m_per_sec)  # 0.9m = 3칸
    check("3칸 전진하면 (1,3)", m.robot_cell == (1, 3))

    print("2) 바코드 1100 (복도 끝 막힘 -> 오른쪽)")
    check("1.5m 거리에서는 반응 안 함", show("1100", 1.5) is None)
    event = show("1100", 0.5)
    check("0.5m 거리에서 인식", event is not None and event["code"] == "1100")
    check("(1,1) 칸 막힘 표시", m.grid[1, 1] == 1)
    check("로봇 위치 (1,2) 로 보정", m.robot_cell == (1, 2))
    check("오른쪽으로 돌아서 E 방향", m.heading == "E")
    check("정지 -> 후진 -> 우회전 순서", commands[:4] == ["safe_stop", "backward", "safe_stop", "right"])
    check("같은 바코드 다시 봐도 무시", show("1100", 0.5) is None)

    print("3) 6칸 전진 후 바코드 1101 (창고 앞 막힘 -> 오른쪽)")
    m.update_motion("forward", 1.8 / m.forward_m_per_sec)
    check("데드레커닝 위치 (7,2)", m.robot_cell == (7, 2))
    event = show("1101", 0.6)
    check("1101 인식 후 S 방향", event is not None and m.heading == "S")

    print("4) 경로 찾기")
    path = m.find_path(m.robot_cell, (0, 0))
    check("막힌 칸을 지나지 않음", path is not None and (1, 1) not in path and (8, 2) not in path)
    print("   ", path)

    out = os.path.join(HERE, "results", "sim_frame.png")
    cv2.imwrite(out, m.draw(fake_frame(label("1101"), 0.6)))
    print("화면 저장:", out)

    print("통과" if not failed else f"실패 {len(failed)}개")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
