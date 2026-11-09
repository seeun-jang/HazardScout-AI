#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pc_map.py

PC(TigerVNC) 서버 + 좌표 매핑 실행

PuTTY:
  cd ~/Adeept_PiCar-Pro/Server
  sudo ~/picar-yolo/bin/python /home/pi/hazard_map/run_pc_map.py

기존 실행(sudo ~/picar-yolo/bin/python GUIServer_custom.py)과 똑같이 동작하고,
영상에 미니맵이 붙고, http://192.168.25.112:5001/map 에서 지도를 볼 수 있다.
PC GUI(~/move/robot_gui.py)는 평소처럼 따로 실행한다.
"""

import sys

from map_launch import launch

if __name__ == "__main__":
    sys.exit(launch(
        kind="pc",
        base_names=["GUIServer_custom_base"],
        wrapper_names=["GUIServer_custom"],
        description="PC 서버 + 좌표 매핑",
    ))
