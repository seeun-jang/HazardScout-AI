#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_mobile_map.py

모바일 서버 + 좌표 매핑 실행

PuTTY:
  cd ~/Adeept_PiCar-Pro/Server
  sudo ~/picar-yolo/bin/python /home/pi/hazard_map_80c7951/run_mobile_map.py

Final 파일 이름이 다르면 --wrapper 로 지정:
  sudo ~/picar-yolo/bin/python /home/pi/hazard_map_80c7951/run_mobile_map.py --wrapper GUIServer_mobile_final_angle_90_80_90

모바일 화면: http://<Pi-IP>:5000  (오른쪽 위 '지도' 버튼)
지도 화면  : http://<Pi-IP>:5001/map
"""

import sys

from map_launch import launch

if __name__ == "__main__":
    sys.exit(launch(
        kind="mobile",
        base_names=["GUIServer_mobile"],
        wrapper_names=["GUIServer_mobile_final", "GUIServer_mobile_final_angle_90_80_90"],
        description="모바일 서버 + 좌표 매핑",
    ))
