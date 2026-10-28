#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barcode_nav.py

팀 서버(GUIServer_custom.py / GUIServer_mobile.py)에 붙이기 위한 연결 모듈
- BarcodeMap(지도) + BarcodeAvoider(회피 동작)를 하나로 묶음
- 서버 쪽에서는 만들 때 한 번, 카메라 루프에서 한 줄만 호출하면 됨

사용 예:
    nav = BarcodeNavigator(robot_ctrl, safe_stop)
    ...
    frame_bgr = nav.on_frame(frame_bgr, current_motion)
"""

import time

from barcode_avoid import BarcodeAvoider
from barcode_map import BarcodeMap


class BarcodeNavigator:
    def __init__(self, robot_ctrl, safe_stop, auto_resume=False,
                 on_event=None, **map_kwargs):
        """
        robot_ctrl, safe_stop : 팀 서버의 함수를 그대로 넘김
        auto_resume           : 회피 후 자동으로 다시 전진 (자율주행 모드)
        on_event              : 바코드 이벤트를 대시보드/음성 안내로 보낼 콜백
        map_kwargs            : BarcodeMap 설정 (forward_m_per_sec, min_width_px 등)
        """
        self.map = BarcodeMap(**map_kwargs)
        self.avoider = BarcodeAvoider(
            robot_ctrl,
            safe_stop,
            barcode_map=self.map,
            auto_resume=auto_resume,
            on_event=on_event,
        )
        self.last_t = None

    def on_frame(self, frame_bgr, current_motion):
        """카메라 루프에서 YOLO 다음에 호출. 미니맵을 그린 프레임을 돌려줌"""
        now = time.time()
        if self.last_t is not None:
            self.map.update_motion(current_motion, now - self.last_t)
        self.last_t = now

        event = self.map.process(frame_bgr)
        if event is not None:
            self.avoider.handle(event)

        return self.map.draw(frame_bgr)

    def cancel(self):
        """사용자가 직접 조작하면 회피 동작 중단"""
        self.avoider.cancel()

    @property
    def busy(self):
        return self.avoider.busy

    def map_dict(self):
        return self.map.to_dict()

    def reset(self):
        self.map.reset()
        self.map.save()
