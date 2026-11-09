#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_hooks.py

팀 서버 파일을 고치지 않고 좌표 매핑을 붙이는 부분.
팀 서버의 Final 파일들(GUIServer_custom.py, GUIServer_mobile_final.py)이
원본 서버를 감싸서 기능을 바꾸는 방식과 똑같이, 실행 중에만 바꾸고 끝나면 되돌린다.

연결 지점은 원본 서버의 AsyncYOLO 클래스 하나다.
  - 카메라 루프는 매 프레임 async_yolo.draw(frame) 을 부른다
      -> 여기서 YOLO 박스를 그린 다음 지도(위치 갱신, 바코드, 미니맵)를 처리
  - YOLO 스레드는 추론마다 detector._run_inference(frame) 을 부른다
      -> 추론 직전 로봇 위치를 기억해 두고, 결과를 위험물 지도에 반영
  - AsyncYOLO 가 만들어지는 시점은 Final 파일이 robot_ctrl 등을 바꾼 뒤라서,
    이때 robot_ctrl 을 한 번 더 감싸 '사용자가 조작하면 회피 중단'을 넣는다
"""

DRIVE_COMMANDS = {"forward", "backward", "left", "right", "backleft", "backright"}
STOP_COMMANDS = {"DS", "TS", "stop_move"}

# kind -> 정지 명령 이름 (서버마다 다름)
STOP_COMMAND = {"pc": "DS", "mobile": "stop_move"}


def neck_pan_provider(*modules):
    """목 A 좌우 각도(도). 0 = 처음 켰을 때 정면"""
    def pan():
        for mod in modules:
            nc = getattr(mod, "neck_controller", None) if mod is not None else None
            if nc is not None:
                with nc.lock:
                    return float(nc.position.get("a_pan", 0))
        return 0.0
    return pan


def install(base, kind, mapper):
    """
    base   : 원본 서버 모듈 (GUIServer_custom_base 또는 GUIServer_mobile)
    kind   : "pc" 또는 "mobile"
    mapper : map_core.HazardMapper
    되돌리기 함수를 돌려준다.
    """
    if not hasattr(base, "AsyncYOLO"):
        print("[지도] 이 서버에는 AsyncYOLO 가 없어 연결할 수 없습니다 (11월 버전 서버 필요)", flush=True)
        return lambda: None

    original_async = base.AsyncYOLO
    state = {"ctrl_installed": False, "errors": 0}

    def install_controls():
        if state["ctrl_installed"]:
            return
        state["ctrl_installed"] = True

        server_ctrl = base.robot_ctrl          # Final 파일이 바꿔 둔 최종 함수
        stop_command = STOP_COMMAND[kind]

        def stop():
            server_ctrl(stop_command)
            try:
                base.safe_stop()               # 조향 바퀴도 가운데로
            except Exception:
                pass

        mapper.attach_controls(server_ctrl, stop)

        def guarded_ctrl(command, *args, **kwargs):
            if command in DRIVE_COMMANDS or command in STOP_COMMANDS:
                if mapper.avoider is not None:
                    mapper.avoider.cancel()
            return server_ctrl(command, *args, **kwargs)

        base.robot_ctrl = guarded_ctrl
        state["server_ctrl"] = server_ctrl
        print("[지도] 주행 연결 완료 (사용자 조작 시 회피 자동 중단)", flush=True)

    def report(where, exc):
        state["errors"] += 1
        if state["errors"] <= 5:
            print(f"[지도] {where} 오류:", repr(exc), flush=True)

    class MappedAsyncYOLO(original_async):
        def __init__(self, detector, *args, **kwargs):
            super().__init__(detector, *args, **kwargs)
            install_controls()

            if detector is not None:
                infer = detector._run_inference

                def mapped_inference(frame):
                    pose = mapper.pose.snapshot()
                    pan = mapper.pan()
                    result = infer(frame)
                    try:
                        mapper.on_inference(result, frame.shape, pose, pan)
                    except Exception as e:
                        report("위험물 지도", e)
                    return result

                detector._run_inference = mapped_inference

        def draw(self, frame):
            frame = super().draw(frame)
            try:
                return mapper.on_frame(frame, getattr(base, "current_motion", "stop"))
            except Exception as e:
                report("지도 처리", e)
                return frame

    base.AsyncYOLO = MappedAsyncYOLO

    def restore():
        base.AsyncYOLO = original_async
        if "server_ctrl" in state:
            base.robot_ctrl = state["server_ctrl"]

    return restore


def add_mobile_map_button(base, port):
    """모바일 화면 오른쪽 위에 '지도' 버튼 추가 (HTML 문자열에 덧붙이기만 함)"""
    html = getattr(base, "MOBILE_HTML", None)
    if not isinstance(html, str) or "</body>" not in html or "hazard-map-link" in html:
        return
    button = (
        '<a id="hazard-map-link" target="_blank" '
        'style="position:fixed;top:10px;right:10px;z-index:9999;padding:8px 12px;'
        'border-radius:10px;background:#1b2027;color:#fff;font:600 14px sans-serif;'
        'text-decoration:none;border:1px solid #3a4250">지도</a>'
        '<script>document.getElementById("hazard-map-link").href='
        f'location.protocol+"//"+location.hostname+":{port}/map";</script>'
    )
    base.MOBILE_HTML = html.replace("</body>", button + "</body>", 1)
