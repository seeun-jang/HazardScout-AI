#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_launch.py

run_pc_map.py / run_mobile_map.py 가 같이 쓰는 실행 코드.
팀 서버 폴더(~/Adeept_PiCar-Pro/Server)의 파일을 그대로 불러와서
좌표 매핑을 붙인 뒤, 팀 서버를 평소처럼 실행한다.
"""

import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SERVER_DIR = "/home/pi/Adeept_PiCar-Pro/Server"


def _import_first(names):
    for name in names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name != name:
                raise          # 파일은 있는데 그 안에서 다른 모듈이 없는 경우는 그대로 알림
    return None


def launch(kind, base_names, wrapper_names, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--server-dir", default=os.environ.get("PICAR_SERVER_DIR", DEFAULT_SERVER_DIR),
                        help="팀 서버 파일이 있는 폴더")
    parser.add_argument("--wrapper", help="Final 서버 파일 이름 (.py 빼고). 기본값은 자동으로 찾음")
    parser.add_argument("--map-port", type=int, default=5001, help="지도 웹 포트")
    parser.add_argument("--no-map", action="store_true", help="지도 없이 팀 서버만 실행 (비교용)")
    args = parser.parse_args()

    server_dir = os.path.abspath(os.path.expanduser(args.server_dir))
    if not os.path.isdir(server_dir):
        print(f"[지도] 서버 폴더가 없습니다: {server_dir}")
        print("       --server-dir 로 팀 서버 폴더를 지정하세요")
        return 1

    # 팀 서버 파일이 상대 경로를 써도 되도록 서버 폴더에서 실행
    os.chdir(server_dir)
    sys.path.insert(0, HERE)
    sys.path.insert(0, server_dir)

    base = _import_first(base_names)
    if base is None:
        print(f"[지도] {server_dir} 에서 {', '.join(n + '.py' for n in base_names)} 를 찾지 못했습니다")
        return 1

    wrapper = _import_first([args.wrapper] if args.wrapper else wrapper_names)
    if wrapper is None:
        print("[지도] Final 서버 파일이 없어 원본 서버만 실행합니다 (목 각도 반영 안 됨)")

    if args.no_map:
        print("[지도] --no-map: 지도 기능 없이 실행")
        return _run(wrapper, base)

    import map_hooks
    import map_web
    from map_core import HazardMapper

    mapper = HazardMapper(pan_provider=map_hooks.neck_pan_provider(wrapper, base))
    restore = map_hooks.install(base, kind, mapper)
    if kind == "mobile":
        map_hooks.add_mobile_map_button(base, args.map_port)
    web = map_web.start(mapper, args.map_port)

    try:
        return _run(wrapper, base)
    finally:
        mapper.close()
        mapper.save()
        restore()
        if web is not None:
            web.shutdown()
        print("[지도] 지도 저장 후 종료:", os.path.join(HERE, "map_state.json"))


def _run(wrapper, base):
    if wrapper is not None:
        entry = getattr(wrapper, "run", None) or getattr(wrapper, "main")
    else:
        entry = base.main
    entry()
    return 0
