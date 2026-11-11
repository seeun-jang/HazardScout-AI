#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PiCar-Pro TigerVNC Tkinter GUI for GUIServer_custom.py.

Run on the Raspberry Pi TigerVNC desktop:
    ~/picar-yolo/bin/python robot_gui.py

Uses the original PC control/video protocols: 127.0.0.1:10223 TCP,
5555 ZeroMQ video, not the mobile HTTP server. Keeps the previous light UI.
Neck axes: A pan +/-90, A tilt +/-80, B tilt +/-90 as set on the PC server.
"""
import base64
import json
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import zmq
from PIL import Image, ImageTk
from fullscreen_gui import FullscreenGUI
from PiCommandPanel import attach_command_panel

SERVER_IP = '127.0.0.1'  # Raspberry Pi desktop viewed via TigerVNC
CONTROL_PORT = 10223
VIDEO_PORT = 5555
DRIVE_SPEED = 60
TURN_SPEED = 50
CAMERA_UPDATE_MS = 80
KEEPALIVE_SECONDS = 0.25

BG = '#F4F6F8'
WHITE = '#FFFFFF'
TEXT = '#191F28'
MUTED = '#8B95A1'
BORDER = '#E5E8EB'
BLUE = '#3182F6'
BLUE_LIGHT = '#EAF2FF'
GREEN = '#20A06B'
GREEN_LIGHT = '#E8F7F0'
RED = '#E34D59'
RED_LIGHT = '#FDECEE'

app_running = True
connected = False
conn = None
conn_lock = threading.RLock()
state_lock = threading.Lock()

requested_move = None
requested_grip = None
requested_neck = {'a_pan': 0, 'a_tilt': 0, 'b_tilt': 0}
light_commands = queue.Queue()
command_wakeup = threading.Event()

last_frame = None
last_frame_time = 0.0
frame_serial = 0
frame_lock = threading.Lock()

pressed_keys = set()
release_jobs = {}
last_drawn_serial = -1
last_photo = None
last_connect_try = 0.0

root = tk.Tk()
root.title('PiCar-Pro Robot Control Center')
root.geometry('1200x920')
root.configure(bg=BG)
root.resizable(False, False)


def on_main_thread(callback):
    try:
        if app_running:
            root.after(0, callback)
    except Exception:
        pass


def set_connection(value):
    global connected, requested_move, requested_grip
    changed = connected != value
    connected = value
    if not value:
        # 즉시 재연결 이전의 조작 상태 제거 (UI 갱신 스케줄보다 먼저 실행).
        with state_lock:
            requested_move = None
            requested_grip = None
            for axis in requested_neck:
                requested_neck[axis] = 0
    if changed:
        def draw():
            status_text.configure(text='연결됨' if value else '연결 안 됨',
                                  fg=GREEN if value else RED)
            connection_dot.configure(bg=GREEN if value else RED)
            # Never resume stale held-motion commands after a reconnect.
            if not value:
                pressed_keys.clear()
                stop_all()
        on_main_thread(draw)


def close_socket():
    global conn
    with conn_lock:
        old, conn = conn, None
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
    set_connection(False)


def connect_if_needed():
    global conn, last_connect_try
    if not app_running:
        return False
    if conn is not None:
        return True
    if time.monotonic() - last_connect_try < 1.5:
        return False
    last_connect_try = time.monotonic()
    try:
        client = socket.create_connection((SERVER_IP, CONTROL_PORT), timeout=2.0)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client.settimeout(1.2)
        with conn_lock:
            conn = client
        set_connection(True)
        print('[PC GUI] 제어 연결 완료')
        return True
    except Exception as exc:
        print('[PC GUI] 연결 대기:', str(exc))
        set_connection(False)
        return False


def send_and_wait(command):
    """Only the command worker sends regular commands; one JSON ACK per command."""
    if conn is None:
        return False
    try:
        with conn_lock:
            client = conn
            if client is None:
                return False
            client.sendall(command.encode('utf-8'))
            data = client.recv(4096)
            if not data:
                raise ConnectionError('제어 연결이 종료되었습니다')
        message = json.loads(data.decode('utf-8'))
        if message.get('status') != 'ok':
            print('[명령 거부]', command, message)
            # Do not repeatedly retry a rejected actuation command.
            close_socket()
            return False
        return True
    except Exception as exc:
        print('[PC GUI 통신 오류]', command, repr(exc))
        close_socket()
        return False


def get_frame():
    with frame_lock:
        return None if last_frame is None else last_frame.copy()


def video_receiver():
    global last_frame, last_frame_time, frame_serial
    context = zmq.Context()
    video = context.socket(zmq.PAIR)
    video.setsockopt(zmq.LINGER, 0)
    video.setsockopt(zmq.RCVHWM, 1)
    video.setsockopt(zmq.RCVTIMEO, 1000)
    try:
        video.bind(f'tcp://*:{VIDEO_PORT}')
        print(f'[PC GUI] 카메라 수신 준비: {VIDEO_PORT}')
        while app_running:
            try:
                encoded = video.recv()
                image_bytes = base64.b64decode(encoded)
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                if image is None:
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                with frame_lock:
                    last_frame = rgb
                    frame_serial += 1
                    last_frame_time = time.monotonic()
            except zmq.Again:
                continue
            except Exception as exc:
                if app_running:
                    print('[PC GUI 영상 오류]', repr(exc))
                    time.sleep(0.3)
    except Exception as exc:
        print('[PC GUI 영상 포트 오류]', repr(exc))
    finally:
        video.close(0)
        context.term()


def set_move(direction):
    global requested_move
    with state_lock:
        requested_move = direction
    if direction:
        speed = DRIVE_SPEED if direction in ('forward', 'backward') else TURN_SPEED
        movement_label.configure(text=direction_to_korean(direction), fg=BLUE)
        speed_label.configure(text=f'{speed}%')
        speed_bar['value'] = speed
        motion_text.configure(text='이동 중', fg=GREEN)
    else:
        movement_label.configure(text='정지', fg=TEXT)
        speed_label.configure(text='0%')
        speed_bar['value'] = 0
        motion_text.configure(text='정지', fg=RED)
    command_wakeup.set()


def direction_to_korean(key):
    return {
        'forward':'전진', 'backward':'후진', 'left':'좌회전',
        'right':'우회전', 'backleft':'후진 좌회전', 'backright':'후진 우회전'
    }.get(key, '정지')


def start_grip(command):
    global requested_grip
    with state_lock:
        requested_grip = command
    grip_label.configure(text='잡는 중' if command == 'grab' else '놓는 중',
                         fg=GREEN if command == 'grab' else BLUE)
    command_wakeup.set()


def stop_grip():
    global requested_grip
    with state_lock:
        requested_grip = None
    grip_label.configure(text='정지', fg=MUTED)
    command_wakeup.set()


def start_neck(axis, direction):
    with state_lock:
        requested_neck[axis] = direction
    neck_status.configure(text=f'{axis} {direction:+d}', fg=BLUE)
    command_wakeup.set()


def stop_neck(axis):
    with state_lock:
        requested_neck[axis] = 0
    neck_status.configure(text='버튼을 놓으면 해당 위치 유지', fg=MUTED)
    command_wakeup.set()


def light_on():
    light_commands.put('light_on')
    light_value.configure(text='ON', fg=GREEN)
    command_wakeup.set()


def light_off():
    light_commands.put('light_off')
    light_value.configure(text='AUTO', fg=BLUE)
    command_wakeup.set()


def command_worker():
    """Single TCP writer: safe stop, grip/neck keepalives, and drive watchdog ping."""
    last_drive = None
    last_grip = None
    last_neck = {name:0 for name in requested_neck}
    move_sent = grip_sent = time.monotonic()
    neck_sent = {name:0.0 for name in requested_neck}
    last_connected = False

    while app_running:
        if not connect_if_needed():
            last_connected = False
            command_wakeup.wait(timeout=0.10)
            command_wakeup.clear()
            continue
        if not last_connected:
            last_connected = True
            last_drive, last_grip = None, None
            last_neck = {name:0 for name in requested_neck}
            move_sent = grip_sent = 0.0
            neck_sent = {name:0.0 for name in requested_neck}

        with state_lock:
            move_now = requested_move
            grip_now = requested_grip
            neck_now = dict(requested_neck)

        now = time.monotonic()
        # Priority 1: stop driving, before other commands.
        if move_now is None and last_drive is not None:
            if send_and_wait('DS'):
                last_drive = None
            else:
                last_connected = False
                continue
        elif move_now:
            if last_drive != move_now:
                if move_now in ('forward', 'backward'):
                    if not send_and_wait(f'wsB {DRIVE_SPEED}'):
                        last_connected = False
                        continue
                if not send_and_wait(move_now):
                    last_connected = False
                    continue
                last_drive = move_now
                move_sent = now
            elif now - move_sent >= KEEPALIVE_SECONDS:
                # PC server watchdog is refreshed without re-issuing motor PWM.
                if not send_and_wait('drive_ping'):
                    last_connected = False
                    continue
                move_sent = now

        for axis in requested_neck:
            direction = neck_now[axis]
            if direction != last_neck[axis] or (direction and now - neck_sent[axis] >= KEEPALIVE_SECONDS):
                if not send_and_wait(f'neck:{axis}:{direction}'):
                    last_connected = False
                    break
                last_neck[axis] = direction
                neck_sent[axis] = now
        if not last_connected:
            continue

        if grip_now != last_grip or (grip_now and now - grip_sent >= KEEPALIVE_SECONDS):
            command = grip_now if grip_now else 'grip_stop'
            if not send_and_wait(command):
                last_connected = False
                continue
            last_grip = grip_now
            grip_sent = now

        while not light_commands.empty():
            try:
                light_cmd = light_commands.get_nowait()
                if not send_and_wait(light_cmd):
                    last_connected = False
                    break
            except queue.Empty:
                break

        command_wakeup.wait(timeout=0.07)
        command_wakeup.clear()


def update_camera():
    global last_drawn_serial, last_photo
    if not app_running:
        return
    with frame_lock:
        current_serial = frame_serial
        current_time = last_frame_time
        image = None if last_frame is None else last_frame.copy()
    if image is not None and current_serial != last_drawn_serial:
        last_drawn_serial = current_serial
        try:
            frame = Image.fromarray(image).resize((520, 292), Image.Resampling.BILINEAR)
            last_photo = ImageTk.PhotoImage(frame)
            camera_label.configure(image=last_photo, text='')
        except Exception as exc:
            print('[카메라 UI 오류]', repr(exc))
    if current_time and time.monotonic() - current_time < 3:
        video_status.configure(text='LIVE', fg=GREEN)
    else:
        video_status.configure(text='영상 대기', fg=MUTED)
    root.after(CAMERA_UPDATE_MS, update_camera)


def stop_all():
    set_move(None)
    stop_grip()
    for axis in requested_neck:
        stop_neck(axis)


def on_key_press(event):
    if isinstance(event.widget, ttk.Entry):
        return
    key = event.keysym.lower()
    if key in release_jobs:
        try:
            root.after_cancel(release_jobs.pop(key))
        except Exception:
            pass
    if key in pressed_keys:
        return
    pressed_keys.add(key)
    if key in ('up', 'down', 'left', 'right'):
        update_arrow_keys()
    elif key in ('z', 'x'):
        start_grip('grab' if key == 'z' else 'loose')
    else:
        axis_dir = {
            'a':('a_pan', -1), 'd':('a_pan', 1),
            'w':('a_tilt', 1), 's':('a_tilt', -1),
            'i':('b_tilt', 1), 'k':('b_tilt', -1),
        }.get(key)
        if axis_dir is not None:
            start_neck(*axis_dir)


def update_arrow_keys():
    up, down = 'up' in pressed_keys, 'down' in pressed_keys
    left, right = 'left' in pressed_keys, 'right' in pressed_keys
    if down and left:
        motion = 'backleft'
    elif down and right:
        motion = 'backright'
    elif up and left:
        motion = 'left'
    elif up and right:
        motion = 'right'
    elif up:
        motion = 'forward'
    elif down:
        motion = 'backward'
    elif left:
        motion = 'left'
    elif right:
        motion = 'right'
    else:
        motion = None
    set_move(motion)


def key_released_later(key):
    release_jobs.pop(key, None)
    pressed_keys.discard(key)
    if key in ('up', 'down', 'left', 'right'):
        update_arrow_keys()
    elif key in ('z', 'x'):
        if not ('z' in pressed_keys or 'x' in pressed_keys):
            stop_grip()
    else:
        map_axis = {'a':'a_pan','d':'a_pan','w':'a_tilt','s':'a_tilt',
                    'i':'b_tilt','k':'b_tilt'}
        if key in map_axis:
            axis = map_axis[key]
            partner = {'a':'d','d':'a','w':'s','s':'w','i':'k','k':'i'}[key]
            if partner not in pressed_keys:
                stop_neck(axis)


def on_key_release(event):
    if isinstance(event.widget, ttk.Entry):
        return
    key = event.keysym.lower()
    if key in release_jobs:
        try:
            root.after_cancel(release_jobs[key])
        except Exception:
            pass
    release_jobs[key] = root.after(30, lambda: key_released_later(key))


def close_program():
    global app_running
    if not app_running:
        return
    stop_all()
    command_wakeup.set()
    # Let the command worker dispatch stop requests, then close connection.
    root.after(450, finish_exit)


def finish_exit():
    global app_running
    try:
        fullscreen.close()
    except Exception:
        pass
    app_running = False
    command_wakeup.set()
    close_socket()
    try:
        root.destroy()
    except Exception:
        pass


def card(parent, title):
    outer = tk.Frame(parent, bg=WHITE, highlightthickness=1,
                     highlightbackground=BORDER)
    tk.Label(outer, text=title, bg=WHITE, fg=TEXT,
             font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', padx=14, pady=(13, 8))
    return outer


def hold_button(parent, title, start, stop, row, col, bg=BLUE_LIGHT):
    b = tk.Button(parent, text=title, width=10, height=2,
                  bg=bg, fg=TEXT, activebackground=BLUE,
                  relief='flat', bd=0,
                  font=('TkDefaultFont', 10, 'bold'))
    b.grid(row=row, column=col, padx=4, pady=4)
    b.bind('<ButtonPress-1>', lambda event: start())
    b.bind('<ButtonRelease-1>', lambda event: stop())
    b.bind('<Leave>', lambda event: stop())
    return b

# ------------------------------------------------------------
# GUI: maintain the original clean white layout and fullscreen flow.
# ------------------------------------------------------------
main = tk.Frame(root, bg=BG)
main.pack(fill='both', expand=True, padx=22, pady=15)
header = tk.Frame(main, bg=BG)
header.pack(fill='x', pady=(0, 12))
tk.Label(header, text='Robot Control Center', bg=BG, fg=TEXT,
         font=('TkDefaultFont', 21, 'bold')).pack(side='left')
status_box = tk.Frame(header, bg=WHITE, padx=12, pady=7)
status_box.pack(side='right')
connection_dot = tk.Label(status_box, bg=RED, width=1)
connection_dot.pack(side='left', padx=(0, 8))
status_text = tk.Label(status_box, text='연결 안 됨', fg=RED, bg=WHITE)
status_text.pack(side='left')

camera_row = tk.Frame(main, bg=BG)
camera_row.pack(fill='x')
robot_card = card(camera_row, '로봇 카메라')
robot_card.pack(side='left', fill='x', padx=(0, 7))
video_status = tk.Label(robot_card, text='영상 대기', bg=WHITE, fg=MUTED)
video_status.pack(anchor='e', padx=14)
video_area = tk.Frame(robot_card, bg='#171A1F', width=520, height=292)
video_area.pack(padx=12, pady=(5, 12))
video_area.pack_propagate(False)
camera_label = tk.Label(video_area, text='카메라 영상 연결 대기',
                        fg=WHITE, bg='#171A1F')
camera_label.pack(fill='both', expand=True)

env_card = card(camera_row, '환경 카메라 (미연결)')
env_card.pack(side='left', fill='both', expand=True, padx=(7, 0))
env_frame = tk.Frame(env_card, bg='#171A1F', width=520, height=292)
env_frame.pack(padx=12, pady=(25, 12), fill='both', expand=True)
env_frame.pack_propagate(False)
tk.Label(env_frame, text='환경 카메라\n추후 연결 예정', bg='#171A1F',
         fg=MUTED).pack(fill='both', expand=True)

control_row = tk.Frame(main, bg=BG)
control_row.pack(fill='both', expand=True, pady=(12, 0))

movement_card = card(control_row, '이동 제어 · 방향키')
movement_card.pack(side='left', fill='both', padx=(0, 6))
movement = tk.Frame(movement_card, bg=WHITE)
movement.pack(padx=12, pady=4)
hold_button(movement, '▲ 전진', lambda:set_move('forward'), lambda:set_move(None), 0, 1)
hold_button(movement, '◀ 좌회전', lambda:set_move('left'), lambda:set_move(None), 1, 0)
tk.Button(movement, text='■ STOP', command=lambda:set_move(None), bg=RED_LIGHT,
          fg=RED, width=10, height=2, relief='flat').grid(row=1, column=1, padx=4, pady=4)
hold_button(movement, '우회전 ▶', lambda:set_move('right'), lambda:set_move(None), 1, 2)
hold_button(movement, '↙ 후진 좌', lambda:set_move('backleft'), lambda:set_move(None), 2, 0)
hold_button(movement, '▼ 후진', lambda:set_move('backward'), lambda:set_move(None), 2, 1)
hold_button(movement, '후진 우 ↘', lambda:set_move('backright'), lambda:set_move(None), 2, 2)

tk.Label(movement_card, text='현재 방향', bg=WHITE, fg=MUTED).pack()
movement_label = tk.Label(movement_card, text='정지', bg=WHITE, fg=TEXT,
                          font=('TkDefaultFont', 13, 'bold'))
movement_label.pack()
motion_text = tk.Label(movement_card, text='정지', bg=WHITE, fg=RED)
motion_text.pack()
speed_label = tk.Label(movement_card, text='0%', bg=WHITE, fg=BLUE)
speed_label.pack()
speed_bar = ttk.Progressbar(movement_card, length=190, maximum=100)
speed_bar.pack(pady=(2, 12))

neck_card = card(control_row, '목 A/B · 누르는 동안 이동')
neck_card.pack(side='left', fill='both', padx=6)
neck_grid = tk.Frame(neck_card, bg=WHITE)
neck_grid.pack(padx=10, pady=4)
neck_items = [
    ('A 왼쪽 (A)', 'a_pan', -1), ('A 오른쪽 (D)', 'a_pan', 1),
    ('A 위 (W)', 'a_tilt', 1), ('A 아래 (S)', 'a_tilt', -1),
    ('B 위 (I)', 'b_tilt', 1), ('B 아래 (K)', 'b_tilt', -1),
]
for index, (label, axis, direction) in enumerate(neck_items):
    hold_button(neck_grid, label,
                lambda a=axis,d=direction: start_neck(a,d),
                lambda a=axis: stop_neck(a), index//2, index%2)
neck_status = tk.Label(neck_card, text='버튼을 놓으면 해당 위치 유지',
                       bg=WHITE, fg=MUTED, wraplength=235)
neck_status.pack(padx=8, pady=(5, 10))
tk.Label(neck_card, text='A 좌우 ±90°  |  A 상하 ±80°\nB 상하 ±90°',
         fg=MUTED, bg=WHITE).pack(pady=(0, 10))

extra_card = card(control_row, '집게 · 라이트')
extra_card.pack(side='left', fill='both', padx=(6, 0), expand=True)
button_box = tk.Frame(extra_card, bg=WHITE)
button_box.pack(padx=12, pady=6)
hold_button(button_box, '잡기 Z', lambda:start_grip('grab'), stop_grip, 0, 0, GREEN_LIGHT)
hold_button(button_box, '놓기 X', lambda:start_grip('loose'), stop_grip, 0, 1)
grip_label = tk.Label(extra_card, text='정지', bg=WHITE, fg=MUTED)
grip_label.pack(pady=(5, 12))
light_box = tk.Frame(extra_card, bg=WHITE)
light_box.pack(pady=4)
tk.Button(light_box, text='후레시 ON', width=12, command=light_on,
          bg=GREEN_LIGHT, relief='flat').pack(side='left', padx=5)
tk.Button(light_box, text='자동 모드', width=12, command=light_off,
          bg=BLUE_LIGHT, relief='flat').pack(side='left', padx=5)
light_value = tk.Label(extra_card, text='AUTO', bg=WHITE, fg=BLUE,
                       font=('TkDefaultFont', 12, 'bold'))
light_value.pack(pady=7)

fullscreen = FullscreenGUI(
    master=root, get_frame=get_frame,
    start_move=set_move, stop_move=lambda:set_move(None),
    start_grip=start_grip, stop_grip=stop_grip,
    light_on=light_on, light_off=light_off,
    neck_start=start_neck, neck_stop=stop_neck,
    get_light_mode=lambda:light_value.cget('text'),
)

toolbar = tk.Frame(main, bg=BG)
toolbar.pack(fill='x', pady=(8, 0))
tk.Label(toolbar, text='PC GUI는 라즈베리파이에서 실행하고 TigerVNC로 화면을 확인합니다.',
         bg=BG, fg=MUTED).pack(side='left')
tk.Button(toolbar, text='카메라 전체화면  F11', command=fullscreen.open,
          bg=BLUE, fg=WHITE, relief='flat', height=2).pack(side='right', padx=4)
tk.Button(toolbar, text='종료', command=close_program,
          bg=WHITE, fg=RED, relief='flat', height=2).pack(side='right', padx=4)

def stop_for_text_input():
    for job in release_jobs.values():
        root.after_cancel(job)
    release_jobs.clear()
    pressed_keys.clear()
    stop_all()


attach_command_panel(root, header, server_ip=SERVER_IP, on_focus=stop_for_text_input)

root.bind('<KeyPress>', on_key_press)
root.bind('<KeyRelease>', on_key_release)
root.bind('<F11>', lambda e: fullscreen.open())
root.bind('<Escape>', lambda e: close_program())
root.bind('<Unmap>', lambda e: stop_all() if e.widget is root else None)
root.protocol('WM_DELETE_WINDOW', close_program)
root.focus_set()

print('[PC GUI] 127.0.0.1:10223 TCP / ZeroMQ:5555, VNC 로컬 사용')
print('[조작] 방향키 주행, WASD 목 A, IK 목 B, ZX 집게, F11 전체화면')

threading.Thread(target=video_receiver, daemon=True, name='PCVideo').start()
threading.Thread(target=command_worker, daemon=True, name='PCControls').start()
root.after(100, update_camera)
root.mainloop()
