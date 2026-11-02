#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TigerVNC PC camera fullscreen; works with robot_gui.py callbacks."""
import tkinter as tk
from PIL import Image, ImageTk

BG = '#10141B'
WHITE = '#FFFFFF'
BLUE = '#3182F6'
GREEN = '#20A06B'


class FullscreenGUI:
    def __init__(self, master, get_frame, start_move, stop_move,
                 start_grip, stop_grip, light_on, light_off,
                 neck_start, neck_stop, get_light_mode):
        self.master = master
        self.get_frame = get_frame
        self.start_move = start_move
        self.stop_move = stop_move
        self.start_grip = start_grip
        self.stop_grip = stop_grip
        self.light_on = light_on
        self.light_off = light_off
        self.neck_start = neck_start
        self.neck_stop = neck_stop
        self.get_light_mode = get_light_mode
        self.window = None
        self.label = None
        self.image_area = None
        self.job = None
        self.last_image = None

    def _hold_button(self, parent, text, start, stop, row, column,
                     color='#EAF2FF', fg='#191F28'):
        btn = tk.Button(parent, text=text, bg=color, fg=fg, width=11, height=2,
                        font=('TkDefaultFont', 10, 'bold'), relief='flat', bd=0)
        btn.grid(row=row, column=column, padx=3, pady=3, sticky='ew')
        btn.bind('<ButtonPress-1>', lambda e: start())
        btn.bind('<ButtonRelease-1>', lambda e: stop())
        btn.bind('<Leave>', lambda e: stop())
        return btn

    def _section(self, parent, title):
        tk.Label(parent, text=title, font=('TkDefaultFont', 12, 'bold'),
                 bg=WHITE, fg='#191F28').pack(anchor='w', pady=(12, 5))
        frame = tk.Frame(parent, bg=WHITE)
        frame.pack(anchor='w', fill='x')
        return frame

    def open(self):
        if self.window is not None:
            if self.window.winfo_exists():
                self.window.lift()
                return
        self.window = tk.Toplevel(self.master)
        self.window.title('PiCar-Pro · 카메라 전체화면')
        self.window.configure(bg=BG)
        self.window.attributes('-fullscreen', True)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.bind('<Escape>', lambda e: self.close())
        self.window.bind('<F11>', lambda e: self.close())

        root = tk.Frame(self.window, bg=BG)
        root.pack(fill='both', expand=True, padx=14, pady=14)
        left = tk.Frame(root, bg=BG)
        left.pack(side='left', fill='both', expand=True)
        tk.Label(left, text='로봇 카메라 · 실시간 영상', bg=BG, fg=WHITE,
                 font=('TkDefaultFont', 14, 'bold')).pack(anchor='w', pady=(0, 8))
        self.image_area = tk.Frame(left, bg='#1F252E')
        self.image_area.pack(fill='both', expand=True)
        self.label = tk.Label(self.image_area, bg='#1F252E', fg=WHITE,
                              text='카메라 영상 연결 대기')
        self.label.pack(fill='both', expand=True)

        side = tk.Frame(root, bg=WHITE, width=340, padx=13, pady=12)
        side.pack(side='right', fill='y', padx=(12, 0))
        side.pack_propagate(False)

        motion = self._section(side, '주행 제어')
        self._hold_button(motion, '▲ 전진', lambda: self.start_move('forward'), self.stop_move, 0, 1)
        self._hold_button(motion, '◀ 좌회전', lambda: self.start_move('left'), self.stop_move, 1, 0)
        tk.Button(motion, text='■ STOP', bg='#FDECEE', fg='#E34D59', width=11, height=2,
                  command=self.stop_move).grid(row=1, column=1, padx=3, pady=3)
        self._hold_button(motion, '우회전 ▶', lambda: self.start_move('right'), self.stop_move, 1, 2)
        self._hold_button(motion, '↙ 후진 좌', lambda: self.start_move('backleft'), self.stop_move, 2, 0)
        self._hold_button(motion, '▼ 후진', lambda: self.start_move('backward'), self.stop_move, 2, 1)
        self._hold_button(motion, '후진 우 ↘', lambda: self.start_move('backright'), self.stop_move, 2, 2)

        neck = self._section(side, '목 A / B 조작 · 누르고 있는 동안 이동')
        neck_items = [
            ('A 왼쪽', 'a_pan', -1), ('A 오른쪽', 'a_pan', 1),
            ('A 위', 'a_tilt', 1), ('A 아래', 'a_tilt', -1),
            ('B 위', 'b_tilt', 1), ('B 아래', 'b_tilt', -1),
        ]
        for index, (label, axis, direction) in enumerate(neck_items):
            self._hold_button(neck, label,
                              lambda a=axis, d=direction: self.neck_start(a, d),
                              lambda a=axis: self.neck_stop(a), index // 2, index % 2)

        grip = self._section(side, '집게')
        self._hold_button(grip, '잡기 Z', lambda: self.start_grip('grab'), self.stop_grip, 0, 0)
        self._hold_button(grip, '놓기 X', lambda: self.start_grip('loose'), self.stop_grip, 0, 1)

        light = self._section(side, '조명')
        tk.Button(light, text='라이트 ON', command=self.light_on, width=12,
                  bg='#E8F7F0', relief='flat').pack(side='left', padx=3)
        tk.Button(light, text='자동 모드', command=self.light_off, width=12,
                  bg='#EAF2FF', relief='flat').pack(side='left', padx=3)
        tk.Button(side, text='전체화면 종료 (ESC / F11)', command=self.close,
                  bg='#E5E8EB', relief='flat', height=2).pack(side='bottom', fill='x')
        self._refresh()
        self.window.focus_force()

    def _refresh(self):
        if self.window is None or not self.window.winfo_exists():
            return
        frame = self.get_frame()
        if frame is not None and self.image_area is not None:
            try:
                h, w = frame.shape[:2]
                aw, ah = self.image_area.winfo_width(), self.image_area.winfo_height()
                if aw > 10 and ah > 10:
                    scale = min(aw / w, ah / h)
                    size = (max(1, int(w * scale)), max(1, int(h * scale)))
                    image = Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
                    self.last_image = ImageTk.PhotoImage(image)
                    self.label.configure(image=self.last_image, text='')
            except Exception as exc:
                print('[전체화면 영상 오류]', repr(exc))
        self.job = self.window.after(100, self._refresh)

    def close(self):
        # 전체화면에서 누르고 있던 버튼이 화면 종료 후 계속 유지되지 않도록 정지.
        try:
            self.stop_move()
            self.stop_grip()
            for axis in ('a_pan', 'a_tilt', 'b_tilt'):
                self.neck_stop(axis)
        except Exception as exc:
            print('[전체화면 정지 처리 경고]', repr(exc))
        if self.window is not None:
            if self.job is not None:
                try:
                    self.window.after_cancel(self.job)
                except Exception:
                    pass
            try:
                self.window.destroy()
            except Exception:
                pass
        self.job = None
        self.window = None
        self.label = None
        self.image_area = None
        self.last_image = None
