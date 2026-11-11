#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI 요청 처리. 터미널 input() 없이 로그만 출력한다."""
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import PiCommandCore as core
import SmoothServo


class ExistingControls:
    """Run timed AI commands through the server's existing actuator owners."""

    def __init__(self, namespace, controllers, stop_drive, drive_started=None):
        self.ns = namespace
        self.controllers = controllers
        self.stop_drive = stop_drive
        self.drive_started = drive_started
        self.motion_enabled = os.environ.get('PICAR_AI_MOTION') == '1'

    def stop(self):
        if self.motion_enabled:
            neck, grip = self.controllers()
            try:
                self.stop_drive()
            finally:
                try:
                    if neck is not None:
                        neck.release_all()
                finally:
                    if grip is not None:
                        grip.release()

    @staticmethod
    def wait(cancel, seconds):
        if cancel.wait(max(0, seconds)):
            raise core.CommandCancelled('수동 조작 또는 중지로 AI 동작을 취소했습니다')

    def execute(self, item, cancel_event):
        item = core.validate_command(item)
        command = item['command']
        duration = item['duration_ms'] / 1000
        if not self.motion_enabled:
            print(f'[AI 미리보기] {command} · {item["duration_ms"]}ms', flush=True)
            return
        self.wait(cancel_event, 0)
        self.stop()
        try:
            if command in core.DRIVE_COMMANDS | core.TURN_COMMANDS:
                self.drive(command, duration, cancel_event)
            elif command in core.NECK_MOTIONS:
                self.move_neck(command, duration, cancel_event)
            elif command in ('grab', 'release'):
                self.move_grip(command, duration, cancel_event)
        finally:
            self.stop()

    def drive(self, command, duration, cancel):
        if not hasattr(self.ns['move'], 'motor1'):
            raise RuntimeError('주행 제어기 준비 중')
        reverse = command in ('backward', 'backleft', 'backright')
        turn = command in core.TURN_COMMANDS
        steering = self.ns['TURN_ANGLE'] * self.ns['Dv'] if turn else 0
        if command in ('right', 'backright'):
            steering = -steering
        speed = core.TURN_SPEED if turn else core.DRIVE_SPEED
        self.ns['drive_command'](command, -1 if reverse else 1, steering, speed)
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if self.drive_started is not None:
                self.drive_started()
            self.wait(cancel, min(.20, end - time.monotonic()))

    def move_neck(self, command, duration, cancel):
        neck, _ = self.controllers()
        if neck is None:
            raise RuntimeError('목 제어기 준비 중')
        axis, pwm_direction = core.NECK_MOTIONS[command]
        signs = SmoothServo.NECK_SIGNS
        step = SmoothServo.NECK_STEP
        limit = core.NECK_MAX_OFFSET
        message = f'neck:{axis}:{pwm_direction * signs[axis]}'
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.wait(cancel, 0)
            with neck.lock:
                position = neck.position[axis]
            if abs(position) > limit:
                raise ValueError(f'{axis} 목을 중앙 기준 ±{limit}도 안으로 수동 조작하세요')
            # The existing worker can have one step in flight as we release it.
            if pwm_direction * position >= limit - step:
                break
            neck.command(message)
            self.wait(cancel, min(.02, end - time.monotonic()))

    def move_grip(self, command, duration, cancel):
        _, grip = self.controllers()
        if grip is None:
            raise RuntimeError('집게 제어기 준비 중')
        direction = core.GRIP_CLOSE_DIRECTION if command == 'grab' else -core.GRIP_CLOSE_DIRECTION
        end = time.monotonic() + duration
        while time.monotonic() < end:
            self.wait(cancel, 0)
            grip.hold(direction)
            self.wait(cancel, min(.20, end - time.monotonic()))


class CommandService:
    def __init__(self, robot, stt=None, key='', analyzer=None):
        self.robot = robot
        self.stt = stt or core.STTClient()
        self.key = key
        self.analyzer = analyzer or core.analyze
        self.lock = threading.Lock()
        self.motion_lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.worker = None
        self.closed = False
        self.avoidance_thread = None
        self.http_server = self.http_thread = None
        self.data = {'state': 'ready', 'busy': False, 'job_id': '', 'text': '', 'command': None,
                     'message': '명령 입력 준비 완료', 'error': '', 'recording_until': 0}

    def status(self):
        with self.lock:
            data = dict(self.data)
        data['remaining'] = max(0, data.pop('recording_until', 0) - time.monotonic())
        data['api_key_configured'] = bool(self.key)
        data['motion_enabled'] = self.robot.motion_enabled
        return data

    def _set(self, **values):
        with self.lock:
            self.data.update(values)

    def submit(self, mode, text=''):
        if mode not in ('text', 'voice'):
            raise ValueError('입력 방식 오류')
        if mode == 'voice':
            text = ''
        if mode == 'text' and (not isinstance(text, str) or not text.strip() or len(text) > 500):
            raise ValueError('텍스트는 1~500자로 입력하세요')
        with self.lock:
            if self.closed:
                raise RuntimeError('서비스 종료 중')
            if self.data['busy']:
                raise RuntimeError('명령 처리 중입니다. 완료 후 다시 입력하세요')
            if self.avoidance_thread is not None and self.avoidance_thread.is_alive():
                raise RuntimeError('지도 자동 회피가 끝난 후 다시 입력하세요')
            if not self.key:
                raise RuntimeError('서버 실행 시 ANTHROPIC_API_KEY를 설정하세요')
            cancel = threading.Event()
            self.cancel_event = cancel
            self.data = {'state': 'loading' if mode == 'voice' else 'analyzing', 'busy': True,
                         'job_id': uuid.uuid4().hex, 'text': text.strip(), 'command': None,
                         'message': '마이크/모델 준비 중' if mode == 'voice' else 'api 전달중',
                         'error': '', 'recording_until': 0}
            self.worker = threading.Thread(target=self._run, args=(mode, text.strip(), cancel),
                                           daemon=True, name='PiCommandWorker')
            self.worker.start()
        return self.status()

    def _check(self, cancel):
        if cancel.is_set() or self.closed:
            raise core.CommandCancelled('수동 조작/중지로 명령을 취소했습니다')

    def _run(self, mode, text, cancel):
        try:
            self._check(cancel)
            if mode == 'voice':
                self.stt.start(cancel)  # 살아 있는 공식 프로세스 재사용
                self._check(cancel)
                self._set(state='recording', message='지금 말하시오',
                          recording_until=time.monotonic() + core.COMMAND_WINDOW_SECONDS)
                text = self.stt.record(cancel)
                self._set(recording_until=0)
                if not text:
                    self._check(cancel)
                    raise ValueError('인식된 문장이 없습니다. 텍스트 입력으로도 사용할 수 있습니다.')
            self._check(cancel)
            print(f'[GUI {mode} 입력] {text}', flush=True)
            self._set(state='analyzing', text=text, message='api 전달중', recording_until=0)
            item = self.analyzer(text, self.key)
            self._check(cancel)
            print('[Claude 분석]', json.dumps(item, ensure_ascii=False), flush=True)
            self._set(command=item)
            if item['command'] != 'none':
                with self.motion_lock:
                    self._check(cancel)
                    suffix = ' 실행 중' if self.robot.motion_enabled else ' 미리보기'
                    self._set(state='executing', message=core.NAMES[item['command']] + suffix)
                    self.robot.execute(item, cancel)
            self._check(cancel)
            prefix = '' if self.robot.motion_enabled else '미리보기 · '
            self._set(state='done', message=prefix + item['reason'])
        except core.CommandCancelled as exc:
            print('[AI 취소]', str(exc), flush=True)
            self._set(state='cancelled', message=str(exc))
        except Exception as exc:
            message = str(exc).replace(self.key, '[키 숨김]') if self.key else str(exc)
            print('[GUI 명령 오류]', message, flush=True)
            self._set(state='error', error=message, message=message)
        finally:
            self._set(busy=False, recording_until=0)

    def manual(self, callback, *args, **kwargs):
        # The map's existing avoidance worker also uses the server command path.
        with self.lock:
            if threading.current_thread().name == 'HazardMapAvoid':
                self.avoidance_thread = threading.current_thread()
            self.cancel_event.set()
        with self.motion_lock:
            return callback(*args, **kwargs)

    def cancel(self):
        with self.lock:
            self.cancel_event.set()
        # An active action stops itself before releasing the motion lock.
        with self.motion_lock:
            return self.status()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.cancel_event.set()
        if self.worker is not None:
            self.worker.join(timeout=core.API_TIMEOUT + 2)
        self.stt.close()
        with self.motion_lock:
            pass  # Wait for any in-flight command's actuator cleanup.
        if self.http_server is not None:
            self.http_server.shutdown()
            self.http_server.server_close()


def handle_command_http(handler, service):
    path = urlsplit(handler.path).path
    if not path.startswith('/api/ai/'):
        return False
    def send(data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        handler.send_response(status)
        handler.send_header('Content-Type', 'application/json; charset=utf-8')
        handler.send_header('Cache-Control', 'no-store')
        handler.send_header('Content-Length', str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    if service is None:
        send({'error': '명령 서비스 준비 중'}, 503)
        return True
    if handler.command == 'GET' and path == '/api/ai/status':
        send(service.status())
        return True
    if handler.command != 'POST' or path not in ('/api/ai/text', '/api/ai/voice', '/api/ai/cancel'):
        send({'error': '지원하지 않는 요청'}, 404)
        return True
    origin = handler.headers.get('Origin')
    if origin and origin != 'http://' + handler.headers.get('Host', ''):
        send({'error': '접속 출처 오류'}, 403)
        return True
    try:
        length = int(handler.headers.get('Content-Length', '0'))
        if not 0 <= length <= 4096:
            raise ValueError('요청 크기 오류')
        data = json.loads(handler.rfile.read(length).decode('utf-8') or '{}')
        if not isinstance(data, dict):
            raise ValueError('JSON 객체 필요')
        if path.endswith('/cancel'):
            send(service.cancel())
        else:
            send(service.submit('voice' if path.endswith('/voice') else 'text', data.get('text', '')), 202)
    except (ValueError, UnicodeDecodeError) as exc:
        send({'error': str(exc)}, 400)
    except RuntimeError as exc:
        send({'error': str(exc)}, 409)
    return True


def start_command_service(robot, local_http=False):
    key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    service = CommandService(robot, key=key)
    if local_http:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if not handle_command_http(self, service):
                    self.send_error(404)

            do_POST = do_GET

            def log_message(self, *_):
                pass

        service.http_server = ThreadingHTTPServer(('127.0.0.1', core.COMMAND_PORT), Handler)
        service.http_thread = threading.Thread(target=service.http_server.serve_forever,
                                              daemon=True, name='PiCommandHTTP')
        service.http_thread.start()
    print('[AI 명령] ' + ('실제 동작 모드' if robot.motion_enabled else '미리보기 모드'), flush=True)
    if not key:
        print('[AI 명령] ANTHROPIC_API_KEY 미설정', flush=True)
    return service
