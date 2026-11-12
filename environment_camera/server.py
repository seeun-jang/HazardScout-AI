"""Standalone laptop HTTP server. Run: python server.py

Provides MJPEG live stream, patrol rounds, notes, red UNKNOWN alerts, review workflow.
NO Internet hosting/security hardening: use only trusted local networks.
"""
import hmac
import json
import os
import re
import signal
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

import cv2
import numpy as np

import config
from storage import Storage
from vision import OverheadDetector

ROOT = Path(__file__).resolve().parent
KEYWORDS = ('미확인', '확인되지 않은', 'unknown', '수상한', '장애물', '이상 물체')


class DemoDetector:
    """Static preview only; never claims to run YOLO or detect real UNKNOWN."""
    def __init__(self, frame):
        self.reference = frame.copy()
        self.model_error = ''
        self.models = []
        self.tracks = {}

    def set_reference(self, frame):
        self.reference = frame.copy()

    def process(self, frame, patrol_id=None, on_alert=None):
        return frame, {'known': 0, 'unknown_visible': 0, 'reference_ready': True}


class Monitor:
    def __init__(self, storage=None, detector=None):
        self.demo_mode = config.DEMO_MODE
        self.db = storage or Storage(config.DEMO_DB_PATH if self.demo_mode else config.DB_PATH)
        self.demo_frame = None
        if self.demo_mode:
            self.demo_frame = cv2.imread(str(config.DEMO_SCENE_PATH))
            if self.demo_frame is None:
                self.demo_frame = np.full((720, 960, 3), (29, 37, 45), dtype=np.uint8)
                cv2.putText(self.demo_frame, 'HAZARDSCOUT / DEMO SCENE',
                            (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 195, 210), 2)
        self.detector = detector or (DemoDetector(self.demo_frame) if self.demo_mode else OverheadDetector())
        self.lock = threading.RLock()
        self.patrol_id = None
        self.last_camera_frame = None
        self.last_jpeg = None
        self.camera_ok = False
        self.camera_error = ''
        self.vision_stats = {'known': 0, 'unknown_visible': 0, 'reference_ready': self.detector.reference is not None}
        self.running = True
        self.worker = None
        self.frame_counter = 0

    def start_worker(self):
        if self.demo_mode:
            self._restore_demo_feed()
            return
        self.worker = threading.Thread(target=self._capture_loop, daemon=True, name='overhead-camera')
        self.worker.start()

    def _restore_demo_feed(self):
        frame = self.demo_frame.copy()
        cv2.rectangle(frame, (7, 7), (frame.shape[1] - 7, frame.shape[0] - 7), (95, 171, 230), 2)
        cv2.putText(frame, 'DEMO MODE / NO LIVE CAMERA / NO YOLO INFERENCE', (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2)
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
        with self.lock:
            self.last_camera_frame = self.demo_frame.copy()
            self.last_jpeg = jpg.tobytes() if ok else None
            self.camera_ok = bool(ok)
            self.camera_error = ''
            self.vision_stats = {'known': 0, 'unknown_visible': 0, 'reference_ready': True}

    def inject_demo_unknown(self):
        if not self.demo_mode:
            raise ValueError('데모 모드에서만 사용할 수 있는 기능입니다.')
        with self.lock:
            if self.patrol_id is None:
                raise ValueError('먼저 데모 순찰을 시작하세요.')
            frame = self.demo_frame.copy()
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = int(w*.32), int(h*.33), int(w*.48), int(h*.49)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (50,65,245), 4)
            cv2.rectangle(frame, (x1,y1-28), (min(w-1,x1+252),y1), (50,65,245), -1)
            cv2.putText(frame, 'UNKNOWN / DEMO', (x1+8, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (255,255,255), 2)
            cv2.putText(frame, 'SIMULATED EVENT / NO YOLO INFERENCE', (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
            if not ok:
                raise ValueError('데모 이미지 생성 실패')
            self.last_jpeg = jpg.tobytes()
            self.vision_stats = {'known': 0, 'unknown_visible': 1, 'reference_ready': True}
            image = self._save_snapshot(frame, 'demo-unknown')
            return self.db.create_alert(self.patrol_id, 'A구역', image=image,
                details='데모 이벤트: 실제 YOLO 검출이 아닌 가상 미확인 후보입니다.', source='DEMO')

    def _save_snapshot(self, frame, prefix):
        name = f'{prefix}-{int(time.time()*1000)}-{os.getpid()}.jpg'
        dest = config.CAPTURE_DIR / name
        cv2.imwrite(str(dest), frame)
        return '/captures/' + name

    def _auto_alert(self, candidate, annotated):
        with self.lock:
            patrol_id = self.patrol_id
            if patrol_id is None:
                return
            # Red UNKNOWN is a proposal, *not* proof of danger.
            image = self._save_snapshot(annotated, 'unknown')
            self.db.create_alert(patrol_id, candidate['zone'], image=image,
                                 details='고정 카메라에서 기준 화면 대비 미분류 변화 감지; 현장 확인 필요',
                                 source='AUTO')

    def _capture_loop(self):
        source = config.CAMERA_SOURCE
        cap = None
        while self.running:
            try:
                if cap is None or not cap.isOpened():
                    backend = cv2.CAP_DSHOW if os.name == 'nt' and isinstance(source, int) else cv2.CAP_ANY
                    cap = cv2.VideoCapture(source, backend)
                    if not cap.isOpened():
                        raise OSError(f'카메라를 열 수 없음 (CAMERA_SOURCE={source!r})')
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, raw = cap.read()
                if not ok or raw is None:
                    raise OSError('카메라 프레임 읽기 실패')
                with self.lock:
                    self.last_camera_frame = raw.copy()
                    patrol_id = self.patrol_id
                annotated, stats = self.detector.process(raw, patrol_id, self._auto_alert)
                ok, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
                with self.lock:
                    if ok:
                        self.last_jpeg = jpg.tobytes()
                    self.vision_stats = stats
                    self.camera_ok = True
                    self.camera_error = ''
                    self.frame_counter += 1
                time.sleep(.035)
            except Exception as error:
                with self.lock:
                    self.camera_ok = False
                    self.camera_error = str(error)
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(2)
        if cap is not None:
            cap.release()

    def capture_reference(self):
        with self.lock:
            if self.patrol_id is not None:
                raise ValueError('순찰 중에는 기준 화면을 변경할 수 없습니다.')
            if self.last_camera_frame is None:
                raise ValueError('카메라 연결 후 다시 시도하세요.')
            self.detector.set_reference(self.last_camera_frame)
            self.vision_stats['reference_ready'] = True

    def start_patrol(self):
        with self.lock:
            if self.patrol_id is not None:
                raise ValueError('이미 순찰 진행 중입니다.')
            if not self.camera_ok:
                raise ValueError('환경카메라가 연결되지 않았습니다.')
            if self.detector.reference is None:
                raise ValueError('순찰 시작 전 비어 있는 현장의 기준 화면을 먼저 저장하세요.')
            if self.db.snapshot()['pending_total'] > 0:
                raise ValueError('관리자 확인 대기 중인 UNKNOWN이 있습니다. 먼저 확인 처리하세요.')
            self.patrol_id = self.db.open_patrol()
            self.detector.tracks.clear()
            if self.demo_mode:
                self._restore_demo_feed()
            return self.patrol_id

    def complete_patrol(self, note, review_required=False, zone='A구역'):
        with self.lock:
            if self.patrol_id is None:
                raise ValueError('진행 중인 순찰이 없습니다.')
            if len(note) > 1000:
                raise ValueError('특이사항은 1000자까지 입력할 수 있습니다.')
            needs_review = review_required or any(word in note.lower() for word in KEYWORDS)
            if zone not in ('A구역', 'B구역', 'C구역'):
                raise ValueError('구역 정보가 올바르지 않습니다.')
            # Text like 'B구역에서 미확인 물체 발견' overrides the select field.
            m = re.search(r'([ABC])\s*구역', note, flags=re.I)
            if m:
                zone = m.group(1).upper() + '구역'
            result = self.db.finish_patrol(self.patrol_id, note, zone if needs_review else None)
            self.patrol_id = None
            self.detector.tracks.clear()
            return result

    def set_worktime(self, value):
        if value:
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    # Local wall-clock time; browser sends datetime-local.
                    dt = dt.astimezone()
                value = dt.isoformat(timespec='minutes')
            except (ValueError, TypeError):
                raise ValueError('작업 시작 날짜와 시간을 확인하세요.')
        self.db.set_setting('worktime', value)
        return value

    def state(self):
        with self.lock:
            state = {
                'demo_mode': self.demo_mode,
                'patrol_active': self.patrol_id is not None,
                'patrol_id': self.patrol_id,
                'camera_ok': self.camera_ok,
                'camera_error': self.camera_error,
                'model_error': self.detector.model_error,
                'model_names': [f"{entry['role']}: {Path(entry['path']).name}" for entry in self.detector.models],
                'vision': self.vision_stats.copy(),
                'worktime': self.db.get_setting('worktime'),
            }
        state.update(self.db.snapshot())
        return state


monitor = None


class Handler(BaseHTTPRequestHandler):
    server_version = 'HazardScout-Laptop/1.0'

    def log_message(self, fmt, *args):
        # Avoid logging pin payloads and excessive stream requests.
        if self.path != '/video':
            super().log_message(fmt, *args)

    def send_bytes(self, content, mimetype, code=200, cache=False):
        self.send_response(code)
        self.send_header('Content-Type', mimetype)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'public,max-age=300' if cache else 'no-store')
        self.send_header('Referrer-Policy', 'same-origin')
        self.end_headers()
        self.wfile.write(content)

    def json_reply(self, payload, code=200):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                        'application/json; charset=utf-8', code)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/state':
            return self.json_reply(monitor.state())
        if path == '/video':
            return self.stream_video()
        static = {
            '/': (ROOT / 'static' / 'index.html', 'text/html; charset=utf-8'),
            '/static/style.css': (ROOT / 'static' / 'style.css', 'text/css; charset=utf-8'),
            '/static/app.js': (ROOT / 'static' / 'app.js', 'application/javascript; charset=utf-8'),
        }
        if path in static:
            file, typ = static[path]
            return self.send_bytes(file.read_bytes(), typ, cache=path != '/')
        if path.startswith('/captures/'):
            filename = unquote(path.split('/')[-1])
            if not re.fullmatch(r'[A-Za-z0-9_.-]+\.jpg', filename):
                return self.json_reply({'error': '잘못된 이미지 경로'}, 400)
            file = config.CAPTURE_DIR / filename
            if not file.is_file():
                return self.json_reply({'error': '이미지를 찾을 수 없음'}, 404)
            return self.send_bytes(file.read_bytes(), 'image/jpeg')
        return self.json_reply({'error': '경로를 찾을 수 없음'}, 404)

    def stream_video(self):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            while monitor.running:
                with monitor.lock:
                    data = monitor.last_jpeg
                if data:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                     str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
                    self.wfile.flush()
                time.sleep(.15)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return

    def read_json(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 16384 or n < 0:
            raise ValueError('요청 크기가 너무 큽니다.')
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            raise ValueError('JSON 요청만 지원합니다.')
        return json.loads(self.rfile.read(n).decode('utf-8')) if n else {}

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if not isinstance(payload, dict):
                raise ValueError('요청 형식이 올바르지 않습니다.')
            if path == '/api/demo/unknown':
                if not config.DEMO_MODE:
                    return self.json_reply({'error': '데모 모드에서만 사용 가능'}, 404)
                new_id = monitor.inject_demo_unknown()
                return self.json_reply({'ok': True, 'alert_id': new_id})
            if path == '/api/reference':
                monitor.capture_reference()
                return self.json_reply({'ok': True, 'message': '기준 화면 저장 완료'})
            if path == '/api/patrol/start':
                pid = monitor.start_patrol()
                return self.json_reply({'ok': True, 'patrol_id': pid})
            if path == '/api/patrol/complete':
                note = payload.get('note', '')
                if not isinstance(note, str) or not isinstance(payload.get('review_required', False), bool):
                    raise ValueError('특이사항 또는 관리자 확인 상태가 올바르지 않습니다.')
                result = monitor.complete_patrol(note.strip(), payload.get('review_required', False),
                                                 payload.get('zone', 'A구역'))
                return self.json_reply({'ok': True, **result})
            if path == '/api/worktime':
                value = payload.get('worktime', '')
                if not isinstance(value, str):
                    raise ValueError('작업 시작 시간이 올바르지 않습니다.')
                return self.json_reply({'ok': True, 'worktime': monitor.set_worktime(value)})
            match = re.fullmatch(r'/api/alerts/(\d+)/resolve', path)
            if match:
                if (not config.ADMIN_PIN or config.ADMIN_PIN.startswith('replace-with-')
                    or (not config.DEMO_MODE and config.ADMIN_PIN == 'DEMO-ONLY-2468')
                    or not hmac.compare_digest(str(payload.get('pin', '')), config.ADMIN_PIN)):
                    return self.json_reply({'error': '관리자 PIN이 올바르지 않거나 설정되지 않았습니다.'}, 403)
                result = payload.get('status')
                if result not in ('CONFIRMED', 'FALSE_ALARM'):
                    raise ValueError('처리 상태는 확인 완료 또는 오탐만 가능합니다.')
                ok = monitor.db.resolve_alert(int(match.group(1)), result)
                if not ok:
                    return self.json_reply({'error': '이미 처리되었거나 존재하지 않는 알림입니다.'}, 409)
                return self.json_reply({'ok': True})
            return self.json_reply({'error': '경로를 찾을 수 없음'}, 404)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self.json_reply({'error': str(error)}, 400)
        except Exception as error:
            print('[SERVER ERROR]', repr(error))
            return self.json_reply({'error': '서버 처리 중 오류가 발생했습니다. 콘솔을 확인하세요.'}, 500)


def main():
    global monitor
    if config.DEMO_MODE and config.HOST not in ('127.0.0.1', 'localhost', '::1'):
        raise RuntimeError('DEMO_MODE=1은 HOST=127.0.0.1에서만 허용합니다.')
    monitor = Monitor()
    monitor.start_worker()
    httpd = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    httpd.daemon_threads = True
    print(f'\nHazardScout Environment Camera: http://127.0.0.1:{config.PORT}')
    print(f'Listening on {config.HOST}:{config.PORT} | Camera: {config.CAMERA_SOURCE}')
    if config.DEMO_MODE:
        print('*** DEMO MODE *** 실제 카메라/YOLO 미실행, 독립 데모 DB 및 현장 사진 사용')
    if config.HOST == '0.0.0.0':
        print('LAN only. Do not expose this unprotected dashboard to the Internet.')
    if not config.ADMIN_PIN:
        print('WARNING: ADMIN_PIN 미설정. 관리자 확인 버튼을 사용할 수 없습니다.')
    try:
        httpd.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.running = False
        httpd.server_close()


if __name__ == '__main__':
    main()
