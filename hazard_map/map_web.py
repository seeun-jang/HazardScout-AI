#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
map_web.py

지도 보기용 작은 웹 서버 (기본 포트 5001)
- PC 서버, 모바일 서버 어느 쪽으로 실행해도 같은 주소로 지도를 볼 수 있음
    http://<라즈베리파이 IP>:5001/map
- GET  /api/map        : 지도 데이터 (JSON) -> 관제 대시보드에서 그대로 사용
- GET  /map.png        : 지도 그림
- POST /api/map/reset  : 지도 초기화 (로봇을 출발 칸에 다시 놓은 뒤)
- POST /api/map/goal   : {"cell": [x, y]} 목표 칸 지정 -> 막힌 칸을 피한 경로 표시
                         {"cell": null} 목표 해제
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

MAP_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HazardScout 지도</title>
<style>
  :root { --bg:#111418; --panel:#1b2027; --line:#2c333d; --text:#e8ecf1; --dim:#9aa4b2;
          --blocked:#d22828; --cone:#ff8c00; --item:#ffd700; --robot:#00c8ff; --visited:#466946; --path:#78ff78; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:system-ui, -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; }
  header { display:flex; align-items:center; gap:10px; padding:12px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; flex:1; }
  .pill { font-size:13px; padding:4px 10px; border-radius:999px; background:var(--panel); color:var(--dim); }
  .pill.ok { color:#7ee787; } .pill.bad { color:#ff7b72; }
  main { display:grid; grid-template-columns:minmax(0, 2fr) minmax(260px, 1fr); gap:16px; padding:16px; }
  @media (max-width: 800px) { main { grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; }
  .card h2 { font-size:14px; margin:0 0 8px; color:var(--dim); font-weight:600; }
  canvas { width:100%; height:auto; display:block; border-radius:8px; background:#282828; touch-action:manipulation; }
  .legend { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; font-size:12px; color:var(--dim); }
  .legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; background:var(--c); }
  .robot { font-size:14px; line-height:1.7; }
  .robot b { color:var(--text); }
  ul { list-style:none; margin:0; padding:0; font-size:13px; }
  li { padding:6px 0; border-bottom:1px solid var(--line); color:var(--dim); }
  li b { color:var(--text); font-weight:600; }
  .btns { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  button { background:#2b3340; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:8px 12px; font-size:14px; }
  button.warn { border-color:#7a2d2d; }
  .hint { font-size:12px; color:var(--dim); margin-top:6px; }
</style>
</head>
<body>
<header>
  <h1>HazardScout 위험 지도</h1>
  <span id="conn" class="pill">연결 중</span>
  <span id="avoid" class="pill" hidden>회피 중</span>
</header>
<main>
  <section class="card">
    <h2>지도 (칸 <span id="cellsize"></span>m)</h2>
    <canvas id="map"></canvas>
    <div class="legend">
      <span style="--c:var(--blocked)">바코드 막힘</span>
      <span style="--c:var(--cone)">통행 막는 위험물</span>
      <span style="--c:var(--item)">위험물·장비</span>
      <span style="--c:var(--robot)">로봇</span>
      <span style="--c:var(--visited)">지나온 칸</span>
      <span style="--c:var(--path)">목표까지 경로</span>
    </div>
    <div class="hint">칸을 누르면 목표로 지정되고, 막힌 곳을 피한 경로가 표시됩니다.</div>
    <div class="btns">
      <button id="clearGoal">목표 해제</button>
      <button id="reset" class="warn">지도 초기화</button>
    </div>
  </section>
  <section>
    <div class="card robot" id="robot"></div>
    <div class="card" style="margin-top:16px"><h2>발견한 위험물</h2><ul id="hazards"></ul></div>
    <div class="card" style="margin-top:16px"><h2>기록</h2><ul id="events"></ul></div>
  </section>
</main>
<script>
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
let last = null;
const ACTION = {left:'왼쪽', right:'오른쪽', back:'유턴', stop:'정지'};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function draw(d) {
  const px = Math.max(18, Math.floor(Math.min(900, canvas.parentElement.clientWidth - 24) / d.grid_w));
  canvas.width = d.grid_w * px; canvas.height = d.grid_h * px;
  ctx.fillStyle = '#282828'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const cell = (c, color) => { ctx.fillStyle = color; ctx.fillRect(c[0]*px+1, c[1]*px+1, px-2, px-2); };
  d.visited.forEach(c => cell(c, css('--visited')));
  d.blocked.forEach(b => cell(b.cell, css('--blocked')));
  ctx.strokeStyle = '#3c3c3c';
  for (let x = 0; x <= d.grid_w; x++) { ctx.beginPath(); ctx.moveTo(x*px, 0); ctx.lineTo(x*px, canvas.height); ctx.stroke(); }
  for (let y = 0; y <= d.grid_h; y++) { ctx.beginPath(); ctx.moveTo(0, y*px); ctx.lineTo(canvas.width, y*px); ctx.stroke(); }
  ctx.fillStyle = '#9aa4b2'; ctx.font = `${Math.max(9, px/3)}px sans-serif`;
  d.barcodes.forEach(b => ctx.fillText(b.code, b.cell[0]*px + 2, b.cell[1]*px + px/3 + 2));
  if (d.path && d.path.length > 1) {
    ctx.strokeStyle = css('--path'); ctx.lineWidth = Math.max(2, px/8); ctx.beginPath();
    d.path.forEach((c, i) => { const x = (c[0]+.5)*px, y = (c[1]+.5)*px; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.lineWidth = 1;
  }
  if (d.goal) { ctx.strokeStyle = css('--path'); ctx.lineWidth = 3; ctx.strokeRect(d.goal[0]*px+3, d.goal[1]*px+3, px-6, px-6); ctx.lineWidth = 1; }
  const s = px / d.cell_size_m;
  d.hazards.forEach(h => {
    ctx.fillStyle = h.blocks_path ? css('--cone') : css('--item');
    ctx.beginPath(); ctx.arc(h.x*s, h.y*s, px/3, 0, Math.PI*2); ctx.fill();
  });
  const r = d.robot, rx = r.x*s, ry = r.y*s, a = r.heading * Math.PI / 180;
  ctx.fillStyle = css('--robot'); ctx.beginPath(); ctx.arc(rx, ry, px/2.6, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(rx, ry);
  ctx.lineTo(rx + Math.sin(a)*px*0.8, ry - Math.cos(a)*px*0.8); ctx.stroke(); ctx.lineWidth = 1;
  canvas.dataset.px = px;
}

function fmtTime(t) { return new Date(t*1000).toLocaleTimeString('ko-KR', {hour12:false}); }

function render(d) {
  last = d;
  document.getElementById('cellsize').textContent = d.cell_size_m;
  draw(d);
  const r = d.robot;
  document.getElementById('robot').innerHTML =
    `<b>로봇 위치</b> 칸 (${r.cell[0]}, ${r.cell[1]}) · ${r.x.toFixed(2)}m, ${r.y.toFixed(2)}m<br>` +
    `<b>방향</b> ${r.heading_name} (${r.heading.toFixed(0)}°) · <b>상태</b> ${r.motion}<br>` +
    `<b>카메라 좌우</b> ${r.camera_pan > 0 ? '+' : ''}${r.camera_pan}° (오른쪽 +)<br>` +
    `<b>막힌 칸</b> ${d.blocked.length}개 · <b>위험물</b> ${d.hazards.length}개` +
    (d.goal ? `<br><b>목표</b> (${d.goal[0]}, ${d.goal[1]}) · ${d.path ? (d.path.length-1)+'칸' : '<span style="color:#ff7b72">갈 수 있는 길 없음</span>'}` : '');
  document.getElementById('avoid').hidden = !r.avoiding;
  const hz = document.getElementById('hazards');
  hz.innerHTML = d.hazards.length ? '' : '<li>아직 없음</li>';
  d.hazards.forEach(h => {
    const li = document.createElement('li');
    li.innerHTML = `<b>${h.name_ko}</b> 칸 (${h.cell[0]}, ${h.cell[1]}) · 신뢰도 ${h.conf} · ${h.hits}회${h.blocks_path ? ' · 통행 막음' : ''}`;
    hz.appendChild(li);
  });
  const ev = document.getElementById('events');
  ev.innerHTML = d.events.length ? '' : '<li>아직 없음</li>';
  d.events.slice(0, 12).forEach(e => {
    const li = document.createElement('li');
    li.innerHTML = `<b>${fmtTime(e.time)}</b> ${e.text}`;
    ev.appendChild(li);
  });
}

async function poll() {
  const pill = document.getElementById('conn');
  try {
    const r = await fetch('/api/map', {cache:'no-store'});
    render(await r.json());
    pill.textContent = '실시간'; pill.className = 'pill ok';
  } catch (e) {
    pill.textContent = '연결 끊김'; pill.className = 'pill bad';
  }
}

async function post(path, body) {
  await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
  poll();
}

canvas.addEventListener('click', ev => {
  if (!last) return;
  const rect = canvas.getBoundingClientRect(), px = canvas.width / last.grid_w;
  const x = Math.floor((ev.clientX - rect.left) * canvas.width / rect.width / px);
  const y = Math.floor((ev.clientY - rect.top) * canvas.height / rect.height / px);
  post('/api/map/goal', {cell:[x, y]});
});
document.getElementById('clearGoal').onclick = () => post('/api/map/goal', {cell:null});
document.getElementById('reset').onclick = () => {
  if (confirm('로봇을 출발 칸에 다시 놓았나요? 지도 기록을 모두 지웁니다.')) post('/api/map/reset');
};
window.addEventListener('resize', () => last && draw(last));
poll(); setInterval(poll, 1000);
</script>
</body>
</html>
"""


def make_handler(mapper):
    class MapHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def _send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=200):
            self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8", status)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/map"):
                self._send(MAP_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/map":
                self._json(mapper.to_dict())
            elif path == "/map.png":
                ok, png = cv2.imencode(".png", mapper.render(cell_px=40))
                if ok:
                    self._send(png.tobytes(), "image/png")
                else:
                    self._json({"status": "error"}, 500)
            else:
                self._json({"status": "error", "error": "없는 주소"}, 404)

        def do_POST(self):
            path = self.path.split("?", 1)[0]

            # 다른 사이트에서 몰래 요청하는 것 막기 (모바일 서버의 음성 API와 같은 방식)
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self._json({"status": "error", "error": "잘못된 접속 출처"}, 403)
                return

            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            except Exception:
                self._json({"status": "error", "error": "잘못된 요청"}, 400)
                return

            if path == "/api/map/reset":
                mapper.reset()
                self._json({"status": "ok"})
            elif path == "/api/map/goal":
                goal = mapper.set_goal(body.get("cell"))
                self._json({"status": "ok", "goal": list(goal) if goal else None})
            else:
                self._json({"status": "error", "error": "없는 주소"}, 404)

    return MapHandler


def start(mapper, port=5001, host=""):
    """백그라운드에서 지도 웹 서버 시작. 실패해도 로봇 서버는 계속 동작"""
    try:
        server = ThreadingHTTPServer((host, port), make_handler(mapper))
    except OSError as e:
        print(f"[지도 웹] 포트 {port} 열기 실패: {e!r}", flush=True)
        return None
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5},
                     daemon=True, name="MapWeb").start()
    print(f"[지도 웹] http://<라즈베리파이-IP>:{port}/map", flush=True)
    return server
