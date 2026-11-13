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
<meta name="theme-color" content="#ffffff">
<title>HazardScout · 로봇 위험 지도</title>
<style>
  :root {
    color-scheme: light;
    --bg:#f5f7fb; --panel:#ffffff; --panel2:#f8fafc; --stroke:#dfe6ef; --line:#e9eef5;
    --text:#17263d; --head:#243854; --soft:#66788e; --faint:#8899ad;
    --accent:#2563eb; --accent-soft:#eaf1ff; --red:#df4351; --red-soft:#fff0f2;
    --green:#188557; --green-soft:#e8f7ef;
    --blocked:#df4351; --cone:#f08c2e; --item:#e2a610; --robot:#2563eb; --visited:#dbe7ff; --path:#16a34a;
    --grid-bg:#fbfcfe; --grid-line:#e6ecf3;
    font-family: Inter, "Pretendard Variable", Pretendard, "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", -apple-system, "Segoe UI", sans-serif;
  }
  * { box-sizing:border-box; }
  html, body { margin:0; background:var(--bg); color:var(--text); }
  button { font:inherit; cursor:pointer; }
  [hidden] { display:none !important; }

  .topbar { height:64px; background:#fff; border-bottom:1px solid #e0e7ef; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 clamp(16px,2.6vw,40px); position:sticky; top:0; z-index:5; }
  .brand { display:flex; align-items:center; gap:12px; }
  .logo { position:relative; width:36px; height:36px; border-radius:11px; background:#e9f0ff; color:#1747a4; display:grid; place-items:center; transform:rotate(-8deg); box-shadow:0 5px 14px #2e5eb116; }
  .logo b { font-size:21px; font-weight:950; letter-spacing:-2px; transform:rotate(8deg); }
  .logo i { position:absolute; right:-3px; bottom:7px; width:11px; height:5px; border-radius:3px; background:#e24c58; transform:rotate(-37deg); }
  .brand-text { display:flex; flex-direction:column; gap:4px; }
  .brand-text strong { font-size:12px; letter-spacing:1.2px; font-weight:900; color:#192c46; }
  .brand-text span { font-size:9px; letter-spacing:1.4px; font-weight:700; color:#718399; }
  .crumb { font-size:10px; font-weight:750; letter-spacing:1px; color:#8494a5; margin-left:22px; padding-left:22px; border-left:1px solid #e3e9f0; }
  .crumb strong { color:#354a66; letter-spacing:0; font-size:11px; margin-left:6px; }
  .top-right { display:flex; align-items:center; gap:12px; }
  .chip { display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border-radius:6px; border:1px solid var(--stroke); background:#fff; color:var(--soft); font-size:10px; font-weight:850; letter-spacing:.9px; white-space:nowrap; }
  .chip .dot { width:7px; height:7px; border-radius:50%; background:#a2b0c0; }
  .chip.ok { color:var(--green); border-color:#bfe6d2; background:var(--green-soft); }
  .chip.ok .dot { background:#2ca374; box-shadow:0 0 0 3px #cdeedd; }
  .chip.bad { color:var(--red); border-color:#f6c4ca; background:var(--red-soft); }
  .chip.bad .dot { background:var(--red); }
  .chip.avoid { color:#fff; border-color:var(--red); background:var(--red); animation:pulse 1.2s ease-in-out infinite; }
  .chip.avoid .dot { background:#fff; }
  @keyframes pulse { 50% { box-shadow:0 0 0 6px #df435122; } }
  time { font-size:12px; color:#5d6f86; font-variant-numeric:tabular-nums; }

  main { max-width:1800px; margin:0 auto; padding:16px clamp(16px,2.6vw,40px) 14px; }
  .intro { display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:12px; }
  .eyebrow { display:flex; align-items:center; gap:8px; font-size:10px; font-weight:900; letter-spacing:2.1px; color:#667e9c; margin-bottom:7px; }
  .eyebrow i { width:9px; height:9px; border:2px solid var(--accent); transform:rotate(45deg); }
  h1 { margin:0; font-size:clamp(21px,1.7vw,26px); letter-spacing:-1.2px; font-weight:840; color:#14243b; }
  h1 small { font-size:11px; font-weight:700; letter-spacing:1.5px; color:#8293a6; margin-left:12px; }
  .intro p { margin:0; font-size:11px; color:#74869b; text-align:right; max-width:420px; line-height:1.6; }

  .layout { display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:16px; align-items:start; }
  .panel { background:var(--panel); border:1px solid var(--stroke); border-radius:11px; padding:15px 16px; box-shadow:0 8px 24px #24365208; min-width:0; }
  .panel-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:10px; }
  .overline { color:#7d8ea3; font-size:9px; letter-spacing:2px; font-weight:850; margin-bottom:6px; }
  .panel-head h3 { margin:0; font-size:16px; letter-spacing:-.5px; font-weight:800; color:var(--head); }
  .tag { font-size:9px; font-weight:850; letter-spacing:.8px; color:#55708f; border:1px solid #d6e0ec; background:#f6f9fd; border-radius:5px; padding:6px 9px; white-space:nowrap; }

  .map-wrap { position:relative; background:var(--grid-bg); border:1px solid #e1e8f1; border-radius:8px; padding:10px; display:flex; justify-content:center; }
  canvas { display:block; touch-action:manipulation; cursor:crosshair; border-radius:4px; }
  .legend { display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:10px; font-size:11px; color:#5f7189; font-weight:600; }
  .legend span { display:inline-flex; align-items:center; gap:6px; }
  .legend span::before { content:""; width:10px; height:10px; border-radius:3px; background:var(--c); }
  .legend .round::before { border-radius:50%; }
  .map-foot { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px; padding-top:10px; border-top:1px solid var(--line); flex-wrap:wrap; }
  .hint { font-size:11px; color:#7a8ca1; }
  .btns { display:flex; gap:8px; }
  .button { border-radius:6px; padding:9px 14px; border:1px solid #d5dfeb; background:#fff; color:#2a405e; font-size:11px; font-weight:850; transition:transform .15s, background .15s; }
  .button:hover { transform:translateY(-1px); background:#f3f7fc; }
  .button.danger { color:#c3283a; border-color:#f3c3ca; background:#fff7f8; }
  .button.danger:hover { background:#ffecef; }

  .side { display:flex; flex-direction:column; gap:12px; min-width:0; }
  .kpis { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .kpi { position:relative; background:var(--panel); border:1px solid var(--stroke); border-radius:10px; padding:13px 14px; overflow:hidden; box-shadow:0 8px 24px #24365206; }
  .kpi .k { font-size:9px; font-weight:850; letter-spacing:1.4px; color:#7d8ea3; }
  .kpi strong { display:block; margin-top:9px; font-size:21px; letter-spacing:-.8px; font-weight:800; color:#14243b; font-variant-numeric:tabular-nums; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .kpi strong small { font-size:12px; font-weight:700; color:#6a7c92; margin-left:3px; }
  .kpi .s { font-size:10px; color:#8193a8; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .kpi::after { content:""; position:absolute; right:0; bottom:12px; width:4px; height:24px; border-radius:3px; background:#c6d3e2; }
  .kpi.blue::after { background:var(--accent); }
  .kpi.red::after { background:var(--red); }
  .kpi.red strong { color:var(--red); }

  .status-row { display:flex; justify-content:space-between; gap:10px; padding:7px 0; border-bottom:1px solid var(--line); font-size:12px; }
  .status-row:last-child { border-bottom:0; padding-bottom:0; }
  .status-row span { color:#7487a0; }
  .status-row b { color:#1d3150; font-weight:750; text-align:right; }
  .warn-text { color:var(--red) !important; }

  ul { list-style:none; margin:0; padding:0; max-height:150px; overflow:auto; scrollbar-color:#cfd9e5 transparent; }
  li { display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--line); font-size:12px; color:#4e6280; line-height:1.5; }
  li:last-child { border-bottom:0; }
  li.empty { justify-content:center; color:#93a3b5; padding:14px 0; font-size:11px; }
  .ico { flex:none; width:26px; height:26px; border-radius:6px; display:grid; place-items:center; font-size:12px; font-weight:900; }
  .ico.hz { background:#fff4e5; color:#c26a0a; }
  .ico.hz.block { background:var(--red-soft); color:var(--red); }
  .li-main { flex:1; min-width:0; }
  .li-main b { color:#1d3150; font-weight:800; }
  .li-sub { font-size:10px; color:#8496aa; margin-top:2px; }
  .badge { font-size:9px; font-weight:850; color:var(--red); border:1px solid #f3c3ca; background:var(--red-soft); border-radius:4px; padding:2px 6px; margin-left:6px; }
  .ev-time { flex:none; font-size:10px; font-weight:750; color:#6a7f99; background:#f1f5fa; border-radius:4px; padding:3px 6px; font-variant-numeric:tabular-nums; }

  footer { display:flex; justify-content:space-between; gap:16px; margin-top:16px; padding-top:14px; border-top:1px solid #e3e9f0; font-size:9px; color:#8494a6; letter-spacing:.9px; font-weight:750; }

  @media (max-width:1100px) { .layout { grid-template-columns:1fr; } .crumb { display:none; } }
  @media (max-width:620px) { .brand-text span, time { display:none; } .kpis { grid-template-columns:1fr 1fr; } }
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <div class="logo" aria-hidden="true"><b>H</b><i></i></div>
    <div class="brand-text"><strong>HAZARDSCOUT</strong><span>ROBOT MAPPING</span></div>
    <div class="crumb">PROJECT / HAZARDSCOUT ›<strong>로봇 위험 지도</strong></div>
  </div>
  <div class="top-right">
    <span id="avoid" class="chip avoid" hidden><span class="dot"></span>바코드 회피 중</span>
    <span id="conn" class="chip"><span class="dot"></span><span id="connText">연결 중</span></span>
    <time id="clock">--:--:--</time>
  </div>
</header>

<main>
  <div class="intro">
    <div>
      <div class="eyebrow"><i></i>ROBOT INTELLIGENCE / LIVE MAP</div>
      <h1>로봇 위험 지도<small>/ GRID MAPPING</small></h1>
    </div>
    <p>바코드로 막힌 칸, YOLO가 찾은 위험물,<br>로봇 위치와 경로를 실시간으로 표시합니다.</p>
  </div>

  <div class="layout">
    <section class="panel">
      <div class="panel-head">
        <div><div class="overline">MAP / 01</div><h3>현장 격자 지도</h3></div>
        <span class="tag" id="gridTag">GRID</span>
      </div>
      <div class="map-wrap" id="mapWrap"><canvas id="map"></canvas></div>
      <div class="legend">
        <span style="--c:var(--blocked)">바코드 막힘</span>
        <span class="round" style="--c:var(--cone)">통행 막는 위험물</span>
        <span class="round" style="--c:var(--item)">위험물·장비</span>
        <span class="round" style="--c:var(--robot)">로봇</span>
        <span style="--c:var(--visited)">지나온 칸</span>
        <span style="--c:var(--path)">목표까지 경로</span>
      </div>
      <div class="map-foot">
        <div class="hint">칸을 누르면 목표로 지정되고, 막힌 곳을 피한 경로가 표시됩니다.</div>
        <div class="btns">
          <button class="button" id="clearGoal">목표 해제</button>
          <button class="button danger" id="reset">지도 초기화</button>
        </div>
      </div>
    </section>

    <aside class="side">
      <div class="kpis">
        <div class="kpi blue"><div class="k">ROBOT CELL</div><strong id="kCell">-</strong><div class="s" id="kPos">위치</div></div>
        <div class="kpi"><div class="k">HEADING</div><strong id="kHead">-</strong><div class="s" id="kMotion">상태</div></div>
        <div class="kpi" id="kBlockedCard"><div class="k">BLOCKED</div><strong id="kBlocked">0<small>칸</small></strong><div class="s">바코드로 막힌 칸</div></div>
        <div class="kpi" id="kHazCard"><div class="k">HAZARDS</div><strong id="kHaz">0<small>개</small></strong><div class="s">발견한 위험물</div></div>
      </div>

      <section class="panel">
        <div class="panel-head"><div><div class="overline">ROBOT / 02</div><h3>로봇 상태</h3></div></div>
        <div id="robot"></div>
      </section>

      <section class="panel">
        <div class="panel-head"><div><div class="overline">DETECTION / 03</div><h3>발견한 위험물</h3></div><span class="tag" id="hazCount">0</span></div>
        <ul id="hazards"></ul>
      </section>

      <section class="panel">
        <div class="panel-head"><div><div class="overline">LOG / 04</div><h3>기록</h3></div></div>
        <ul id="events"></ul>
      </section>
    </aside>
  </div>

  <footer><span>HAZARDSCOUT — ROBOT GRID MAPPING</span><span>RASPBERRY PI · PORT 5001</span></footer>
</main>

<script>
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
let last = null, cellPx = 20;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MOTION = {forward:'전진', backward:'후진', left:'좌회전', right:'우회전', backleft:'후진 좌회전', backright:'후진 우회전', stop:'정지'};
const HEAD_KO = {N:'북', E:'동', S:'남', W:'서'};

function roundRect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function draw(d) {
  const wrap = document.getElementById('mapWrap');
  const availW = wrap.clientWidth - 22;
  const availH = Math.max(260, window.innerHeight - (wrap.getBoundingClientRect().top + window.scrollY) - 122);
  const px = Math.max(10, Math.floor(Math.min(availW / d.grid_w, availH / d.grid_h)));
  cellPx = px;
  const W = d.grid_w * px, H = d.grid_h * px, dpr = window.devicePixelRatio || 1;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  const cell = (c, color, inset) => { ctx.fillStyle = color; roundRect(c[0]*px + inset, c[1]*px + inset, px - inset*2, px - inset*2, Math.min(4, px/5)); ctx.fill(); };
  d.visited.forEach(c => cell(c, css('--visited'), 1));

  ctx.strokeStyle = css('--grid-line'); ctx.lineWidth = 1;
  for (let x = 0; x <= d.grid_w; x++) { ctx.beginPath(); ctx.moveTo(x*px + .5, 0); ctx.lineTo(x*px + .5, H); ctx.stroke(); }
  for (let y = 0; y <= d.grid_h; y++) { ctx.beginPath(); ctx.moveTo(0, y*px + .5); ctx.lineTo(W, y*px + .5); ctx.stroke(); }
  ctx.strokeStyle = '#d3dce8'; ctx.strokeRect(.5, .5, W - 1, H - 1);

  // 바코드 자리 (아직 안 읽은 곳은 점선)
  const blockedCodes = new Set(d.blocked.map(b => b.code));
  d.barcodes.forEach(b => {
    if (blockedCodes.has(b.code)) return;
    ctx.setLineDash([3, 3]); ctx.strokeStyle = '#e7a1a9'; ctx.lineWidth = 1.5;
    roundRect(b.cell[0]*px + 2, b.cell[1]*px + 2, px - 4, px - 4, 3); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  });
  d.blocked.forEach(b => cell(b.cell, css('--blocked'), 1.5));

  // 바코드 번호 표시
  const fs = Math.max(9, Math.min(12, px * 0.42));
  ctx.font = `800 ${fs}px Inter, "Malgun Gothic", sans-serif`; ctx.textBaseline = 'middle';
  d.barcodes.forEach(b => {
    const done = blockedCodes.has(b.code), tw = ctx.measureText(b.code).width + 8;
    let tx = (b.cell[0] + .5) * px - tw / 2, ty = b.cell[1] * px - fs - 4;
    if (ty < 2) ty = (b.cell[1] + 1) * px + 3;
    tx = Math.max(2, Math.min(W - tw - 2, tx));
    ctx.fillStyle = done ? css('--blocked') : '#ffffff'; ctx.strokeStyle = done ? css('--blocked') : '#efb3ba';
    roundRect(tx, ty, tw, fs + 4, 3); ctx.fill(); ctx.stroke();
    ctx.fillStyle = done ? '#ffffff' : '#c3283a'; ctx.fillText(b.code, tx + 4, ty + (fs + 4) / 2 + .5);
  });

  if (d.path && d.path.length > 1) {
    ctx.strokeStyle = css('--path'); ctx.lineWidth = Math.max(2.5, px / 6); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.globalAlpha = .9; ctx.beginPath();
    d.path.forEach((c, i) => { const x = (c[0] + .5)*px, y = (c[1] + .5)*px; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.globalAlpha = 1; ctx.lineWidth = 1;
  }
  if (d.goal) {
    ctx.strokeStyle = css('--path'); ctx.lineWidth = 2.5; ctx.setLineDash([4, 3]);
    roundRect(d.goal[0]*px + 2, d.goal[1]*px + 2, px - 4, px - 4, 4); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  }

  const s = px / d.cell_size_m;
  d.hazards.forEach(h => {
    const color = h.blocks_path ? css('--cone') : css('--item'), x = h.x*s, y = h.y*s, r = Math.max(4, px / 2.8);
    ctx.fillStyle = color + '33'; ctx.beginPath(); ctx.arc(x, y, r * 1.7, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.lineWidth = 1;
  });

  const r = d.robot, rx = r.x*s, ry = r.y*s, a = r.heading * Math.PI / 180, rr = Math.max(6, px / 2.2);
  ctx.fillStyle = r.avoiding ? '#df435126' : '#2563eb22';
  ctx.beginPath(); ctx.arc(rx, ry, rr * 2, 0, Math.PI*2); ctx.fill();
  ctx.save(); ctx.translate(rx, ry); ctx.rotate(a);
  ctx.fillStyle = r.avoiding ? css('--red') : css('--robot');
  ctx.beginPath(); ctx.arc(0, 0, rr, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.moveTo(0, -rr * .62); ctx.lineTo(rr * .42, rr * .3); ctx.lineTo(-rr * .42, rr * .3); ctx.closePath(); ctx.fill();
  ctx.restore(); ctx.lineWidth = 1;
}

const pad = n => String(n).padStart(2, '0');
function hms(dt) { return `${pad(dt.getHours())}:${pad(dt.getMinutes())}:${pad(dt.getSeconds())}`; }
function fmtTime(t) { return hms(new Date(t*1000)); }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function render(d) {
  last = d;
  const r = d.robot;
  document.getElementById('gridTag').textContent = `${d.grid_w}×${d.grid_h} · 칸 ${d.cell_size_m}m`;
  draw(d);

  document.getElementById('kCell').textContent = `(${r.cell[0]}, ${r.cell[1]})`;
  document.getElementById('kPos').textContent = `${r.x.toFixed(2)}m, ${r.y.toFixed(2)}m`;
  document.getElementById('kHead').innerHTML = `${esc(r.heading_name)}<small>${r.heading.toFixed(0)}°</small>`;
  document.getElementById('kMotion').textContent = r.avoiding ? '바코드 회피 중' : (MOTION[r.motion] || r.motion);
  document.getElementById('kBlocked').innerHTML = `${d.blocked.length}<small>칸</small>`;
  document.getElementById('kHaz').innerHTML = `${d.hazards.length}<small>개</small>`;
  document.getElementById('kBlockedCard').classList.toggle('red', d.blocked.length > 0);
  document.getElementById('kHazCard').classList.toggle('red', d.hazards.some(h => h.blocks_path));

  const pan = r.camera_pan;
  let goal = '<b>없음</b>';
  if (d.goal) goal = d.path ? `<b>(${d.goal[0]}, ${d.goal[1]}) · ${d.path.length - 1}칸</b>` : `<b class="warn-text">(${d.goal[0]}, ${d.goal[1]}) · 갈 수 있는 길 없음</b>`;
  document.getElementById('robot').innerHTML =
    `<div class="status-row"><span>주행 상태</span><b class="${r.avoiding ? 'warn-text' : ''}">${r.avoiding ? '바코드 회피 중' : esc(MOTION[r.motion] || r.motion)}</b></div>` +
    `<div class="status-row"><span>방향</span><b>${esc(HEAD_KO[r.heading_name] || r.heading_name)}쪽 (${r.heading.toFixed(0)}°)</b></div>` +
    `<div class="status-row"><span>카메라 좌우</span><b class="${Math.abs(pan) > 15 ? 'warn-text' : ''}">${pan > 0 ? '+' : ''}${pan}°${Math.abs(pan) > 15 ? ' · 회피 안 함' : ''}</b></div>` +
    `<div class="status-row"><span>목표</span>${goal}</div>`;
  document.getElementById('avoid').hidden = !r.avoiding;

  const hz = document.getElementById('hazards');
  document.getElementById('hazCount').textContent = d.hazards.length;
  hz.innerHTML = d.hazards.length ? '' : '<li class="empty">아직 발견한 위험물이 없습니다</li>';
  d.hazards.forEach(h => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="ico hz ${h.blocks_path ? 'block' : ''}">!</span>` +
      `<div class="li-main"><b>${esc(h.name_ko)}</b>${h.blocks_path ? '<span class="badge">통행 막음</span>' : ''}` +
      `<div class="li-sub">칸 (${h.cell[0]}, ${h.cell[1]}) · 신뢰도 ${h.conf} · ${h.hits}회</div></div>`;
    hz.appendChild(li);
  });

  const ev = document.getElementById('events');
  ev.innerHTML = d.events.length ? '' : '<li class="empty">아직 기록이 없습니다</li>';
  d.events.slice(0, 12).forEach(e => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="ev-time">${fmtTime(e.time)}</span><div class="li-main">${esc(e.text)}</div>`;
    ev.appendChild(li);
  });
}

async function poll() {
  const chip = document.getElementById('conn'), text = document.getElementById('connText');
  try {
    const r = await fetch('/api/map', {cache:'no-store'});
    render(await r.json());
    text.textContent = 'LIVE · 실시간'; chip.className = 'chip ok';
  } catch (e) {
    text.textContent = '연결 끊김'; chip.className = 'chip bad';
  }
}

async function post(path, body) {
  await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
  poll();
}

canvas.addEventListener('click', ev => {
  if (!last) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((ev.clientX - rect.left) / (rect.width / last.grid_w));
  const y = Math.floor((ev.clientY - rect.top) / (rect.height / last.grid_h));
  post('/api/map/goal', {cell:[x, y]});
});
document.getElementById('clearGoal').onclick = () => post('/api/map/goal', {cell:null});
document.getElementById('reset').onclick = () => {
  if (confirm('로봇을 출발 칸에 다시 놓았나요? 지도 기록을 모두 지웁니다.')) post('/api/map/reset');
};
window.addEventListener('resize', () => last && draw(last));
const tick = () => { document.getElementById('clock').textContent = hms(new Date()); };
tick(); setInterval(tick, 1000);
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
