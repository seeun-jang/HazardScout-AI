'use strict';
const $ = id => document.getElementById(id);
let latest = null;
let selectedAlert = null;
let toastTimer;
let isBusy = false;
let lastDataHash = '';

function esc(value){return String(value ?? '').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));}
function showToast(message, error = false){const el=$('toast');el.textContent=message;el.className=error?'error':'';el.style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.style.display='none',4100);}
async function api(path, body){
  const options={method:body===undefined?'GET':'POST',cache:'no-store'};
  if(body!==undefined){options.headers={'Content-Type':'application/json'};options.body=JSON.stringify(body);}
  const response=await fetch(path,options);
  let result;try{result=await response.json();}catch{throw new Error('서버 응답을 읽을 수 없습니다.');}
  if(!response.ok)throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function fmtTime(t){if(!t)return '—';const d=new Date(t);return Number.isNaN(d.getTime())?'—':d.toLocaleString('ko-KR',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});}
function fmtNow(){const d=new Date();$('clock').textContent=d.toLocaleString('ko-KR',{hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});$('today-date').textContent=d.toLocaleDateString('ko-KR',{year:'numeric',month:'2-digit',day:'2-digit',weekday:'short'});}
function updateCountdown(){
  if(!latest||!latest.worktime){$('countdown').textContent='--:--';$('countdown-sub').textContent='시작 시각을 설정하세요';$('counter-line').style.width='0%';return;}
  const target=new Date(latest.worktime).getTime();if(Number.isNaN(target)){return;}
  const diff=Math.max(0,target-Date.now());const hours=Math.floor(diff/3600000), mins=Math.floor(diff%3600000/60000),sec=Math.floor(diff%60000/1000);
  $('countdown').textContent=diff<=0?'00:00':hours>0?`${hours}h ${String(mins).padStart(2,'0')}m`:`${String(mins).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
  $('countdown-sub').textContent=diff<=0?'작업 시작 시각 도래':hours>0?'작업 시작까지 남은 시간':'작업 시작까지 남은 분 · 초';
  $('counter-line').style.width=diff<=0?'100%':`${Math.max(5,100-Math.min(100,diff/3600000*20))}%`;
}
function renderAlerts(alerts){
  $('alert-count').textContent=`${alerts.length} EVENTS`;
  if(!alerts.length){$('alerts-list').innerHTML='<div class="empty-state">탐지된 UNKNOWN 기록이 없습니다.</div>';return;}
  $('alerts-list').innerHTML=alerts.slice(0,12).map(a=>{
    const pending=a.status==='PENDING';const status=pending?'관리자 확인 대기':a.status==='CONFIRMED'?'확인 완료':'오탐 처리';
    const icon=pending?'ic-alert':'ic-check';
    return `<article class="alert-item"><div class="alert-item-icon ${pending?'':'resolved'}"><svg><use href="#${icon}"/></svg></div>
    <div class="alert-item-main"><div class="alert-item-title">${esc(a.zone)} <span class="hot">/ UNKNOWN</span> ${a.source==='DEMO'?'<span class="hot">· DEMO</span>':''}</div>
      <div class="alert-item-meta">${a.source==='AUTO'?'환경 카메라 자동 탐지':a.source==='DEMO'?'시뮬레이션 알림':'수동 보고'} · 순찰 #${Number(a.patrol_id)||0}</div>
      ${a.details?`<div class="alert-item-details">${esc(a.details)}</div>`:''}
      <div class="alert-item-status ${pending?'':'resolved'}">${pending?'● ': '✓ '}${status}</div>
      <div class="alert-actions">${a.image?`<a href="${encodeURI(a.image)}" target="_blank" rel="noopener noreferrer" class="mini-button">기록 이미지</a>`:''}${pending?`<button type="button" class="mini-button" data-alert-id="${Number(a.id)||0}">관리자 확인</button>`:''}</div></div><time>${esc(fmtTime(a.created_at))}</time></article>`;
  }).join('');
  document.querySelectorAll('[data-alert-id]').forEach(button=>button.addEventListener('click',()=>{
    selectedAlert=Number(button.dataset.alertId);const a=alerts.find(x=>x.id===selectedAlert);
    $('review-info').textContent=`${a?.zone||'구역 미확인'} / UNKNOWN 알림 #${selectedAlert}을 현장에서 확인한 뒤 상태를 선택하세요.`;
    $('admin-pin').value='';$('review-dialog').showModal();
  }));
}
function renderPatrols(items){
  $('history-count').textContent=`${items.length} ROUNDS`;
  if(!items.length){$('patrols-list').innerHTML='<div class="empty-state">아직 저장된 순찰 기록이 없습니다.</div>';return;}
  $('patrols-list').innerHTML=items.map(p=>{
    const map={IN_PROGRESS:['순찰 중','progress'],REVIEW_REQUIRED:['관리자 확인 필요','warn'],REVIEW_DONE:['검토 완료',''],NORMAL:['정상 완료','']};const [label,cls]=map[p.status]||['완료',''];
    return `<article class="history-item"><div class="history-item-top"><strong class="history-item-title">PATROL <span style="color:#8297ac">/</span> #${Number(p.id)||0}</strong><span class="history-item-status ${cls}">${label}</span></div>
      <div class="history-time">${fmtTime(p.started_at)} — ${fmtTime(p.completed_at)} · UNKNOWN ${Number(p.unknown_total)||0}건</div>${p.note?`<div class="history-note">${esc(p.note)}</div>`:''}</article>`;
  }).join('');
}
function setStep(id,done){$(id).classList.toggle('done',!!done);}
function render(state){
  const priorWork=latest?.worktime, priorMode=latest?.demo_mode;latest=state;
  const demo=!!state.demo_mode, active=!!state.patrol_active, pending=(Number(state.pending_total)||0)>0;
  const cam=!!state.camera_ok, modelNames=Array.isArray(state.model_names)?state.model_names:[];
  $('demo-banner').hidden=!demo;
  $('demo-unknown-btn').hidden=!demo;
  $('demo-unknown-btn').disabled=!active;
  $('camera-pill').className='status-chip '+(demo?'demo':cam?'online':'');
  $('camera-pill').innerHTML=`<span class="status-dot"></span> ${demo?'DEMO / SAMPLE FEED':cam?'CAMERA ONLINE':'CAMERA OFFLINE'}`;
  $('feed-mode').textContent=demo?'DEMO / NOT LIVE':cam?'LIVE ANALYSIS':'NO SIGNAL';
  $('camera-feed').hidden=!cam;
  $('offline-placeholder').hidden=cam;
  $('video-stage').classList.toggle('is-offline',!cam);
  $('feed-live').classList.toggle('on',cam&&!demo);
  $('feed-live').lastElementChild.textContent=demo?'SAMPLE':cam?'LIVE':'STANDBY';
  $('kpi-camera').textContent=demo?'데모 영상':cam?'연결됨':'연결 대기';
  $('kpi-camera-sub').textContent=demo?'실제 카메라 사용 안 함':cam?'영상 정상 수신':'USB / IP 카메라 연결 필요';
  $('kpi-camera-indicator').style.background=demo?'#caa46d':cam?'#52c99a':'#526375';
  $('patrol-state').textContent=active?`진행 중 #${state.patrol_id}`:'대기 중';
  $('kpi-patrol-sub').textContent=active?'현재 환경 감시 중': '새로운 순찰 시작 가능 여부 확인';
  $('kpi-known').textContent=demo?'—':String(state.vision?.known??0).padStart(2,'0');
  $('pending-total').innerHTML=`${Number(state.pending_total)||0} <small>건</small>`;
  $('kpi-alert-sub').textContent=pending?'관리자 확인 후 작업 진행':'미확인 경고 없음';
  $('kpi-alert-card').classList.toggle('has-alerts',pending);
  $('sidebar-pending').textContent=String(state.pending_total||0);
  $('reference-status').textContent=state.vision?.reference_ready?'기준 화면 저장됨':'기준 화면 저장 필요';
  $('object-count').textContent=demo?'DEMO · 분석 미실행':`KNOWN ${state.vision?.known||0} · UNKNOWN ${state.vision?.unknown_visible||0}`;
  $('model-short').textContent=demo?'SIMULATION':modelNames.length?`${modelNames.length}개 모델 활성`:'미설정';
  $('side-cam').textContent=demo?'데모 모드':cam?'온라인':'오프라인';
  $('side-cam-dot').className='health-dot '+(demo?'':cam?'good':'bad');
  $('side-model').textContent=demo?'미사용':state.model_error?'오류':modelNames.length?'활성':'미설정';
  $('side-model-dot').className='health-dot '+(demo?'':state.model_error?'bad':modelNames.length?'good':'');
  $('start-btn').disabled=active||!cam||!state.vision?.reference_ready||pending;
  $('reference-btn').disabled=active||!cam;
  $('finish-btn').disabled=!active;
  $('submit-report-btn').disabled=!active;
  $('report-tag').textContent=active?`PATROL #${state.patrol_id} ACTIVE`:'PATROL STANDBY';
  $('report-tag').classList.toggle('active',active);
  if(pending){$('main-status').textContent='관리자 확인이 필요합니다.';$('main-subtitle').textContent=`미확인 물체 ${state.pending_total}건이 접수되었습니다. 현장 점검 및 관리자 확인 후 작업을 진행하세요.`;$('mission-kicker').textContent='SAFETY REVIEW REQUIRED';}
  else if(active){$('main-status').textContent='순찰이 진행 중입니다.';$('main-subtitle').textContent=`순찰 #${state.patrol_id} · 환경카메라 관제 및 이상 물체 감시 중`;$('mission-kicker').textContent='MISSION IN PROGRESS';}
  else {$('main-status').textContent='순찰을 시작하십시오.';$('main-subtitle').textContent=cam?'기준 화면을 저장하고 작업을 시작하세요.':'환경카메라를 연결하면 실시간 관제를 시작할 수 있습니다.';$('mission-kicker').textContent=demo?'DEMO READY':'MISSION READY';}
  $('mission-hint').textContent=pending?'REVIEW REQUIRED':active?'MONITORING ACTIVE':demo?'SIMULATION STANDBY':'SYSTEM STANDBY';
  document.querySelector('.mission-panel').classList.toggle('alert-mode',pending);
  $('review-label').textContent=`${Number(state.pending_total)||0} PENDING`;
  $('urgent-alert').classList.toggle('needs-review',pending);
  $('urgent-alert').querySelector('strong').textContent=pending?'관리자 확인 필요':'현재 구역 정상';
  $('urgent-alert').querySelector('span').textContent=pending?`UNKNOWN ${state.pending_total}건 확인 대기 중 · 작업 전 관리자 확인 요청`:'현재 확인 대기 중인 UNKNOWN이 없습니다.';
  setStep('step-reference',state.vision?.reference_ready);
  setStep('step-camera',cam);
  setStep('step-patrol',active||((state.patrols||[]).some(x=>x.completed_at)));
  setStep('step-review',!pending && (state.patrols||[]).some(x=>x.completed_at));
  $('schedule-hint').textContent=state.model_error&&!demo?'모델 상태: '+state.model_error:state.worktime?'시각을 변경하면 카운트다운이 즉시 갱신됩니다.':'시각을 등록하면 실시간 카운트다운이 표시됩니다.';
  if(state.worktime!==priorWork && document.activeElement!==$('worktime'))$('worktime').value=state.worktime?state.worktime.slice(0,16):'';
  if(priorMode!==undefined&&priorMode!==demo&&cam){$('camera-feed').src=`/video?mode=${demo?'demo':'live'}&t=${Date.now()}`;}
  const json=JSON.stringify([state.alerts||[],state.patrols||[]]);
  if(json!==lastDataHash){lastDataHash=json;renderAlerts(state.alerts||[]);renderPatrols(state.patrols||[]);}
  updateCountdown();
}async function refresh(){try{render(await api('/api/state'));}catch(err){if(!latest)showToast(`서버 연결 실패: ${err.message}`,true);$('side-cam').textContent='서버 확인 필요';}}
async function action(fn,success){if(isBusy)return false;isBusy=true;try{await fn();showToast(success);await refresh();return true;}catch(err){showToast(err.message,true);return false;}finally{isBusy=false;}}
$('reference-btn').addEventListener('click',()=>{
  if(!confirm('현재 카메라 화면을 기준 장면으로 저장할까요? 로봇과 신규 미확인 물체가 없는 상태에서만 저장하세요.'))return;
  action(()=>api('/api/reference',{}),'기준 화면을 저장했습니다.');
});
$('start-btn').addEventListener('click',()=>action(()=>api('/api/patrol/start',{}),'순찰이 시작되었습니다.'));
$('finish-btn').addEventListener('click',()=>{$('report').scrollIntoView({behavior:'smooth',block:'start'});$('patrol-note').focus({preventScroll:true});showToast('순찰 종료 내용을 작성한 뒤 보고서 저장 버튼을 누르세요.');});
$('submit-report-btn').addEventListener('click',async()=>{
  const note=$('patrol-note').value.trim(),zone=$('patrol-zone').value,review_required=$('manual-review').checked;
  const result=await action(()=>api('/api/patrol/complete',{note,zone,review_required}),'순찰 회차가 저장되었습니다. 관리자 확인 목록도 확인하세요.');
  if(result){$('patrol-note').value='';$('manual-review').checked=false;$('note-count').textContent='0 / 1000';}
});
$('schedule-form').addEventListener('submit',e=>{e.preventDefault();action(()=>api('/api/worktime',{worktime:$('worktime').value}),'작업 시작 시간을 저장했습니다.');});
$('patrol-note').addEventListener('input',()=>{$('note-count').textContent=`${$('patrol-note').value.length} / 1000`;});
async function resolveAlert(status){
  if(!selectedAlert)return;
  const pin=$('admin-pin').value;
  if(!pin){showToast('관리자 PIN을 입력하세요.',true);return;}
  const ok=await action(()=>api(`/api/alerts/${selectedAlert}/resolve`,{pin,status}),status==='CONFIRMED'?'관리자 확인 완료로 기록했습니다.':'오탐 처리로 기록했습니다.');
  if(ok){$('review-dialog').close();selectedAlert=null;}
}
$('confirmed-btn').addEventListener('click',()=>resolveAlert('CONFIRMED'));
$('false-alarm-btn').addEventListener('click',()=>resolveAlert('FALSE_ALARM'));
$('review-form').addEventListener('submit',e=>{e.preventDefault();resolveAlert('CONFIRMED');});
$('demo-unknown-btn').addEventListener('click',()=>{
  if(!confirm('DEMO MODE 전용입니다. 실제 탐지가 아닌 가상 UNKNOWN 관리자 확인 알림을 생성할까요?'))return;
  action(()=>api('/api/demo/unknown',{}),'DEMO UNKNOWN이 기록되었습니다. 관리자 확인 센터를 확인하세요.');
});
document.querySelectorAll('.nav-link').forEach(link=>link.addEventListener('click',()=>{
  document.querySelectorAll('.nav-link').forEach(x=>x.classList.remove('active'));
  link.classList.add('active');
}));
setInterval(()=>{fmtNow();updateCountdown();},1000);
setInterval(refresh,2500);
fmtNow();refresh();
