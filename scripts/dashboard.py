#!/usr/bin/env python3
"""
암호화폐 자동매매 리모트 컨트롤 대시보드

iPhone에서 접속 가능한 웹 대시보드.
포트폴리오 조회, 시장 데이터, 긴급정지, 매매 이력 확인.
"""

import base64
import io
import json
import os
import subprocess
import sys
import socket
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template_string, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

app = Flask(__name__)
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python3")


def run_script(name, *args):
    """프로젝트 스크립트 실행 후 JSON 결과 반환"""
    try:
        r = subprocess.run(
            [VENV_PYTHON, str(PROJECT_ROOT / "scripts" / name), *args],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr.strip()}
    except Exception as e:
        return {"error": str(e)}


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Crypto Trading Bot</title>
<style>
:root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3; --dim: #8b949e; --green: #3fb950; --red: #f85149; --blue: #58a6ff; --yellow: #d29922; }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 16px; padding-top: env(safe-area-inset-top, 16px); }
h1 { font-size: 20px; text-align: center; padding: 12px 0; }
h1 span { color: var(--yellow); }
.status-bar { display: flex; justify-content: center; gap: 12px; margin-bottom: 16px; font-size: 12px; }
.badge { padding: 4px 10px; border-radius: 12px; font-weight: 600; }
.badge-safe { background: #0d1f0d; color: var(--green); border: 1px solid #1a3a1a; }
.badge-danger { background: #2d0f0f; color: var(--red); border: 1px solid #4a1a1a; }
.cards { display: flex; flex-direction: column; gap: 12px; max-width: 500px; margin: 0 auto; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.card h2 { font-size: 14px; color: var(--dim); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
.price { font-size: 32px; font-weight: 700; }
.change-up { color: var(--green); }
.change-down { color: var(--red); }
.metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 14px; }
.metric:last-child { border-bottom: none; }
.metric .label { color: var(--dim); }
.metric .value { font-weight: 600; }
.btn-row { display: flex; gap: 8px; margin-top: 12px; }
.btn { flex: 1; padding: 12px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity .2s; }
.btn:active { opacity: 0.7; }
.btn-red { background: var(--red); color: #fff; }
.btn-green { background: var(--green); color: #000; }
.btn-blue { background: var(--blue); color: #000; }
.btn-dim { background: var(--border); color: var(--text); }
.fgi-bar { height: 8px; border-radius: 4px; background: linear-gradient(to right, var(--red), var(--yellow), var(--green)); margin-top: 8px; position: relative; }
.fgi-marker { position: absolute; top: -4px; width: 16px; height: 16px; background: #fff; border-radius: 50%; border: 2px solid var(--bg); transform: translateX(-50%); }
.holding { padding: 8px 0; border-bottom: 1px solid var(--border); }
.holding:last-child { border-bottom: none; }
.holding-header { display: flex; justify-content: space-between; }
.holding-name { font-weight: 600; }
.spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border); border-top: 2px solid var(--blue); border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.refresh-time { text-align: center; font-size: 11px; color: var(--dim); margin-top: 12px; }
.toast { position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); background: var(--card); border: 1px solid var(--border); padding: 10px 20px; border-radius: 8px; font-size: 13px; display: none; z-index: 100; }
</style>
</head>
<body>
<h1>🤖 <span>Crypto Bot</span> Remote</h1>
<div class="status-bar">
  <span class="badge" id="dryRunBadge">-</span>
  <span class="badge" id="emergencyBadge">-</span>
</div>
<div class="cards">
  <!-- 시세 -->
  <div class="card" id="priceCard">
    <h2>BTC / KRW</h2>
    <div class="price" id="btcPrice">--</div>
    <div id="btcChange" style="font-size:14px; margin-top:4px;">--</div>
  </div>
  <!-- 포트폴리오 -->
  <div class="card" id="portfolioCard">
    <h2>Portfolio</h2>
    <div class="metric"><span class="label">KRW 잔고</span><span class="value" id="krwBal">--</span></div>
    <div class="metric"><span class="label">총 평가</span><span class="value" id="totalEval">--</span></div>
    <div class="metric"><span class="label">수익률</span><span class="value" id="totalPnl">--</span></div>
    <div id="holdingsList"></div>
  </div>
  <!-- 지표 -->
  <div class="card">
    <h2>Indicators</h2>
    <div class="metric"><span class="label">RSI (14)</span><span class="value" id="rsi">--</span></div>
    <div class="metric"><span class="label">SMA 20</span><span class="value" id="sma20">--</span></div>
    <div class="metric"><span class="label">MACD</span><span class="value" id="macd">--</span></div>
    <div class="metric"><span class="label">Stochastic %K</span><span class="value" id="stoch">--</span></div>
  </div>
  <!-- Fear & Greed -->
  <div class="card">
    <h2>Fear & Greed Index</h2>
    <div style="display:flex; justify-content:space-between; align-items:baseline;">
      <span class="price" id="fgiValue" style="font-size:28px;">--</span>
      <span id="fgiLabel" style="font-size:14px; color:var(--dim);">--</span>
    </div>
    <div class="fgi-bar"><div class="fgi-marker" id="fgiMarker" style="left:50%;"></div></div>
  </div>
  <!-- 컨트롤 -->
  <div class="card">
    <h2>Controls</h2>
    <div class="btn-row">
      <button class="btn btn-blue" onclick="refreshAll()">🔄 새로고침</button>
      <button class="btn btn-dim" onclick="runAnalysis()">📊 분석 실행</button>
    </div>
    <div class="btn-row">
      <button class="btn btn-red" id="emergencyBtn" onclick="toggleEmergency()">🛑 긴급정지</button>
      <button class="btn btn-dim" onclick="toggleDryRun()">💡 DRY_RUN 전환</button>
    </div>
  </div>
  <!-- 전략 -->
  <div class="card">
    <h2>Strategy Summary</h2>
    <div style="font-size: 13px; color: var(--dim); line-height: 1.6;" id="strategyInfo">
      매수: FGI≤30, RSI≤30, SMA20 -5%<br>
      매도: +15% 수익 / -5% 손절 / FGI≥75 / RSI≥70<br>
      관망: 조건 미충족 또는 4시간 미경과
    </div>
  </div>
</div>
<div class="refresh-time" id="refreshTime">--</div>
<div class="toast" id="toast"></div>

<script>
const API = '';

function fmt(n) {
  if (n === undefined || n === null) return '--';
  return Number(n).toLocaleString('ko-KR');
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2500);
}

async function fetchJSON(path) {
  try {
    const r = await fetch(API + path);
    return await r.json();
  } catch(e) { return {error: e.message}; }
}

async function loadMarket() {
  const d = await fetchJSON('/api/market');
  if (d.error) return;
  document.getElementById('btcPrice').textContent = '₩' + fmt(d.current_price);
  const pct = (d.change_rate_24h * 100).toFixed(2);
  const el = document.getElementById('btcChange');
  el.textContent = (pct >= 0 ? '+' : '') + pct + '% (24h)';
  el.className = pct >= 0 ? 'change-up' : 'change-down';
  if (d.indicators) {
    document.getElementById('rsi').textContent = d.indicators.rsi_14?.toFixed(1) || '--';
    document.getElementById('sma20').textContent = '₩' + fmt(d.indicators.sma_20);
    document.getElementById('macd').textContent = d.indicators.macd?.macd || '--';
    document.getElementById('stoch').textContent = d.indicators.stochastic?.k?.toFixed(1) || '--';
  }
}

async function loadPortfolio() {
  const d = await fetchJSON('/api/portfolio');
  if (d.error) { document.getElementById('krwBal').textContent = d.error; return; }
  document.getElementById('krwBal').textContent = '₩' + fmt(Math.round(d.krw_balance));
  document.getElementById('totalEval').textContent = '₩' + fmt(Math.round(d.total_eval));
  const pnl = d.total_profit_loss_pct;
  const pnlEl = document.getElementById('totalPnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl + '%';
  pnlEl.className = 'value ' + (pnl >= 0 ? 'change-up' : 'change-down');
  let html = '';
  (d.holdings || []).forEach(h => {
    const cls = h.profit_loss_pct >= 0 ? 'change-up' : 'change-down';
    html += '<div class="holding"><div class="holding-header"><span class="holding-name">' + h.currency + '</span><span class="' + cls + '">' + (h.profit_loss_pct >= 0 ? '+' : '') + h.profit_loss_pct + '%</span></div><div style="font-size:12px;color:var(--dim);">' + h.balance.toFixed(8) + ' @ ₩' + fmt(h.avg_buy_price) + ' → ₩' + fmt(h.current_price) + '</div></div>';
  });
  document.getElementById('holdingsList').innerHTML = html;
}

async function loadFGI() {
  const d = await fetchJSON('/api/fgi');
  if (d.error) return;
  document.getElementById('fgiValue').textContent = d.current.value;
  document.getElementById('fgiLabel').textContent = d.current.classification;
  document.getElementById('fgiMarker').style.left = d.current.value + '%';
}

async function loadStatus() {
  const d = await fetchJSON('/api/status');
  const dryBadge = document.getElementById('dryRunBadge');
  const emBadge = document.getElementById('emergencyBadge');
  dryBadge.textContent = d.dry_run ? '🧪 DRY RUN' : '🔴 LIVE';
  dryBadge.className = 'badge ' + (d.dry_run ? 'badge-safe' : 'badge-danger');
  emBadge.textContent = d.emergency_stop ? '🛑 STOPPED' : '✅ ACTIVE';
  emBadge.className = 'badge ' + (d.emergency_stop ? 'badge-danger' : 'badge-safe');
}

async function refreshAll() {
  showToast('데이터 로딩 중...');
  await Promise.all([loadMarket(), loadPortfolio(), loadFGI(), loadStatus()]);
  document.getElementById('refreshTime').textContent = '마지막 업데이트: ' + new Date().toLocaleTimeString('ko-KR');
  showToast('업데이트 완료');
}

async function toggleEmergency() {
  const d = await fetchJSON('/api/toggle/emergency_stop');
  showToast(d.message);
  loadStatus();
}

async function toggleDryRun() {
  const d = await fetchJSON('/api/toggle/dry_run');
  showToast(d.message);
  loadStatus();
}

async function runAnalysis() {
  showToast('분석 실행 중...');
  const d = await fetchJSON('/api/analyze');
  showToast(d.status || d.error || '완료');
}

refreshAll();
setInterval(refreshAll, 60000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/qr")
def qr_page():
    import qrcode
    ip = get_local_ip()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    dashboard_url = f"http://{ip}:{port}"
    remote_url_file = PROJECT_ROOT / "data" / "remote_url.txt"
    try:
        claude_url = remote_url_file.read_text().strip()
    except FileNotFoundError:
        claude_url = ""

    qr_items = [
        {"label": "Dashboard (LTE/원격)", "url": "https://dashboard.wwwmoksu.com"},
        {"label": "Dashboard (로컬)", "url": dashboard_url},
    ]
    if claude_url and "pending" not in claude_url:
        qr_items.append({"label": "Claude Code Remote", "url": claude_url})
    qr_data = []
    for item in qr_items:
        img = qrcode.make(item["url"])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qr_data.append({
            "label": item["label"],
            "url": item["url"],
            "b64": base64.b64encode(buf.getvalue()).decode(),
        })
    return render_template_string(QR_HTML, qr_items=qr_data)


QR_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QR Code - Crypto Bot</title>
<style>
body { background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 100vh; margin: 0; padding: 20px; }
h1 { font-size: 22px; margin-bottom: 8px; }
p { color: #8b949e; font-size: 14px; margin-bottom: 24px; }
.qr-grid { display: flex; flex-wrap: wrap; gap: 32px; justify-content: center; }
.qr-card { display: flex; flex-direction: column; align-items: center; }
.qr-label { font-size: 16px; font-weight: 700; margin-bottom: 12px; }
.qr-box { background: #fff; padding: 16px; border-radius: 16px; }
.qr-box img { width: 220px; height: 220px; display: block; }
.qr-url { margin-top: 12px; font-size: 14px; font-weight: 600; color: #58a6ff; }
.hint { margin-top: 24px; font-size: 13px; color: #8b949e; }
</style>
</head>
<body>
<h1>Crypto Bot Remote</h1>
<p>iPhone 카메라로 QR코드를 스캔하세요</p>
<div class="qr-grid">
{% for item in qr_items %}
  <div class="qr-card">
    <div class="qr-label">{{ item.label }}</div>
    <div class="qr-box">
      <img src="data:image/png;base64,{{ item.b64 }}" alt="{{ item.label }}">
    </div>
    <div class="qr-url">{{ item.url }}</div>
  </div>
{% endfor %}
</div>
<div class="hint">같은 Wi-Fi 네트워크에 연결되어 있어야 합니다</div>
</body>
</html>
"""


@app.route("/api/market")
def api_market():
    return jsonify(run_script("collect_market_data.py"))


@app.route("/api/portfolio")
def api_portfolio():
    return jsonify(run_script("get_portfolio.py"))


@app.route("/api/fgi")
def api_fgi():
    return jsonify(run_script("collect_fear_greed.py"))


@app.route("/api/status")
def api_status():
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    return jsonify({
        "dry_run": os.environ.get("DRY_RUN", "true").lower() == "true",
        "emergency_stop": os.environ.get("EMERGENCY_STOP", "false").lower() == "true",
        "max_trade_amount": int(os.environ.get("MAX_TRADE_AMOUNT", "100000")),
        "max_daily_trades": int(os.environ.get("MAX_DAILY_TRADES", "3")),
    })


@app.route("/api/toggle/<key>")
def api_toggle(key):
    allowed = {"emergency_stop": "EMERGENCY_STOP", "dry_run": "DRY_RUN"}
    env_key = allowed.get(key)
    if not env_key:
        return jsonify({"error": "invalid key"}), 400

    env_path = PROJECT_ROOT / ".env"
    content = env_path.read_text()
    current = os.environ.get(env_key, "false").lower() == "true"
    new_val = "false" if current else "true"

    old = f"{env_key}={'true' if current else 'false'}"
    new = f"{env_key}={new_val}"
    content = content.replace(old, new)
    env_path.write_text(content)

    os.environ[env_key] = new_val
    label = "긴급정지" if key == "emergency_stop" else "DRY_RUN"
    return jsonify({"message": f"{label}: {new_val}", "value": new_val})


@app.route("/rc")
def rc_redirect():
    """최신 RC URL로 리다이렉트 -- 아이폰 홈화면 바로가기용"""
    url_file = PROJECT_ROOT / "data" / "remote_url.txt"
    try:
        if url_file.exists():
            url = url_file.read_text().strip()
            if url and url.startswith("https://"):
                return redirect(url)
    except Exception:
        pass
    return "<h2>RC 세션 없음</h2><p>워치독이 재시작 중이거나 세션이 없습니다.</p>", 503


@app.route("/api/rc-health")
def api_rc_health():
    """Remote Control 건강 상태 (watchdog v4)"""
    health_file = PROJECT_ROOT / "data" / ".rc_health.json"
    url_file = PROJECT_ROOT / "data" / "remote_url.txt"
    keepalive_file = PROJECT_ROOT / "data" / ".rc_keepalive_ts"
    fail_file = PROJECT_ROOT / "data" / ".rc_fail_count"

    result = {
        "status": "unknown",
        "detail": "",
        "url": "",
        "last_keepalive": None,
        "keepalive_age_sec": None,
        "fail_count": 0,
        "ts": "",
    }

    try:
        if health_file.exists():
            import json as _json
            result.update(_json.loads(health_file.read_text()))
    except Exception:
        pass

    try:
        if url_file.exists():
            result["url"] = url_file.read_text().strip()
    except Exception:
        pass

    try:
        if keepalive_file.exists():
            import time
            last_ts = int(keepalive_file.read_text().strip())
            result["last_keepalive"] = last_ts
            result["keepalive_age_sec"] = int(time.time()) - last_ts
    except Exception:
        pass

    try:
        if fail_file.exists():
            result["fail_count"] = int(fail_file.read_text().strip())
    except Exception:
        pass

    return jsonify(result)


@app.route("/api/analyze")
def api_analyze():
    try:
        market = run_script("collect_market_data.py")
        fgi = run_script("collect_fear_greed.py")
        portfolio = run_script("get_portfolio.py")
        return jsonify({
            "status": "분석 완료",
            "price": market.get("current_price"),
            "rsi": market.get("indicators", {}).get("rsi_14"),
            "fgi": fgi.get("current", {}).get("value"),
            "krw_balance": portfolio.get("krw_balance"),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/docs/<path:filename>")
def download_doc(filename):
    """docs/ 폴더 파일 다운로드"""
    return send_from_directory(PROJECT_ROOT / "docs", filename, as_attachment=True)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    ip = get_local_ip()
    print(f"\n  Dashboard: http://{ip}:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
