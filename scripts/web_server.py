#!/usr/bin/env python3
"""
암호화폐 자동매매 웹 대시보드 서버

- 대시보드: 포트폴리오, 최근 결정, 시장 데이터
- 리모트 컨트롤: 긴급정지, 전략 변경, 수동 실행
- QR 코드: 아이폰 접속용

실행: .venv\\Scripts\\python.exe scripts\\web_server.py
"""

import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import hashlib
import secrets

from dotenv import load_dotenv
import requests

# Windows 인코딩
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

WEB_DIR = PROJECT_DIR / "web"
PORT = 5555

# 간단한 토큰 인증: .env의 WEB_AUTH_TOKEN 또는 자동 생성
AUTH_TOKEN = os.environ.get("WEB_AUTH_TOKEN", "")
AUTH_ENABLED = bool(AUTH_TOKEN)

# 인증 불필요 경로 (QR 페이지만)
AUTH_EXEMPT_PATHS = {"/qr.html"}


def get_local_ip():
    """로컬 네트워크 IP 주소를 반환한다."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def read_env():
    """현재 .env 파일의 안전장치 설정을 읽는다."""
    env_file = PROJECT_DIR / ".env"
    config = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*([A-Z_]+)\s*=\s*(.*)\s*$", line)
            if m:
                config[m.group(1)] = m.group(2).strip()
    return config


def update_env_value(key, value):
    """특정 .env 키의 값을 변경한다."""
    env_file = PROJECT_DIR / ".env"
    content = env_file.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(\s*{key}\s*=\s*)(.*)$", re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(rf"\g<1>{value}", content)
    else:
        content += f"\n{key}={value}\n"
    env_file.write_text(content, encoding="utf-8")
    os.environ[key] = value


def supabase_get(table, params=""):
    """Supabase REST API GET 요청."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return []
    try:
        r = requests.get(
            f"{url}/rest/v1/{table}?{params}",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_portfolio():
    """포트폴리오 조회."""
    try:
        python = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
        result = subprocess.run(
            [python, str(PROJECT_DIR / "scripts" / "get_portfolio.py")],
            capture_output=True, text=True, timeout=30, encoding="utf-8"
        )
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}


def api_market():
    """시장 데이터 간략 조회."""
    try:
        python = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
        result = subprocess.run(
            [python, str(PROJECT_DIR / "scripts" / "collect_market_data.py")],
            capture_output=True, text=True, timeout=30, encoding="utf-8"
        )
        return json.loads(result.stdout)
    except Exception as e:
        return {"error": str(e)}


def api_fgi():
    """Fear & Greed Index 조회 (Alternative.me 직접 호출)."""
    try:
        r = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": "7", "format": "json"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()["data"]
        current = data[0]
        return {
            "value": int(current["value"]),
            "label": current["value_classification"],
            "timestamp": current["timestamp"],
            "history": [
                {"value": int(d["value"]), "label": d["value_classification"]}
                for d in data
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def api_decisions():
    """최근 의사결정 10건."""
    return supabase_get("decisions", "select=*&order=created_at.desc&limit=10")


def api_status():
    """현재 설정 상태."""
    config = read_env()
    return {
        "dry_run": config.get("DRY_RUN", "true"),
        "emergency_stop": config.get("EMERGENCY_STOP", "false"),
        "max_trade_amount": config.get("MAX_TRADE_AMOUNT", "100000"),
        "max_daily_trades": config.get("MAX_DAILY_TRADES", "3"),
        "max_position_ratio": config.get("MAX_POSITION_RATIO", "0.5"),
        "strategy": _get_active_strategy(),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _get_active_strategy():
    """strategy.md에서 활성 전략을 읽는다."""
    strat_file = PROJECT_DIR / "strategy.md"
    if strat_file.exists():
        for line in strat_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("## 활성 전략:"):
                return line.replace("## 활성 전략:", "").strip()
    return "unknown"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        pass  # 로그 무시

    def _check_auth(self):
        """토큰 인증 확인. 인증 실패 시 True(차단), 성공 시 False."""
        if not AUTH_ENABLED:
            return False
        parsed = urlparse(self.path)
        if parsed.path in AUTH_EXEMPT_PATHS:
            return False
        # ?token=xxx 쿼리 파라미터 또는 Authorization 헤더
        qs = parse_qs(parsed.query)
        token = qs.get("token", [None])[0]
        if not token:
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
        if token == AUTH_TOKEN:
            return False
        # 인증 실패
        body = json.dumps({"error": "unauthorized"}).encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_GET(self):
        if self._check_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._handle_api(path)
        elif path == "/" or path == "":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self._check_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            data = {}

        if path == "/api/emergency-stop":
            current = os.environ.get("EMERGENCY_STOP", "false")
            new_val = "false" if current == "true" else "true"
            update_env_value("EMERGENCY_STOP", new_val)
            self._json_response({"emergency_stop": new_val})

        elif path == "/api/dry-run":
            current = os.environ.get("DRY_RUN", "true")
            new_val = "false" if current == "true" else "true"
            update_env_value("DRY_RUN", new_val)
            self._json_response({"dry_run": new_val})

        elif path == "/api/strategy":
            strategy = data.get("strategy", "")
            if strategy in ["conservative", "moderate", "aggressive"]:
                names = {
                    "conservative": "🛡️ 보수적 (conservative)",
                    "moderate": "⚖️ 보통 (moderate)",
                    "aggressive": "🔥 공격적 (aggressive)",
                }
                _update_strategy(strategy, names[strategy])
                self._json_response({"strategy": names[strategy]})
            else:
                self._json_response({"error": "invalid strategy"}, 400)

        elif path == "/api/run":
            # 비동기로 분석 실행
            threading.Thread(target=_run_analysis, daemon=True).start()
            self._json_response({"status": "started"})

        else:
            self._json_response({"error": "not found"}, 404)

    def _handle_api(self, path):
        handlers = {
            "/api/portfolio": api_portfolio,
            "/api/market": api_market,
            "/api/fgi": api_fgi,
            "/api/decisions": api_decisions,
            "/api/status": api_status,
        }
        handler = handlers.get(path)
        if handler:
            self._json_response(handler())
        else:
            self._json_response({"error": "not found"}, 404)

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def _update_strategy(key, display_name):
    """strategy.md의 활성 전략 마커를 변경한다."""
    strat_file = PROJECT_DIR / "strategy.md"
    content = strat_file.read_text(encoding="utf-8")
    # 활성 전략 헤더 변경
    content = re.sub(
        r"^## 활성 전략:.*$",
        f"## 활성 전략: {display_name}",
        content,
        flags=re.MULTILINE,
    )
    # ← 현재 활성 마커 이동
    content = re.sub(r" ← 현재 활성", "", content)
    marker_map = {
        "conservative": "보수적 전략",
        "moderate": "보통 전략",
        "aggressive": "공격적 전략",
    }
    target = marker_map[key]
    content = re.sub(
        rf"(## [^\n]*{target}[^\n]*)",
        rf"\1 ← 현재 활성",
        content,
    )
    strat_file.write_text(content, encoding="utf-8")


def _run_analysis():
    """수동 분석 실행 (백그라운드)."""
    try:
        python = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
        scripts = [
            "collect_market_data.py",
            "collect_fear_greed.py",
            "get_portfolio.py",
        ]
        for s in scripts:
            subprocess.run(
                [python, str(PROJECT_DIR / "scripts" / s)],
                capture_output=True, timeout=30,
            )
        # 텔레그램 알림
        subprocess.run(
            [python, str(PROJECT_DIR / "scripts" / "notify_telegram.py"),
             "trade", "수동 분석 실행됨 (웹 대시보드)", "웹 리모트 컨트롤에서 수동 실행"],
            capture_output=True, timeout=15,
        )
    except Exception:
        pass


def main():
    local_ip = get_local_ip()
    token_qs = f"?token={AUTH_TOKEN}" if AUTH_ENABLED else ""
    dashboard_url = f"http://{local_ip}:{PORT}{token_qs}"
    remote_url = f"http://{local_ip}:{PORT}/remote.html{token_qs}"
    qr_url = f"http://{local_ip}:{PORT}/qr.html"

    print(f"\n{'='*50}")
    print(f"  암호화폐 자동매매 대시보드")
    print(f"{'='*50}")
    print(f"  대시보드:     {dashboard_url}")
    print(f"  리모트 컨트롤: {remote_url}")
    print(f"  QR 코드:      {qr_url}")
    if AUTH_ENABLED:
        print(f"  인증 토큰:    {AUTH_TOKEN}")
    else:
        print(f"  인증: 비활성 (.env에 WEB_AUTH_TOKEN 설정으로 활성화)")
    print(f"{'='*50}")
    print(f"  Ctrl+C 로 종료\n")

    server = http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료.")
        server.server_close()


if __name__ == "__main__":
    main()
