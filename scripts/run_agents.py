#!/usr/bin/env python3
"""
Python 기반 에이전트 모드 파이프라인 (Cross-platform 지원)
기존 run_agents.sh를 대체하며, Windows/Linux 어디서든 안전하게 동작합니다.

사용법:
  python scripts/run_agents.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 부모 디렉토리 sys.path 추가 및 상대 임포트 준비
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
import requests

from agents.external_data import ExternalDataAgent
from agents.orchestrator import Orchestrator

KST = timezone(timedelta(hours=9))


def log(msg: str):
    print(f"[{datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr)


def notify_error(msg: str, detail: str):
    log(f"ERROR: {msg}")
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/notify_telegram.py", "error", msg, detail],
            cwd=str(PROJECT_DIR),
            check=False,
            capture_output=True
        )
    except Exception as e:
        log(f"텔레그램 전송 실패: {e}")


async def run_script(script_name: str) -> dict:
    import subprocess
    """별도 프로세스로 스크립트를 실행하여 JSON 결과 반환"""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, f"scripts/{script_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_DIR)
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            log(f"스크립트 {script_name} 실패: {stderr.decode('utf-8', errors='ignore')}")
            return {"error": f"{script_name} 실패: {proc.returncode}"}
        
        return json.loads(stdout.decode('utf-8'))
    except Exception as e:
        log(f"스크립트 {script_name} 실행 중 예외: {e}")
        return {"error": str(e)}


async def collect_internal_data() -> tuple[dict, dict, dict]:
    """Phase 1: 마켓, 포트폴리오, AI 시그널 동시 수집"""
    log("Phase 1: 내부 데이터 수집...")
    results = await asyncio.gather(
        run_script("collect_market_data.py"),
        run_script("get_portfolio.py"),
        run_script("collect_ai_signal.py"),
        return_exceptions=True
    )
    
    market_data = results[0] if not isinstance(results[0], Exception) else {"error": "market_data 수집 실패"}
    portfolio = results[1] if not isinstance(results[1], Exception) else {"error": "portfolio 수집 실패"}
    ai_signal = results[2] if not isinstance(results[2], Exception) else {"error": "ai_signal 수집 실패"}
    
    log("Phase 1 완료.")
    return market_data, portfolio, ai_signal


def main():
    load_dotenv(PROJECT_DIR / ".env")
    
    # 1. 수동 긴급 정지 확인
    if os.environ.get("EMERGENCY_STOP", "false").lower() == "true":
        log("[STOP] 사용자 EMERGENCY_STOP 활성화됨. 실행 중단.")
        notify_error("EMERGENCY_STOP", "사용자 긴급 정지 활성화로 에이전트 실행 중단")
        sys.exit(1)
        
    # 2. 감독 자동 긴급 정지 확인
    auto_emergency_file = PROJECT_DIR / "data" / "auto_emergency.json"
    if auto_emergency_file.exists():
        try:
            with open(auto_emergency_file, "r", encoding="utf-8") as f:
                auto_em = json.load(f)
            if auto_em.get("active"):
                reason = auto_em.get("reason", "알 수 없음")
                log(f"[STOP] 감독 자동 긴급정지 활성 중: {reason}")
                log("[STOP] Orchestrator가 해제 조건을 평가합니다...")
        except Exception:
            pass
            
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = PROJECT_DIR / "data" / "snapshots" / timestamp
    log_dir = PROJECT_DIR / "logs" / "executions"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log("═══ Python 에이전트 모드 시작 ═══")
    
    # Phase 1
    market_data, portfolio, ai_signal = asyncio.run(collect_internal_data())
    
    with open(snapshot_dir / "market_data.json", "w", encoding="utf-8") as f:
        json.dump(market_data, f, ensure_ascii=False)
    with open(snapshot_dir / "portfolio.json", "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False)
    with open(snapshot_dir / "ai_signal.json", "w", encoding="utf-8") as f:
        json.dump(ai_signal, f, ensure_ascii=False)
        
    # Phase 2: 파이프라인 실행
    log("Phase 2: 에이전트 파이프라인 실행...")
    
    try:
        ext_agent = ExternalDataAgent(snapshot_dir=snapshot_dir)
        external_data = ext_agent.collect_all()
        log(f"외부 데이터 수집 완료 ({external_data.get('collection_time_sec', 0)}초)")
        
        # 데이터 병합
        market_data["ai_composite_signal"] = ai_signal.get("ai_composite_signal", ai_signal.get("composite_signal", {}))
        
        fgi_data = external_data.get("sources", {}).get("fear_greed", {})
        market_data["fear_greed"] = fgi_data.get("current", {})
        
        news_data = external_data.get("sources", {}).get("news", {})
        news_sentiment = external_data.get("sources", {}).get("news_sentiment", {})
        market_data["news"] = news_data
        market_data["news"]["overall_sentiment"] = news_sentiment.get("overall_sentiment", "neutral")
        market_data["news"]["sentiment_score"] = news_sentiment.get("sentiment_score", 0)
        
        # Supabase 과거 결정 (선택사항)
        past_decisions = []
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if supabase_url and supabase_key:
            try:
                resp = requests.get(
                    f"{supabase_url}/rest/v1/decisions",
                    params={"select": "*", "order": "created_at.desc", "limit": "10"},
                    headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    past_decisions = resp.json()
            except Exception as e:
                log(f"Supabase 조회 실패: {e}")
                
        # 포트폴리오 메타 데이터 주입
        btc_info = portfolio.get("coins", {}).get("BTC", portfolio.get("btc", {}))
        total_eval = portfolio.get("total_evaluation", 1)
        btc_eval = btc_info.get("evaluation", 0) if isinstance(btc_info, dict) else 0
        portfolio["btc_ratio"] = btc_eval / total_eval if total_eval > 0 else 0
        portfolio["btc"] = btc_info if isinstance(btc_info, dict) else {}
        
        # 오케스트레이터 실행
        orchestrator = Orchestrator()
        result = orchestrator.run(
            market_data=market_data,
            external_data=external_data,
            portfolio=portfolio,
            past_decisions=past_decisions,
        )
        log(f"결정: {result['decision']['decision']} by {result['active_agent']}")
        
        output = {
            "timestamp": external_data.get("timestamp", datetime.now(KST).isoformat()),
            "active_agent": result["active_agent"],
            "switch": result.get("switch"),
            "decision": result["decision"],
            "external_data_summary": {
                "collection_time_sec": external_data.get("collection_time_sec"),
                "errors": external_data.get("errors"),
                "signal": external_data.get("external_signal", {}),
            },
            "snapshot_dir": str(snapshot_dir),
        }
        
    except Exception as e:
        log(f"Phase 2 실패: {e}")
        import traceback
        traceback.print_exc()
        notify_error("Agent Pipeline", f"에이전트 파이프라인 실패: {e}")
        sys.exit(1)
        
    log("Phase 2 완료.")
    
    agent_result_path = snapshot_dir / "agent_result.json"
    with open(agent_result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    # Phase 3: 매매 실행
    decision = output["decision"]["decision"]
    reason = output["decision"]["reason"]
    agent_name = output["active_agent"]
    switch_sw = output.get("switch")
    switch_info = f"전략 전환: {switch_sw['from']} → {switch_sw['to']} ({switch_sw['reason']})" if switch_sw else "전환 없음"
    
    log(f"Phase 3: 매매 실행 — {decision} ({agent_name})")
    
    trade_params = output["decision"].get("trade_params", {})
    market = trade_params.get("market", "KRW-BTC")
    
    import subprocess
    trade_log = str(log_dir / f"trade_{timestamp}.log")
    
    if decision == "buy":
        amount = trade_params.get("amount", 0)
        is_dca = trade_params.get("is_dca", False)
        if int(amount) > 0:
            dca_tag = " [DCA]" if is_dca else ""
            log(f"매수 실행: {market} {amount} KRW{dca_tag}")
            with open(trade_log, "w", encoding="utf-8") as tf:
                subprocess.run([sys.executable, "scripts/execute_trade.py", "bid", market, str(amount)], cwd=str(PROJECT_DIR), stdout=tf, stderr=subprocess.STDOUT)
    elif decision == "sell":
        volume = trade_params.get("volume", 0)
        if float(volume) > 0:
            log(f"매도 실행: {market} {volume} BTC")
            with open(trade_log, "w", encoding="utf-8") as tf:
                subprocess.run([sys.executable, "scripts/execute_trade.py", "ask", market, str(volume)], cwd=str(PROJECT_DIR), stdout=tf, stderr=subprocess.STDOUT)
    else:
        log("관망 결정. 매매 없음.")
        
    # Phase 4: 텔레그램 알림 (풍부한 정보 전달 피드백 루프 포함)
    log("Phase 4: 텔레그램 알림...")
    buy_score = output["decision"].get("buy_score", {}).get("total", "N/A")
    confidence = round(output["decision"].get("confidence", 0) * 100)
    current_price = market_data.get("ticker", {}).get("trade_price", 0)
    fgi = market_data.get("fear_greed", {}).get("value", "N/A")
    krw_bal = float(portfolio.get("krw", {}).get("balance", 0))
    btc_bal = float(portfolio.get("btc", {}).get("balance", 0))
    btc_ratio = round(portfolio.get("btc_ratio", 0) * 100, 1)

    # 피드백 상태 조회
    state_file = PROJECT_DIR / "data" / "orchestrator_state.json"
    user_bias = 0.0
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                user_bias = state.get("feedback_bias", 0.0)
        except Exception:
            pass

    summary_msg = f"[{agent_name}] {decision.upper()} 결정 ({confidence}%)"
    detail_msg = (
        f"💡 근거: {reason}\n\n"
        f"📊 시장 현황:\n"
        f"- 현재가: {int(current_price):,}원\n"
        f"- 매수점수: {buy_score}/100\n"
        f"- 탐욕지수(FGI): {fgi}\n\n"
        f"💼 포트폴리오:\n"
        f"- 자산 비율: BTC {btc_ratio}% / KRW {100-btc_ratio:.1f}%\n"
        f"- KRW 잔고: {int(krw_bal):,}원\n"
        f"- BTC 잔고: {btc_bal:.6f} BTC\n\n"
        f"🤖 감독 상태:\n"
        f"- 사용자 피드백 Bias: {user_bias:+.2f} (음수:보수적, 양수:공격적)\n"
        f"- {switch_info}\n\n"
        f"ℹ️ 피드백을 주시려면 'python scripts/feedback.py +0.5' 를 실행하세요."
    )
    
    try:
        subprocess.run([sys.executable, "scripts/notify_telegram.py", "trade", summary_msg, detail_msg], cwd=str(PROJECT_DIR), check=False)
    except Exception:
        pass
        
    # Phase 5: Supabase 기록
    if supabase_url and supabase_key:
        log("Phase 5: Supabase 기록...")
        try:
            row = {
                "decision": decision,
                "confidence": output["decision"].get("confidence", 0),
                "reason": output["decision"].get("reason", ""),
                "buy_score": output["decision"].get("buy_score", {}).get("total", 0),
                "agent_name": agent_name,
                "current_price": market_data.get("ticker", {}).get("trade_price", 0),
                "snapshot_dir": str(snapshot_dir),
            }
            requests.post(
                f"{supabase_url}/rest/v1/decisions",
                json=row,
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                timeout=5,
            )
            log("Supabase 기록 완료")
        except Exception as e:
            log(f"Supabase 기록 실패: {e}")
            
    # Phase 6: 전환 성과 평가
    log("Phase 6: 전환 성과 평가...")
    subprocess.run([sys.executable, "scripts/evaluate_switches.py"], cwd=str(PROJECT_DIR), check=False)
    
    log("═══ 에이전트 모드 완료 ═══")
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
