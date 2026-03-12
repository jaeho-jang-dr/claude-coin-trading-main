#!/usr/bin/env python3
"""
Python 기반 에이전트 모드 파이프라인 (Cross-platform 지원)
기존 run_agents.sh를 대체하며, Windows/Linux 어디서든 안전하게 동작합니다.

사용법:
  python scripts/run_agents.py
"""

import hide_console
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


def _get_active_agent() -> str:
    """agent_state.json에서 현재 활성 에이전트 이름을 읽는다."""
    try:
        state_path = PROJECT_DIR / "data" / "agent_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state.get("active_agent", "conservative")
    except Exception:
        pass
    return "conservative"


def _action_to_direction(action: float) -> str:
    """연속 행동값 [-1, 1]을 방향 문자열로 변환."""
    if action > 0.3:
        return "buy"
    elif action < -0.3:
        return "sell"
    return "hold"


def get_rl_advisory(market_data: dict, external_data: dict,
                    portfolio: dict, agent_state: dict) -> dict | None:
    """Phase 2.5: RL 모델 어드바이저리 시그널 (다중 모델 앙상블).

    사용 가능한 모든 RL 모델의 시그널을 수집하고, 가중 평균 앙상블로
    최종 어드바이저리를 생성한다. 개별 모델은 모두 선택적(try/except).

    지원 모델:
      1. SB3 (PPO/SAC/TD3) -- data/rl_models/best/best_model.zip
      2. Decision Transformer -- data/rl_models/transformer/dt_model.pt
      3. Multi-Agent Consensus -- data/rl_models/multi_agent/
      4. Offline RL (CQL/BCQ) -- data/rl_models/offline/
    """
    advisories = {}

    # 공통 StateEncoder (한 번만 초기화)
    encoder = None
    obs = None
    try:
        from rl_hybrid.rl.state_encoder import StateEncoder
        encoder = StateEncoder()
        obs = encoder.encode(market_data, external_data, portfolio, agent_state)
    except Exception as e:
        log(f"RL StateEncoder 초기화 실패: {e}")

    # ── 1. 기존 SB3 모델 (PPO/SAC/TD3) ──
    try:
        from rl_hybrid.rl.policy import SB3_AVAILABLE
        if SB3_AVAILABLE and obs is not None:
            model_path = str(PROJECT_DIR / "data" / "rl_models" / "best" / "best_model")
            if (PROJECT_DIR / "data" / "rl_models" / "best" / "best_model.zip").exists():
                # 알고리즘 감지: model_info.json이 있으면 해당 알고리즘 사용
                algo = "ppo"  # 기본값 (폴백)
                info_path = PROJECT_DIR / "data" / "rl_models" / "best" / "model_info.json"
                if info_path.exists():
                    try:
                        with open(info_path) as f:
                            model_info = json.load(f)
                        algo = model_info.get("algorithm", "ppo")
                    except (json.JSONDecodeError, KeyError):
                        pass

                from rl_hybrid.rl.train import get_trader_class
                TraderClass = get_trader_class(algo)
                trader = TraderClass(env=None, model_path=model_path)
                sb3_action = trader.predict(obs)
                advisories["sb3"] = {
                    "action": round(float(sb3_action), 4),
                    "source": f"sb3_{algo}",
                }
                log(f"  SB3({algo}): action={sb3_action:.4f}")
    except Exception as e:
        log(f"  SB3 advisory 실패: {e}")

    # ── 2. Decision Transformer ──
    try:
        dt_model_path = PROJECT_DIR / "data" / "rl_models" / "transformer" / "dt_model.pt"
        if dt_model_path.exists() and obs is not None:
            from rl_hybrid.rl.decision_transformer import DTPredictor
            predictor = DTPredictor(model_path=str(dt_model_path))
            # 활성 에이전트에 맞게 리스크 프로파일 설정
            active_agent = agent_state.get("active_agent", _get_active_agent())
            predictor.set_risk_profile(active_agent)
            # 시장 상황으로 RTG 동적 조정
            danger = agent_state.get("danger_score", 50)
            opportunity = agent_state.get("opportunity_score", 50)
            predictor.adjust_rtg_for_market(danger, opportunity)
            dt_action = predictor.predict(obs)
            advisories["dt"] = {
                "action": round(float(dt_action), 4),
                "source": "decision_transformer",
            }
            log(f"  DT: action={dt_action:.4f} (agent={active_agent})")
    except Exception as e:
        log(f"  Decision Transformer advisory 실패: {e}")

    # ── 3. Multi-Agent Consensus ──
    try:
        from rl_hybrid.config import SystemConfig
        cfg = SystemConfig()
        if cfg.multi_agent.enabled:
            from rl_hybrid.rl.multi_agent_consensus import MultiAgentPredictor
            ma_predictor = MultiAgentPredictor()
            ma_result = ma_predictor.predict(
                market_data=market_data,
                external_data=external_data,
                portfolio=portfolio,
                agent_state=agent_state,
            )
            if ma_result and ma_result.get("action", 0) != 0:
                advisories["multi_agent"] = {
                    "action": round(float(ma_result["action"]), 4),
                    "source": "multi_agent_consensus",
                    "consensus": ma_result.get("consensus"),
                }
                log(f"  Multi-Agent: action={ma_result['action']:.4f}")
    except Exception as e:
        log(f"  Multi-Agent Consensus advisory 실패: {e}")

    # ── 4. Offline RL (CQL/BCQ) ──
    try:
        offline_dir = PROJECT_DIR / "data" / "rl_models" / "offline"
        if offline_dir.exists() and obs is not None:
            # 최신 모델 파일 찾기 (cql_*.pt 또는 bcq_*.pt)
            import glob
            offline_models = sorted(
                glob.glob(str(offline_dir / "*.pt")),
                key=os.path.getmtime,
                reverse=True,
            )
            if offline_models:
                best_offline = offline_models[0]
                offline_algo = "cql" if "cql" in os.path.basename(best_offline).lower() else "bcq"
                if offline_algo == "cql":
                    from rl_hybrid.rl.offline_rl import CQLTrainer
                    trainer = CQLTrainer()
                    trainer.load(best_offline)
                    offline_action = trainer.predict(obs)
                else:
                    from rl_hybrid.rl.offline_rl import BCQTrainer
                    trainer = BCQTrainer()
                    trainer.load(best_offline)
                    offline_action = trainer.predict(obs)
                advisories["offline"] = {
                    "action": round(float(offline_action), 4),
                    "source": f"offline_{offline_algo}",
                }
                log(f"  Offline RL({offline_algo}): action={offline_action:.4f}")
    except Exception as e:
        log(f"  Offline RL advisory 실패: {e}")

    # ── 5. 앙상블: 사용 가능한 모든 시그널의 가중 평균 ──
    if not advisories:
        return None

    actions = [v["action"] for v in advisories.values() if "action" in v]
    ensemble_action = sum(actions) / len(actions) if actions else 0.0
    ensemble_direction = _action_to_direction(ensemble_action)

    # ── 6. DB 기록: 앙상블 추론 결과 ──
    try:
        from rl_hybrid.rl.rl_db_logger import log_prediction as _log_pred

        # 개별 모델 액션/버전 추출
        sb3_info = advisories.get("sb3", {})
        dt_info = advisories.get("dt", {})
        ma_info = advisories.get("multi_agent", {})
        offline_info = advisories.get("offline", {})

        # SB3 버전: model_info.json에서 읽기
        sb3_ver = None
        try:
            _info_path = PROJECT_DIR / "data" / "rl_models" / "best" / "model_info.json"
            if _info_path.exists():
                with open(_info_path) as _f:
                    sb3_ver = json.load(_f).get("version_id")
        except Exception:
            pass

        # Multi-Agent 세부 액션 추출
        ma_consensus = ma_info.get("consensus", {}) if ma_info else {}
        ma_scalp = ma_consensus.get("scalp_action") if isinstance(ma_consensus, dict) else None
        ma_swing = ma_consensus.get("swing_action") if isinstance(ma_consensus, dict) else None

        # 시장 컨텍스트
        ticker = market_data.get("ticker", {})
        indicators = market_data.get("indicators", {})
        fgi_data = market_data.get("fear_greed", {})

        _log_pred(
            cycle_id=_CYCLE_ID,
            ensemble_action=round(ensemble_action, 4),
            ensemble_direction=ensemble_direction,
            num_models=len(advisories),
            sb3_action=sb3_info.get("action"),
            sb3_version=sb3_ver,
            dt_action=dt_info.get("action"),
            dt_version=None,
            multi_agent_action=ma_info.get("action"),
            multi_agent_direction=_action_to_direction(ma_info["action"]) if ma_info.get("action") is not None else None,
            multi_agent_scalp_action=ma_scalp,
            multi_agent_swing_action=ma_swing,
            offline_action=offline_info.get("action"),
            offline_version=None,
            btc_price=ticker.get("trade_price"),
            rsi_14=indicators.get("rsi_14"),
            fgi=fgi_data.get("value"),
            danger_score=agent_state.get("danger_score"),
            opportunity_score=agent_state.get("opportunity_score"),
        )
    except Exception as _db_err:
        log(f"RL 추론 DB 기록 실패 (비치명적): {_db_err}")

    return {
        "action": round(ensemble_action, 4),
        "abs_action": round(abs(ensemble_action), 4),
        "direction": ensemble_direction,
        "models": advisories,
        "sources": list(advisories.keys()),
        "num_models": len(advisories),
    }


# cycle_id: 이 파이프라인 실행의 모든 DB 기록을 연결하는 키
try:
    from scripts.cycle_id import make_cycle_id, set_cycle_id
    _CYCLE_ID = make_cycle_id("agent")
    set_cycle_id(_CYCLE_ID)
except Exception:
    _CYCLE_ID = datetime.now(KST).strftime("%Y%m%d-%H%M") + "-agent"


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


def supabase_headers() -> dict:
    """Supabase REST API 공통 헤더"""
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def save_execution_log(
    execution_mode: str,
    duration_ms: int,
    data_sources: dict | None = None,
    errors: dict | None = None,
    raw_output: str | None = None,
    decision_id: str | None = None,
    phases_completed: list | None = None,
) -> bool:
    """execution_logs 테이블에 파이프라인 실행 기록을 저장한다."""
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not supabase_key:
        log("execution_logs 저장 스킵: Supabase 미설정")
        return False

    has_errors = bool(errors and errors.get("pipeline_errors"))
    row = {
        "execution_mode": execution_mode,
        "duration_ms": duration_ms,
        "data_sources": json.dumps(data_sources or {}, ensure_ascii=False),
        "errors": json.dumps(errors or {}, ensure_ascii=False),
        "cycle_id": _CYCLE_ID,
        "success": not has_errors,
        "execution_started_at": datetime.now(KST).isoformat(),
    }
    if phases_completed:
        row["phases_completed"] = json.dumps(phases_completed, ensure_ascii=False)
    if raw_output:
        row["raw_output"] = raw_output[:10000]  # 10KB 제한
    if decision_id:
        row["decision_id"] = decision_id

    try:
        resp = requests.post(
            f"{supabase_url}/rest/v1/execution_logs",
            json=row,
            headers=supabase_headers(),
            timeout=10,
        )
        if resp.status_code in (200, 201):
            log("execution_logs 기록 완료")
            return True
        else:
            log(f"execution_logs 기록 실패 (HTTP {resp.status_code}): {resp.text[:300]}")
            return False
    except Exception as e:
        log(f"execution_logs 기록 예외: {e}")
        return False


def save_market_data_record(market_data: dict, external_data: dict) -> bool:
    """market_data 테이블에 시장 데이터 스냅샷을 저장한다.

    모든 수집된 지표를 포함:
    - 가격, 거래량, 변화율
    - RSI, SMA20 기술지표
    - FGI 값/분류
    - 뉴스 감성 점수
    """
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not supabase_url or not supabase_key:
        log("market_data 저장 스킵: Supabase 미설정")
        return False

    ticker = market_data.get("ticker", {})
    indicators = market_data.get("indicators", {})
    fgi = market_data.get("fear_greed", {})
    news = market_data.get("news", {})

    price = ticker.get("trade_price") or market_data.get("current_price", 0)
    volume_24h = ticker.get("acc_trade_volume_24h")
    change_rate = ticker.get("signed_change_rate")

    row = {
        "market": "KRW-BTC",
        "price": int(price) if price else 0,
        "volume_24h": float(volume_24h) if volume_24h else None,
        "change_rate_24h": float(change_rate) if change_rate else None,
        "fear_greed_value": fgi.get("value"),
        "fear_greed_class": fgi.get("value_classification"),
        "rsi_14": indicators.get("rsi_14"),
        "sma_20": int(indicators.get("sma_20")) if indicators.get("sma_20") else None,
        "news_sentiment": news.get("overall_sentiment", "neutral"),
        "cycle_id": _CYCLE_ID,
    }

    try:
        resp = requests.post(
            f"{supabase_url}/rest/v1/market_data",
            json=row,
            headers=supabase_headers(),
            timeout=10,
        )
        if resp.status_code in (200, 201):
            log("market_data 기록 완료")
            return True
        else:
            log(f"market_data 기록 실패 (HTTP {resp.status_code}): {resp.text[:300]}")
            return False
    except Exception as e:
        log(f"market_data 기록 예외: {e}")
        return False


def main():
    load_dotenv(PROJECT_DIR / ".env")

    # 파이프라인 시작 시간 기록
    pipeline_start = time.time()
    pipeline_errors = []
    data_sources_used = []

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

    # 데이터 소스 추적
    if "error" not in market_data:
        data_sources_used.append("market_data")
    else:
        pipeline_errors.append({"phase": "phase1", "source": "market_data", "error": market_data.get("error")})
    if "error" not in portfolio:
        data_sources_used.append("portfolio")
    else:
        pipeline_errors.append({"phase": "phase1", "source": "portfolio", "error": portfolio.get("error")})
    if "error" not in ai_signal:
        data_sources_used.append("ai_signal")
    else:
        pipeline_errors.append({"phase": "phase1", "source": "ai_signal", "error": ai_signal.get("error")})
    
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
        # NVT Signal을 최상위에 배치 (StateEncoder 호환)
        nvt_data = external_data.get("sources", external_data).get("nvt", {})
        external_data["nvt_signal"] = nvt_data.get("nvt_signal", 100.0)
        log(f"외부 데이터 수집 완료 ({external_data.get('collection_time_sec', 0)}초)")
        data_sources_used.append("external_data")
        ext_errors = external_data.get("errors", [])
        if ext_errors:
            pipeline_errors.append({"phase": "phase2", "source": "external_data", "errors": ext_errors})
        
        # 데이터 병합
        market_data["ai_composite_signal"] = ai_signal.get("ai_composite_signal", ai_signal.get("composite_signal", {}))
        
        fgi_data = external_data.get("sources", {}).get("fear_greed", {})
        market_data["fear_greed"] = fgi_data.get("current", {})
        
        news_data = external_data.get("sources", {}).get("news", {})
        news_sentiment = external_data.get("sources", {}).get("news_sentiment", {})
        market_data["news"] = news_data
        market_data["news"]["overall_sentiment"] = news_sentiment.get("overall_sentiment", "neutral")
        market_data["news"]["sentiment_score"] = news_sentiment.get("sentiment_score", 0)
        
        # RAG: 현재 시장과 유사한 과거 경험 조회 (LIMIT 10 → 벡터 유사도 Top 3)
        past_decisions = []
        try:
            import subprocess as _sp
            rag_result = _sp.run(
                [sys.executable, "scripts/recall_rag.py", "--json", "--top", "3"],
                cwd=str(PROJECT_DIR),
                capture_output=True, text=True, timeout=30,
            )
            if rag_result.returncode == 0 and rag_result.stdout.strip():
                rag_data = json.loads(rag_result.stdout)
                if isinstance(rag_data, dict) and rag_data.get("results"):
                    past_decisions = rag_data["results"]
                    log(f"RAG: 유사 과거 경험 {len(past_decisions)}건 조회")
                elif isinstance(rag_data, list) and rag_data:
                    past_decisions = rag_data
                    log(f"RAG: 유사 과거 경험 {len(rag_data)}건 조회")
        except Exception as e:
            log(f"RAG 조회 실패 (fallback): {e}")

        # RAG 실패 시 기존 방식 fallback (최근 5건만)
        if not past_decisions:
            supabase_url = os.environ.get("SUPABASE_URL", "")
            supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
            if supabase_url and supabase_key:
                try:
                    resp = requests.get(
                        f"{supabase_url}/rest/v1/decisions",
                        params={"select": "id,decision,reason,confidence,current_price,profit_loss,created_at", "order": "created_at.desc", "limit": "5"},
                        headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        past_decisions = resp.json()
                        log(f"RAG fallback: 최근 {len(past_decisions)}건 조회")
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
            # DB 연결용 ID
            "external_signal_id": ext_agent.saved_signal_id,
            "buy_score_id": result.get("buy_score_id"),
        }
        
    except Exception as e:
        log(f"Phase 2 실패: {e}")
        import traceback
        traceback.print_exc()
        pipeline_errors.append({"phase": "phase2", "source": "orchestrator", "error": str(e)})
        # 실패해도 execution_logs는 기록
        duration_ms = int((time.time() - pipeline_start) * 1000)
        save_execution_log(
            execution_mode="dry_run" if os.environ.get("DRY_RUN", "true").lower() == "true" else "execute",
            duration_ms=duration_ms,
            data_sources={"sources": data_sources_used},
            errors={"pipeline_errors": pipeline_errors, "fatal": str(e)},
            phases_completed=["phase1"],
        )
        notify_error("Agent Pipeline", f"에이전트 파이프라인 실패: {e}")
        sys.exit(1)
        
    log("Phase 2 완료.")

    # Phase 2.5: RL 모델 어드바이저리
    rl_advisory = None
    agent_state_for_rl = {}
    try:
        market_state = result.get("market_state", {})
        agent_state_for_rl = {
            "active_agent": output.get("active_agent", "conservative"),
            "danger_score": market_state.get("danger_score", 50),
            "opportunity_score": market_state.get("opportunity_score", 50),
            "consecutive_losses": market_state.get("consecutive_losses", 0),
        }
        rl_advisory = get_rl_advisory(market_data, external_data, portfolio, agent_state_for_rl)
        if rl_advisory:
            sources = ", ".join(rl_advisory.get("sources", []))
            log(f"RL advisory: {rl_advisory['direction']} (action={rl_advisory['action']:.4f}, models=[{sources}])")
            output["rl_advisory"] = rl_advisory

            # 에이전트 결정과 RL 방향 비교하여 confidence 조정
            agent_decision = output["decision"]["decision"]
            rl_dir = rl_advisory["direction"]
            rl_strength = rl_advisory["abs_action"]
            orig_conf = output["decision"].get("confidence", 0.5)

            if agent_decision == rl_dir:
                # 일치: confidence 부스트 (최대 +10%)
                boost = min(0.10, rl_strength * 0.15)
                output["decision"]["confidence"] = min(1.0, orig_conf + boost)
                log(f"RL 일치 → confidence {orig_conf:.2f} → {output['decision']['confidence']:.2f}")
            elif rl_dir == "hold":
                pass  # RL이 hold이면 간섭 안 함
            elif agent_decision == "hold" and rl_strength > 0.5:
                # 에이전트 관망인데 RL이 강한 시그널 → advisory 메모만
                output["decision"]["rl_override_hint"] = rl_dir
                log(f"RL 강한 시그널({rl_dir}, {rl_strength:.2f}) -- advisory 메모 추가")
            else:
                # 불일치: confidence 감소 (최대 -20%)
                dampen = min(0.20, rl_strength * 0.25)
                output["decision"]["confidence"] = max(0.0, orig_conf - dampen)
                log(f"RL 불일치({rl_dir}) → confidence {orig_conf:.2f} → {output['decision']['confidence']:.2f}")
        else:
            log("RL advisory: 모델 없음 또는 비활성")
    except Exception as e:
        log(f"Phase 2.5 RL advisory 예외: {e}")

    agent_result_path = snapshot_dir / "agent_result.json"
    with open(agent_result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Phase 3: 매매 실행
    decision = output["decision"]["decision"]
    reason = output["decision"]["reason"]
    agent_name = output["active_agent"]
    switch_sw = output.get("switch")
    switch_info = f"전략 전환: {switch_sw['from']} → {switch_sw['to']} ({switch_sw['reason']})" if switch_sw else "전환 없음"
    
    log(f"Phase 3: 매매 실행 -- {decision} ({agent_name})")
    
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
    if rl_advisory:
        rl_parts = [f"- RL 앙상블: {rl_advisory['direction']}({rl_advisory['action']:+.4f}) [{rl_advisory.get('num_models', 1)}모델]"]
        for src, info in rl_advisory.get("models", {}).items():
            rl_parts.append(f"  · {src}: {info['action']:+.4f}")
        rl_line = "\n".join(rl_parts) + "\n"
    else:
        rl_line = ""
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
        f"- {switch_info}\n"
        f"{rl_line}\n"
        f"ℹ️ 피드백을 주시려면 'python scripts/feedback.py +0.5' 를 실행하세요."
    )
    
    try:
        subprocess.run([sys.executable, "scripts/notify_telegram.py", "trade", summary_msg, detail_msg], cwd=str(PROJECT_DIR), check=False)
    except Exception:
        pass
        
    # Phase 5: Supabase 기록 (decisions + market_data + execution_logs)
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if supabase_url and supabase_key:
        log("Phase 5: Supabase 기록...")

        # 5a. decisions 테이블
        resp = None
        try:
            DECISION_MAP = {"buy": "매수", "sell": "매도", "hold": "관망"}
            dec = output["decision"]
            buy_score = dec.get("buy_score", {})
            ext_summary = dec.get("external_signal_summary", {})

            reason_parts = [dec.get("reason", "")]
            if agent_name:
                reason_parts.append(f"에이전트: {agent_name}")
            if buy_score:
                reason_parts.append(f"매수점수: {buy_score.get('total', '?')}/{buy_score.get('threshold', '?')}")
            if ext_summary:
                reason_parts.append(f"외부시그널: {ext_summary.get('fusion_signal', '?')}({ext_summary.get('total_score', '?')})")

            raw_conf = float(dec.get("confidence", 0))
            confidence_val = raw_conf / 100.0 if raw_conf > 1 else raw_conf

            decision_row = {
                "market": "KRW-BTC",
                "decision": DECISION_MAP.get(decision, decision),
                "confidence": round(confidence_val, 2),
                "reason": " | ".join(reason_parts),
                "current_price": int(market_data.get("ticker", {}).get("trade_price", 0)) or None,
                "rsi_value": market_data.get("indicators", {}).get("rsi_14"),
                "fear_greed_value": market_data.get("fear_greed", {}).get("value"),
                "sma20_price": int(market_data.get("indicators", {}).get("sma_20")) if market_data.get("indicators", {}).get("sma_20") else None,
                "market_data_snapshot": json.dumps({
                    "buy_score": buy_score,
                    "external": ext_summary,
                    "rl_advisory": rl_advisory,
                    "snapshot_dir": str(snapshot_dir),
                }, ensure_ascii=False),
                "cycle_id": _CYCLE_ID,
                "source": "agent",
            }
            # 외부 정보 및 매수 점수와 직접 연결 (FK)
            if output.get("external_signal_id"):
                decision_row["external_signal_id"] = output["external_signal_id"]
            if output.get("buy_score_id"):
                decision_row["buy_score_id"] = output["buy_score_id"]
            decision_headers = supabase_headers()
            decision_headers["Prefer"] = "return=representation"
            resp = requests.post(
                f"{supabase_url}/rest/v1/decisions",
                json=decision_row,
                headers=decision_headers,
                timeout=10,
            )
            if resp.status_code in (200, 201):
                log("decisions 기록 완료")
                # 임베딩 생성 (RAG 벡터검색용)
                try:
                    resp_data_for_emb = resp.json()
                    emb_decision_id = None
                    if isinstance(resp_data_for_emb, list) and resp_data_for_emb:
                        emb_decision_id = resp_data_for_emb[0].get("id")
                    elif isinstance(resp_data_for_emb, dict):
                        emb_decision_id = resp_data_for_emb.get("id")

                    if emb_decision_id:
                        from scripts.save_decision import generate_state_embedding, _update_embedding_via_sql
                        emb_data = {
                            "current_price": market_data.get("ticker", {}).get("trade_price"),
                            "change_rate_24h": market_data.get("ticker", {}).get("signed_change_rate"),
                            "rsi_14": market_data.get("indicators", {}).get("rsi_14"),
                            "sma_20": market_data.get("indicators", {}).get("sma_20"),
                            "fear_greed_value": market_data.get("fear_greed", {}).get("value"),
                            "volume_24h": market_data.get("ticker", {}).get("acc_trade_volume_24h"),
                            "news_sentiment": market_data.get("news", {}).get("overall_sentiment"),
                        }
                        emb_text, emb_vector = generate_state_embedding(emb_data)
                        if emb_vector:
                            _update_embedding_via_sql(emb_decision_id, emb_vector, emb_text)
                            log("임베딩 생성 완료 (RAG)")
                except Exception as emb_e:
                    log(f"임베딩 생성 실패 (비치명적): {emb_e}")

                # RL prediction에 decision_id 연결
                if emb_decision_id and rl_advisory:
                    try:
                        from rl_hybrid.rl.rl_db_logger import log_prediction as _log_pred_final
                        _log_pred_final(
                            decision_id=emb_decision_id,
                            cycle_id=_CYCLE_ID,
                            ensemble_action=round(rl_advisory.get("action", 0), 4),
                            ensemble_direction=rl_advisory.get("direction"),
                            num_models=rl_advisory.get("num_models", 0),
                            btc_price=market_data.get("ticker", {}).get("trade_price"),
                            rsi_14=market_data.get("indicators", {}).get("rsi_14"),
                            fgi=market_data.get("fear_greed", {}).get("value"),
                            danger_score=market_state.get("danger_score"),
                            opportunity_score=market_state.get("opportunity_score"),
                        )
                        log("RL prediction DB 기록 (decision_id 연결)")
                    except Exception as _rl_db_err:
                        log(f"RL prediction DB 기록 실패 (비치명적): {_rl_db_err}")
            else:
                log(f"decisions 기록 실패 (HTTP {resp.status_code}): {resp.text[:300]}")
                pipeline_errors.append({"phase": "phase5", "source": "decisions", "error": resp.text[:300]})
        except Exception as e:
            log(f"decisions 기록 예외: {e}")
            pipeline_errors.append({"phase": "phase5", "source": "decisions", "error": str(e)})

        # 5a-2. market_context_log 테이블 (결정 시점 전체 시장 스냅샷)
        try:
            decision_id = None
            if resp is not None and resp.ok:
                try:
                    resp_data = resp.json()
                    if isinstance(resp_data, list) and len(resp_data) > 0:
                        decision_id = resp_data[0].get("id")
                    elif isinstance(resp_data, dict):
                        decision_id = resp_data.get("id")
                except Exception:
                    pass

            from scripts.save_decision import save_market_context
            market_state = result.get("market_state", {})
            save_market_context(
                decision_id=decision_id,
                market_data=market_data,
                external_data=external_data,
                portfolio=portfolio,
                agent_state={
                    "active_agent": agent_name,
                    "danger_score": market_state.get("danger_score"),
                    "opportunity_score": market_state.get("opportunity_score"),
                },
            )
            log("market_context_log 기록 완료")
        except Exception as e:
            log(f"market_context_log 기록 예외: {e}")
            pipeline_errors.append({"phase": "phase5", "source": "market_context_log", "error": str(e)})

        # 5b. market_data 테이블
        try:
            save_market_data_record(market_data, external_data)
        except Exception as e:
            log(f"market_data 기록 예외: {e}")
            pipeline_errors.append({"phase": "phase5", "source": "market_data", "error": str(e)})

        # 5c. execution_logs 테이블
        try:
            duration_ms = int((time.time() - pipeline_start) * 1000)
            dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
            if decision in ("buy", "sell") and not dry_run:
                exec_mode = "execute"
            elif decision in ("buy", "sell") and dry_run:
                exec_mode = "dry_run"
            else:
                exec_mode = "analyze"

            save_execution_log(
                execution_mode=exec_mode,
                duration_ms=duration_ms,
                data_sources={
                    "sources": data_sources_used,
                    "agent": agent_name,
                    "decision": decision,
                    "snapshot_dir": str(snapshot_dir),
                },
                errors={"pipeline_errors": pipeline_errors} if pipeline_errors else None,
                raw_output=json.dumps(output, ensure_ascii=False)[:10000],
                phases_completed=["phase1", "phase2", "phase2.5", "phase3", "phase4", "phase5"],
            )
        except Exception as e:
            log(f"execution_logs 기록 예외: {e}")

    # Phase 6: 전환 성과 평가
    log("Phase 6: 전환 성과 평가...")
    subprocess.run([sys.executable, "scripts/evaluate_switches.py"], cwd=str(PROJECT_DIR), check=False)
    
    # Phase 6.5: RL 온라인 학습 버퍼
    if rl_advisory:
        try:
            from rl_hybrid.rl.online_buffer import OnlineExperienceBuffer
            log("Phase 6.5: RL 온라인 학습 버퍼...")
            buf = OnlineExperienceBuffer()
            buf.add_experience(
                market_data=market_data,
                external_data=external_data,
                portfolio=portfolio,
                agent_state=agent_state_for_rl,
                rl_action=rl_advisory["action"],
                agent_decision=decision,
            )
            stats = buf.get_stats()
            log(f"온라인 버퍼: {stats['total']}/{stats['trigger_size']}건 (outcome: {stats['outcome_filled']}건)")
            if buf.should_train():
                log("Phase 6.5: RL 미세 학습 시작...")
                train_result = buf.micro_train()
                log(f"RL 미세 학습: {train_result.get('message', 'N/A')}")
        except Exception as e:
            log(f"Phase 6.5 온라인 학습 예외: {e}")

    # Phase 6.7: Parameter Self-Tuning
    try:
        from rl_hybrid.config import SystemConfig as _STConfig
        if _STConfig().self_tuning.enabled:
            log("Phase 6.7: 파라미터 자동 튜닝...")
            from rl_hybrid.rl.self_tuning_rl import run_parameter_tuning
            # 현재 에이전트 파라미터 및 시장 레짐 수집
            _market_state = result.get("market_state", {})
            _perf_metrics = {
                "recent_decision": decision,
                "danger_score": _market_state.get("danger_score", 50),
                "opportunity_score": _market_state.get("opportunity_score", 50),
            }
            _market_regime = {
                "fgi": market_data.get("fear_greed", {}).get("value"),
                "rsi": market_data.get("indicators", {}).get("rsi_14"),
                "change_rate_24h": market_data.get("ticker", {}).get("signed_change_rate"),
            }
            tuning_result = run_parameter_tuning(
                performance_metrics=_perf_metrics,
                market_regime=_market_regime,
            )
            if tuning_result:
                log(f"Phase 6.7: {tuning_result.get('status', '?')} -- {tuning_result.get('message', 'N/A')}")
    except Exception as e:
        log(f"Phase 6.7 Self-tuning 스킵: {e}")

    log("═══ 에이전트 모드 완료 ═══")
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--dry-run", action="store_true", help="DRY_RUN=true 강제 (포어그라운드 테스트용)")
    _args = _parser.parse_args()
    if _args.dry_run:
        os.environ["DRY_RUN"] = "true"
    main()
