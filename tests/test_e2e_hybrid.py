"""LLM-RL 하이브리드 시스템 End-to-End 테스트

전체 파이프라인을 7단계로 검증:
  1. 시장 데이터 수집
  2. 포트폴리오 조회
  3. 외부 데이터 수집 (ExternalDataAgent)
  4. 에이전트 파이프라인 (Orchestrator)
  5. RL 추론 (DistributedTrainer)
  6. LLM/RAG 분석 (Gemini + pgvector)
  7. 결정 융합 (DecisionBlender)
"""

import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

results = {}


def phase1_market_data():
    """시장 데이터 수집"""
    print("\n[Phase 1] 시장 데이터 수집...")
    proc = subprocess.run(
        [sys.executable, "scripts/collect_market_data.py"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        print(f"  [FAIL] {proc.stderr[:150]}")
        return None

    data = json.loads(proc.stdout)
    price = data.get("current_price", 0)
    ind = data.get("indicators", {})
    change = data.get("change_rate_24h", 0)
    print(f"  BTC: {price:,.0f} KRW | 24h: {change*100:+.2f}%")
    macd = ind.get("macd", 0)
    macd_val = macd.get("macd", 0) if isinstance(macd, dict) else macd
    print(f"  RSI: {ind.get('rsi_14', 0):.1f} | SMA20: {ind.get('sma_20', 0):,.0f} | MACD: {macd_val:,.0f}")
    print("  [OK]")
    results["market_data"] = True
    return data


def phase2_portfolio():
    """포트폴리오 조회"""
    print("\n[Phase 2] 포트폴리오 조회...")
    proc = subprocess.run(
        [sys.executable, "scripts/get_portfolio.py"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        print(f"  [FAIL] {proc.stderr[:150]}")
        return None

    data = json.loads(proc.stdout)
    total = data.get("total_krw", 0) or data.get("total_balance", 0)
    krw = data.get("krw_balance", 0)
    print(f"  총 자산: {total:,.0f} KRW | 현금: {krw:,.0f} KRW")
    coins = data.get("coins", [])
    for c in coins[:3]:
        curr = c.get("currency", "?")
        bal = c.get("balance", 0)
        pnl = c.get("profit_rate", 0) * 100
        print(f"  {curr}: {bal} (수익: {pnl:+.2f}%)")
    print("  [OK]")
    results["portfolio"] = True
    return data


def phase3_external_data(market_data):
    """외부 데이터 수집"""
    print("\n[Phase 3] 외부 데이터 수집 (ExternalDataAgent)...")
    try:
        from agents.external_data import ExternalDataAgent
        ext_agent = ExternalDataAgent()
        data = ext_agent.collect_all()

        sources = data.get("sources", {})
        signal = data.get("external_signal", {})
        print(f"  수집 소스: {len(sources)}개")
        print(f"  종합 시그널: {signal.get('total_score', 'N/A')} ({signal.get('signal', 'N/A')})")
        fgi = sources.get("fgi", {})
        if isinstance(fgi, dict) and fgi.get("value"):
            print(f"  FGI: {fgi.get('value')} ({fgi.get('classification', 'N/A')})")
        print("  [OK]")
        results["external_data"] = True
        return data
    except Exception as e:
        print(f"  [FAIL] {e}")
        return {"sources": {}, "external_signal": {"total_score": 0, "signal": "neutral"}}


def phase4_orchestrator(market_data, external_data, portfolio):
    """에이전트 파이프라인"""
    print("\n[Phase 4] 에이전트 파이프라인 (Orchestrator)...")
    try:
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        result = orch.run(
            market_data=market_data,
            external_data=external_data,
            portfolio=portfolio,
        )

        if not result:
            print("  [FAIL] Orchestrator returned None")
            return None

        decision = result.get("decision", {})
        ms = result.get("market_state", {})
        print(f"  활성 에이전트: {ms.get('active_agent', 'N/A')}")
        print(f"  Danger: {ms.get('danger_score', 0)} | Opportunity: {ms.get('opportunity_score', 0)}")
        print(f"  결정: {decision.get('decision', 'N/A')} (conf: {decision.get('confidence', 0):.2f})")
        reason = decision.get("reason", "N/A")
        print(f"  근거: {reason[:80]}")
        print("  [OK]")
        results["orchestrator"] = True
        return result
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return {"decision": {"decision": "hold", "confidence": 0.5}, "market_state": {}}


def phase5_rl_inference(market_data, external_data, portfolio, agent_result):
    """RL 추론"""
    print("\n[Phase 5] RL 추론 (DistributedTrainer)...")
    try:
        from rl_hybrid.rl.state_encoder import StateEncoder
        from rl_hybrid.rl.distributed_trainer import DistributedTrainer

        encoder = StateEncoder()
        trainer = DistributedTrainer(obs_dim=42)
        trainer.load_model()

        ms = agent_result.get("market_state", {}) if agent_result else {}
        agent_state = {
            "danger_score": ms.get("danger_score", 30),
            "opportunity_score": ms.get("opportunity_score", 30),
            "cascade_risk": (agent_result or {}).get("drop_context", {}).get("cascade_risk", 20),
            "consecutive_losses": ms.get("consecutive_losses", 0),
        }

        obs = encoder.encode(
            market_data=market_data,
            external_data=external_data,
            portfolio=portfolio,
            agent_state=agent_state,
        )

        action, log_prob, value = trainer.predict(obs, deterministic=True)

        if action > 0.5: interp = "strong_buy"
        elif action > 0.2: interp = "cautious_buy"
        elif action < -0.5: interp = "strong_sell"
        elif action < -0.2: interp = "cautious_sell"
        else: interp = "hold"

        print(f"  관측 벡터: {obs.shape} (42d)")
        print(f"  Action: {action:.4f} ({interp})")
        print(f"  Value: {value:.4f} | LogProb: {log_prob:.4f}")
        params = sum(p.numel() for p in trainer.model.parameters())
        print(f"  모델 파라미터: {params:,}")
        print("  [OK]")
        results["rl_inference"] = True
        return {"action": action, "value": value, "log_prob": log_prob, "interpretation": interp}
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return None


def phase6_llm_rag(market_data, external_data):
    """LLM/RAG 분석"""
    print("\n[Phase 6] LLM/RAG 분석 (Gemini + pgvector)...")
    try:
        from rl_hybrid.rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()

        analysis = pipeline.analyze_and_store(
            cycle_id=f"e2e_test_{int(time.time())}",
            market_data=market_data,
            external_data=external_data,
        )

        if analysis:
            print(f"  Market Regime: {analysis.get('market_regime', 'N/A')}")
            print(f"  Action: {analysis.get('recommended_action', 'N/A')}")
            print(f"  Confidence: {analysis.get('confidence', 'N/A')}")
            signals = analysis.get("key_signals", [])[:3]
            print(f"  Key Signals: {', '.join(str(s) for s in signals)}")
            print(f"  Embedding ID: {analysis.get('_embedding_id', 'N/A')}")
            print(f"  Pipeline Time: {analysis.get('_pipeline_time', 'N/A')}s")
            print("  [OK]")
            results["llm_rag"] = True
            return analysis
        else:
            print("  [WARN] Analysis returned None (quota/timeout)")
            return None
    except Exception as e:
        print(f"  [FAIL] {e}")
        return None


def phase7_blender(agent_result, rl_prediction, llm_analysis, portfolio):
    """결정 융합"""
    print("\n[Phase 7] 결정 융합 (DecisionBlender)...")
    try:
        from rl_hybrid.rl.decision_blender import DecisionBlender
        blender = DecisionBlender()

        blended = blender.blend(
            agent_result=agent_result.get("decision") if agent_result else None,
            rl_prediction=rl_prediction,
            llm_analysis=llm_analysis,
            portfolio=portfolio,
            market_state=agent_result.get("market_state") if agent_result else None,
        )

        print(f"  최종 결정: {blended.decision}")
        print(f"  신뢰도: {blended.confidence:.3f}")
        print(f"  Action Value: {blended.action_value:.4f}")
        print(f"  Agent: {blended.agent_decision} | RL: {blended.rl_action:.3f}")
        print(f"  근거: {blended.reason[:100]}")
        print(f"  매매 파라미터: {blended.trade_params}")
        print("  [OK]")
        results["blender"] = True
        return blended
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return None


def print_summary():
    """결과 요약"""
    print("\n" + "=" * 60)
    print(" End-to-End 테스트 결과 요약")
    print("=" * 60)

    phases = [
        ("Phase 1: 시장 데이터 수집", "market_data"),
        ("Phase 2: 포트폴리오 조회", "portfolio"),
        ("Phase 3: 외부 데이터 수집", "external_data"),
        ("Phase 4: 에이전트 파이프라인", "orchestrator"),
        ("Phase 5: RL 추론", "rl_inference"),
        ("Phase 6: LLM/RAG 분석", "llm_rag"),
        ("Phase 7: 결정 융합", "blender"),
    ]

    passed = 0
    for name, key in phases:
        ok = results.get(key, False)
        icon = "[v]" if ok else "[X]"
        print(f"  {icon} {name}")
        if ok:
            passed += 1

    print(f"\n  Total: {passed}/{len(phases)} PASSED")
    print("=" * 60)


if __name__ == "__main__":
    start = time.time()

    market_data = phase1_market_data()
    if not market_data:
        market_data = {"current_price": 100000000, "indicators": {"rsi_14": 50}, "change_rate_24h": 0}

    portfolio = phase2_portfolio()
    if not portfolio:
        portfolio = {"krw_balance": 1000000, "total_krw": 1000000, "btc_ratio": 0}

    external_data = phase3_external_data(market_data)

    agent_result = phase4_orchestrator(market_data, external_data, portfolio)

    rl_prediction = phase5_rl_inference(market_data, external_data, portfolio, agent_result)

    llm_analysis = phase6_llm_rag(market_data, external_data)

    blended = phase7_blender(agent_result, rl_prediction, llm_analysis, portfolio)

    elapsed = time.time() - start
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print_summary()
