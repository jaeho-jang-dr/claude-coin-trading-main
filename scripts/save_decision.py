#!/usr/bin/env python3
"""
매매 결정을 Supabase에 저장하는 스크립트

cron_run.sh에서 claude -p 응답을 파싱하여 decisions 테이블에 기록한다.
이전 결정의 사후 성과도 함께 업데이트한다.

사용법:
  echo "$CLAUDE_RESPONSE" | python3 scripts/save_decision.py
  python3 scripts/save_decision.py "$CLAUDE_RESPONSE"
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── RAG 임베딩 ──────────────────────────────────────────────────────────────

def generate_state_embedding(market_data: dict) -> tuple[str, list] | tuple[None, None]:
    """시장 상태를 영문 텍스트로 요약하고 OpenAI 임베딩 벡터를 생성한다.

    Args:
        market_data: current_price, rsi_14, fear_greed_value, sma_20,
                     change_rate_24h, news_sentiment, funding_rate, volume_24h 등

    Returns:
        (embedding_text, embedding_vector) 또는 실패 시 (None, None)
    """
    try:
        text = build_embedding_text(market_data)
        if not text:
            return None, None

        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        vector = resp.data[0].embedding
        return text, vector
    except Exception as e:
        print(f"[save_decision] 임베딩 생성 실패: {e}", file=sys.stderr)
        return None, None


def build_embedding_text(market_data: dict) -> str:
    """시장 데이터를 임베딩용 영문 텍스트로 변환한다.

    save_decision.py와 recall_rag.py가 동일한 텍스트를 생성하도록 공유 함수로 분리.
    """
    parts = []

    # Price
    price = market_data.get("current_price")
    change = market_data.get("change_rate_24h")
    if price:
        price_str = f"Bitcoin price is {int(price):,} KRW"
        if change is not None:
            pct = change * 100 if abs(change) < 1 else change
            price_str += f" (24h change: {pct:+.1f}%)"
        parts.append(price_str + ".")

    # RSI
    rsi_val = market_data.get("rsi_14") or (market_data.get("indicators", {}) or {}).get("rsi_14")
    if rsi_val is not None:
        rsi_val = float(rsi_val)
        if rsi_val < 30:
            label = "oversold"
        elif rsi_val > 70:
            label = "overbought"
        else:
            label = "neutral"
        parts.append(f"RSI is {rsi_val:.1f} ({label}).")

    # Fear & Greed
    fgi = market_data.get("fear_greed_value")
    if fgi is None:
        fg = market_data.get("fear_greed", {})
        if isinstance(fg, dict):
            fgi = fg.get("value") or (fg.get("current", {}) or {}).get("value")
    if fgi is not None:
        fgi = int(fgi)
        if fgi <= 25:
            fgi_label = "Extreme Fear"
        elif fgi <= 45:
            fgi_label = "Fear"
        elif fgi <= 55:
            fgi_label = "Neutral"
        elif fgi <= 75:
            fgi_label = "Greed"
        else:
            fgi_label = "Extreme Greed"
        parts.append(f"Fear and Greed Index is {fgi} ({fgi_label}).")

    # SMA
    sma = market_data.get("sma20_price") or market_data.get("sma_20") or (market_data.get("indicators", {}) or {}).get("sma_20")
    if sma and price:
        direction = "above" if float(price) > float(sma) else "below"
        bias = "bullish" if direction == "above" else "bearish"
        parts.append(f"SMA20 is {direction} price ({bias}).")

    # News sentiment
    ns = market_data.get("news_sentiment")
    if ns is not None:
        try:
            ns_val = float(ns)
            if ns_val > 0.2:
                label = "positive"
            elif ns_val < -0.2:
                label = "negative"
            else:
                label = "neutral"
            parts.append(f"News sentiment is {label} ({ns_val:+.2f}).")
        except (ValueError, TypeError):
            if isinstance(ns, str):
                parts.append(f"News sentiment is {ns}.")

    # Funding rate
    fr = market_data.get("funding_rate")
    if fr is not None:
        try:
            fr_val = float(fr)
            bias = "bullish" if fr_val > 0 else "bearish"
            parts.append(f"Funding rate is {fr_val:+.4f}% ({bias}).")
        except (ValueError, TypeError):
            pass

    # Volume
    vol = market_data.get("volume_24h")
    if vol is not None:
        try:
            vol_f = float(vol)
            parts.append(f"24h volume is {vol_f:,.0f} BTC.")
        except (ValueError, TypeError):
            pass

    # Kimchi premium
    kp = market_data.get("kimchi_premium")
    if kp is not None:
        try:
            kp_val = float(kp)
            parts.append(f"Kimchi premium is {kp_val:+.2f}%.")
        except (ValueError, TypeError):
            pass

    # Long/Short ratio
    ls = market_data.get("long_short_ratio")
    if ls is not None:
        try:
            ls_val = float(ls)
            parts.append(f"Long/Short ratio is {ls_val:.2f}.")
        except (ValueError, TypeError):
            pass

    return " ".join(parts)


def _update_embedding_via_sql(decision_id: str, embedding: list, embedding_text: str):
    """psycopg2로 직접 SQL을 실행하여 벡터 임베딩을 업데이트한다.

    Supabase REST API가 pgvector 타입을 지원하지 않으므로 직접 DB 연결이 필요.
    """
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("[save_decision] SUPABASE_DB_URL 미설정 -- 임베딩 저장 건너뜀", file=sys.stderr)
        return

    import psycopg2
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            "UPDATE decisions SET state_embedding = %s::vector, embedding_text = %s WHERE id = %s",
            (str(embedding), embedding_text, str(decision_id)),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"[save_decision] 임베딩 SQL 업데이트 실패: {e}", file=sys.stderr)
    finally:
        if conn:
            conn.close()

KST = timezone(timedelta(hours=9))

# cycle_id: 같은 실행 사이클의 모든 DB 기록을 연결하는 키
_cycle_id = None

def _get_cycle_id() -> str:
    """현재 사이클의 cycle_id를 반환 (최초 호출 시 생성)."""
    global _cycle_id
    if _cycle_id is None:
        try:
            from scripts.cycle_id import get_or_create_cycle_id
            _cycle_id = get_or_create_cycle_id("llm")
        except Exception:
            _cycle_id = datetime.now(KST).strftime("%Y%m%d-%H%M") + "-llm"
    return _cycle_id

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def supabase_post(table: str, row: dict) -> dict | None:
    """Supabase 테이블에 INSERT. 실패 시 stderr에 로그."""
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=supabase_headers(),
            json=row,
            timeout=10,
        )
        if not r.ok:
            print(f"[save_decision] {table} INSERT 실패 ({r.status_code}): {r.text[:500]}", file=sys.stderr)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[save_decision] {table} INSERT 예외: {e}", file=sys.stderr)
        return None


def _try_parse(s: str) -> dict | None:
    """JSON 파싱 시도. 후행 쉼표 등 경미한 오류도 정리 후 재시도."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Claude가 간혹 후행 쉼표를 출력 (,} 또는 ,])
    cleaned = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def extract_json_from_response(text: str) -> dict | None:
    """claude -p 응답에서 JSON 블록을 추출한다.

    5단계 fallback으로 다양한 Claude 출력 형식에 대응:
    1) ```json 코드펜스
    2) ``` 코드펜스 (json 태그 없이)
    3) 전체 텍스트가 JSON
    4) 가장 큰 { ... } 블록 (중첩 지원)
    5) 불완전 JSON 복구 시도
    """
    # 1) ```json 코드펜스 (가장 일반적)
    m = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        result = _try_parse(m.group(1))
        if result:
            return result

    # 2) ``` 코드펜스 (json 태그 없이)
    m = re.search(r"```\s*\n(\{.*\})\n\s*```", text, re.DOTALL)
    if m:
        result = _try_parse(m.group(1))
        if result:
            return result

    # 3) 전체 텍스트가 JSON
    result = _try_parse(text.strip())
    if result:
        return result

    # 4) 텍스트 내에서 가장 큰 { ... } 블록 찾기 (depth 기반)
    best = None
    best_len = 0
    depth = 0
    start = -1
    in_string = False
    escape_next = False
    for i, c in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                parsed = _try_parse(candidate)
                if parsed and len(candidate) > best_len:
                    best = parsed
                    best_len = len(candidate)
                start = -1

    if best:
        return best

    # 5) 최후: 불완전 JSON 복구 (닫는 괄호 부족 시)
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
    if start >= 0 and depth > 0:
        # 닫는 괄호 추가하여 복구 시도
        candidate = text[start:] + "}" * depth
        result = _try_parse(candidate)
        if result:
            return result

    return None


def map_decision(raw: str) -> str:
    """다양한 결정 표현을 DB enum으로 매핑."""
    raw_lower = raw.lower().strip()
    if raw_lower in ("hold", "관망"):
        return "관망"
    if raw_lower in ("buy", "bid", "매수", "strong_buy"):
        return "매수"
    if raw_lower in ("sell", "ask", "매도", "strong_sell", "reduce"):
        return "매도"
    print(f"[save_decision] 알 수 없는 결정값 '{raw}' → 관망으로 처리", file=sys.stderr)
    return "관망"


def _get_nested(data: dict, *keys, default=None):
    """여러 키 후보 중 첫 번째로 존재하는 값을 반환."""
    for key in keys:
        if "." in key:
            parts = key.split(".")
            val = data
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p)
                else:
                    val = None
                    break
            if val is not None:
                return val
        elif key in data and data[key] is not None:
            return data[key]
    return default


def _safe_float(val, default=None):
    """안전하게 float 변환. 실패 시 default 반환."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=None):
    """안전하게 int 변환. 실패 시 default 반환."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def save_market_context(
    decision_id: str | None = None,
    market_data: dict | None = None,
    external_data: dict | None = None,
    portfolio: dict | None = None,
    agent_state: dict | None = None,
) -> dict | None:
    """매매 결정 시점의 전체 시장 컨텍스트를 market_context_log에 저장한다.

    모든 데이터 소스(기술지표, 감성, 파생, 온체인, 매크로, 포트폴리오, 에이전트 상태)를
    개별 컬럼 + raw_data JSONB로 이중 보관한다.
    """
    market_data = market_data or {}
    external_data = external_data or {}
    portfolio = portfolio or {}
    agent_state = agent_state or {}

    sources = external_data.get("sources", {})

    # ── Price ──
    ticker = market_data.get("ticker", {})
    btc_price = _safe_int(
        ticker.get("trade_price")
        or market_data.get("current_price")
        or market_data.get("price")
    )
    btc_24h_change = _safe_float(
        ticker.get("signed_change_rate", market_data.get("change_rate_24h", 0))
    )
    if btc_24h_change is not None and abs(btc_24h_change) < 1:
        btc_24h_change = round(btc_24h_change * 100, 2)  # 비율 → 퍼센트
    btc_volume = _safe_float(
        ticker.get("acc_trade_volume_24h")
        or market_data.get("volume_24h")
    )

    # ── Technical indicators ──
    indicators = market_data.get("indicators", {})
    rsi_14 = _safe_float(indicators.get("rsi_14", indicators.get("rsi")))
    sma_20 = _safe_int(indicators.get("sma_20", indicators.get("sma20")))
    sma_vs_price = None
    if btc_price and sma_20:
        sma_vs_price = "above" if btc_price > sma_20 else "below"

    macd = indicators.get("macd", {})
    if isinstance(macd, dict):
        macd_value = _safe_float(macd.get("macd", macd.get("value")))
        macd_signal_val = _safe_float(macd.get("signal"))
        macd_histogram = _safe_float(macd.get("histogram", macd.get("hist")))
    else:
        macd_value = macd_signal_val = macd_histogram = None

    bb = indicators.get("bollinger_bands", indicators.get("bb", {}))
    if isinstance(bb, dict):
        bb_upper = _safe_int(bb.get("upper"))
        bb_lower = _safe_int(bb.get("lower"))
        bb_position = None
        if btc_price and bb_upper and bb_lower:
            if btc_price >= bb_upper:
                bb_position = "above_upper"
            elif btc_price <= bb_lower:
                bb_position = "below_lower"
            else:
                bb_position = "inside"
    else:
        bb_upper = bb_lower = bb_position = None

    adx_value = _safe_float(indicators.get("adx", indicators.get("adx_value")))
    adx_regime = None
    if adx_value is not None:
        if adx_value >= 25:
            adx_regime = "trending"
        elif adx_value < 20:
            adx_regime = "ranging"
        else:
            adx_regime = "transitioning"

    # ── Sentiment ──
    fgi = market_data.get("fear_greed", sources.get("fear_greed", {}).get("current", {}))
    fgi_value = _safe_int(fgi.get("value") if isinstance(fgi, dict) else fgi)
    fgi_class = fgi.get("value_classification", "") if isinstance(fgi, dict) else None

    news = market_data.get("news", sources.get("news_sentiment", {}))
    news_sentiment_score = _safe_float(news.get("sentiment_score", news.get("score")))
    news_positive = _safe_int(news.get("positive", news.get("positive_count")))
    news_negative = _safe_int(news.get("negative", news.get("negative_count")))
    news_neutral_count = _safe_int(news.get("neutral", news.get("neutral_count")))

    # ── Binance Derivatives ──
    binance = sources.get("binance_sentiment", {})
    funding = binance.get("funding_rate", {})
    funding_rate = _safe_float(funding.get("current_rate", funding.get("rate")))
    ls = binance.get("top_trader_long_short", {})
    long_short_ratio = _safe_float(ls.get("current_ratio", ls.get("ratio")))
    oi_data = binance.get("open_interest", {})
    open_interest = _safe_float(oi_data.get("value", oi_data.get("open_interest")))
    oi_change_pct = _safe_float(oi_data.get("change_pct"))
    kimchi = binance.get("kimchi_premium", {})
    kimchi_premium = _safe_float(kimchi.get("premium_pct", kimchi.get("kimchi_premium")))

    # ── Whale / On-chain ──
    whale = sources.get("whale_tracker", {})
    whale_score = whale.get("whale_score", {})
    whale_flow = whale_score.get("direction") if isinstance(whale_score, dict) else None
    large_tx_count = _safe_int(whale.get("large_tx_count", whale_score.get("large_tx_count") if isinstance(whale_score, dict) else None))
    exchange_net_flow = _safe_float(whale.get("exchange_net_flow"))

    # ── Macro ──
    macro = sources.get("macro", {})
    macro_analysis = macro.get("analysis", macro)
    sp500_trend = macro_analysis.get("sp500_trend", macro_analysis.get("sp500"))
    dxy_trend = macro_analysis.get("dxy_trend", macro_analysis.get("dxy"))
    gold_trend = macro_analysis.get("gold_trend", macro_analysis.get("gold"))
    macro_sentiment_val = macro_analysis.get("macro_sentiment", macro_analysis.get("overall"))

    # ── ETH/BTC ──
    eth_btc = market_data.get("eth_btc", sources.get("eth_btc", {}))
    eth_btc_ratio = _safe_float(eth_btc.get("ratio", eth_btc.get("eth_btc_ratio")))
    eth_btc_trend = eth_btc.get("trend", eth_btc.get("signal"))

    # ── Portfolio state ──
    krw_info = portfolio.get("krw", {})
    btc_info = portfolio.get("btc", portfolio.get("coins", {}).get("BTC", {}))
    krw_balance = _safe_int(krw_info.get("balance", portfolio.get("krw_balance")))
    btc_balance = _safe_float(btc_info.get("balance") if isinstance(btc_info, dict) else None)
    btc_avg_price = _safe_int(btc_info.get("avg_buy_price") if isinstance(btc_info, dict) else None)
    total_asset_value = _safe_int(portfolio.get("total_eval", portfolio.get("total_evaluation")))
    position_ratio = _safe_float(portfolio.get("btc_ratio"))

    # ── Agent state ──
    active_agent = agent_state.get("active_agent", agent_state.get("agent_name"))
    danger_score = _safe_int(agent_state.get("danger_score"))
    opportunity_score = _safe_int(agent_state.get("opportunity_score"))

    # ── Build row ──
    row = {
        "decision_id": decision_id,
        "cycle_id": _get_cycle_id(),
        "btc_price": btc_price,
        "btc_24h_change": btc_24h_change,
        "btc_volume": btc_volume,
        "rsi_14": rsi_14,
        "sma_20": sma_20,
        "sma_vs_price": sma_vs_price,
        "macd_value": macd_value,
        "macd_signal": macd_signal_val,
        "macd_histogram": macd_histogram,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_position": bb_position,
        "adx_value": adx_value,
        "adx_regime": adx_regime,
        "fgi_value": fgi_value,
        "fgi_class": fgi_class,
        "news_sentiment": news_sentiment_score,
        "news_positive": news_positive,
        "news_negative": news_negative,
        "news_neutral": news_neutral_count,
        "funding_rate": funding_rate,
        "long_short_ratio": long_short_ratio,
        "open_interest": open_interest,
        "oi_change_pct": oi_change_pct,
        "kimchi_premium": kimchi_premium,
        "whale_flow": whale_flow,
        "large_tx_count": large_tx_count,
        "exchange_net_flow": exchange_net_flow,
        "sp500_trend": sp500_trend,
        "dxy_trend": dxy_trend,
        "gold_trend": gold_trend,
        "macro_sentiment": macro_sentiment_val,
        "eth_btc_ratio": eth_btc_ratio,
        "eth_btc_trend": eth_btc_trend,
        "krw_balance": krw_balance,
        "btc_balance": btc_balance,
        "btc_avg_price": btc_avg_price,
        "total_asset_value": total_asset_value,
        "position_ratio": position_ratio,
        "active_agent": active_agent,
        "danger_score": danger_score,
        "opportunity_score": opportunity_score,
        "raw_data": json.dumps({
            "market_data": market_data,
            "external_data": external_data,
            "portfolio": portfolio,
            "agent_state": agent_state,
        }, ensure_ascii=False, default=str),
    }

    # None 값 제거 (Supabase는 NULL로 처리하지만 명시적 null 전송 방지)
    row = {k: v for k, v in row.items() if v is not None}

    return supabase_post("market_context_log", row)


def save_decision(data: dict) -> dict | None:
    """decisions 테이블에 저장.

    Claude의 실제 출력 구조에 맞춘 필드 매핑:
    - current_price: 최상위 current_price
    - buy_score: buy_score.total / buy_score.fgi.value 등
    - fear_greed_value: buy_score.fgi.value
    - rsi_value: buy_score.rsi.value
    - sma20_price: 시장 데이터에서 추출 (별도 저장 안 하면 NULL 허용)
    """
    buy_score = data.get("buy_score", {})
    fgi_obj = buy_score.get("fgi", {}) if isinstance(buy_score.get("fgi"), dict) else {}
    rsi_obj = buy_score.get("rsi", {}) if isinstance(buy_score.get("rsi"), dict) else {}

    # confidence: Claude가 85 같은 정수로 출력 → 0.85로 변환 (DB는 DECIMAL(3,2))
    raw_conf = float(data.get("confidence", 0))
    confidence = raw_conf / 100.0 if raw_conf > 1 else raw_conf

    row = {
        "market": data.get("market", "KRW-BTC"),
        "decision": map_decision(data.get("decision", "hold")),
        "confidence": round(confidence, 2),
        "reason": data.get("reason", ""),
        "current_price": data.get("current_price"),
        "fear_greed_value": fgi_obj.get("value"),
        "rsi_value": rsi_obj.get("value"),
        "sma20_price": data.get("sma20_price"),
        "executed": data.get("executed", data.get("trade_executed", False)),
        "cycle_id": data.get("cycle_id") or _get_cycle_id(),
        "source": data.get("source", "llm"),
        # 전체 JSON을 execution_result에 저장 (감사 추적용)
        "execution_result": json.dumps(data, ensure_ascii=False),
        # buy_score + ai_signal + portfolio를 market_data_snapshot에 저장
        "market_data_snapshot": json.dumps({
            "buy_score": buy_score,
            "ai_composite_signal": data.get("ai_composite_signal"),
            "portfolio_status": data.get("portfolio_status", data.get("portfolio")),
            "risk_alerts": data.get("risk_alerts", []),
            "eth_btc_signal": data.get("eth_btc_signal"),
            "strategy_switch": data.get("strategy_switch_recommendation"),
        }, ensure_ascii=False),
    }

    # 매매 실행된 경우 금액 기록
    trade_details = data.get("trade_details", {})
    if trade_details.get("amount"):
        row["trade_amount"] = trade_details["amount"]
    elif data.get("trade_amount"):
        row["trade_amount"] = data["trade_amount"]

    result = supabase_post("decisions", row)

    # 결정 저장 성공 시 market_context_log + 임베딩 저장
    if result:
        decision_id = None
        if isinstance(result, list) and len(result) > 0:
            decision_id = result[0].get("id")
        elif isinstance(result, dict):
            decision_id = result.get("id")

        # market_context_log 저장
        try:
            save_market_context(
                decision_id=decision_id,
                market_data=data.get("_market_data", {}),
                external_data=data.get("_external_data", {}),
                portfolio=data.get("_portfolio", {}),
                agent_state=data.get("_agent_state", {}),
            )
        except Exception as e:
            print(f"[save_decision] market_context_log 저장 실패: {e}", file=sys.stderr)

        # RAG 임베딩 생성 및 저장 (실패해도 매매 파이프라인에 영향 없음)
        if decision_id:
            try:
                emb_data = {**data}
                # _market_data가 있으면 더 풍부한 데이터 소스 사용
                raw_market = data.get("_market_data", {})
                if raw_market:
                    emb_data.setdefault("current_price", raw_market.get("current_price"))
                    emb_data.setdefault("change_rate_24h", raw_market.get("change_rate_24h"))
                    emb_data.setdefault("volume_24h", raw_market.get("volume_24h"))
                    if "indicators" in raw_market:
                        emb_data.setdefault("indicators", raw_market["indicators"])
                emb_text, emb_vector = generate_state_embedding(emb_data)
                if emb_text and emb_vector:
                    _update_embedding_via_sql(decision_id, emb_vector, emb_text)
            except Exception as e:
                print(f"[save_decision] RAG 임베딩 저장 실패 (무시): {e}", file=sys.stderr)

    return result


def update_past_performance():
    """이전 결정의 사후 성과를 업데이트한다.

    profit_loss가 NULL인 과거 결정을 찾아,
    결정 시점 가격 vs 현재 가격으로 성과를 기록한다.
    """
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/decisions"
        "?profit_loss=is.null&order=created_at.desc&limit=20",
        headers=supabase_headers(),
        timeout=10,
    )
    if not r.ok:
        return

    decisions = r.json()
    if not decisions:
        return

    # 현재 BTC 가격
    try:
        ticker = requests.get(
            "https://api.upbit.com/v1/ticker?markets=KRW-BTC", timeout=5
        ).json()[0]
        current_price = ticker["trade_price"]
    except Exception:
        return

    # 결정 유형별로 그룹화하여 일괄 업데이트 (N+1 → 최대 3회 API 호출)
    groups: dict[str, list[str]] = {}  # profit_loss_value -> [id, ...]
    for d in decisions:
        decision_price = d.get("current_price")
        if not decision_price or decision_price == 0:
            continue

        decision_type = d.get("decision", "관망")
        price_change_pct = (current_price - decision_price) / decision_price * 100

        if decision_type == "관망":
            profit_loss = -price_change_pct
        elif decision_type == "매수":
            profit_loss = price_change_pct
        elif decision_type == "매도":
            profit_loss = -price_change_pct
        else:
            profit_loss = 0

        pl_rounded = str(round(profit_loss, 2))
        groups.setdefault(pl_rounded, []).append(str(d["id"]))

    # 같은 profit_loss 값을 가진 결정들을 한 번에 업데이트
    for pl_value, ids in groups.items():
        id_filter = ",".join(f"'{i}'" for i in ids)
        try:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/decisions?id=in.({id_filter})",
                headers=supabase_headers(),
                json={"profit_loss": float(pl_value)},
                timeout=10,
            )
        except Exception as e:
            print(f"[save_decision] 성과 일괄 업데이트 실패: {e}", file=sys.stderr)


def mark_feedback_applied():
    """프롬프트에 주입된 미반영 피드백을 applied=true로 갱신."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/feedback?applied=eq.false&select=id",
            headers=supabase_headers(),
            timeout=10,
        )
        if not r.ok:
            return
        feedbacks = r.json()
        if not feedbacks:
            return

        ids = ",".join(f"'{f['id']}'" for f in feedbacks)
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/feedback?id=in.({ids})",
            headers=supabase_headers(),
            json={
                "applied": True,
                "applied_at": datetime.now(KST).isoformat(),
            },
            timeout=10,
        )
        print(f"[save_decision] {len(feedbacks)}건 피드백 applied 처리", file=sys.stderr)
    except Exception as e:
        print(f"[save_decision] 피드백 applied 갱신 실패: {e}", file=sys.stderr)


def save_portfolio_snapshot():
    """현재 포트폴리오 스냅샷을 저장한다."""
    try:
        import subprocess
        venv_python = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        result = subprocess.run(
            [python_cmd, "scripts/get_portfolio.py"],
            capture_output=True, text=True, timeout=30,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if result.returncode != 0:
            return
        portfolio = json.loads(result.stdout)

        row = {
            "total_krw": int(portfolio.get("krw_balance", 0)),
            "total_crypto_value": int(
                sum(h.get("eval_amount", 0) for h in portfolio.get("holdings", []))
            ),
            "total_value": int(portfolio.get("total_eval", 0)),
            "holdings": json.dumps(portfolio.get("holdings", []), ensure_ascii=False),
            "cycle_id": _get_cycle_id(),
        }

        supabase_post("portfolio_snapshots", row)
    except Exception:
        pass


def save_execution_log_record(data: dict, duration_ms: int = 0, decision_id: str | None = None) -> dict | None:
    """execution_logs 테이블에 파이프라인 실행 기록을 저장한다."""
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    raw_decision = data.get("decision", "hold").lower()

    if raw_decision in ("buy", "bid", "매수", "sell", "ask", "매도", "strong_buy", "strong_sell", "reduce") and not dry_run:
        exec_mode = "execute"
    elif raw_decision in ("buy", "bid", "매수", "sell", "ask", "매도", "strong_buy", "strong_sell", "reduce") and dry_run:
        exec_mode = "dry_run"
    else:
        exec_mode = "analyze"

    row = {
        "execution_mode": exec_mode,
        "duration_ms": duration_ms if duration_ms > 0 else None,
        "data_sources": json.dumps({
            "sources": ["market_data", "portfolio", "fear_greed", "news"],
            "mode": "llm_fallback",
        }, ensure_ascii=False),
        "raw_output": json.dumps(data, ensure_ascii=False)[:10000],
        "cycle_id": _get_cycle_id(),
    }
    if decision_id:
        row["decision_id"] = decision_id

    return supabase_post("execution_logs", row)


def save_market_data_record(data: dict) -> dict | None:
    """market_data 테이블에 시장 데이터를 저장한다.

    Claude LLM 응답에서 추출 가능한 시장 데이터를 기록.
    """
    buy_score = data.get("buy_score", {})
    fgi_obj = buy_score.get("fgi", {}) if isinstance(buy_score.get("fgi"), dict) else {}
    rsi_obj = buy_score.get("rsi", {}) if isinstance(buy_score.get("rsi"), dict) else {}

    price = data.get("current_price", 0)
    if not price:
        return None

    row = {
        "market": data.get("market", "KRW-BTC"),
        "price": int(price),
        "fear_greed_value": fgi_obj.get("value"),
        "fear_greed_class": fgi_obj.get("classification"),
        "rsi_14": rsi_obj.get("value"),
        "sma_20": int(data.get("sma20_price")) if data.get("sma20_price") else None,
        "news_sentiment": data.get("news_sentiment", "neutral"),
        "cycle_id": _get_cycle_id(),
    }
    return supabase_post("market_data", row)


def main():
    start_time = time.time()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print(json.dumps({"error": "SUPABASE 환경변수 미설정"}))
        sys.exit(1)

    # 입력: 파이프 또는 인자
    if len(sys.argv) > 1:
        raw_text = sys.argv[1]
    else:
        raw_text = sys.stdin.read()

    data = extract_json_from_response(raw_text)
    if not data:
        print(json.dumps({"error": "JSON 파싱 실패", "raw": raw_text[:500]}))
        sys.exit(1)

    # 1. 이전 결정 성과 업데이트
    try:
        update_past_performance()
    except Exception as e:
        print(f"[save_decision] 성과 업데이트 실패: {e}", file=sys.stderr)

    # 2. 현재 결정 저장
    saved = save_decision(data)
    if saved:
        print(json.dumps({"success": True, "saved": saved}, ensure_ascii=False, default=str))
    else:
        print(json.dumps({"success": False, "error": "decisions INSERT 실패"}))
        sys.exit(1)

    # 3. 포트폴리오 스냅샷 저장
    try:
        save_portfolio_snapshot()
    except Exception:
        pass

    # 4. 사용된 피드백 applied 처리
    try:
        mark_feedback_applied()
    except Exception:
        pass

    # 5. market_data 기록
    try:
        save_market_data_record(data)
    except Exception as e:
        print(f"[save_decision] market_data 기록 실패: {e}", file=sys.stderr)

    # 6. execution_logs 기록
    try:
        duration_ms = int((time.time() - start_time) * 1000)
        decision_id = None
        if saved:
            if isinstance(saved, list) and len(saved) > 0:
                decision_id = saved[0].get("id")
            elif isinstance(saved, dict):
                decision_id = saved.get("id")
        save_execution_log_record(data, duration_ms=duration_ms, decision_id=decision_id)
    except Exception as e:
        print(f"[save_decision] execution_logs 기록 실패: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
