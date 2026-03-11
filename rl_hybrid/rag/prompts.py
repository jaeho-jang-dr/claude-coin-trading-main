"""Gemini API 프롬프트 템플릿 — 시장 분석용"""

MARKET_ANALYSIS_PROMPT = """당신은 비트코인 시장 전문 분석가입니다. 아래 데이터를 종합적으로 분석하여 JSON 형식으로 응답하세요.

## 시장 데이터
{market_data}

## 외부 지표
{external_data}

## 과거 유사 분석 (RAG)
{rag_context}

## 분석 요청
위 데이터를 기반으로 다음 JSON을 작성하세요:

```json
{{
  "market_regime": "accumulation | distribution | trending_up | trending_down | ranging",
  "confidence": 0.0 ~ 1.0,
  "key_signals": ["시그널1", "시그널2", ...],
  "risk_assessment": "low | medium | high | extreme",
  "recommended_action": "strong_buy | cautious_buy | hold | cautious_sell | strong_sell",
  "reasoning": "판단 근거 (2~3문장)",
  "time_horizon": "1h | 4h | 12h | 24h",
  "danger_score_adjustment": -10 ~ +10,
  "opportunity_score_adjustment": -10 ~ +10
}}
```

반드시 JSON만 출력하세요. 다른 텍스트 없이 JSON 블록만 응답하세요.
"""

EMBEDDING_TEXT_TEMPLATE = """시장분석 {timestamp}
체제: {market_regime} | 신뢰도: {confidence}
핵심시그널: {key_signals}
위험도: {risk_assessment} | 권고: {recommended_action}
근거: {reasoning}
BTC가격: {btc_price} | RSI: {rsi} | FGI: {fgi}
"""
