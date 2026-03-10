#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 전체 데이터 수집 + 프롬프트 생성 파이프라인
# cron에서 claude -p에 전달할 프롬프트를 stdout으로 출력한다.
#
# 사용법 (cron — LLM 프롬프트 모드):
#   bash scripts/run_analysis.sh 2>/dev/null | claude -p --dangerously-skip-permissions
#
# 사용법 (에이전트 모드 — Python 에이전트가 직접 판단):
#   bash scripts/run_analysis.sh --agent
#
# 사용법 (수동 분석):
#   bash scripts/run_analysis.sh
# ──────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# --agent 모드: 에이전트 파이프라인으로 위임
if [ "${1:-}" = "--agent" ]; then
  exec bash scripts/run_agents.sh
fi

# .env 로드
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Python 가상환경 활성화
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

# 긴급 정지 확인
if [ "${EMERGENCY_STOP:-false}" = "true" ]; then
  echo "EMERGENCY_STOP 활성화됨. 실행 중단." >&2
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR="data/snapshots/${TIMESTAMP}"
mkdir -p "$SNAPSHOT_DIR" "logs/executions"

echo "[$(date)] 데이터 수집 시작..." >&2

# 1. 시장 데이터 수집
python3 scripts/collect_market_data.py > "${SNAPSHOT_DIR}/market_data.json" 2>/dev/null \
  || echo '{"error":"market_data 수집 실패"}' > "${SNAPSHOT_DIR}/market_data.json"

# 2. Fear & Greed Index 수집
python3 scripts/collect_fear_greed.py > "${SNAPSHOT_DIR}/fear_greed.json" 2>/dev/null \
  || echo '{"error":"fear_greed 수집 실패"}' > "${SNAPSHOT_DIR}/fear_greed.json"

# 3. 뉴스 수집 + 압축 (토큰 절감: 10-15KB → 2-4KB)
python3 scripts/collect_news.py > "${SNAPSHOT_DIR}/news_full.json" 2>/dev/null \
  || echo '{"error":"news 수집 실패"}' > "${SNAPSHOT_DIR}/news_full.json"
python3 scripts/summarize_news.py "${SNAPSHOT_DIR}/news_full.json" > "${SNAPSHOT_DIR}/news.json" 2>/dev/null \
  || cp "${SNAPSHOT_DIR}/news_full.json" "${SNAPSHOT_DIR}/news.json"

# 4. 차트 캡처
python3 scripts/capture_chart.py > "${SNAPSHOT_DIR}/chart_paths.json" 2>/dev/null \
  || echo '{"error":"chart 캡처 실패"}' > "${SNAPSHOT_DIR}/chart_paths.json"

# 5. 포트폴리오 조회
python3 scripts/get_portfolio.py > "${SNAPSHOT_DIR}/portfolio.json" 2>/dev/null \
  || echo '{"error":"portfolio 조회 실패"}' > "${SNAPSHOT_DIR}/portfolio.json"

# 6. AI 복합 시그널 수집
python3 scripts/collect_ai_signal.py > "${SNAPSHOT_DIR}/ai_signal.json" 2>/dev/null \
  || echo '{"error":"ai_signal 수집 실패"}' > "${SNAPSHOT_DIR}/ai_signal.json"

# 7. 온체인 데이터 수집 (Binance 선물 + mempool)
python3 scripts/collect_onchain_data.py > "${SNAPSHOT_DIR}/onchain.json" 2>/dev/null \
  || echo '{"error":"onchain 수집 실패"}' > "${SNAPSHOT_DIR}/onchain.json"

# 8. 고래 추적 (mempool.space — 블록체인 대규모 이동, 무료)
python3 scripts/whale_tracker.py > "${SNAPSHOT_DIR}/whale_tracker.json" 2>/dev/null \
  || echo '{"error":"whale_tracker 수집 실패"}' > "${SNAPSHOT_DIR}/whale_tracker.json"

# 9. 바이낸스 심리 지표 + 김치 프리미엄 (무료, 키 불필요)
python3 scripts/binance_sentiment.py > "${SNAPSHOT_DIR}/binance_sentiment.json" 2>/dev/null \
  || echo '{"error":"binance_sentiment 수집 실패"}' > "${SNAPSHOT_DIR}/binance_sentiment.json"

# 11. CoinGecko 거래량 이상 감지 (crypto-signals, 무료)
python3 scripts/collect_crypto_signals.py > "${SNAPSHOT_DIR}/crypto_signals.json" 2>/dev/null \
  || echo '{"error":"crypto_signals 수집 실패"}' > "${SNAPSHOT_DIR}/crypto_signals.json"

# 12. CoinMarketCap (MCMP 대체 - 글로벌/매크로 지표)
python3 scripts/collect_coinmarketcap.py > "${SNAPSHOT_DIR}/coinmarketcap.json" 2>/dev/null \
  || echo '{"error":"coinmarketcap 수집 실패"}' > "${SNAPSHOT_DIR}/coinmarketcap.json"

echo "[$(date)] 데이터 수집 완료. 외부 시그널 종합 중..." >&2

# 10. 외부 지표 종합 점수 산출 (Data Fusion)
python3 scripts/calculate_external_signal.py "${SNAPSHOT_DIR}" > "${SNAPSHOT_DIR}/external_signal.json" 2>/dev/null \
  || echo '{"error":"external_signal 산출 실패"}' > "${SNAPSHOT_DIR}/external_signal.json"

echo "[$(date)] 프롬프트 생성 중..." >&2

# 데이터 로드
STRATEGY=$(cat strategy.md)
MARKET_DATA=$(cat "${SNAPSHOT_DIR}/market_data.json")
FEAR_GREED=$(cat "${SNAPSHOT_DIR}/fear_greed.json")
NEWS=$(cat "${SNAPSHOT_DIR}/news.json")
PORTFOLIO=$(cat "${SNAPSHOT_DIR}/portfolio.json")
AI_SIGNAL=$(cat "${SNAPSHOT_DIR}/ai_signal.json")
ONCHAIN=$(cat "${SNAPSHOT_DIR}/onchain.json")
WHALE_TRACKER=$(cat "${SNAPSHOT_DIR}/whale_tracker.json")
BINANCE_SENTIMENT=$(cat "${SNAPSHOT_DIR}/binance_sentiment.json")
EXTERNAL_SIGNAL=$(cat "${SNAPSHOT_DIR}/external_signal.json")
CRYPTO_SIGNALS=$(cat "${SNAPSHOT_DIR}/crypto_signals.json")
COINMARKETCAP=$(cat "${SNAPSHOT_DIR}/coinmarketcap.json")

# RAG: 현재 시장과 유사한 과거 경험 조회 (벡터 유사도 Top 3)
PAST_DECISIONS="[]"
RAG_OUTPUT=$(python3 scripts/recall_rag.py --json --top 3 2>/dev/null) || true
if [ -n "$RAG_OUTPUT" ] && [ "$RAG_OUTPUT" != "[]" ]; then
  PAST_DECISIONS="$RAG_OUTPUT"
  echo "  RAG: 유사 과거 경험 조회 완료" >&2
else
  # RAG 실패 시 fallback (최근 5건, 필수 컬럼만)
  if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
    PAST_DECISIONS=$(curl -s \
      "${SUPABASE_URL}/rest/v1/decisions?select=id,decision,reason,confidence,current_price,profit_loss,created_at&order=created_at.desc&limit=5" \
      -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
      -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
      2>/dev/null || echo "[]")
    echo "  RAG fallback: 최근 결정 조회" >&2
  fi
fi

# 미반영 피드백 조회
FEEDBACK="[]"
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  FEEDBACK=$(curl -s \
    "${SUPABASE_URL}/rest/v1/feedback?select=*&applied=eq.false&order=created_at.desc" \
    -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
    2>/dev/null || echo "[]")
fi

# 초단타(스캘프) 최근 성과 조회 — 크로스 학습용
SCALP_PERFORMANCE="[]"
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  SCALP_PERFORMANCE=$(curl -s \
    "${SUPABASE_URL}/rest/v1/scalp_trades?select=strategy,side,amount,pnl,price,created_at&order=created_at.desc&limit=20" \
    -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
    2>/dev/null || echo "[]")
fi

# 고래 감지 최근 동향 — 크로스 학습용
WHALE_ACTIVITY="[]"
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  WHALE_ACTIVITY=$(curl -s \
    "${SUPABASE_URL}/rest/v1/whale_detections?select=side,amount,price,ratio,created_at&order=created_at.desc&limit=10" \
    -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
    2>/dev/null || echo "[]")
fi

# 로컬 피드백 편향치 (사용자 개입)
USER_BIAS_STATE="{}"
if [ -f data/orchestrator_state.json ]; then
  USER_BIAS_STATE=$(cat data/orchestrator_state.json 2>/dev/null || echo "{}")
fi

# ETH/BTC 비율 및 도미넌스 데이터 수집
ETH_DATA=$(python3 -c "
import requests, json, statistics
try:
    btc = requests.get('https://api.upbit.com/v1/ticker', params={'markets': 'KRW-BTC'}, timeout=5).json()[0]
    eth = requests.get('https://api.upbit.com/v1/ticker', params={'markets': 'KRW-ETH'}, timeout=5).json()[0]
    btc_d = requests.get('https://api.upbit.com/v1/candles/days', params={'market': 'KRW-BTC', 'count': '60'}, timeout=5).json()
    eth_d = requests.get('https://api.upbit.com/v1/candles/days', params={'market': 'KRW-ETH', 'count': '60'}, timeout=5).json()
    btc_p = [c['trade_price'] for c in reversed(btc_d)]
    eth_p = [c['trade_price'] for c in reversed(eth_d)]
    ratios = [e/b for b, e in zip(btc_p, eth_p)]
    mean_r = statistics.mean(ratios)
    std_r = statistics.stdev(ratios)
    z = (ratios[-1] - mean_r) / std_r
    print(json.dumps({
        'eth_price': eth['trade_price'],
        'eth_change_24h': round(eth['signed_change_rate']*100, 2),
        'eth_btc_ratio': round(ratios[-1], 6),
        'eth_btc_ratio_avg60': round(mean_r, 6),
        'eth_btc_z_score': round(z, 2),
        'btc_volume_24h': round(btc['acc_trade_price_24h']/1e8),
        'eth_volume_24h': round(eth['acc_trade_price_24h']/1e8),
    }))
except: print('{}')
" 2>/dev/null || echo '{}')

# 이전 결정 성과 평가 데이터 조회
PERFORMANCE_REVIEW="[]"
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  PERFORMANCE_REVIEW=$(curl -s \
    "${SUPABASE_URL}/rest/v1/decisions?select=decision,confidence,current_price,profit_loss,reason,created_at&profit_loss=not.is.null&order=created_at.desc&limit=10" \
    -H "apikey: ${SUPABASE_SERVICE_ROLE_KEY}" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
    2>/dev/null || echo "[]")
fi

echo "[$(date)] 프롬프트 생성 완료 (ETH/성과 포함)" >&2

# 프롬프트를 stdout으로 출력
cat <<PROMPT_EOF
당신은 암호화폐 자동매매 AI 트레이더입니다.
아래 데이터를 종합 분석하고, 전략에 따라 매매 결정을 내려주세요.

═══════════════════════════════════════════
[매매 전략]
═══════════════════════════════════════════
${STRATEGY}

═══════════════════════════════════════════
[시장 데이터 - OHLCV, 기술지표]
═══════════════════════════════════════════
${MARKET_DATA}

═══════════════════════════════════════════
[ETH/BTC 비율 및 시장 구조]
═══════════════════════════════════════════
${ETH_DATA}

═══════════════════════════════════════════
[공포탐욕지수]
═══════════════════════════════════════════
${FEAR_GREED}

═══════════════════════════════════════════
[최신 뉴스 (24시간)]
═══════════════════════════════════════════
${NEWS}

═══════════════════════════════════════════
[현재 포트폴리오]
═══════════════════════════════════════════
${PORTFOLIO}

═══════════════════════════════════════════
[AI 복합 시그널]
═══════════════════════════════════════════
${AI_SIGNAL}

═══════════════════════════════════════════
[온체인 데이터 (선물 시장 + 네트워크)]
═══════════════════════════════════════════
${ONCHAIN}

펀딩레이트/롱숏비율은 역발상 지표로 활용:
- 롱 과열(펀딩레이트 높음) → 하락 조정 가능성 (매수 주의)
- 숏 과열(펀딩레이트 음수) → 숏스퀴즈 가능성 (매수 기회)
- OI 급증 + 가격 하락 = 숏 포지션 증가 → 반등 가능

═══════════════════════════════════════════
[고래 추적 — 블록체인 대규모 BTC 이동]
═══════════════════════════════════════════
${WHALE_TRACKER}
주의: 고래 데이터는 단독 매매 신호로 사용하지 마세요.
거래소 순유입(bearish) + 다른 약세 지표 겹침 시에만 매도 가중치를 부여하세요.
거래소 순유출(bullish) + 다른 강세 지표 겹침 시에만 매수 가중치를 부여하세요.

═══════════════════════════════════════════
[바이낸스 파생상품 심리 + 김치 프리미엄]
═══════════════════════════════════════════
${BINANCE_SENTIMENT}
해석 가이드:
- 롱/숏 비율 1.5+ & 펀딩비 양수 = 롱 과밀 → 조정 경고
- 롱/숏 비율 0.7- & 펀딩비 음수 = 숏 과밀 → 숏 스퀴즈(반등) 가능
- 김치 프리미엄 5%+ = 국내 FOMO → 과열 경고
- 김치 프리미엄 -3%  = 디스카운트 → 매수 기회 가능

═══════════════════════════════════════════
[CoinGecko 거래량 이상 감지 (Crypto Signals)]
═══════════════════════════════════════════
${CRYPTO_SIGNALS}
해석:
- BTC vol_mcap_ratio > 5% = 비정상적 거래량 급증 → 큰 움직임 예고
- anomaly_alerts에 주요 토큰 포함 시 시장 전체 변동성 증가 시그널
- ETH anomaly_level이 HIGH/CRITICAL이면 알트 과열 경고

═══════════════════════════════════════════
[CoinMarketCap (매크로 시장 환경)]
═══════════════════════════════════════════
${COINMARKETCAP}
해석:
- btc_dominance 상승 + 시장 전체 시총 상승 = 안정적 비트코인 상승 (매수 긍정)
- btc_dominance 하락 + ETH 상승 = 알트시즌 초입 가능성
- 총 시가총액(total_market_cap) 감소 = 자본 이탈(리스크 오프)

═══════════════════════════════════════════
[★ 외부 지표 종합 시그널 (Data Fusion)]
═══════════════════════════════════════════
${EXTERNAL_SIGNAL}
★ strategy_bonus 값을 매수 점수에 직접 가산하세요.
★ fusion.signal이 strong_buy/strong_sell이면 해당 방향 가중치를 강화하세요.
★ fusion.signal이 mixed이면 관망 우선 고려하세요.

═══════════════════════════════════════════
[★ 가장 유사했던 과거 경험 (RAG 벡터 검색)]
═══════════════════════════════════════════
${PAST_DECISIONS}

═══════════════════════════════════════════
[이전 결정 사후 성과 평가]
═══════════════════════════════════════════
${PERFORMANCE_REVIEW}

위 성과 데이터에서 profit_loss 값의 의미:
- 관망 결정: 양수 = 안 사서 다행(가격 하락), 음수 = 기회 놓침(가격 상승)
- 매수 결정: 양수 = 수익, 음수 = 손실
- 매도 결정: 양수 = 잘 팔았음(가격 하락), 음수 = 너무 일찍 팔았음(가격 상승)
이전 결정의 정확도를 분석하고, 같은 실수를 반복하지 마세요.

═══════════════════════════════════════════
[초단타(스캘프) 최근 거래 성과]
═══════════════════════════════════════════
${SCALP_PERFORMANCE}

위 데이터는 실시간 초단타 봇의 최근 거래 기록입니다.
- 고래 추종, 급변동 리바운드, 뉴스 반응 3가지 전략을 사용합니다.
- 초단타 성과가 좋은 전략 방향(매수/매도)은 단타 판단의 보조 참고 자료로 활용하세요.
- 초단타가 연속 손실 중이면 시장 변동성이 높아 단타도 보수적으로 접근하세요.

═══════════════════════════════════════════
[고래 거래 최근 동향]
═══════════════════════════════════════════
${WHALE_ACTIVITY}

고래(3000만원+ 거래)의 최근 방향성을 파악하세요.
매수/매도 비율이 편중되면 시장 방향성 판단에 참고하세요.

═══════════════════════════════════════════
[사용자 피드백 (미반영)]
═══════════════════════════════════════════
${FEEDBACK}

═══════════════════════════════════════════
[사용자 수동 피드백 상태 (Bias)]
═══════════════════════════════════════════
${USER_BIAS_STATE}

═══════════════════════════════════════════
[현재 시각]
═══════════════════════════════════════════
$(date '+%Y-%m-%d %H:%M:%S KST')

═══════════════════════════════════════════
[지시사항]
═══════════════════════════════════════════

1. 위 모든 데이터를 종합하여 시장 상황을 분석하세요.

2. **매수 조건은 점수제입니다.** 전략 문서의 점수 기준표에 따라 각 조건의 점수를 계산하고,
   합산 점수가 임계점 이상이면 매수하세요. JSON에 buy_score 필드를 포함하세요.

3. 사용자 피드백이 있다면 반드시 반영하세요.

4. **전략 전환 판단**: 전략 문서의 "전략 전환 가이드라인"을 기준으로
   현재 전략이 적절한지 평가하세요. 전환이 필요하면 strategy_switch_recommendation에
   구체적인 전환 방향과 근거를 제시하세요 (실제 전환은 사용자 승인 후).

5. **이전 결정 성과 리뷰**: profit_loss 데이터를 분석하여 최근 판단의 정확도를 평가하고,
   이번 결정에 반영하세요.

6. **ETH/BTC 비율 모니터링**: ETH/BTC z-score가 -2 이하 또는 +2 이상이면
   시장 구조 변화 신호로 보고 risk_alerts에 포함하세요.

7. **외부 지표 Data Fusion**: 고래/파생상품 데이터는 단독 매매 신호가 아닙니다.
   반드시 2개 이상의 지표가 겹칠 때만 가중치를 부여하세요:
   - 고래 활동 활발 + 롱 과밀 + FGI 탐욕 → 조정 경고, 매도 가중치 강화
   - 숏 과밀 + 음수 펀딩비 + RSI 과매도 + FGI 공포 → 숏 스퀴즈, 매수 가중치 강화
   - 김치 프리미엄 5%+ + 다른 과열 지표 → 고점 경고
   JSON에 external_signal_analysis 필드를 포함하세요.

8. 결정을 내린 후, 아래 순서대로 실행하세요:

   a) 결정이 매수 또는 매도인 경우:
      python3 scripts/execute_trade.py [bid|ask] KRW-BTC [금액|수량]

   b) 텔레그램 알림 전송:
      python3 scripts/notify_telegram.py trade "[결정 요약]" "[상세 근거]"

9. 최종 결과를 JSON 형식으로 출력하세요. 반드시 다음 필드를 포함:
   - decision, confidence, reason, buy_score (점수 내역)
   - external_signal_analysis (고래+파생상품+김치프리미엄 Data Fusion 판단)
   - strategy_switch_recommendation (전환 제안 또는 "없음")
   - performance_review (이전 결정 정확도 요약)
   - eth_btc_signal (ETH/BTC 비율 분석)
PROMPT_EOF
