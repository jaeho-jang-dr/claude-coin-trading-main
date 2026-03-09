#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# 에이전트 모드 파이프라인
#
# 기존 run_analysis.sh(LLM 프롬프트 조립 방식)와 달리,
# Python 에이전트가 직접 데이터 수집 → 전략 전환 → 매매 판단을 수행한다.
#
# 사용법:
#   bash scripts/run_agents.sh              # 일반 실행
#   DRY_RUN=true bash scripts/run_agents.sh # 시뮬레이션
# ──────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# .env 로드
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Python 가상환경 활성화
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
  source .venv/Scripts/activate
fi

# 긴급 정지 확인 — 사용자 수동
if [ "${EMERGENCY_STOP:-false}" = "true" ]; then
  echo "[STOP] 사용자 EMERGENCY_STOP 활성화됨. 실행 중단." >&2
  python3 scripts/notify_telegram.py error "EMERGENCY_STOP" "사용자 긴급 정지 활성화로 에이전트 실행 중단" 2>/dev/null || true
  exit 1
fi

# 긴급 정지 확인 — 감독 자동 (data/auto_emergency.json)
AUTO_EMERGENCY_FILE="${PROJECT_DIR}/data/auto_emergency.json"
if [ -f "$AUTO_EMERGENCY_FILE" ]; then
  IS_ACTIVE=$(python3 -c "import json; d=json.load(open('$AUTO_EMERGENCY_FILE','r',encoding='utf-8')); print(d.get('active',False))" 2>/dev/null || echo "False")
  if [ "$IS_ACTIVE" = "True" ]; then
    REASON=$(python3 -c "import json; d=json.load(open('$AUTO_EMERGENCY_FILE','r',encoding='utf-8')); print(d.get('reason','알 수 없음'))" 2>/dev/null || echo "알 수 없음")
    echo "[STOP] 감독 자동 긴급정지 활성 중: $REASON" >&2
    echo "[STOP] Orchestrator가 해제 조건을 평가합니다..." >&2
    # Orchestrator 내부에서 해제 여부를 판단하므로 여기서는 차단하지 않고
    # Phase 2까지는 진행시킨다 (Orchestrator가 해제 가능 여부를 확인해야 하므로)
  fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR="data/snapshots/${TIMESTAMP}"
LOG_DIR="logs/executions"
mkdir -p "$SNAPSHOT_DIR" "$LOG_DIR"

echo "[$(date)] ═══ 에이전트 모드 시작 ═══" >&2

# ── Phase 1: 내부 시장 데이터 수집 (기존 스크립트 활용) ──
echo "[$(date)] Phase 1: 내부 데이터 수집..." >&2

# 병렬 수집
python3 scripts/collect_market_data.py > "${SNAPSHOT_DIR}/market_data.json" 2>/dev/null &
PID_MARKET=$!

python3 scripts/get_portfolio.py > "${SNAPSHOT_DIR}/portfolio.json" 2>/dev/null &
PID_PORTFOLIO=$!

python3 scripts/collect_ai_signal.py > "${SNAPSHOT_DIR}/ai_signal.json" 2>/dev/null &
PID_AI=$!

# 병렬 완료 대기 + 에러 핸들링
wait $PID_MARKET 2>/dev/null || echo '{"error":"market_data 수집 실패"}' > "${SNAPSHOT_DIR}/market_data.json"
wait $PID_PORTFOLIO 2>/dev/null || echo '{"error":"portfolio 조회 실패"}' > "${SNAPSHOT_DIR}/portfolio.json"
wait $PID_AI 2>/dev/null || echo '{"error":"ai_signal 수집 실패"}' > "${SNAPSHOT_DIR}/ai_signal.json"

echo "[$(date)] Phase 1 완료." >&2

# ── Phase 2: 에이전트 실행 (Python) ──
echo "[$(date)] Phase 2: 에이전트 파이프라인 실행..." >&2

AGENT_RESULT=$(python3 -c "
import json, sys, os
sys.path.insert(0, '.')

from agents.external_data import ExternalDataAgent
from agents.orchestrator import Orchestrator
from pathlib import Path

snapshot_dir = Path('${SNAPSHOT_DIR}')

# 1) 외부 데이터 수집 (ExternalDataAgent)
print('[Agent] 외부 데이터 수집 중...', file=sys.stderr)
ext_agent = ExternalDataAgent(snapshot_dir=snapshot_dir)
external_data = ext_agent.collect_all()
print(f'[Agent] 외부 데이터 수집 완료 ({external_data[\"collection_time_sec\"]}초, 에러: {external_data[\"errors\"]})', file=sys.stderr)

# 2) 내부 데이터 로드
market_data = json.loads(open(snapshot_dir / 'market_data.json', encoding='utf-8').read())
portfolio = json.loads(open(snapshot_dir / 'portfolio.json', encoding='utf-8').read())
ai_signal = json.loads(open(snapshot_dir / 'ai_signal.json', encoding='utf-8').read())

# market_data에 ai_composite_signal 주입
market_data['ai_composite_signal'] = ai_signal.get('composite_signal', {})

# FGI 정보를 market_data에 주입 (에이전트가 참조)
fgi_data = external_data.get('sources', {}).get('fear_greed', {})
market_data['fear_greed'] = fgi_data.get('current', {})

# 뉴스 정보를 market_data에 주입
news_data = external_data.get('sources', {}).get('news', {})
news_sentiment = external_data.get('sources', {}).get('news_sentiment', {})
market_data['news'] = news_data
market_data['news']['overall_sentiment'] = news_sentiment.get('overall_sentiment', 'neutral')
market_data['news']['sentiment_score'] = news_sentiment.get('sentiment_score', 0)

# 3) 과거 결정 조회 (Supabase)
past_decisions = []
supabase_url = os.getenv('SUPABASE_URL', '')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')
if supabase_url and supabase_key:
    try:
        import requests
        resp = requests.get(
            f'{supabase_url}/rest/v1/decisions',
            params={'select': '*', 'order': 'created_at.desc', 'limit': '10'},
            headers={
                'apikey': supabase_key,
                'Authorization': f'Bearer {supabase_key}',
            },
            timeout=10,
        )
        if resp.status_code == 200:
            past_decisions = resp.json()
    except Exception as e:
        print(f'[Agent] Supabase 조회 실패: {e}', file=sys.stderr)

# 4) BTC 비중 계산
btc_info = portfolio.get('coins', {}).get('BTC', portfolio.get('btc', {}))
total_eval = portfolio.get('total_evaluation', 1)
btc_eval = btc_info.get('evaluation', 0) if isinstance(btc_info, dict) else 0
portfolio['btc_ratio'] = btc_eval / total_eval if total_eval > 0 else 0
portfolio['btc'] = btc_info if isinstance(btc_info, dict) else {}

# 5) Orchestrator 실행
print('[Agent] Orchestrator 실행 중...', file=sys.stderr)
orchestrator = Orchestrator()
result = orchestrator.run(
    market_data=market_data,
    external_data=external_data,
    portfolio=portfolio,
    past_decisions=past_decisions,
)
print(f'[Agent] 결정: {result[\"decision\"][\"decision\"]} by {result[\"active_agent\"]}', file=sys.stderr)

# 6) 전체 결과 조립
output = {
    'timestamp': external_data['timestamp'],
    'active_agent': result['active_agent'],
    'switch': result.get('switch'),
    'decision': result['decision'],
    'external_data_summary': {
        'collection_time_sec': external_data['collection_time_sec'],
        'errors': external_data['errors'],
        'signal': external_data.get('external_signal', {}),
    },
    'snapshot_dir': str(snapshot_dir),
}

print(json.dumps(output, ensure_ascii=False, indent=2))
" 2>&2)

AGENT_EXIT=$?

if [ $AGENT_EXIT -ne 0 ]; then
  echo "[$(date)] Phase 2 실패 (exit $AGENT_EXIT)" >&2
  python3 scripts/notify_telegram.py error "Agent Pipeline" "에이전트 파이프라인 실패 (exit $AGENT_EXIT)" 2>/dev/null || true
  exit 1
fi

echo "[$(date)] Phase 2 완료." >&2

# 결과 저장
echo "$AGENT_RESULT" > "${SNAPSHOT_DIR}/agent_result.json"

# ── Phase 3: 매매 실행 ──
DECISION=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['decision'])")
REASON=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['reason'])")
AGENT_NAME=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['active_agent'])")
SWITCH_INFO=$(echo "$AGENT_RESULT" | python3 -c "
import sys,json
r=json.load(sys.stdin)
sw = r.get('switch')
if sw:
    print(f\"전략 전환: {sw['from']} → {sw['to']} ({sw['reason']})\")
else:
    print('전환 없음')
")

echo "[$(date)] Phase 3: 매매 실행 — $DECISION ($AGENT_NAME)" >&2

if [ "$DECISION" = "buy" ]; then
  SIDE=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['trade_params'].get('side','bid'))")
  MARKET=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['trade_params'].get('market','KRW-BTC'))")
  AMOUNT=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['trade_params'].get('amount',0))")
  IS_DCA=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['trade_params'].get('is_dca',False))")

  if [ "$AMOUNT" -gt 0 ] 2>/dev/null; then
    DCA_TAG=""
    [ "$IS_DCA" = "True" ] && DCA_TAG=" [DCA]"
    echo "[$(date)] 매수 실행: $MARKET $AMOUNT KRW${DCA_TAG}" >&2
    python3 scripts/execute_trade.py bid "$MARKET" "$AMOUNT" 2>&1 | tee -a "$LOG_DIR/trade_${TIMESTAMP}.log" >&2
  fi

elif [ "$DECISION" = "sell" ]; then
  MARKET=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['trade_params'].get('market','KRW-BTC'))")
  VOLUME=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision']['trade_params'].get('volume',0))")

  if python3 -c "assert float('$VOLUME') > 0" 2>/dev/null; then
    echo "[$(date)] 매도 실행: $MARKET $VOLUME BTC" >&2
    python3 scripts/execute_trade.py ask "$MARKET" "$VOLUME" 2>&1 | tee -a "$LOG_DIR/trade_${TIMESTAMP}.log" >&2
  fi

else
  echo "[$(date)] 관망 결정. 매매 없음." >&2
fi

# ── Phase 4: 텔레그램 알림 ──
echo "[$(date)] Phase 4: 텔레그램 알림..." >&2

BUY_SCORE=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['decision'].get('buy_score',{}).get('total','N/A'))")
CONFIDENCE=$(echo "$AGENT_RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(round(r['decision'].get('confidence',0)*100))")

SUMMARY="${AGENT_NAME} | ${DECISION} (${CONFIDENCE}%)"
DETAIL="근거: ${REASON}
매수점수: ${BUY_SCORE}
${SWITCH_INFO}"

python3 scripts/notify_telegram.py trade "$SUMMARY" "$DETAIL" 2>/dev/null || true

# ── Phase 5: Supabase 기록 ──
if [ -n "${SUPABASE_URL:-}" ] && [ -n "${SUPABASE_SERVICE_ROLE_KEY:-}" ]; then
  echo "[$(date)] Phase 5: Supabase 기록..." >&2
  python3 -c "
import json, os, requests, sys

result = json.loads(open('${SNAPSHOT_DIR}/agent_result.json', encoding='utf-8').read())
decision = result['decision']

row = {
    'decision': decision['decision'],
    'confidence': decision.get('confidence', 0),
    'reason': decision.get('reason', ''),
    'buy_score': decision.get('buy_score', {}).get('total', 0),
    'agent_name': result.get('active_agent', ''),
    'current_price': json.loads(open('${SNAPSHOT_DIR}/market_data.json', encoding='utf-8').read()).get('ticker', {}).get('trade_price', 0),
    'snapshot_dir': '${SNAPSHOT_DIR}',
}

try:
    resp = requests.post(
        os.environ['SUPABASE_URL'] + '/rest/v1/decisions',
        json=row,
        headers={
            'apikey': os.environ['SUPABASE_SERVICE_ROLE_KEY'],
            'Authorization': 'Bearer ' + os.environ['SUPABASE_SERVICE_ROLE_KEY'],
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal',
        },
        timeout=10,
    )
    print(f'[Agent] Supabase 기록 완료 (HTTP {resp.status_code})', file=sys.stderr)
except Exception as e:
    print(f'[Agent] Supabase 기록 실패: {e}', file=sys.stderr)
" 2>&2 || true
fi

# ── Phase 6: 과거 전환 성과 평가 (학습 데이터 축적) ──
echo "[$(date)] Phase 6: 전환 성과 평가..." >&2
python3 scripts/evaluate_switches.py 2>&2 || true

echo "[$(date)] ═══ 에이전트 모드 완료 ═══" >&2
echo "$AGENT_RESULT"
