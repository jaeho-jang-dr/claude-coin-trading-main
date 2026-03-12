#!/usr/bin/env python3
"""
거래 기록 리콜 스크립트 -- 최소 토큰으로 과거 결정 조회

사용법:
  python3 scripts/recall.py today              # 오늘 거래 요약
  python3 scripts/recall.py daily 7            # 최근 7일 일별 요약
  python3 scripts/recall.py cycle 20260310-1400-agent  # 특정 사이클 상세
  python3 scripts/recall.py last 5             # 최근 5건 거래
  python3 scripts/recall.py wins               # 수익 난 거래만
  python3 scripts/recall.py losses             # 손실 난 거래만
  python3 scripts/recall.py scalp today        # 오늘 초단타
  python3 scripts/recall.py scalp last 10      # 최근 10건 초단타
  python3 scripts/recall.py search "whale"     # 사유에 키워드 검색
  python3 scripts/recall.py stats              # 전체 통계
  python3 scripts/recall.py agent-perf         # 에이전트별 성과
  python3 scripts/recall.py retro              # 회고 리포트 (정확도/놓친기회/나쁜거래)
  python3 scripts/recall.py near-miss          # 니어미스 + AI거부 분석
  python3 scripts/recall.py filters            # 필터 효과 분석
  python3 scripts/recall.py health             # 시스템 건강도 (24h)
  python3 scripts/recall.py veto               # AI 거부권 분석
  python3 scripts/recall.py tag whale          # 태그로 전체 테이블 검색
  python3 scripts/recall.py tag loss,extreme_fear  # 복수 태그 (AND)
"""

import json, os, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
KST = timezone(timedelta(hours=9))


def headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }


def query_raw(table_or_view: str, params: dict = None) -> list:
    """Supabase REST API query -- 파라미터 그대로 전달"""
    url = f"{SUPABASE_URL}/rest/v1/{table_or_view}"
    r = requests.get(url, headers=headers(), params=params or {}, timeout=15)
    if r.ok:
        return r.json()
    print(f"[error] {r.status_code}: {r.text[:200]}", file=sys.stderr)
    return []


def query(table_or_view: str, params: dict = None, limit: int = 50) -> list:
    """Supabase REST API query (PostgREST format)"""
    url = f"{SUPABASE_URL}/rest/v1/{table_or_view}"
    p = params or {}
    # Set default order based on table/view
    if "order" not in p:
        if table_or_view == "v_daily_summary":
            p["order"] = "trade_date.desc"
        elif table_or_view == "v_scalp_recall":
            p["order"] = "entry_time.desc"
        elif table_or_view == "scalp_trade_log":
            p["order"] = "entry_time.desc"
        else:
            p["order"] = "created_at.desc"
    if "limit" not in p:
        p["limit"] = str(limit)
    r = requests.get(url, headers=headers(), params=p, timeout=15)
    if r.ok:
        return r.json()
    print(f"[error] {r.status_code}: {r.text[:200]}", file=sys.stderr)
    return []


def fmt_krw(v):
    """숫자를 KRW 포맷으로"""
    if v is None:
        return "-"
    return f"{int(v):,}"


def fmt_pct(v):
    if v is None:
        return "-"
    return f"{float(v):+.2f}%"


def cmd_today():
    """오늘 거래 요약"""
    today = datetime.now(KST).strftime("%Y-%m-%d")

    # 단타 (decisions via view)
    rows = query("v_trade_recall", {
        "created_at": f"gte.{today}T00:00:00+09:00",
        "order": "created_at.asc",
    })

    print(f"=== {today} 거래 기록 ===")
    if not rows:
        print("단타 거래 없음")
    else:
        for r in rows:
            t = r.get("created_at", "")[:16]
            action = r.get("action", "?")
            price = fmt_krw(r.get("price"))
            score = r.get("buy_total", "-")
            agent = r.get("agent_type", "?")
            pnl = fmt_krw(r.get("profit_loss"))
            fgi = r.get("fgi_value", "?")
            rsi = r.get("rsi_14", "?")
            reason = (r.get("reason") or "")[:60]
            print(f"  {t} | {action} | {price}원 | 점수:{score} | {agent} | FGI:{fgi} RSI:{rsi} | PnL:{pnl} | {reason}")

    # 초단타 (scalp)
    scalps = query("v_scalp_recall", {
        "entry_time": f"gte.{today}T00:00:00+09:00",
        "order": "entry_time.asc",
    })
    if scalps:
        print(f"\n--- 초단타 {len(scalps)}건 ---")
        total_pnl = 0
        for s in scalps:
            t = (s.get("entry_time") or "")[:16]
            strategy = s.get("strategy", "?")
            pnl_krw = s.get("pnl_krw", 0) or 0
            pnl_pct = fmt_pct(s.get("pnl_pct"))
            reason = (s.get("exit_reason") or "")[:30]
            lesson = (s.get("lesson") or "")[:40]
            total_pnl += pnl_krw
            print(f"  {t} | {strategy} | {fmt_krw(pnl_krw)}원 ({pnl_pct}) | {reason} | {lesson}")
        print(f"  합계: {fmt_krw(total_pnl)}원")


def cmd_daily(days=7):
    """최근 N일 일별 요약"""
    rows = query("v_daily_summary", {"limit": str(days)})
    print(f"=== 최근 {days}일 요약 ===")
    print(f"  {'날짜':<12} {'매수':>4} {'매도':>4} {'관망':>4} {'PnL':>10} {'초단타':>4} {'스PnL':>10} {'FGI':>4} {'RSI':>5} {'점수':>5}")
    for r in rows:
        d = str(r.get("trade_date") or "?")
        buys = r.get("buys") or 0
        sells = r.get("sells") or 0
        holds = r.get("holds") or 0
        total_pnl = fmt_krw(r.get("total_pnl"))
        scalp_trades = r.get("scalp_trades") or 0
        scalp_pnl = fmt_krw(r.get("scalp_pnl"))
        avg_fgi = str(r.get("avg_fgi") or "?")
        avg_rsi = str(r.get("avg_rsi") or "?")
        avg_score = str(r.get("avg_buy_score") or "?")
        print(
            f"  {d:<12}"
            f" {buys:>4}"
            f" {sells:>4}"
            f" {holds:>4}"
            f" {total_pnl:>10}"
            f" {scalp_trades:>4}"
            f" {scalp_pnl:>10}"
            f" {avg_fgi:>4}"
            f" {avg_rsi:>5}"
            f" {avg_score:>5}"
        )


def cmd_last(n=5):
    """최근 N건 거래"""
    rows = query("v_trade_recall", {"limit": str(n)})
    for r in rows:
        print(f"\n--- {r.get('created_at', '')[:19]} [{r.get('source', '?')}] ---")
        print(f"  결정: {r.get('action')} | 가격: {fmt_krw(r.get('price'))} | 금액: {fmt_krw(r.get('trade_amount'))}")
        print(f"  점수: {r.get('buy_total', '-')} (FGI:{r.get('buy_fgi', '-')} RSI:{r.get('buy_rsi', '-')} SMA:{r.get('buy_sma', '-')} 뉴스:{r.get('buy_news', '-')} 외부:{r.get('buy_ext', '-')})")
        print(f"  에이전트: {r.get('agent_type', '?')} | 위험:{r.get('danger_score', '?')} 기회:{r.get('opportunity_score', '?')}")
        print(f"  시장: FGI={r.get('fgi_value', '?')} RSI={r.get('rsi_14', '?')} 펀딩={r.get('funding_rate', '?')} L/S={r.get('long_short_ratio', '?')} 김프={r.get('kimchi_premium', '?')}%")
        print(f"  외부: fusion={r.get('ext_fusion', '?')}({r.get('ext_signal', '?')}) 바이낸스={r.get('ext_binance', '?')} 고래={r.get('ext_whale', '?')} 매크로={r.get('ext_macro', '?')}")
        print(f"  사유: {(r.get('reason') or '')[:120]}")
        if r.get("profit_loss"):
            print(f"  손익: {fmt_krw(r.get('profit_loss'))}원")


def cmd_cycle(cycle_id):
    """특정 사이클 상세"""
    rows = query("v_cycle_detail", {"cycle_id": f"eq.{cycle_id}"})
    if not rows:
        print(f"cycle_id '{cycle_id}' 없음")
        return
    for r in rows:
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str))


def cmd_wins():
    """수익 거래만"""
    rows = query("decisions", {
        "profit_loss": "gt.0",
        "order": "profit_loss.desc",
        "limit": "20",
        "select": "created_at,decision,current_price,trade_amount,profit_loss,reason,cycle_id",
    })
    print(f"=== 수익 거래 TOP {len(rows)} ===")
    for r in rows:
        print(f"  {r.get('created_at', '')[:16]} | {r.get('decision')} | {fmt_krw(r.get('profit_loss'))}원 | {(r.get('reason') or '')[:60]}")


def cmd_losses():
    """손실 거래만"""
    rows = query("decisions", {
        "profit_loss": "lt.0",
        "order": "profit_loss.asc",
        "limit": "20",
        "select": "created_at,decision,current_price,trade_amount,profit_loss,reason,cycle_id",
    })
    print(f"=== 손실 거래 TOP {len(rows)} ===")
    for r in rows:
        print(f"  {r.get('created_at', '')[:16]} | {r.get('decision')} | {fmt_krw(r.get('profit_loss'))}원 | {(r.get('reason') or '')[:60]}")


def cmd_scalp(sub="today", n=10):
    """초단타 조회"""
    if sub == "today":
        today = datetime.now(KST).strftime("%Y-%m-%d")
        rows = query("v_scalp_recall", {
            "session_date": f"eq.{today}",
            "order": "entry_time.asc",
        })
    else:
        rows = query("v_scalp_recall", {"limit": str(n)})

    print(f"=== 초단타 {len(rows)}건 ===")
    for s in rows:
        t = (s.get("entry_time") or "")[:16]
        print(
            f"  {t}"
            f" | {s.get('strategy', '?'):>5}"
            f" | {fmt_krw(s.get('amount_krw'))}원"
            f" | {fmt_pct(s.get('pnl_pct'))} ({fmt_krw(s.get('pnl_krw'))}원)"
            f" | {s.get('exit_reason', ''):>10}"
            f" | trend:{s.get('market_trend', '?')} FGI:{s.get('fgi_value', '?')}"
        )
        if s.get("lesson"):
            print(f"         교훈: {s['lesson'][:60]}")


def cmd_tag(tag_query):
    """태그로 전체 테이블 통합 검색 (v_tag_index 뷰 사용)"""
    tags = [t.strip() for t in tag_query.split(",")]

    # PostgreSQL array contains operator: tags @> ARRAY['whale','loss']
    # PostgREST: use cs (contains) operator for arrays
    params = {
        "tags": f"cs.{{{','.join(tags)}}}",
        "order": "ts.desc",
        "limit": "30",
    }
    rows = query_raw("v_tag_index", params)
    print(f"=== 태그 [{', '.join(tags)}] 검색: {len(rows)}건 ===")
    for r in rows:
        ts = (r.get("ts") or "")[:16]
        tbl = r.get("table_name", "?")
        src = r.get("source", "?")
        row_tags = r.get("tags", [])
        summary = (r.get("summary") or "")[:70]
        print(f"  {ts} | {tbl:<8} | {src:<6} | {row_tags}")
        if summary:
            print(f"           {summary}")


def cmd_search(keyword):
    """사유에 키워드 검색"""
    rows = query("decisions", {
        "reason": f"ilike.*{keyword}*",
        "limit": "20",
        "select": "created_at,decision,current_price,profit_loss,reason,cycle_id",
    })
    print(f"=== '{keyword}' 검색 결과 {len(rows)}건 ===")
    for r in rows:
        print(f"  {r.get('created_at', '')[:16]} | {r.get('decision')} | PnL:{fmt_krw(r.get('profit_loss'))} | {(r.get('reason') or '')[:80]}")


def cmd_stats():
    """전체 통계"""
    all_d = query("decisions", {"select": "id", "limit": "10000"})
    buys = query("decisions", {"decision": "eq.매수", "select": "profit_loss", "limit": "10000"})
    sells = query("decisions", {"decision": "eq.매도", "select": "profit_loss", "limit": "10000"})
    scalps = query("scalp_trade_log", {"select": "pnl_krw,strategy", "limit": "10000"})

    buy_pnl = sum(float(b.get("profit_loss") or 0) for b in buys)
    sell_pnl = sum(float(s.get("profit_loss") or 0) for s in sells)
    scalp_pnl = sum(int(s.get("pnl_krw") or 0) for s in scalps)

    print("=== 전체 통계 ===")
    print(f"  총 결정: {len(all_d)}건 (매수 {len(buys)}, 매도 {len(sells)})")
    print(f"  단타 PnL: {fmt_krw(buy_pnl + sell_pnl)}원")
    print(f"  초단타: {len(scalps)}건, PnL: {fmt_krw(scalp_pnl)}원")

    # Strategy breakdown for scalps
    by_strategy = {}
    for s in scalps:
        st = s.get("strategy", "?")
        by_strategy.setdefault(st, {"count": 0, "pnl": 0})
        by_strategy[st]["count"] += 1
        by_strategy[st]["pnl"] += int(s.get("pnl_krw") or 0)
    for st, v in sorted(by_strategy.items()):
        print(f"    {st}: {v['count']}건, {fmt_krw(v['pnl'])}원")


def cmd_agent_perf():
    """에이전트별 성과"""
    rows = query("buy_score_detail", {"select": "agent_type,action,total_score", "limit": "10000"})
    perf = {}
    for r in rows:
        agent = r.get("agent_type", "?")
        perf.setdefault(agent, {"count": 0, "buys": 0, "avg_score": []})
        perf[agent]["count"] += 1
        if r.get("action") in ("매수", "buy"):
            perf[agent]["buys"] += 1
        if r.get("total_score"):
            perf[agent]["avg_score"].append(float(r["total_score"]))

    print("=== 에이전트 성과 ===")
    for agent, v in sorted(perf.items()):
        avg = sum(v["avg_score"]) / len(v["avg_score"]) if v["avg_score"] else 0
        print(f"  {agent}: {v['count']}건 (매수 {v['buys']}) | 평균점수: {avg:.1f}")


def cmd_retrospective():
    """회고 리포트: 결정 정확도 + 놓친 기회 + 나쁜 거래"""
    # Query v_decision_accuracy
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_decision_accuracy", headers=headers(), timeout=15)
    if r.ok and r.json():
        print("=== 결정 정확도 ===")
        for row in r.json():
            print(f"  {row['decision']:<6} | 1h: {row.get('accuracy_1h','?')}% (avg {row.get('avg_1h_pct','?')}%) | 4h: {row.get('accuracy_4h','?')}% | 24h: {row.get('accuracy_24h','?')}%")
    else:
        print("=== 결정 정확도 === (데이터 없음 또는 뷰 미생성)")

    # Missed opportunities
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_missed_opportunities", headers=headers(), params={"limit": "5"}, timeout=15)
    if r.ok and r.json():
        print(f"\n=== 놓친 기회 TOP {len(r.json())} ===")
        for row in r.json():
            print(f"  {row['created_at'][:16]} | +{row.get('outcome_24h_pct',0):.1f}% 상승 | FGI:{row.get('fear_greed_value','?')}")
    else:
        print("\n=== 놓친 기회 === (데이터 없음)")

    # Bad trades
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_bad_trades", headers=headers(), params={"limit": "5"}, timeout=15)
    if r.ok and r.json():
        print(f"\n=== 나쁜 매수 TOP {len(r.json())} ===")
        for row in r.json():
            print(f"  {row['created_at'][:16]} | {row.get('outcome_4h_pct',0):+.1f}% | 금액: {row.get('trade_amount',0):,}")
    else:
        print("\n=== 나쁜 매수 === (데이터 없음)")


def cmd_near_miss():
    """니어미스 분석"""
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_near_miss_analysis", headers=headers(), params={"limit": "20"}, timeout=15)
    if r.ok and r.json():
        print(f"=== 니어미스 + AI거부 {len(r.json())}건 ===")
        for row in r.json():
            print(f"  {row['recorded_at'][:16]} | {row.get('agent_type','?')} | 점수:{row.get('total_score','?')}/{row.get('threshold','?')} ({row.get('points_from_threshold','')}pt) | {row.get('evaluation','?')}")
            if row.get('was_ai_vetoed'):
                print(f"    AI거부: {row.get('ai_veto_reason','?')} | 4h후: {row.get('outcome_4h_pct','?')}%")
    else:
        print("=== 니어미스 === (데이터 없음 또는 뷰 미생성)")


def cmd_filters():
    """필터 효과 분석"""
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_filter_effectiveness", headers=headers(), timeout=15)
    if r.ok and r.json():
        print("=== 필터 효과 ===")
        for row in r.json():
            print(f"  {row.get('block_filter','?')}: {row.get('total_blocked',0)}건 차단 | 손실방지: {row.get('filter_save_rate','?')}% | 평균결과: {row.get('avg_outcome_if_traded','?')}%")
    else:
        print("=== 필터 효과 === (데이터 없음 또는 뷰 미생성)")


def cmd_health():
    """시스템 건강도"""
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_system_health", headers=headers(), timeout=15)
    if r.ok and r.json():
        print("=== 시스템 건강도 (24h) ===")
        for row in r.json():
            print(f"  {row.get('execution_mode','?')}: {row.get('total_runs',0)}회 | 성공률: {row.get('success_rate','?')}% | 평균: {row.get('avg_duration_ms','?')}ms | 최근: {(row.get('last_run') or '')[:16]}")
    else:
        print("=== 시스템 건강도 === (최근 24시간 데이터 없음)")


def cmd_veto():
    """AI 거부권 분석"""
    r = requests.get(f"{SUPABASE_URL}/rest/v1/v_ai_veto_effectiveness", headers=headers(), timeout=15)
    if r.ok and r.json():
        print("=== AI 거부권 효과 ===")
        for row in r.json():
            print(f"  {row.get('ai_veto_reason','?')}: {row.get('total_vetoes',0)}건 | 정확도: {row.get('veto_accuracy_pct','?')}% | 매수했다면: {row.get('avg_outcome_if_bought','?')}%")
    else:
        print("=== AI 거부권 === (데이터 없음 또는 뷰 미생성)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "today":
        cmd_today()
    elif cmd == "daily":
        cmd_daily(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    elif cmd == "last":
        cmd_last(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif cmd == "cycle":
        if len(sys.argv) < 3:
            print("cycle_id를 지정하세요")
            return
        cmd_cycle(sys.argv[2])
    elif cmd == "wins":
        cmd_wins()
    elif cmd == "losses":
        cmd_losses()
    elif cmd == "scalp":
        sub = sys.argv[2] if len(sys.argv) > 2 else "today"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        cmd_scalp(sub, n)
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("검색어를 지정하세요")
            return
        cmd_search(sys.argv[2])
    elif cmd == "stats":
        cmd_stats()
    elif cmd == "agent-perf":
        cmd_agent_perf()
    elif cmd == "retro":
        cmd_retrospective()
    elif cmd == "near-miss":
        cmd_near_miss()
    elif cmd == "filters":
        cmd_filters()
    elif cmd == "health":
        cmd_health()
    elif cmd == "veto":
        cmd_veto()
    elif cmd == "tag":
        if len(sys.argv) < 3:
            print("태그를 지정하세요 (예: whale, loss,extreme_fear)")
            return
        cmd_tag(sys.argv[2])
    else:
        print(f"알 수 없는 명령: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
