"""주간 성과 리포트 + 모델 성능 모니터링

매시간 retrospective 실행 후, 주간 단위로 종합 리포트를 텔레그램으로 발송한다.
모델 성능 저하 감지 시 즉시 알림.

사용법:
  python scripts/performance_report.py              # 매시간 모니터링 (retrospective + 알림)
  python scripts/performance_report.py --weekly      # 주간 종합 리포트
  python scripts/performance_report.py --backtest    # 최근 30일 백테스트
"""

import hide_console
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
KST = timezone(timedelta(hours=9))
PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")
if not os.path.exists(PYTHON):
    PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python")


def supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def supabase_get(table, params=None):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**supabase_headers(), "Prefer": "count=exact"},
        params=params or {},
        timeout=15,
    )
    count = 0
    cr = r.headers.get("content-range", "")
    if "/" in cr:
        try:
            count = int(cr.split("/")[1])
        except (ValueError, IndexError):
            pass
    return r.json() if r.ok else [], count


def send_telegram(message: str, parse_mode="MarkdownV2"):
    """텔레그램 알림 전송"""
    try:
        result = subprocess.run(
            [PYTHON, str(PROJECT_DIR / "scripts" / "notify_telegram.py"),
             "report", "Performance Report", message],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_DIR),
        )
        return result.returncode == 0
    except Exception:
        return False


def send_telegram_raw(text: str):
    """텔레그램 직접 전송 (escape 없이)"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_USER_ID", "")
    if not token or not chat_id:
        print(text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        print(text)


# ─── 매시간 모니터링 ───


def hourly_monitor():
    """매시간 실행: retrospective + 성능 저하 감지"""
    now = datetime.now(KST)
    print(f"[{now:%H:%M}] 매시간 모니터링 시작")

    # 1. retrospective 실행
    print("  retrospective 실행 중...")
    result = subprocess.run(
        [PYTHON, str(PROJECT_DIR / "scripts" / "retrospective.py")],
        capture_output=True, text=True, timeout=120,
        cwd=str(PROJECT_DIR),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    if result.stdout:
        print(f"  {result.stdout.strip()}")

    # 2. 최근 24시간 성과 체크
    cutoff = (now - timedelta(hours=24)).isoformat()
    decisions, _ = supabase_get("decisions", {
        "select": "decision,outcome_1h_pct,was_correct_1h",
        "created_at": f"gt.{cutoff}",
        "outcome_1h_pct": "not.is.null",
    })

    if not decisions:
        print("  최근 24시간 평가된 결정 없음")
        return

    # 3. 성능 저하 감지
    correct = sum(1 for d in decisions if d.get("was_correct_1h"))
    total = len(decisions)
    accuracy = correct / total if total > 0 else 0
    avg_outcome = np.mean([d["outcome_1h_pct"] for d in decisions if d.get("outcome_1h_pct") is not None])

    print(f"  24h 정확도: {accuracy:.0%} ({correct}/{total}) | 평균 1h 변동: {avg_outcome:+.2f}%")

    # 매수 결정만 필터
    buys = [d for d in decisions if d.get("decision") in ("매수", "buy")]
    if buys:
        buy_correct = sum(1 for d in buys if d.get("was_correct_1h"))
        buy_accuracy = buy_correct / len(buys) if buys else 0
        if buy_accuracy < 0.3 and len(buys) >= 3:
            alert = (
                f"<b>⚠️ 매수 정확도 저하 경고</b>\n\n"
                f"최근 24h 매수 정확도: {buy_accuracy:.0%} ({buy_correct}/{len(buys)})\n"
                f"임계값: 30% 미만\n"
                f"시간: {now:%Y-%m-%d %H:%M}"
            )
            send_telegram_raw(alert)
            print(f"  ⚠️ 매수 정확도 경고 발송: {buy_accuracy:.0%}")

    # 연속 손실 감지
    recent, _ = supabase_get("decisions", {
        "select": "decision,was_correct_1h",
        "order": "created_at.desc",
        "limit": "5",
        "decision": "in.(매수,buy)",
        "was_correct_1h": "not.is.null",
    })
    consecutive_wrong = 0
    for d in recent:
        if not d.get("was_correct_1h"):
            consecutive_wrong += 1
        else:
            break

    if consecutive_wrong >= 3:
        alert = (
            f"<b>🚨 연속 손실 경고</b>\n\n"
            f"최근 매수 {consecutive_wrong}건 연속 손실\n"
            f"전략 점검 권장\n"
            f"시간: {now:%Y-%m-%d %H:%M}"
        )
        send_telegram_raw(alert)
        print(f"  🚨 연속 {consecutive_wrong}회 손실 경고 발송")

    print(f"[{now:%H:%M}] 모니터링 완료")


# ─── 주간 리포트 ───


def weekly_report():
    """주간 종합 성과 리포트"""
    now = datetime.now(KST)
    week_ago = (now - timedelta(days=7)).isoformat()

    # 1. 주간 결정 통계
    decisions, total_decisions = supabase_get("decisions", {
        "select": "decision,confidence,outcome_1h_pct,outcome_4h_pct,outcome_24h_pct,was_correct_1h,was_correct_4h,was_correct_24h,source",
        "created_at": f"gt.{week_ago}",
        "limit": "500",
    })

    # 2. 포트폴리오 변화
    snapshots, _ = supabase_get("portfolio_snapshots", {
        "select": "total_value,total_krw,total_crypto_value,created_at",
        "created_at": f"gt.{week_ago}",
        "order": "created_at.asc",
        "limit": "100",
    })

    # 3. 결정 분석
    buy_decisions = [d for d in decisions if d.get("decision") in ("매수", "buy")]
    hold_decisions = [d for d in decisions if d.get("decision") in ("관망", "hold")]
    sell_decisions = [d for d in decisions if d.get("decision") in ("매도", "sell")]

    def calc_accuracy(items, window="1h"):
        key = f"was_correct_{window}"
        evaluated = [d for d in items if d.get(key) is not None]
        if not evaluated:
            return None, 0
        correct = sum(1 for d in evaluated if d.get(key))
        return correct / len(evaluated), len(evaluated)

    acc_1h, n_1h = calc_accuracy(decisions, "1h")
    acc_4h, n_4h = calc_accuracy(decisions, "4h")
    acc_24h, n_24h = calc_accuracy(decisions, "24h")

    buy_acc, buy_n = calc_accuracy(buy_decisions, "1h")
    hold_acc, hold_n = calc_accuracy(hold_decisions, "1h")

    # 4. 포트폴리오 수익률
    portfolio_return = None
    if len(snapshots) >= 2:
        first_val = snapshots[0].get("total_value", 0)
        last_val = snapshots[-1].get("total_value", 0)
        if first_val > 0:
            portfolio_return = (last_val - first_val) / first_val * 100

    # 5. 리포트 생성
    lines = [
        f"<b>📊 주간 성과 리포트</b>",
        f"<i>{(now - timedelta(days=7)):%m/%d} ~ {now:%m/%d}</i>",
        "",
        f"<b>📋 결정 통계</b>",
        f"  총 결정: {total_decisions}건",
        f"  매수: {len(buy_decisions)} | 관망: {len(hold_decisions)} | 매도: {len(sell_decisions)}",
        "",
        f"<b>🎯 정확도</b>",
    ]

    if acc_1h is not None:
        lines.append(f"  1h: {acc_1h:.0%} ({n_1h}건)")
    if acc_4h is not None:
        lines.append(f"  4h: {acc_4h:.0%} ({n_4h}건)")
    if acc_24h is not None:
        lines.append(f"  24h: {acc_24h:.0%} ({n_24h}건)")

    lines.append("")
    lines.append("<b>📈 유형별 정확도 (1h)</b>")
    if buy_acc is not None:
        emoji = "✅" if buy_acc >= 0.5 else "⚠️"
        lines.append(f"  {emoji} 매수: {buy_acc:.0%} ({buy_n}건)")
    if hold_acc is not None:
        emoji = "✅" if hold_acc >= 0.5 else "⚠️"
        lines.append(f"  {emoji} 관망: {hold_acc:.0%} ({hold_n}건)")

    if portfolio_return is not None:
        emoji = "📈" if portfolio_return >= 0 else "📉"
        lines.append("")
        lines.append(f"<b>{emoji} 포트폴리오</b>")
        lines.append(f"  주간 수익률: {portfolio_return:+.2f}%")
        lines.append(f"  시작: {snapshots[0].get('total_value', 0):,.0f}원")
        lines.append(f"  현재: {snapshots[-1].get('total_value', 0):,.0f}원")

    # 6. 놓친 기회 분석
    missed, _ = supabase_get("decisions", {
        "select": "current_price,outcome_24h_pct,reason",
        "created_at": f"gt.{week_ago}",
        "decision": "in.(관망,hold)",
        "outcome_24h_pct": "gt.1.5",
        "order": "outcome_24h_pct.desc",
        "limit": "3",
    })
    if missed:
        lines.append("")
        lines.append(f"<b>💡 놓친 기회 TOP {len(missed)}</b>")
        for m in missed:
            lines.append(f"  +{m.get('outcome_24h_pct', 0):.1f}% (24h) @ {m.get('current_price', 0):,}원")

    report_text = "\n".join(lines)
    send_telegram_raw(report_text)
    print(report_text)
    return report_text


# ─── 백테스트 ───


def backtest_30d():
    """최근 30일 에이전트 결정을 시장 실제와 비교"""
    now = datetime.now(KST)
    month_ago = (now - timedelta(days=30)).isoformat()

    print("=== 최근 30일 백테스트 ===\n")

    # 1. 모든 결정 조회
    decisions, total = supabase_get("decisions", {
        "select": "id,created_at,decision,confidence,current_price,outcome_1h_pct,outcome_4h_pct,outcome_24h_pct,was_correct_1h,was_correct_4h,was_correct_24h,source,reason",
        "created_at": f"gt.{month_ago}",
        "order": "created_at.asc",
        "limit": "500",
    })

    print(f"총 결정: {total}건\n")

    if not decisions:
        print("결정 데이터 없음")
        return

    # 2. 소스별 분석
    by_source = {}
    for d in decisions:
        src = d.get("source", "unknown")
        by_source.setdefault(src, []).append(d)

    for src, items in by_source.items():
        evaluated_1h = [d for d in items if d.get("was_correct_1h") is not None]
        evaluated_4h = [d for d in items if d.get("was_correct_4h") is not None]
        evaluated_24h = [d for d in items if d.get("was_correct_24h") is not None]

        print(f"[{src}] 총 {len(items)}건")

        if evaluated_1h:
            acc = sum(1 for d in evaluated_1h if d["was_correct_1h"]) / len(evaluated_1h)
            avg = np.mean([d["outcome_1h_pct"] for d in evaluated_1h])
            print(f"  1h 정확도: {acc:.0%} ({len(evaluated_1h)}건) | 평균 변동: {avg:+.3f}%")

        if evaluated_4h:
            acc = sum(1 for d in evaluated_4h if d["was_correct_4h"]) / len(evaluated_4h)
            avg = np.mean([d["outcome_4h_pct"] for d in evaluated_4h])
            print(f"  4h 정확도: {acc:.0%} ({len(evaluated_4h)}건) | 평균 변동: {avg:+.3f}%")

        if evaluated_24h:
            acc = sum(1 for d in evaluated_24h if d["was_correct_24h"]) / len(evaluated_24h)
            avg = np.mean([d["outcome_24h_pct"] for d in evaluated_24h])
            print(f"  24h 정확도: {acc:.0%} ({len(evaluated_24h)}건) | 평균 변동: {avg:+.3f}%")

        # 결정 유형별
        by_decision = {}
        for d in items:
            dec = d.get("decision", "?")
            by_decision.setdefault(dec, []).append(d)

        for dec, dec_items in by_decision.items():
            e1 = [d for d in dec_items if d.get("was_correct_1h") is not None]
            if e1:
                acc = sum(1 for d in e1 if d["was_correct_1h"]) / len(e1)
                print(f"    {dec}: {acc:.0%} ({len(e1)}건)")
        print()

    # 3. 시뮬레이션: 에이전트를 따랐을 때 vs 관망
    print("=== 시뮬레이션: 에이전트 추종 vs 관망 ===\n")

    buy_decisions = [d for d in decisions
                     if d.get("decision") in ("매수", "buy")
                     and d.get("outcome_4h_pct") is not None]
    sell_decisions = [d for d in decisions
                      if d.get("decision") in ("매도", "sell")
                      and d.get("outcome_4h_pct") is not None]
    hold_missed = [d for d in decisions
                   if d.get("decision") in ("관망", "hold")
                   and d.get("outcome_24h_pct") is not None
                   and d["outcome_24h_pct"] > 1.0]

    if buy_decisions:
        avg_buy_4h = np.mean([d["outcome_4h_pct"] for d in buy_decisions])
        win_rate = sum(1 for d in buy_decisions if d["outcome_4h_pct"] > 0) / len(buy_decisions)
        print(f"매수 결정 {len(buy_decisions)}건:")
        print(f"  4h 승률: {win_rate:.0%}")
        print(f"  4h 평균 수익: {avg_buy_4h:+.3f}%")
        print(f"  추정 누적 수익: {sum(d['outcome_4h_pct'] for d in buy_decisions):+.3f}%")

    if hold_missed:
        avg_missed = np.mean([d["outcome_24h_pct"] for d in hold_missed])
        print(f"\n놓친 기회 (관망 후 24h +1% 이상): {len(hold_missed)}건")
        print(f"  평균 놓친 수익: +{avg_missed:.2f}%")

    # 4. 포트폴리오 추이
    snapshots, snap_count = supabase_get("portfolio_snapshots", {
        "select": "total_value,created_at",
        "created_at": f"gt.{month_ago}",
        "order": "created_at.asc",
        "limit": "200",
    })

    if len(snapshots) >= 2:
        values = [s["total_value"] for s in snapshots if s.get("total_value")]
        if values:
            start_val = values[0]
            end_val = values[-1]
            max_val = max(values)
            min_val = min(values)
            drawdown = (max_val - min_val) / max_val * 100 if max_val > 0 else 0
            total_return = (end_val - start_val) / start_val * 100

            print(f"\n=== 포트폴리오 (30일) ===")
            print(f"  시작: {start_val:,.0f}원")
            print(f"  현재: {end_val:,.0f}원")
            print(f"  수익률: {total_return:+.2f}%")
            print(f"  최대 낙폭: {drawdown:.2f}%")
            print(f"  스냅샷: {len(values)}개")

    # 5. 텔레그램 요약
    summary = (
        f"<b>📊 30일 백테스트 완료</b>\n\n"
        f"총 결정: {total}건\n"
    )
    if buy_decisions:
        win_rate = sum(1 for d in buy_decisions if d["outcome_4h_pct"] > 0) / len(buy_decisions)
        summary += f"매수 승률(4h): {win_rate:.0%} ({len(buy_decisions)}건)\n"
    if hold_missed:
        summary += f"놓친 기회: {len(hold_missed)}건 (평균 +{avg_missed:.1f}%)\n"
    if len(snapshots) >= 2 and values:
        summary += f"포트폴리오: {total_return:+.2f}%\n"

    send_telegram_raw(summary)


def main():
    parser = argparse.ArgumentParser(description="성과 모니터링")
    parser.add_argument("--weekly", action="store_true", help="주간 리포트")
    parser.add_argument("--backtest", action="store_true", help="30일 백테스트")
    parser.add_argument("--daemon", action="store_true", help="매시간 데몬 모드")
    args = parser.parse_args()

    if args.weekly:
        weekly_report()
    elif args.backtest:
        backtest_30d()
    elif args.daemon:
        print("성과 모니터링 데몬 시작 (1시간 간격)")
        while True:
            try:
                hourly_monitor()
            except Exception as e:
                print(f"모니터링 에러: {e}")
            time.sleep(3600)
    else:
        hourly_monitor()


if __name__ == "__main__":
    main()
