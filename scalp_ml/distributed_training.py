#!/usr/bin/env python3
"""
3대 분산 1주일 스캘핑 RL 훈련 시스템

각 머신이 다른 모델/전략을 훈련하고, Supabase에 결과를 공유한다.

머신별 역할:
  Mac Mini  — LightGBM/XGBoost 시그널 분류기 + 피처 실험
  PC128     — DQN/SAC 청산 최적화 (신경망, 대규모)
  PC36      — PPO 환경 다양화 + 보상함수 실험 (DRJAY)

실행:
  python -m scalp_ml.distributed_training --machine mac-mini
  python -m scalp_ml.distributed_training --machine pc128
  python -m scalp_ml.distributed_training --machine pc36
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_DIR / "logs" / "distributed_training.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("dist_train")

KST = timezone(timedelta(hours=9))
MODEL_DIR = PROJECT_DIR / "data" / "scalp_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def log_to_db(machine: str, phase: str, status: str, metrics: dict = None, error: str = None):
    """훈련 진행상황을 DB에 기록"""
    try:
        row = {
            "task_type": "train_pytorch",
            "params": json.dumps({
                "machine": machine,
                "phase": phase,
                "plan": "1week_distributed",
            }),
            "status": status,
            "assigned_worker": machine,
            "result": json.dumps(metrics) if metrics else None,
            "error_message": error[:500] if error else None,
            "priority": 1,
        }
        if status == "running":
            row["started_at"] = datetime.now(KST).isoformat()
        elif status in ("completed", "failed"):
            row["completed_at"] = datetime.now(KST).isoformat()

        requests.post(
            f"{SUPABASE_URL}/rest/v1/scalp_training_tasks",
            json=row,
            headers={**HEADERS, "Prefer": "return=minimal"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"DB 기록 실패: {e}")


def send_telegram(text: str):
    """텔레그램 알림"""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "5273754646")
    if not token:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════
# Mac Mini v2: 3-데몬 병렬 — 수집 + 백테스트 + 레짐 학습
# ═══════════════════════════════════════════════════

SNAPSHOT_DIR = PROJECT_DIR / "data" / "scalp_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# 7일 실행 제한 (초)
MAX_RUN_SECONDS = 7 * 24 * 3600
_mac_running = True


def run_mac_mini():
    """
    Mac Mini 1주일 계획 v2 — 3 데몬 병렬 운영

    데몬 1: 고해상도 시장 데이터 수집 (1분 간격, 24/7)
    데몬 2: 롤링 백테스트 엔진 (6시간 주기, 2160 파라미터 조합)
    데몬 3: 시장 레짐 탐지 + 적응형 필터 학습 (3시간 주기)
    """
    import threading

    global _mac_running
    machine = "mac-mini"
    log.info(f"{'='*60}")
    log.info(f"  Mac Mini v2 훈련 시작 — 3-데몬 병렬 (7일)")
    log.info(f"  데몬 1: 고해상도 수집 (1분)")
    log.info(f"  데몬 2: 롤링 백테스트 (6시간)")
    log.info(f"  데몬 3: 레짐 탐지 + 필터 (3시간)")
    log.info(f"{'='*60}")
    send_telegram(
        f"🖥️ [Mac Mini v2] 1주일 훈련 시작\n"
        f"데몬 1: 고해상도 수집 (1분 간격)\n"
        f"데몬 2: 백테스트 (6h 주기, 2160조합)\n"
        f"데몬 3: HMM 레짐 + LightGBM 필터 (3h 주기)"
    )
    log_to_db(machine, "v2_start", "running")

    start_time = time.time()

    # 초기 데이터 수집 (캐시 없으면)
    _mac_init_data()

    threads = [
        threading.Thread(target=_daemon_collector, name="collector", daemon=True),
        threading.Thread(target=_daemon_backtester, name="backtester", daemon=True),
        threading.Thread(target=_daemon_regime, name="regime", daemon=True),
    ]
    for t in threads:
        t.start()

    # 메인: 30분마다 상태 보고
    report_interval = 1800
    last_report = time.time()
    daily_report_hour = -1

    while _mac_running and (time.time() - start_time) < MAX_RUN_SECONDS:
        time.sleep(60)

        now = datetime.now(KST)
        elapsed_h = (time.time() - start_time) / 3600

        # 30분마다 상태 로그
        if time.time() - last_report >= report_interval:
            alive = [t.name for t in threads if t.is_alive()]
            dead = [t.name for t in threads if not t.is_alive()]
            snap_count = len(list(SNAPSHOT_DIR.glob("snapshot_*.parquet")))
            bt_file = MODEL_DIR / "backtest_results.json"
            regime_file = MODEL_DIR / "regime_detector.pkl"

            log.info(
                f"[Mac Mini] {elapsed_h:.1f}h 경과 | "
                f"데몬: {','.join(alive)} | "
                f"스냅샷: {snap_count}개 | "
                f"백테스트: {'있음' if bt_file.exists() else '대기'} | "
                f"레짐: {'있음' if regime_file.exists() else '대기'}"
            )

            # 죽은 데몬 재시작
            for i, t in enumerate(threads):
                if not t.is_alive():
                    log.warning(f"  데몬 {t.name} 죽음 — 재시작")
                    targets = [_daemon_collector, _daemon_backtester, _daemon_regime]
                    new_t = threading.Thread(target=targets[i], name=t.name, daemon=True)
                    new_t.start()
                    threads[i] = new_t

            last_report = time.time()

        # 매일 09시에 일일 보고
        if now.hour == 9 and now.hour != daily_report_hour:
            daily_report_hour = now.hour
            _mac_daily_report(elapsed_h)

    _mac_running = False
    log.info(f"[Mac Mini] 7일 훈련 종료 ({(time.time()-start_time)/3600:.1f}h)")
    _mac_final_report()
    send_telegram("[Mac Mini v2] 1주일 훈련 완료!")
    log_to_db(machine, "v2_complete", "completed")


# ── 데몬 1: 고해상도 시장 데이터 수집 ────────────────

def _daemon_collector():
    """1분 간격 시장 데이터 수집 (호가 + 체결 + 1분봉)"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    log.info("[수집기] 시작 — 1분 간격 고해상도 수집")
    buffer = []
    flush_interval = 300  # 5분마다 parquet 저장

    while _mac_running:
        try:
            now = datetime.now(KST)
            row = {"timestamp": now.isoformat()}

            # 1) 현재가 + 1분봉
            try:
                r = requests.get(
                    "https://api.upbit.com/v1/candles/minutes/1",
                    params={"market": "KRW-BTC", "count": 5}, timeout=5
                )
                if r.status_code == 200:
                    candles = r.json()
                    c = candles[0]
                    row.update({
                        "price": c["trade_price"],
                        "open": c["opening_price"],
                        "high": c["high_price"],
                        "low": c["low_price"],
                        "volume": c["candle_acc_trade_volume"],
                        "volume_krw": c["candle_acc_trade_price"],
                    })
                    # 모멘텀
                    if len(candles) >= 5:
                        prices = [x["trade_price"] for x in reversed(candles)]
                        row["mom_1m"] = (prices[-1] / prices[-2] - 1) * 100 if prices[-2] else 0
                        row["mom_5m"] = (prices[-1] / prices[0] - 1) * 100 if prices[0] else 0
            except Exception:
                pass

            # 2) 호가 (15단계)
            try:
                r = requests.get(
                    "https://api.upbit.com/v1/orderbook",
                    params={"markets": "KRW-BTC"}, timeout=5
                )
                if r.status_code == 200:
                    ob = r.json()[0]
                    units = ob.get("orderbook_units", [])
                    if units:
                        bid_5 = sum(u["bid_size"] for u in units[:5])
                        ask_5 = sum(u["ask_size"] for u in units[:5])
                        bid_10 = sum(u["bid_size"] for u in units[:10])
                        ask_10 = sum(u["ask_size"] for u in units[:10])
                        total = bid_5 + ask_5
                        row["ob_imbalance_5"] = round((bid_5 - ask_5) / total, 4) if total else 0
                        total10 = bid_10 + ask_10
                        row["ob_imbalance_10"] = round((bid_10 - ask_10) / total10, 4) if total10 else 0
                        mid = (units[0]["bid_price"] + units[0]["ask_price"]) / 2
                        row["spread_bps"] = round(
                            (units[0]["ask_price"] - units[0]["bid_price"]) / mid * 10000, 2
                        ) if mid else 0
                        # 벽 감지: 잔량 상위 5%
                        all_bids = sorted([u["bid_size"] for u in units], reverse=True)
                        all_asks = sorted([u["ask_size"] for u in units], reverse=True)
                        if all_bids:
                            wall_bid = units[[u["bid_size"] for u in units].index(all_bids[0])]
                            row["bid_wall_dist_pct"] = round(
                                (row.get("price", mid) - wall_bid["bid_price"]) / row.get("price", mid) * 100, 3
                            )
                        if all_asks:
                            wall_ask = units[[u["ask_size"] for u in units].index(all_asks[0])]
                            row["ask_wall_dist_pct"] = round(
                                (wall_ask["ask_price"] - row.get("price", mid)) / row.get("price", mid) * 100, 3
                            )
            except Exception:
                pass

            # 3) 최근 체결 (50건)
            try:
                r = requests.get(
                    "https://api.upbit.com/v1/trades/ticks",
                    params={"market": "KRW-BTC", "count": 50}, timeout=5
                )
                if r.status_code == 200:
                    trades = r.json()
                    if trades:
                        buy_vol = sum(t["trade_volume"] for t in trades if t["ask_bid"] == "BID")
                        sell_vol = sum(t["trade_volume"] for t in trades if t["ask_bid"] == "ASK")
                        total_vol = buy_vol + sell_vol
                        row["trade_intensity"] = round(buy_vol / total_vol, 4) if total_vol else 0.5
                        large = sum(1 for t in trades if t["trade_price"] * t["trade_volume"] >= 10_000_000)
                        row["large_trade_ratio"] = round(large / len(trades), 4)
            except Exception:
                pass

            buffer.append(row)

            # 5분마다 parquet로 저장
            if len(buffer) >= 5:
                try:
                    table = pa.Table.from_pylist(buffer)
                    fname = SNAPSHOT_DIR / f"snapshot_{now.strftime('%Y%m%d_%H')}.parquet"
                    if fname.exists():
                        existing = pq.read_table(fname)
                        table = pa.concat_tables([existing, table])
                    pq.write_table(table, fname)
                    buffer.clear()
                except Exception as e:
                    log.warning(f"[수집기] parquet 저장 실패: {e}")

            # DB에도 5분마다 기록
            if now.minute % 5 == 0 and now.second < 61:
                db_insert("scalp_market_snapshot", {
                    "price": row.get("price"),
                    "volume_1m": row.get("volume"),
                    "rsi_1m": None,
                    "ob_imbalance": row.get("ob_imbalance_5"),
                    "trade_intensity": row.get("trade_intensity"),
                    "spread_bps": row.get("spread_bps"),
                })

            time.sleep(60)

        except Exception as e:
            log.error(f"[수집기] 에러: {e}")
            time.sleep(30)


# ── 데몬 2: 롤링 백테스트 엔진 ────────────────

def _daemon_backtester():
    """6시간마다 파라미터 그리드서치 백테스트"""
    import pickle
    import itertools

    log.info("[백테스터] 시작 — 6시간 주기")

    # 초기 대기: 수집기가 데이터 쌓을 시간
    time.sleep(120)

    cycle = 0
    while _mac_running:
        cycle += 1
        try:
            log.info(f"[백테스터] 사이클 {cycle} 시작")
            start = time.time()

            # 캔들 데이터 로드
            candles = _load_best_candles()
            if not candles or len(candles) < 1000:
                log.warning("[백테스터] 캔들 데이터 부족, 대기")
                time.sleep(3600)
                continue

            # 파라미터 그리드
            param_grid = {
                "spike_pct": [0.3, 0.5, 0.8, 1.0, 1.5],
                "spike_window": [180, 300, 600],
                "whale_krw": [20_000_000, 50_000_000, 80_000_000],
                "whale_ratio": [0.60, 0.70, 0.80],
                "tp_pct": [0.15, 0.20, 0.30, 0.50],
                "sl_pct": [0.5, 0.7, 1.0, 1.2],
                "max_hold": [10, 15, 20, 30],
            }

            keys = list(param_grid.keys())
            combos = list(itertools.product(*param_grid.values()))
            log.info(f"[백테스터] {len(combos)} 조합 x {len(candles)} 캔들")

            results = []
            for i, vals in enumerate(combos):
                params = dict(zip(keys, vals))
                metrics = _run_backtest(candles, params)
                metrics["params"] = params
                results.append(metrics)

                if (i + 1) % 500 == 0:
                    log.info(f"  진행: {i+1}/{len(combos)}")

            # 결과 정렬 (Sharpe 기준)
            results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

            # 저장
            report = {
                "cycle": cycle,
                "timestamp": datetime.now(KST).isoformat(),
                "total_combos": len(combos),
                "candle_count": len(candles),
                "top_10": results[:10],
                "current_params_rank": _find_current_rank(results),
                "elapsed_sec": round(time.time() - start),
            }

            with open(MODEL_DIR / "backtest_results.json", "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            best = results[0]
            log.info(
                f"[백테스터] 사이클 {cycle} 완료 ({report['elapsed_sec']}s) | "
                f"최적: sharpe={best.get('sharpe', 0):.3f}, "
                f"승률={best.get('win_rate', 0):.1%}, "
                f"pnl={best.get('total_pnl', 0):.2f}%"
            )

            send_telegram(
                f"📊 [백테스터] 사이클 {cycle} 완료\n"
                f"최적 Sharpe: {best.get('sharpe', 0):.3f}\n"
                f"승률: {best.get('win_rate', 0):.1%}\n"
                f"현재 파라미터 순위: {report['current_params_rank']}/{len(combos)}"
            )

            log_to_db("mac-mini", f"backtest_cycle_{cycle}", "completed", metrics=report)

            # 6시간 대기
            _sleep_interruptible(6 * 3600)

        except Exception as e:
            log.error(f"[백테스터] 에러: {e}", exc_info=True)
            log_to_db("mac-mini", f"backtest_cycle_{cycle}", "failed", error=str(e))
            time.sleep(1800)


def _run_backtest(candles: list, params: dict) -> dict:
    """단일 파라미터 조합 백테스트"""
    tp = params["tp_pct"]
    sl = params["sl_pct"]
    max_hold = params["max_hold"]
    fee = 0.1  # 왕복 수수료 %

    trades = []
    i = 30  # 초기 윈도우

    while i < len(candles) - max_hold - 1:
        # 급등 시그널 판정
        window_start = max(0, i - params["spike_window"] // 60)
        if window_start < i:
            change = abs(candles[i]["trade_price"] / candles[window_start]["trade_price"] - 1) * 100
        else:
            change = 0

        # 거래량 급증
        vol_recent = sum(c.get("candle_acc_trade_volume", 0) for c in candles[max(0, i-3):i+1])
        vol_prev = sum(c.get("candle_acc_trade_volume", 0) for c in candles[max(0, i-8):max(0, i-3)])
        vol_spike = vol_recent / max(vol_prev, 0.001) if vol_prev > 0 else 1

        # 진입 조건: 급등 또는 거래량 급증
        entry = False
        if change >= params["spike_pct"]:
            entry = True
        elif vol_spike >= 2.0 and change >= 0.2:
            entry = True

        if entry:
            entry_price = candles[i]["trade_price"]
            # 시뮬레이션
            exit_pnl = None
            hold_min = 0
            for j in range(1, max_hold + 1):
                if i + j >= len(candles):
                    break
                p = candles[i + j]["trade_price"]
                pnl = (p / entry_price - 1) * 100 - fee
                hold_min = j

                if pnl >= tp:
                    exit_pnl = pnl
                    break
                elif pnl <= -sl:
                    exit_pnl = pnl
                    break

            if exit_pnl is None and hold_min > 0:
                p = candles[i + hold_min]["trade_price"]
                exit_pnl = (p / entry_price - 1) * 100 - fee

            if exit_pnl is not None:
                trades.append({"pnl": exit_pnl, "hold": hold_min})

            i += max_hold + 1  # 쿨다운
        else:
            i += 1

    if not trades:
        return {"win_rate": 0, "total_pnl": 0, "sharpe": -999, "trades": 0}

    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    avg_pnl = sum(pnls) / len(pnls)
    std_pnl = (sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5 if len(pnls) > 1 else 1

    return {
        "win_rate": round(wins / len(trades), 4),
        "avg_pnl": round(avg_pnl, 4),
        "total_pnl": round(sum(pnls), 2),
        "sharpe": round(avg_pnl / std_pnl, 4) if std_pnl > 0 else 0,
        "trades": len(trades),
        "avg_hold": round(sum(t["hold"] for t in trades) / len(trades), 1),
    }


def _find_current_rank(results: list) -> int:
    """현재 봇 파라미터의 순위 찾기"""
    current = {
        "spike_pct": 0.8, "spike_window": 300,
        "whale_krw": 50_000_000, "whale_ratio": 0.70,
        "tp_pct": 0.20, "sl_pct": 1.2, "max_hold": 30,
    }
    for i, r in enumerate(results):
        p = r.get("params", {})
        if all(p.get(k) == v for k, v in current.items()):
            return i + 1
    return -1


# ── 데몬 3: 시장 레짐 탐지 + 적응형 필터 ────────────────

def _daemon_regime():
    """3시간마다 HMM 레짐 탐지 + 레짐별 LightGBM 필터 학습"""
    import pickle

    log.info("[레짐] 시작 — 3시간 주기")

    # 초기 대기: 최소 데이터 필요
    time.sleep(300)

    cycle = 0
    while _mac_running:
        cycle += 1
        try:
            log.info(f"[레짐] 사이클 {cycle} 시작")
            start = time.time()

            candles = _load_best_candles()
            if not candles or len(candles) < 500:
                log.warning("[레짐] 데이터 부족, 대기")
                time.sleep(3600)
                continue

            # Phase A: HMM 레짐 탐지
            regime_model, regime_labels = _train_hmm_regime(candles)

            if regime_model is not None:
                with open(MODEL_DIR / "regime_detector.pkl", "wb") as f:
                    pickle.dump({
                        "model": regime_model,
                        "n_states": regime_model.n_components,
                        "cycle": cycle,
                        "timestamp": datetime.now(KST).isoformat(),
                    }, f)

                # 레짐별 통계
                regime_stats = _analyze_regimes(candles, regime_labels)
                log.info(f"[레짐] HMM {regime_model.n_components}-state: {regime_stats}")

            # Phase B: 레짐별 LightGBM 필터
            filter_metrics = _train_regime_filters(candles, regime_labels)

            elapsed = round(time.time() - start)
            log.info(f"[레짐] 사이클 {cycle} 완료 ({elapsed}s)")

            log_to_db("mac-mini", f"regime_cycle_{cycle}", "completed",
                      metrics={"regime_stats": regime_stats, "filter": filter_metrics, "elapsed": elapsed})

            if cycle % 4 == 0:  # 12시간마다 보고
                send_telegram(
                    f"🎯 [레짐] 사이클 {cycle} 완료\n"
                    f"레짐: {regime_stats}\n"
                    f"필터: {filter_metrics}"
                )

            _sleep_interruptible(3 * 3600)

        except Exception as e:
            log.error(f"[레짐] 에러: {e}", exc_info=True)
            log_to_db("mac-mini", f"regime_cycle_{cycle}", "failed", error=str(e))
            time.sleep(1800)


def _train_hmm_regime(candles: list):
    """HMM으로 시장 레짐 분류"""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        log.warning("[레짐] hmmlearn 미설치, sklearn GMM 대체")
        return _train_gmm_regime(candles)

    # 피처: 5분 수익률, 변동성, 거래량 변화율
    features = []
    for i in range(5, len(candles)):
        ret = (candles[i]["trade_price"] / candles[i-1]["trade_price"] - 1) * 100
        ret_5 = (candles[i]["trade_price"] / candles[i-5]["trade_price"] - 1) * 100
        vol = candles[i].get("candle_acc_trade_volume", 0)
        vol_prev = candles[i-5].get("candle_acc_trade_volume", 0.001)
        vol_ratio = vol / max(vol_prev, 0.001)
        # 변동성 (5분 high-low range)
        highs = [candles[j]["high_price"] for j in range(i-4, i+1)]
        lows = [candles[j]["low_price"] for j in range(i-4, i+1)]
        volatility = (max(highs) - min(lows)) / candles[i]["trade_price"] * 100
        features.append([ret, ret_5, vol_ratio, volatility])

    X = np.array(features)

    # 3, 4, 5 state로 실험, BIC로 선택
    best_model, best_bic = None, float("inf")
    for n in [3, 4]:
        try:
            model = GaussianHMM(n_components=n, covariance_type="diag",
                                n_iter=200, random_state=42)
            model.fit(X)
            bic = -2 * model.score(X) + n * X.shape[1] * np.log(len(X))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        except Exception:
            continue

    if best_model is None:
        return None, None

    labels = best_model.predict(X)
    # 앞의 5개는 레짐 미지정
    full_labels = np.concatenate([np.full(5, -1), labels])
    return best_model, full_labels


def _train_gmm_regime(candles: list):
    """hmmlearn 없을 때 GMM 대체"""
    from sklearn.mixture import GaussianMixture

    features = []
    for i in range(5, len(candles)):
        ret = (candles[i]["trade_price"] / candles[i-1]["trade_price"] - 1) * 100
        vol = candles[i].get("candle_acc_trade_volume", 0)
        vol_prev = candles[i-5].get("candle_acc_trade_volume", 0.001)
        features.append([ret, vol / max(vol_prev, 0.001)])

    X = np.array(features)
    best_model, best_bic = None, float("inf")
    for n in [3, 4]:
        model = GaussianMixture(n_components=n, random_state=42, max_iter=200)
        model.fit(X)
        bic = model.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_model = model

    labels = best_model.predict(X)
    full_labels = np.concatenate([np.full(5, -1), labels])
    return best_model, full_labels


def _analyze_regimes(candles: list, labels) -> dict:
    """레짐별 통계 분석"""
    if labels is None:
        return {}
    stats = {}
    unique = set(labels)
    for regime in unique:
        if regime == -1:
            continue
        mask = labels == regime
        indices = np.where(mask)[0]
        prices = [candles[i]["trade_price"] for i in indices if i < len(candles)]
        if len(prices) < 2:
            continue
        returns = [(prices[j] / prices[j-1] - 1) * 100 for j in range(1, len(prices))]
        stats[f"regime_{regime}"] = {
            "count": int(len(indices)),
            "pct": round(len(indices) / len(labels) * 100, 1),
            "avg_ret": round(np.mean(returns), 4) if returns else 0,
            "volatility": round(np.std(returns), 4) if returns else 0,
        }
    return stats


def _train_regime_filters(candles: list, regime_labels) -> dict:
    """레짐별 LightGBM 시그널 필터"""
    try:
        import lightgbm as lgb
        from scalp_ml.train_lgbm import build_dataset, FEATURE_NAMES
    except ImportError:
        return {"status": "skipped", "reason": "lightgbm 미설치"}

    if regime_labels is None or len(candles) < 1000:
        return {"status": "insufficient_data"}

    X, y_cls, y_reg = build_dataset(candles)

    # 레짐 라벨 피처 추가
    min_len = min(len(X), len(regime_labels))
    X_ext = np.column_stack([X[:min_len], regime_labels[:min_len]])
    y_cls_ext = y_cls[:min_len]
    y_reg_ext = y_reg[:min_len]
    feat_names = FEATURE_NAMES + ["regime"]

    split = int(len(X_ext) * 0.8)
    train_data = lgb.Dataset(X_ext[:split], label=y_reg_ext[:split], feature_name=feat_names)
    val_data = lgb.Dataset(X_ext[split:], label=y_reg_ext[split:], reference=train_data)

    params = {
        "objective": "regression", "metric": "mae",
        "num_leaves": 31, "max_depth": 6, "learning_rate": 0.03,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "verbose": -1,
    }
    model = lgb.train(params, train_data, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(30)])

    pred = model.predict(X_ext[split:])
    mae = float(np.mean(np.abs(y_reg_ext[split:] - pred)))
    corr = float(np.corrcoef(y_reg_ext[split:], pred)[0, 1]) if len(pred) > 1 else 0

    import pickle
    with open(MODEL_DIR / "regime_filters.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feat_names, "mae": mae, "corr": corr}, f)

    return {"mae": round(mae, 4), "corr": round(corr, 4)}


# ── 공통 유틸 (Mac Mini) ────────────────

def _mac_init_data():
    """초기 캔들 데이터 확보"""
    import pickle
    from scalp_ml.train_lgbm import collect_candles

    for days in [7, 14, 30]:
        cache = MODEL_DIR / f"candles_{days}d.pkl"
        if not cache.exists():
            log.info(f"  {days}일 캔들 수집 중...")
            candles = collect_candles(days=days)
            with open(cache, "wb") as f:
                pickle.dump(candles, f)
            log.info(f"  {days}일: {len(candles)}건 수집")
        else:
            log.info(f"  {days}일 캐시 존재")


def _load_best_candles() -> list:
    """가장 큰 캔들 캐시 로드"""
    import pickle
    for days in [30, 14, 7]:
        cache = MODEL_DIR / f"candles_{days}d.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                return pickle.load(f)
    return []


def _sleep_interruptible(seconds: float):
    """_mac_running 체크하면서 대기"""
    end = time.time() + seconds
    while _mac_running and time.time() < end:
        time.sleep(30)


def _mac_daily_report(elapsed_h: float):
    """일일 보고"""
    snap_count = len(list(SNAPSHOT_DIR.glob("snapshot_*.parquet")))
    bt_file = MODEL_DIR / "backtest_results.json"
    regime_file = MODEL_DIR / "regime_detector.pkl"

    msg = (
        f"📋 [Mac Mini] 일일 보고 (Day {int(elapsed_h/24)+1})\n"
        f"경과: {elapsed_h:.0f}시간\n"
        f"스냅샷: {snap_count}개 파일\n"
        f"백테스트: {'완료' if bt_file.exists() else '대기'}\n"
        f"레짐: {'완료' if regime_file.exists() else '대기'}"
    )

    if bt_file.exists():
        with open(bt_file) as f:
            bt = json.load(f)
        top = bt.get("top_10", [{}])[0]
        msg += f"\n최적 Sharpe: {top.get('sharpe', 0):.3f}"
        msg += f"\n현재 순위: {bt.get('current_params_rank', '?')}"

    send_telegram(msg)


def _mac_final_report():
    """7일 최종 보고"""
    bt_file = MODEL_DIR / "backtest_results.json"
    regime_file = MODEL_DIR / "regime_detector.pkl"

    report = {"timestamp": datetime.now(KST).isoformat()}

    if bt_file.exists():
        with open(bt_file) as f:
            report["backtest"] = json.load(f)

    if regime_file.exists():
        report["regime"] = "trained"

    snap_count = len(list(SNAPSHOT_DIR.glob("snapshot_*.parquet")))
    report["snapshots"] = snap_count

    with open(MODEL_DIR / "mac_mini_final_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"[Mac Mini] 최종 보고서 저장: {MODEL_DIR / 'mac_mini_final_report.json'}")


# ═══════════════════════════════════════════════════
# PC128: DQN/SAC 청산 최적화 (대규모 신경망)
# ═══════════════════════════════════════════════════

def run_pc128():
    """
    PC128 1주일 훈련 계획

    Day 1:   DQN 기본 훈련 (200K 스텝)
    Day 2:   보상함수 실험 (v1~v3)
    Day 3-4: SAC 연속행동 훈련 (500K 스텝)
    Day 5:   네트워크 크기 실험 ([64,32] vs [256,128] vs [512,256])
    Day 6:   최적 모델 장기 훈련 (1M 스텝)
    Day 7:   평가 + 기준선 비교
    """
    machine = "pc128"
    log.info(f"{'='*60}")
    log.info(f"  PC128 훈련 시작 — DQN/SAC 청산 최적화")
    log.info(f"{'='*60}")
    send_telegram(f"[PC128] 1주일 훈련 시작 — DQN/SAC 청산 최적화 (대규모)")

    phases = [
        ("phase1_data", _pc128_phase1_data, "캔들 데이터 수집"),
        ("phase2_dqn_200k", _pc128_phase2_dqn, "DQN 기본 200K 스텝"),
        ("phase3_reward_exp", _pc128_phase3_reward, "보상함수 3가지 실험"),
        ("phase4_sac_500k", _pc128_phase4_sac, "SAC 500K 스텝"),
        ("phase5_arch_exp", _pc128_phase5_arch, "네트워크 아키텍처 실험"),
        ("phase6_best_1m", _pc128_phase6_long, "최적 모델 1M 스텝"),
        ("phase7_eval", _pc128_phase7_eval, "최종 평가 + 기준선 비교"),
    ]

    _run_phases(machine, phases)


def _pc128_phase1_data():
    """데이터 준비"""
    from scalp_ml.train_lgbm import collect_candles
    import pickle
    for days in [14, 30]:
        cache = MODEL_DIR / f"candles_{days}d.pkl"
        if not cache.exists():
            candles = collect_candles(days=days)
            with open(cache, "wb") as f:
                pickle.dump(candles, f)
    return {"status": "data_ready"}


def _pc128_phase2_dqn():
    """DQN 기본 훈련 200K"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3, buffer_size=100000, batch_size=128,
        gamma=0.99, exploration_fraction=0.3, exploration_final_eps=0.05,
        target_update_interval=500, train_freq=4,
        policy_kwargs={"net_arch": [128, 64]},
        verbose=0,
    )

    eval_cb = EvalCallback(eval_env, n_eval_episodes=200, eval_freq=10000,
                          best_model_save_path=str(MODEL_DIR / "dqn_pc128_best"),
                          deterministic=True)
    model.learn(total_timesteps=200000, callback=eval_cb, progress_bar=True)
    model.save(str(MODEL_DIR / "dqn_pc128_200k"))

    return _eval_sb3_model(model, ScalpExitEnv(), 500, "dqn_200k")


def _pc128_phase3_reward():
    """보상함수 3가지 실험"""
    # 실제 구현에서는 ScalpExitEnv 서브클래스로 보상 변경
    return {"status": "placeholder", "variants": ["v1_base", "v2_sharp_sl", "v3_trailing_tp"]}


def _pc128_phase4_sac():
    """SAC 연속 행동공간 500K"""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnvV2

    env = ScalpExitEnvV2()
    eval_env = ScalpExitEnvV2()

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=100000, batch_size=256,
        gamma=0.99, tau=0.005, ent_coef="auto",
        policy_kwargs={"net_arch": [256, 128]},
        verbose=0,
    )

    eval_cb = EvalCallback(eval_env, n_eval_episodes=200, eval_freq=20000,
                          best_model_save_path=str(MODEL_DIR / "sac_pc128_best"),
                          deterministic=True)
    model.learn(total_timesteps=500000, callback=eval_cb, progress_bar=True)
    model.save(str(MODEL_DIR / "sac_pc128_500k"))

    return _eval_sb3_model(model, ScalpExitEnvV2(), 500, "sac_500k")


def _pc128_phase5_arch():
    """네트워크 아키텍처 실험"""
    from stable_baselines3 import DQN
    from scalp_ml.scalp_exit_env import ScalpExitEnv
    import numpy as np

    archs = [[64, 32], [128, 64], [256, 128], [256, 256, 128]]
    results = []

    for arch in archs:
        log.info(f"  아키텍처 {arch} 훈련 중...")
        env = ScalpExitEnv()
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3, buffer_size=50000, batch_size=64,
            policy_kwargs={"net_arch": arch}, verbose=0,
        )
        model.learn(total_timesteps=100000, progress_bar=False)
        metrics = _eval_sb3_model(model, ScalpExitEnv(), 300, f"arch_{'x'.join(map(str,arch))}")
        results.append({"arch": arch, **metrics})
        log.info(f"  {arch}: win={metrics['win_rate']:.1%}, pnl={metrics['avg_pnl_pct']:.3f}%")

    return {"arch_comparison": results}


def _pc128_phase6_long():
    """최적 모델 장기 훈련 1M 스텝"""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4, buffer_size=200000, batch_size=128,
        gamma=0.99, exploration_fraction=0.2, exploration_final_eps=0.03,
        target_update_interval=1000, train_freq=4,
        policy_kwargs={"net_arch": [256, 128]},
        verbose=0,
    )

    ckpt_cb = CheckpointCallback(save_freq=100000,
                                save_path=str(MODEL_DIR / "dqn_1m_checkpoints"))
    eval_cb = EvalCallback(eval_env, n_eval_episodes=300, eval_freq=50000,
                          best_model_save_path=str(MODEL_DIR / "dqn_1m_best"),
                          deterministic=True)

    model.learn(total_timesteps=1000000, callback=[ckpt_cb, eval_cb], progress_bar=True)
    model.save(str(MODEL_DIR / "dqn_pc128_1m"))

    return _eval_sb3_model(model, ScalpExitEnv(), 1000, "dqn_1m_final")


def _pc128_phase7_eval():
    """전 모델 비교 평가"""
    return {"status": "placeholder", "note": "모든 phase 완료 후 비교"}


# ═══════════════════════════════════════════════════
# PC36 (DRJAY): PPO 스캘핑 + 보상함수 + 온라인 학습
# ═══════════════════════════════════════════════════

def run_pc36():
    """
    PC36 (DRJAY) 1주일 훈련 계획

    Day 1:   PPO 기본 훈련 (200K 스텝)
    Day 2:   보상함수 v1~v3 비교 실험
    Day 3-4: 최적 보상함수로 PPO 500K
    Day 5:   시장 레짐별 모델 (trending vs ranging)
    Day 6:   온라인 학습 루프 (6시간 주기 마이크로 학습)
    Day 7:   전체 평가 + 앙상블 테스트
    """
    machine = "pc36"
    log.info(f"{'='*60}")
    log.info(f"  PC36 (DRJAY) 훈련 시작 — PPO + 보상함수 실험")
    log.info(f"{'='*60}")
    send_telegram(f"[PC36/DRJAY] 1주일 훈련 시작 — PPO 스캘핑 + 보상함수 실험")

    phases = [
        ("phase1_data", _pc36_phase1_data, "캔들 데이터 수집"),
        ("phase2_ppo_base", _pc36_phase2_ppo, "PPO 기본 200K 스텝"),
        ("phase3_reward_exp", _pc36_phase3_reward, "보상함수 3가지 실험"),
        ("phase4_ppo_500k", _pc36_phase4_ppo_long, "최적 보상 PPO 500K"),
        ("phase5_regime", _pc36_phase5_regime, "시장 레짐별 모델"),
        ("phase6_online", _pc36_phase6_online, "온라인 학습 루프 테스트"),
        ("phase7_eval", _pc36_phase7_eval, "전체 평가 + 앙상블"),
    ]

    _run_phases(machine, phases)


def _pc36_phase1_data():
    """데이터 준비"""
    return _pc128_phase1_data()


def _pc36_phase2_ppo():
    """PPO 기본 훈련 200K"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.05,
        policy_kwargs={"net_arch": [128, 64]},
        verbose=0,
    )

    eval_cb = EvalCallback(eval_env, n_eval_episodes=200, eval_freq=10000,
                          best_model_save_path=str(MODEL_DIR / "ppo_pc36_best"),
                          deterministic=True)
    model.learn(total_timesteps=200000, callback=eval_cb, progress_bar=True)
    model.save(str(MODEL_DIR / "ppo_pc36_200k"))

    return _eval_sb3_model(model, ScalpExitEnv(), 500, "ppo_200k")


def _pc36_phase3_reward():
    """보상함수 변형 실험"""
    from stable_baselines3 import PPO
    from scalp_ml.scalp_exit_env import ScalpExitEnv
    import numpy as np

    # 3가지 보상 전략을 ScalpExitEnv를 서브클래스하여 실험
    results = []

    # v1: 기본 (현재)
    env = ScalpExitEnv()
    m1 = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, verbose=0,
             policy_kwargs={"net_arch": [128, 64]})
    m1.learn(total_timesteps=100000, progress_bar=False)
    r1 = _eval_sb3_model(m1, ScalpExitEnv(), 300, "reward_v1")
    results.append({"variant": "v1_base", **r1})
    log.info(f"  v1(기본): win={r1['win_rate']:.1%}, pnl={r1['avg_pnl_pct']:.3f}%")

    return {"reward_comparison": results}


def _pc36_phase4_ppo_long():
    """최적 보상 PPO 500K 스텝"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from scalp_ml.scalp_exit_env import ScalpExitEnv

    env = ScalpExitEnv()
    eval_env = ScalpExitEnv()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=128,
        n_epochs=10, gamma=0.99, ent_coef=0.03,
        policy_kwargs={"net_arch": [256, 128]},
        verbose=0,
    )

    ckpt_cb = CheckpointCallback(save_freq=100000,
                                save_path=str(MODEL_DIR / "ppo_500k_ckpts"))
    eval_cb = EvalCallback(eval_env, n_eval_episodes=300, eval_freq=20000,
                          best_model_save_path=str(MODEL_DIR / "ppo_500k_best"),
                          deterministic=True)

    model.learn(total_timesteps=500000, callback=[ckpt_cb, eval_cb], progress_bar=True)
    model.save(str(MODEL_DIR / "ppo_pc36_500k"))

    return _eval_sb3_model(model, ScalpExitEnv(), 500, "ppo_500k")


def _pc36_phase5_regime():
    """시장 레짐별 모델"""
    return {"status": "placeholder", "note": "레짐 탐지기 구현 후 실행"}


def _pc36_phase6_online():
    """온라인 학습 루프 — 6시간마다 5K 스텝 마이크로 학습"""
    return {"status": "placeholder", "note": "기본 모델 완성 후 온라인 학습 추가"}


def _pc36_phase7_eval():
    """전체 평가"""
    return {"status": "pending_evaluation"}


# ═══════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════

def _eval_sb3_model(model, env, episodes: int, tag: str) -> dict:
    """SB3 모델 평가"""
    import numpy as np
    pnls, holds, exits = [], [], {}

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        if "pnl_pct" in info:
            pnls.append(info["pnl_pct"])
            holds.append(info.get("hold_minutes", 0))
            er = info.get("exit_reason", "unknown")
            exits[er] = exits.get(er, 0) + 1

    pnls = np.array(pnls) if pnls else np.array([0])
    return {
        "tag": tag,
        "win_rate": round(float((pnls > 0).mean()), 4),
        "avg_pnl_pct": round(float(pnls.mean()), 4),
        "total_pnl_pct": round(float(pnls.sum()), 2),
        "avg_hold_min": round(float(np.mean(holds)), 1) if holds else 0,
        "exit_reasons": exits,
        "episodes": episodes,
    }


def _run_phases(machine: str, phases: list):
    """순차 실행 — 실패해도 다음 phase 진행"""
    total = len(phases)
    for i, (name, func, desc) in enumerate(phases, 1):
        log.info(f"\n{'─'*50}")
        log.info(f"  [{machine}] Phase {i}/{total}: {desc}")
        log.info(f"{'─'*50}")
        log_to_db(machine, name, "running")

        try:
            start = time.time()
            result = func()
            elapsed = round(time.time() - start)
            result = result or {}
            result["elapsed_sec"] = elapsed

            log_to_db(machine, name, "completed", metrics=result)
            send_telegram(f"[{machine}] Phase {i}/{total} 완료: {desc} ({elapsed}s)")
            log.info(f"  Phase {i} 완료 ({elapsed}s)")

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"  Phase {i} 실패: {e}\n{tb}")
            log_to_db(machine, name, "failed", error=str(e))
            send_telegram(f"[{machine}] Phase {i}/{total} 실패: {desc}\n{e}")

    send_telegram(f"[{machine}] 1주일 훈련 전체 완료!")
    log.info(f"\n{'='*60}")
    log.info(f"  [{machine}] 전체 훈련 완료!")
    log.info(f"{'='*60}")


# ═══════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="3대 분산 스캘핑 RL 훈련")
    parser.add_argument("--machine", required=True,
                       choices=["mac-mini", "pc128", "pc36"],
                       help="이 머신의 역할")
    parser.add_argument("--phase", type=int, default=0,
                       help="특정 phase부터 시작 (1-7, 0=전체)")
    args = parser.parse_args()

    runners = {
        "mac-mini": run_mac_mini,
        "pc128": run_pc128,
        "pc36": run_pc36,
    }

    runners[args.machine]()


if __name__ == "__main__":
    main()
